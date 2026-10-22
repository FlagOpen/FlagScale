// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

function parseCommand(body) {
  const match = /^\/ci run ([a-z][a-z0-9_-]*)$/.exec(body.trim());
  return match ? match[1] : null;
}

function parseRequest(body) {
  const platform = parseCommand(body);
  if (platform) return { kind: 'run', platform };
  const match = /^\/ci rerun-job ([1-9][0-9]*)$/.exec(body.trim());
  if (!match || !Number.isSafeInteger(Number(match[1]))) return null;
  return { kind: 'rerun-job', jobId: Number(match[1]) };
}

function currentRun(run, pr, context, workflowId) {
  return run.workflow_id === workflowId && run.event === 'pull_request' &&
    run.repository?.full_name === `${context.repo.owner}/${context.repo.repo}` &&
    run.head_repository?.full_name === pr.head.repo.full_name &&
    run.head_branch === pr.head.ref && run.head_sha === pr.head.sha &&
    // Fork workflow runs sometimes have an empty pull_requests array. In that
    // case the exact repository, branch and immutable SHA are the provenance.
    (!run.pull_requests?.length || run.pull_requests.some(item => item.number === pr.number));
}

module.exports = async ({ github, context, core }) => {
  const request = parseRequest(context.payload.comment.body);
  if (!request) return;
  const number = context.payload.issue.number;
  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: number });
  if (pr.state !== 'open' || !pr.head.repo) {
    core.setFailed('The PR must be open and its head repository available.');
    return;
  }
  const commenter = context.payload.comment.user;
  const { data: permission } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...context.repo, username: commenter.login
  });
  const maintainer = ['admin', 'maintain', 'write'].includes(permission.permission);
  const author = commenter.type === 'User' && Number.isSafeInteger(commenter.id) &&
    commenter.id === pr.user?.id;
  if (!maintainer && !author) {
    core.notice('CI requests require repository write permission or ownership of this PR.');
    return;
  }
  const { data: workflow } = await github.rest.actions.getWorkflow({
    ...context.repo, workflow_id: 'ci.yml'
  });
  if (workflow.state !== 'active') {
    core.setFailed('CI must be enabled before requesting tests.');
    return;
  }
  // Authors can repeat an already-executed current-head CI; they cannot use a
  // privileged workflow_dispatch to bypass initial fork workflow approval.
  if (!maintainer && request.kind === 'run') {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...context.repo, workflow_id: workflow.id, event: 'pull_request',
      head_sha: pr.head.sha, per_page: 100
    });
    let executed = false;
    for (const run of runs.filter(run => currentRun(run, pr, context, workflow.id) &&
        run.status === 'completed' && ['success', 'failure'].includes(run.conclusion))) {
      const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
        ...context.repo, run_id: run.id, filter: 'latest', per_page: 100
      });
      if (jobs.some(job => job.name.includes(' / test / ') && job.conclusion === 'success')) {
        executed = true;
        break;
      }
    }
    if (!executed) {
      core.setFailed('PR authors must wait for an approved current-head platform CI run to complete.');
      return;
    }
  }
  const unchanged = async () => {
    const { data: latest } = await github.rest.pulls.get({ ...context.repo, pull_number: number });
    return latest.state === 'open' && latest.head.repo?.full_name === pr.head.repo.full_name &&
      latest.head.sha === pr.head.sha && latest.head.ref === pr.head.ref;
  };
  if (request.kind === 'rerun-job') {
    const { data: job } = await github.rest.actions.getJobForWorkflowRun({
      ...context.repo, job_id: request.jobId
    });
    const { data: run } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: job.run_id });
    if (!currentRun(run, pr, context, workflow.id) || run.status !== 'completed' ||
        run.conclusion !== 'failure' || job.status !== 'completed' || job.conclusion !== 'failure') {
      core.setFailed('Only failed CI jobs from this PR current head can be rerun.');
      return;
    }
    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
      ...context.repo, run_id: run.id, filter: 'latest', per_page: 100
    });
    if (!jobs.some(item => item.id === job.id && item.conclusion === 'failure')) {
      core.setFailed('Job is from an old attempt; use the latest failed job ID.');
      return;
    }
    const { data: latestRun } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: run.id });
    if (!await unchanged() || latestRun.run_attempt !== run.run_attempt || latestRun.status !== 'completed') {
      core.setFailed('PR head or run attempt changed; request again using the current job.');
      return;
    }
    await github.rest.actions.reRunJobForWorkflowRun({ ...context.repo, job_id: job.id });
    core.notice(`Requested rerun of job ${job.id} and its downstream jobs in run ${run.id}.`);
    return;
  }
  // Workflow and platform list both come from the trusted default branch.
  const { data: configs } = await github.rest.repos.getContent({
    ...context.repo, path: '.github/configs', ref: context.payload.repository.default_branch
  });
  const platforms = configs.filter(file => file.name.endsWith('.yml') &&
    !['template.yml', 'image_sources.yml', 'platforms.yml'].includes(file.name))
    .map(file => file.name.slice(0, -4));
  const target = request.platform;
  if (target !== 'all' && !platforms.includes(target)) {
    core.setFailed(`Unknown platform: ${target}. Available: all, ${platforms.join(', ')}`);
    return;
  }
  if (!await unchanged()) {
    core.setFailed('PR head changed; send the command again.');
    return;
  }
  await github.rest.actions.createWorkflowDispatch({
    ...context.repo, workflow_id: 'ci-request.yml',
    ref: context.payload.repository.default_branch,
    inputs: { pr_number: String(pr.number), head_sha: pr.head.sha, platform: target }
  });
  await github.rest.issues.createComment({
    ...context.repo, issue_number: pr.number,
    body: `CI request accepted: **${target}**, commit \`${pr.head.sha}\`. ` +
      `[View requested runs](${context.payload.repository.html_url}/actions/workflows/ci-request.yml). ` +
      'This supplemental run does not replace the CI Complete merge gate.'
  });
};
module.exports.parseCommand = parseCommand;
module.exports.parseRequest = parseRequest;
