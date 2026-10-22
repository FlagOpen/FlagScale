// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.
// Deliberately conservative: unclassified failures never authorize a rerun.
const RECOVERY = /^Recover and validate prepared (megatron|te) runtime$/;
const TRANSIENT = /\b(ETIMEDOUT|ECONNRESET|EAI_AGAIN)\b|socket hang up|connection timed out|request timed out|Unable to download artifact[^\n]*(?:500|502|503|504)/i;
const INVALID = /RECORD mismatch|Wheel CRC failure|BadZipFile|Missing RECORD hash|Artifact not found|Unable to find any artifacts|\b(?:401|403|404)\b/i;

function classify(job, log) {
  if (job.conclusion !== 'failure') return 'ignore';
  const failed = (job.steps || []).filter(step => step.conclusion === 'failure');
  if (failed.length === 0) return 'unknown';
  // Aggregates are consequences of root failures, not independent retry reasons.
  const leaf = job.name.split(' / ').at(-1);
  const aggregate = {
    all_tests_complete: 'Verify all tests passed',
    'CI Complete': 'Verify required checks'
  }[leaf];
  if (aggregate && failed.every(step => step.name === aggregate)) return 'aggregate';
  if (!failed.every(step => RECOVERY.test(step.name))) return 'unknown';
  if (typeof log !== 'string') return 'unknown';
  // Never borrow a timeout warning from an earlier cache/install/test step.
  // Missing/invalid timestamps or a changed log format fail closed.
  for (const step of failed) {
    const start = Date.parse(step.started_at), end = Date.parse(step.completed_at);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return 'unknown';
    const scoped = log.split('\n').filter(line => {
      const stamp = /^(\d{4}-\d\d-\d\dT\S+)/.exec(line);
      const time = stamp ? Date.parse(stamp[1]) : NaN;
      return time >= start && time <= end + 999;
    }).join('\n');
    if (!/Z\s+CI_INFRA_RECOVERY_DOWNLOAD_FAILED\s*$/m.test(scoped) ||
        INVALID.test(scoped) || !TRANSIENT.test(scoped)) return 'unknown';
  }
  return 'transient';
}

async function retry({ github, context, core }) {
  const repo = context.repo;
  const expected = context.payload.workflow_run;
  const skip = message => core.notice(`No automatic retry: ${message}`);
  const { data: run } = await github.rest.actions.getWorkflowRun({ ...repo, run_id: expected.id });
  const branch = context.payload.repository.default_branch;
  // Only default-branch push/schedule CI is eligible. PRs and dispatches can execute
  // untrusted source with different provenance; leave those to maintainers.
  if (run.run_attempt !== 1 || run.status !== 'completed' || run.conclusion !== 'failure' ||
      !['push', 'schedule'].includes(run.event) || run.head_branch !== branch ||
      run.head_sha !== expected.head_sha || run.repository.full_name !== context.payload.repository.full_name) {
    return skip('run is not an unchanged first attempt on the default branch.');
  }
  const { data: workflow } = await github.rest.actions.getWorkflow({ ...repo, workflow_id: 'ci.yml' });
  if (run.workflow_id !== workflow.id) return skip('workflow is not ci.yml.');
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...repo, run_id: run.id, filter: 'latest', per_page: 100
  });
  if (jobs.some(job => ['cancelled', 'timed_out', 'action_required', 'startup_failure', 'stale'].includes(job.conclusion))) {
    return skip('run includes cancellation, a timeout, or an unresolved job.');
  }
  const failed = jobs.filter(job => job.conclusion === 'failure');
  let transient = 0;
  for (const job of failed) {
    let log = '';
    if ((job.steps || []).some(step => step.conclusion === 'failure' && RECOVERY.test(step.name))) {
      const response = await github.request('GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs', {
        ...repo, job_id: job.id
      });
      log = typeof response.data === 'string' ? response.data : Buffer.from(response.data).toString('utf8');
    }
    const kind = classify(job, log);
    if (kind === 'unknown') return skip(`job ${job.id} is not a proven dependency transfer failure.`);
    if (kind === 'transient') transient++;
  }
  if (!transient) return skip('no proven transient failure.');
  // Close the common manual-rerun/new-commit race before the mutation.
  const { data: current } = await github.rest.actions.getWorkflowRun({ ...repo, run_id: run.id });
  const { data: tip } = await github.rest.repos.getBranch({ ...repo, branch });
  if (current.run_attempt !== 1 || current.status !== 'completed' || current.conclusion !== 'failure' ||
      current.head_sha !== run.head_sha || tip.commit.sha !== run.head_sha) {
    return skip('attempt or default-branch commit changed during inspection.');
  }
  await github.rest.actions.reRunWorkflowFailedJobs({ ...repo, run_id: run.id });
  core.notice(`Requested one retry of failed jobs in CI run ${run.id}; ${transient} dependency transfer failure(s).`);
}
module.exports = retry;
module.exports.classify = classify;
