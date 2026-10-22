// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const command = require('../../.github/scripts/ci/slash_command.js');

function fixture(permission = 'write', body = '/ci run ascend') {
  const calls = [];
  const sha = 'a'.repeat(40);
  const context = { repo: { owner: 'team', repo: 'FlagScale' }, payload: {
    comment: { body, user: { login: 'member', id: 1, type: 'User' } }, issue: { number: 42 },
    repository: { default_branch: 'main', html_url: 'https://github.com/team/FlagScale' }
  } };
  const github = { rest: {
    repos: {
      getCollaboratorPermissionLevel: async () => ({ data: { permission } }),
      getContent: async args => {
        calls.push(['configs', args]);
        return { data: ['ascend.yml', 'cuda.yml', 'ppu.yml', 'template.yml', 'image_sources.yml']
          .map(name => ({ name })) };
      }
    },
    pulls: { get: async () => ({ data: { number: 42, user: { id: 2 }, state: 'open', head: {
      sha, ref: 'feature', repo: { full_name: 'contributor/FlagScale' }
    } } }) },
    actions: {
      getWorkflow: async () => ({ data: { id: 7, state: 'active' } }),
      listWorkflowRuns() {},
      listJobsForWorkflowRun() {},
      createWorkflowDispatch: async args => calls.push(['dispatch', args])
    },
    issues: { createComment: async args => calls.push(['comment', args]) }
  } };
  const core = { notice: value => calls.push(['notice', value]), setFailed: value => calls.push(['failed', value]) };
  return { github, context, core, calls, sha };
}

for (const input of ['/ci run ascend; echo injected', 'please /ci run all', '/ci run ../../secret',
  '/ci run all\n/ci run cuda', '/ci run', '/ci run ASCEND']) {
  test(`reject malformed command ${JSON.stringify(input)}`, () => {
    assert.equal(command.parseCommand(input), null);
  });
}

test('dispatch pins current PR commit and trusted workflow branch', async () => {
  const f = fixture();
  await command(f);
  const dispatched = f.calls.find(([name]) => name === 'dispatch')[1];
  assert.equal(dispatched.workflow_id, 'ci-request.yml');
  assert.equal(dispatched.ref, 'main');
  assert.deepEqual(dispatched.inputs, { pr_number: '42', head_sha: f.sha, platform: 'ascend' });
  assert.equal(f.calls.find(([name]) => name === 'configs')[1].ref, 'main');
  assert.match(f.calls.find(([name]) => name === 'comment')[1].body, /does not replace/);
});

for (const role of ['read', 'triage', 'none']) {
  test(`no dispatch for ${role} permission`, async () => {
    const f = fixture(role);
    await command(f);
    assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
    assert.equal(f.calls.some(([name]) => name === 'comment'), false);
  });
}

for (const target of ['all', 'ppu']) {
  test(`allow ${target}`, async () => {
    const f = fixture('admin', `/ci run ${target}`);
    await command(f);
    assert.equal(f.calls.find(([name]) => name === 'dispatch')[1].inputs.platform, target);
  });
}

test('unknown and metadata platforms cannot dispatch', async () => {
  for (const target of ['typo', 'template', 'image_sources']) {
    const f = fixture('write', `/ci run ${target}`);
    await command(f);
    assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
    assert.equal(f.calls.some(([name]) => name === 'failed'), true);
  }
});

test('closed PR cannot dispatch', async () => {
  const f = fixture();
  f.github.rest.pulls.get = async () => ({ data: { state: 'closed' } });
  await command(f);
  assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
});

const resolveSource = require('../../.github/scripts/ci/request_source.js');
function requestFixture() {
  const f = fixture();
  const outputs = {};
  f.prNumber = '42';
  f.headSha = f.sha;
  f.core.setOutput = (name, value) => { outputs[name] = value; };
  f.core.summary = {
    addHeading() { return this; }, addRaw() { return this; }, async write() {}
  };
  f.outputs = outputs;
  return f;
}

test('requested workflow resolves source repository through PR API', async () => {
  const f = requestFixture();
  await resolveSource(f);
  assert.deepEqual(f.outputs, { repository: 'contributor/FlagScale', sha: f.sha });
});

test('requested workflow rejects stale SHA', async () => {
  const f = requestFixture();
  f.headSha = 'b'.repeat(40);
  await assert.rejects(resolveSource(f), /head changed/);
  assert.deepEqual(f.outputs, {});
});

test('requested workflow rejects closed or deleted PR sources', async () => {
  for (const pr of [{ state: 'closed', head: {} }, { state: 'open', head: { repo: null } }]) {
    const f = requestFixture();
    f.github.rest.pulls.get = async () => ({ data: pr });
    await assert.rejects(resolveSource(f), /closed, unavailable/);
  }
});

test('requested workflow rejects invalid PR and SHA inputs', async () => {
  for (const [prNumber, headSha] of [['1;echo', 'a'.repeat(40)], ['42', 'main'], ['0', 'a'.repeat(40)]]) {
    const f = requestFixture();
    await assert.rejects(resolveSource({ ...f, prNumber, headSha }), /Expected a PR number/);
  }
});

function rerunFixture(permission = 'read', body = '/ci rerun-job 123') {
  const f = fixture(permission, body);
  f.context.payload.comment.user.id = 2; // This PR's author.
  f.run = { id: 10, workflow_id: 7, event: 'pull_request', head_sha: f.sha,
    head_branch: 'feature', head_repository: { full_name: 'contributor/FlagScale' },
    repository: { full_name: 'team/FlagScale' }, pull_requests: [],
    status: 'completed', conclusion: 'failure', run_attempt: 1 };
  f.job = { id: 123, run_id: 10, status: 'completed', conclusion: 'failure',
    name: 'Platform pipeline (kunlunxin) / test / benchmark' };
  Object.assign(f.github.rest.actions, {
    getJobForWorkflowRun: async () => ({ data: f.job }),
    getWorkflowRun: async () => ({ data: f.run }),
    reRunJobForWorkflowRun: async args => f.calls.push(['rerun', args])
  });
  f.github.paginate = async method => method === f.github.rest.actions.listWorkflowRuns
    ? [f.run] : [f.job, { name: 'Platform pipeline (kunlunxin) / test / checkout_and_config', conclusion: 'success' }];
  return f;
}

test('author can request their own already-executed platform CI', async () => {
  const f = rerunFixture('read', '/ci run ppu');
  await command(f);
  assert.equal(f.calls.find(([name]) => name === 'dispatch')[1].inputs.platform, 'ppu');
});

test('author cannot bypass initial fork approval or skipped platform execution', async () => {
  for (const mode of ['approval', 'no-platform']) {
    const f = rerunFixture('read', '/ci run ppu');
    if (mode === 'approval') f.run.conclusion = 'action_required';
    else f.github.paginate = async method => method === f.github.rest.actions.listWorkflowRuns
      ? [f.run] : [{ name: 'Lint', conclusion: 'success' }];
    await command(f);
    assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
    assert.equal(f.calls.some(([name]) => name === 'failed'), true);
  }
});

test('author requests only the selected failed job and its downstream jobs', async () => {
  const f = rerunFixture();
  await command(f);
  assert.deepEqual(f.calls.find(([name]) => name === 'rerun')[1], { owner: 'team', repo: 'FlagScale', job_id: 123 });
  assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
});

test('unrelated reader and bot cannot use author privileges', async () => {
  for (const identity of [{ id: 3, type: 'User' }, { id: 2, type: 'Bot' }]) {
    const f = rerunFixture(); Object.assign(f.context.payload.comment.user, identity);
    await command(f); assert.equal(f.calls.some(([name]) => name === 'rerun'), false);
  }
});

test('reject stale commits, other repositories/PRs/workflows, active or successful jobs', async () => {
  const changes = [
    f => { f.run.head_sha = 'b'.repeat(40); },
    f => { f.run.head_repository.full_name = 'other/FlagScale'; },
    f => { f.run.head_branch = 'other'; },
    f => { f.run.repository.full_name = 'other/FlagScale'; },
    f => { f.run.pull_requests = [{ number: 99 }]; },
    f => { f.run.workflow_id = 999; },
    f => { f.run.event = 'push'; },
    f => { f.run.status = 'in_progress'; },
    f => { f.job.conclusion = 'success'; },
    f => { f.github.paginate = async () => [{ ...f.job, id: 999 }]; }
  ];
  for (const change of changes) {
    const f = rerunFixture(); change(f); await command(f);
    assert.equal(f.calls.some(([name]) => name === 'rerun'), false);
  }
});

test('recheck head and attempt before a rerun mutation', async () => {
  for (const mode of ['head', 'attempt']) {
    const f = rerunFixture(); let reads = 0;
    if (mode === 'head') {
      const get = f.github.rest.pulls.get;
      f.github.rest.pulls.get = async () => {
        const result = await get();
        if (++reads > 1) result.data.head.sha = 'b'.repeat(40);
        return result;
      };
    } else {
      f.github.rest.actions.getWorkflowRun = async () => ({ data: ++reads > 1
        ? { ...f.run, run_attempt: 2 } : f.run });
    }
    await command(f); assert.equal(f.calls.some(([name]) => name === 'rerun'), false);
  }
});

test('new commands reject extra lines, shell text, zero and unsafe job IDs', () => {
  for (const body of ['/ci rerun-job 0', '/ci rerun-job 9007199254740992', '/ci rerun-job 123; ls',
    '/ci rerun-job 123\n/ci run all', 'Re-run failed jobs', 'gh run rerun --job 123']) {
    assert.equal(command.parseRequest(body), null);
  }
});
