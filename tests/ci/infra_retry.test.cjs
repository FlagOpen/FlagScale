// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.
const test = require('node:test');
const assert = require('node:assert/strict');
const retry = require('../../.github/scripts/ci/infra_retry.cjs');
const root = (name = 'Recover and validate prepared megatron runtime') => ({
  id: 9, name: 'Platform pipeline (ppu) / test / unit-ppu', conclusion: 'failure',
  steps: [{ name, conclusion: 'failure', started_at: '2026-10-21T01:00:00Z', completed_at: '2026-10-21T01:01:00Z' }]
});
const stamp = text => text.split('\n').map(line => '2026-10-21T01:00:30.000Z ' + line).join('\n');
const log = stamp('Error: read ECONNRESET\nCI_INFRA_RECOVERY_DOWNLOAD_FAILED\nInvalid megatron runtime: missing files');
const aggregate = { id: 10, name: 'CI Complete', conclusion: 'failure', steps: [
  { name: 'Verify required checks', conclusion: 'failure' }
] };

test('only an exhausted transfer in the failing recovery step is eligible', () => {
  assert.equal(retry.classify(root(), log), 'transient');
  assert.equal(retry.classify(root('Run unit tests'), log), 'unknown');
  assert.equal(retry.classify(root(), 'ECONNRESET'), 'unknown');
  assert.equal(retry.classify(root(), 'CI_INFRA_RECOVERY_DOWNLOAD_FAILED'), 'unknown');
  assert.equal(retry.classify({ ...root(), steps: [] }, log), 'unknown');
  for (const failure of ['RECORD mismatch', 'Wheel CRC failure', 'Artifact not found', '403 Forbidden']) {
    assert.equal(retry.classify(root(), log + '\n' + stamp(failure)), 'unknown');
  }
  assert.equal(retry.classify(aggregate, ''), 'aggregate');
  assert.equal(retry.classify(root(), '2026-10-21T00:59:00Z ECONNRESET\n' +
    stamp('CI_INFRA_RECOVERY_DOWNLOAD_FAILED')), 'unknown');
  assert.equal(retry.classify(root(), 'ECONNRESET\nCI_INFRA_RECOVERY_DOWNLOAD_FAILED'), 'unknown');
});

function fixture(jobs = [root(), aggregate], overrides = {}) {
  const run = { id: 1, workflow_id: 2, event: 'push', head_branch: 'main', head_sha: 'a',
    repository: { full_name: 'org/repo' }, run_attempt: 1, status: 'completed', conclusion: 'failure', ...overrides };
  const calls = [];
  let reads = 0;
  const github = {
    rest: {
      actions: {
        getWorkflowRun: async () => ({ data: ++reads > 1 && overrides.changed ? { ...run, run_attempt: 2 } : run }),
        getWorkflow: async () => ({ data: { id: 2 } }),
        listJobsForWorkflowRun: () => {},
        reRunWorkflowFailedJobs: async params => calls.push(params)
      },
      repos: { getBranch: async () => ({ data: { commit: { sha: overrides.tip || 'a' } } }) }
    },
    paginate: async () => jobs,
    request: async () => ({ data: log })
  };
  const context = { repo: { owner: 'org', repo: 'repo' }, payload: {
    workflow_run: { id: 1, head_sha: 'a' }, repository: { full_name: 'org/repo', default_branch: 'main' }
  } };
  return { github, context, core: { notice: () => {} }, calls };
}

test('retry uses failed-jobs API only, preserving the successful jobs', async () => {
  const f = fixture(); await retry(f);
  assert.deepEqual(f.calls, [{ owner: 'org', repo: 'repo', run_id: 1 }]);
});

test('mixed test/performance failures, unknowns and aggregates alone never retry', async () => {
  for (const jobs of [[root(), root('Run benchmark tests'), aggregate], [aggregate],
    [root(), { ...root(), conclusion: 'cancelled' }]]) {
    const f = fixture(jobs); await retry(f); assert.equal(f.calls.length, 0);
  }
});

test('fork/PR/dispatch, old attempts and superseded commits never retry', async () => {
  for (const change of [{ event: 'pull_request' }, { event: 'workflow_dispatch' }, { run_attempt: 2 },
    { head_branch: 'feature' }, { repository: { full_name: 'fork/repo' } },
    { tip: 'new' }, { changed: true }, { workflow_id: 999 }, { head_sha: 'new' }]) {
    const f = fixture(undefined, change); await retry(f); assert.equal(f.calls.length, 0);
  }
});

test('API errors stop inspection before rerunning anything', async () => {
  const f = fixture(); f.github.request = async () => { throw new Error('logs unavailable'); };
  await assert.rejects(retry(f), /logs unavailable/); assert.equal(f.calls.length, 0);
});

test('scheduled Nightly keeps bounded transfer recovery and verifies its workflow ID', async () => {
  const f = fixture([root(), { ...aggregate, name: 'Nightly Complete' }], { name: 'Nightly', event: 'schedule' });
  f.github.rest.actions.getWorkflow = async params => {
    assert.equal(params.workflow_id, 'nightly.yml');
    return { data: { id: 2 } };
  };
  await retry(f); assert.equal(f.calls.length, 1);
  for (const change of [{ name: 'Nightly', event: 'push' },
    { name: 'Nightly', event: 'workflow_dispatch' },
    { name: 'Nightly', event: 'schedule', workflow_id: 999 }]) {
    const denied = fixture(undefined, change); await retry(denied);
    assert.equal(denied.calls.length, 0);
  }
});
