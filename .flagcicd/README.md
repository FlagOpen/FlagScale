# FlagScale test capabilities and rollout

This adapter describes FlagScale; it does not vendor Torch-FL or TE-FL's
execution framework. The existing dependency preparation, reusable workflows,
platform images, test exclusions, launchers and coverage calculation remain
authoritative. No Weekly or release workflow is introduced.

## Capability levels

| Level | Meaning | Existing evidence and limitations |
| --- | --- | --- |
| L0 | Static/configuration/plan checks | pre-commit, platform routing and inventory validation |
| L1 | Hardware-free isolated tests and package validation | `tests/ci/` Python and Node tests; a dedicated package acceptance lane is not implemented |
| L2 | Single-device runtime and short functional tests | Classify actual configured training topology; no universal device smoke suite exists |
| L3 | Distributed, multi-device and component integration | Existing torchrun unit envelope, multi-device train and hetero-train cases |
| L4 | Performance, model quality, endurance and release acceptance | Existing benchmark cases only; complete release/endurance acceptance is not implemented |

Levels are capabilities, not schedules. Nightly runs **all currently enabled
configured tests**, including benchmark (L4), rather than stopping at L3.
Inference and serving remain disabled; RL remains disconnected. Their presence
on disk is not evidence of execution. Missing required capabilities must not be
reported as passed. Time budgets remain in existing platform/workflow settings.

The legacy unit job mixes isolated and device-dependent tests. Its execution
envelope is L3 because it retains configured multi-process torchrun. Individual
test semantics are not inferred from a directory or a name containing GPU.
Splitting verified CPU tests into a separate pytest lane is future work.

## What "smoke" means here

Smoke means a bounded check that a particular operation can start and produce a
basic valid result. It does not establish convergence, stability or performance.
There is no unified `smoke/` suite in this repository. In particular:

- `tests/unit_tests/test_basic.py::test_import` contains only `pass`; it is a
  placeholder, **not** an import or device smoke check.
- `tests/functional_tests/train/qwen3/conf/train/0_6b_ascend.yaml` sets
  `model.train_iters: 10`. Its launcher configuration at
  `tests/functional_tests/train/qwen3/conf/0_6b_ascend.yaml` uses **two** processes,
  so this is an L3 short distributed training check, not single-device L2.
- `tests/functional_tests/train/qwen3/conf/0_6b_hygon.yaml` declares one process
  on one node. It is a single-device functional candidate (L2), still requiring
  its real image, data, runtime and metric comparison; local CPU tests do not
  prove that it passes on hardware.
- `tests/unit_tests/models/kerv/test_ops.py` mixes CPU fallback checks and other
  runtime tests. It cannot be relabeled wholesale as a generic device smoke.
- `tests/unit_tests/train/megatron/test_distributed_backends.py` contains mocked
  backend selection checks; its name alone does not prove a real collective ran.

These paths explain existing tests. This change does not invent new smoke tests
or claim that a placeholder satisfies a capability.

## Ownership and migration

`project.yaml` references existing platform configuration, `test-tiers.json`
declares capability semantics and routing groups, and `test-inventory.json`
classifies tracked test files (including helpers and placeholders). CI validates
the inventory, so adding/removing a test requires updating its classification.
Use `git add` for new tests before local inventory validation: CI sees committed
files, and the validator deliberately ignores unrelated untracked files.

Source-to-group routes are deliberately small. Unknown/shared dependencies,
platform/build/CI changes, missing diffs and unsupported selection fall back to
full execution. Platform labels only add coverage. Retained platform exclusions
are applied to both full and affected selection; no routing policy enables an
excluded test. Changes to excluded tests require review of that platform's gap.

The default PR rollout is **shadow**: record the proposed affected selection but
execute the legacy full test set on selected platforms. After hardware review,
repository variable `CI_AFFECTED_TESTS_ENABLED=true` enables affected execution.
This variable cannot be enabled by a PR label or a workflow input.

## Execution and evidence

Both entry workflows call the existing `_platform_pipeline.yml` dependency/test
chain. The `test_plan` input is optional; empty preserves legacy callers such as
image validation and authorized comment requests. Ordinary CI continues using
the immutable event SHA fallback, so explicitly requested PR source retains its
existing cache-write isolation. Plan consumers compare the source SHA and
configuration fingerprint with their checkout before executing selected tests.

The planner's hash identifies content; it is **not** a signature or a replacement
for GitHub's workflow trust and fork-approval boundaries. This adapter does not
introduce `pull_request_target`, privileged candidate execution, a central
trusted Core distribution, or a cross-repository release acceptance gate.

Planned unit tests retain the same torchrun process count and coverage setup.
They emit JUnit per rank, and empty/all-skipped runs fail. Functional cases use
the existing matrix, launcher and result comparator; the platform gate requires
each planned job category to succeed and permits only deliberately empty
functional matrices to skip. This is job/matrix completion evidence, **not** the
future cross-repository group/platform report protocol. Existing functional
exit codes remain authoritative; their result validation is not replaced.

The daily 02:00 Asia/Shanghai schedule moves from `CI` to `Nightly`, avoiding
duplicate schedules. Nightly always selects all enabled configured cases;
main pushes and manual `CI` runs retain full execution. Nightly has independent
non-cancelling concurrency and a `Nightly Complete` check, while PRs retain
`CI Complete`. Notifications and the default-disabled infrastructure retry
controller recognize the new scheduled entry without admitting PRs/dispatches.

No resource parallelism is added: unit topology and existing workflow ordering
remain unchanged. This does not claim cross-run physical-device reservation;
runner scheduling and explicit shared-resource locks still require hardware
validation. Coverage source roots, calculation, thresholds and upload endpoints
are unchanged. Consult the plan's profile/mode/unit_mode before interpreting a
partial PR coverage sample as a full-regression measurement.

## Local inspection

From the repository root, with PyYAML installed:

```bash
python .github/scripts/ci/flagcicd_contract.py
python .github/scripts/ci/flagcicd_plan.py plan --profile nightly --file /tmp/flagscale-plan.json
python .github/scripts/ci/flagcicd_plan.py slice --file /tmp/flagscale-plan.json --platform ascend
python -m pytest --confcutdir=tests/ci tests/ci -q
node --test tests/ci/*.test.cjs
```

Before enabling affected execution, review proposed routes on real platform
runs, including fork approval, cached dependency recovery, retained exclusions,
rank-specific reports, benchmark baselines and the platform's short training
case. CPU planner tests cannot certify those hardware paths.
