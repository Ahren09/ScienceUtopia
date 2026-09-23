# Authorized pending-six continuation

This is a separately frozen execution of six jobs that never started in the failed V6 queue. The September 21 approval of the pending-only proposal authorizes this scope. A and B remain excluded. This is not parent-queue resume, checkpoint resume, retry, replacement, or retrospective adoption.

The exact order is C, D, P1F1, P0F1, P1F0, P0F0. Their original job IDs, command arrays, environment, output roots, artifact ordering, predicates and scientific parameters are retained. Only C/D's twenty exact native checkpoint artifacts gain `max_json_bytes=536870912`. Other predicate JSONs retain 64 MiB. The approved capacity candidate is reused without changing its parser or integrity checks.

The new worktree is `/home/ubuntu/Workspace/PeerReview/ScienceUtopia-llm-pending6` on `codex/llm-james-pending6`. The queue ID is `james_pending_seed42_n1200_20260921_r1`. The private root, Python, HF cache, original scientific job paths and command arrays, model/revision/BF16, TP2, prompts, seed 42 and sequential funding protocol remain unchanged.

The original mechanism command arrays name the absolute frozen-parent `src/james_mechanisms.py`. Those bytes remain intact. Pending mode permits only that exact parent-plan-bound mechanism entrypoint outside the new worktree, and continues checking its original source pin. Relative influx commands execute the unchanged scientific copies in the new worktree. This exception does not permit arbitrary external scripts.

## Plan descriptor

The new top-level `continuation` object has exactly these fields:

```json
{
  "schema_version": 1,
  "kind": "v6_pending_six_v1",
  "parent_plan": {"path": "P/outputs/logs/ec2_preparation/james_ec2_queue_plan_v6.json", "sha256": "382dbb5168196c2b6adb15f2ed3ef9bb273389d7c816b65e6057ee524ed9e3e0"},
  "parent_status": {"path": "P/outputs/logs/james_ec2_queue/james_seed42_n1200_20260919_v6/status.json", "sha256": "5cb068350039d8dd3d2a1a581d5e5a591721c45992faefe314dc10c96ce9f4cf"},
  "parent_source_freeze": {"path": "P/outputs/logs/ec2_preparation/v6_source_freeze.json", "sha256": "e692b66b13347faa1b6024fdd01fbed531bd00043675a602121b12a0a20bdd4b"},
  "authorization": {"path": "P/outputs/logs/ec2_preparation/pending6_execution_authorization_20260921.json", "sha256": "bc82df78b9eb570b694b413d9fabafc124427634185761a157b73c767cbe762c"}
}
```

`P` denotes the absolute private root in this illustration. Actual paths must be absolute and must equal those exact paths relative to the plan's private root. The approval file is copied byte-for-byte without adding execution-status fields. Its proposal hash is `cf888fa2a056257dae27916c86498de3af8227a9fb4f6444c8ef5274a78fe4ce`.

All other parent plan fields match exactly except `worktree`, `queue_id`, the six-job list and this descriptor. The capacity additions are the sole permitted changes within those jobs. Different subsets, order, settings, predicates, paths, or capacity values fail validation.

## Binding and admission

`parse_plan` checks the descriptor and exact membership shape. `validate_pending_continuation(plan, expected=None, fresh_jobs=...)` reads the actual bound records and checks the original failed terminal state, clean stopped servers, exact `{"status":"pending"}` records for all six jobs, and absence of original owned sessions. No launch identity, even a null field, is accepted in a pending parent job.

The validator checks the original repository HEAD/main/branch and clean tracked status, all 87 original frozen source hashes, and matching copies of those sources in the new worktree except the adapted controller. It records the new controller, test and documentation separately. Other prospective infrastructure additions, including the parent's capacity-resource utility, enter the ordinary new-worktree `source_hashes` freeze. The original source map remains unchanged.

No-symlink descriptor walks, regular-file ownership/single-link checks, whole-file hashes and file identities bind the records and sources. Both worktree outputs aliases are captured before and after binding. The startup snapshot is subsequently compared without refreshing its baseline.

Parent validation and output freshness precede runtime setup, queue-directory creation and locking. Parent bindings are rechecked before servers, before each scientific client, again after launch-intent journaling immediately before spawn, before and after artifact admission, and at the start and end of final validation. Existing output roots, even empty directories or broken symlinks, reject a launch.

The original client commands and their unchanged source implement raw request, sequential-selection, native-ledger and positive-coverage validation before successful client exit. The controller retains its original complete artifact predicates and hashing, readiness probes, server manifests, source checks and receipt revalidation. The adapter does not substitute a summary-only implementation for client checks or claim that its generic artifact gate independently reconstructs the raw audit joins.

New receipts record the startup parent-binding digest and explicit pending-only scope. Final status may say this six-job queue is complete, but explicitly records `full_first8_complete=false` and `all12_complete=false`. It never writes parent records or invents A/B receipts.

## Resource and deadline scope

The unchanged idle telemetry checks, shared UUID locks, durable reservation intents, PID identities, cancellation gates and owned-only cleanup govern TP2 allocation. In this mode pair order prefers GPU 5 then GPU 6. If that pair is unavailable, only other verified idle disjoint pairs are eligible. No running THIRD job is targeted, no GPU sharing is introduced, and no TP1/TP3 substitution occurs.

The pending-only outer deadline is fixed at `len(PENDING_JOB_IDS) * TIMEOUT_SECONDS`, or **432 hours**, from new queue-state creation. It aggregates six controller-enforced 72-hour client allowances for potentially serial execution. It is not a runtime prediction and is not guaranteed to cover all waiting and probe overhead. There is no arbitrary deadline field, extension, or pending-mode resume. Original eight-job queues retain their 72-hour outer budget.

**Every pending-mode scientific client is individually bounded to 72 hours from its persisted `started_epoch_ns`.** C/D retain their original `--timeout_hours 72.0` arguments. The actual P/F commands have no timeout flag, so the controller supplies their required bound without altering those commands. The controller checks active-client expiry before polling, before process creation, and before accepting an exit, including exit code zero. Expiry fails the queue through its existing owned-process cleanup. The original launch timestamp is never refreshed and no retry or resume is allowed.

Blocking readiness probes are capped by the earliest active-client time remaining, and client deadlines are checked around those probes. Expiry is handled at the controller's checking boundaries with the existing cleanup grace. A receipt records when the controller observed a successful exit and the fixed client timeout. Final validation checks that stored observation against the original launch timestamp, so later artifact hashing or six-job final validation does not grant additional client runtime or invalidate an earlier timely exit merely because wall time has advanced.

The 512 MiB allowance is a prospective file-size capacity amendment, not a bound on parser RAM or proof of sufficient capacity for unseen C/D histories. The parent's separately reviewed synthetic resource check is distinct from the new continuation tests.

## Verification scope

The new `src/test_james_pending_continuation.py` exercises this contract with synthetic files, actual temporary Git repositories and real file/lock checks. Fixture hashes replace only the four production trust anchors. Parent and admission validators remain active. It covers exact membership/settings/capacity, parent status and pending identities, source and alias changes, freshness, rejection before writes or launches, all lifecycle checks, receipt guard retention, GPU preference, the fixed outer deadline and resume rejection.

Only these new tests are in scope here. No existing 357-test suite, the fifteen capacity cases, real A diagnostic, GPU/model request, or scientific run is repeated. Independent review and the parent's actual-runtime validation, source freeze and launch follow this implementation.

Earlier local result before the individual-client deadline correction: **19 tests passed, zero skipped, 7.963 seconds**. That receipt remains historical evidence and does not establish the subsequently identified P/F timeout requirement.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -B -m unittest discover -s src -p test_james_pending_continuation.py -v
```

The supplied plan SHA256 `ae6b1ab2d43481f6c2b59996dd1a54f9e85948ba16db47d2b5a7e4bff7b73d72` passes schema parsing. Its jobs equal original parent jobs `[2:]` after removing exactly twenty 512 MiB capacity fields. A separate local byte comparison confirms that only the adapted controller differs among the original 87 pinned sources. These local checks do not replace actual EC2 parent binding and runtime validation.

The subsequent independent review found that P/F had relied on the old whole-queue deadline. The corrected fixture now gives timeout arguments only to C/D and preserves the actual P/F `run` command form with no timeout flag. After adding the controller's individual 72-hour enforcement, **four targeted cases passed locally, zero skipped, in 2.567 seconds**:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/local/bin/python3 -B -m unittest -v \
  test_james_pending_continuation.PendingContinuationTests.test_fixed_432_hour_outer_budget_no_resume_or_extension \
  test_james_pending_continuation.PendingContinuationTests.test_expired_pf_fails_before_poll_and_uses_owned_cleanup_without_reset \
  test_james_pending_continuation.PendingContinuationTests.test_overdue_zero_exit_rejected_at_finish_before_admission_and_cleanup \
  test_james_pending_continuation.PendingContinuationTests.test_client_deadline_exact_boundary_historical_evidence_and_probe_budget
```

The expired-running and overdue-zero-exit cases execute the actual queue failure/finally-cleanup route with synthetic process handles and mocked OS termination. Fail-fast sentinels prohibit GPU queries or artifact admission. The overdue-zero-exit case directs the loop to the real `finish_job` method to check its independent guard. Both cases assert that admission was never invoked, cleanup addressed only those owned handles, no receipt appeared, and the persisted launch timestamp was unchanged. The remaining cases check the 432-hour outer budget, absent P/F timeout flags, no resume/extension, exact 72-hour boundary, historical exit-observation validation, and active-client time remaining for probe bounds. No full 19-case rerun, capacity-resource rerun, old-world diagnostic, or inference was performed for this correction.
