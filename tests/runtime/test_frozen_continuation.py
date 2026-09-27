"""New pending-six contract tests only. No GPU, model, real artifacts, or old suites.

Each fixture uses real temporary files, an actual clean Git parent, 87 synthetic
source pins, and the production validator. Only the four approval hash anchors
are substituted with fixture hashes. Admission functions are never stubbed out.
"""
from __future__ import annotations

from utopia.utils.paths import project_root

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

SOURCE = (project_root(__file__) / 'utopia/runtime/experiment_queue.py')
ALL_IDS = ("influx_james1202_v6_A_seed42", "influx_james1202_v6_B_seed42",
           "influx_james1202_v6_C_seed42", "influx_james1202_v6_D_seed42",
           "james_mechanisms_v6_n1200_y6_P1F1_seed42", "james_mechanisms_v6_n1200_y6_P0F1_seed42",
           "james_mechanisms_v6_n1200_y6_P1F0_seed42", "james_mechanisms_v6_n1200_y6_P0F0_seed42")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PendingContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="james-pending6-cpu-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.parent = self.root / "parent"
        self.new = self.root / "ScienceUtopia-llm-pending6"
        self.private = self.root / "private"
        for root in (self.parent, self.new, self.private):
            root.mkdir(mode=0o700)
        (self.private / "outputs").mkdir(mode=0o700)
        for root in (self.parent, self.new):
            (root / "src").mkdir()
            (root / "utopia/runtime").mkdir(parents=True)
            (root / "utopia/experiments").mkdir(parents=True)
            (root / "tests/runtime").mkdir(parents=True)
            (root / "outputs").symlink_to(self.private / "outputs", target_is_directory=True)
        shutil.copyfile(SOURCE, self.new / "utopia/runtime/experiment_queue.py")
        shutil.copyfile(Path(__file__), self.new / "tests/runtime/test_frozen_continuation.py")
        (self.new / "docs").mkdir()
        (self.new / "docs/frozen_continuation.md").write_text("Synthetic fixture documentation.\n")
        module_name = "pending_fixture_" + self.id().rsplit(".", 1)[-1]
        spec = importlib.util.spec_from_file_location(module_name, self.new / "utopia/runtime/experiment_queue.py")
        self.q = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = self.q
        self.addCleanup(sys.modules.pop, module_name, None)
        spec.loader.exec_module(self.q)
        source_pins = {}
        for i in range(86):
            name = f"src/scientific_{i:02d}.py" if i != 85 else "utopia/experiments/funding_feedback.py"
            for root in (self.parent, self.new):
                (root / name).write_text(f"# Synthetic source {i}\n")
            source_pins[name] = sha(self.parent / name)
        (self.parent / "utopia/runtime/experiment_queue.py").write_text("# Synthetic original controller\n")
        source_pins["utopia/runtime/experiment_queue.py"] = sha(self.parent / "utopia/runtime/experiment_queue.py")
        self.git("init", "-q", "-b", "main")
        self.git("add", "src", "utopia")
        self.git("-c", "user.name=CPU Fixture", "-c", "user.email=fixture@invalid",
                 "commit", "-qm", "Synthetic parent")
        self.head = self.git("rev-parse", "HEAD").strip()
        self.git("checkout", "-qb", "codex/synthetic-parent")
        self.parent_spec = {
            "schema_version": 1, "seed": 42, "queue_id": "synthetic_v6_parent",
            "worktree": str(self.parent), "private_root": str(self.private),
            "funding_selection_protocol": self.q.SEQUENTIAL_PROTOCOL,
            "funding_probe_fixture": {"path": str(self.private / "outputs/fixture.json"),
                                      "sha256": "0" * 64},
            "jobs": [],
        }
        for job_id in ALL_IDS:
            influx = job_id.startswith("influx")
            root = self.private / "outputs" / job_id
            artifacts = [{"path": str(root / "status.json"),
                          "equals": {"/status": "complete", "/seed": 42, "/years": 6}}]
            if influx:
                artifacts = [{"path": str(root / f"checkpoint_year_{year}.json"),
                              "equals": {"/initial_industry_count": 2, "/initial_university_count": 1200,
                                         "/phase": 5, "/year": year}} for year in range(1, 11)]
            self.parent_spec["jobs"].append({
                "id": job_id, "family": "influx" if influx else "mechanisms", "seed": 42,
                "command": [str(self.private / "env/bin/python"),
                            "src/scientific_00.py" if influx else str(self.parent / "utopia/experiments/funding_feedback.py")]
                           + ([] if influx else ["run"])
                           + ["--seed", "42", "--server_url", "__SERVER_URL__",
                              "--server_manifest", "__SERVER_MANIFEST__"]
                           + (["--timeout_hours", "72"] if influx else []),
                "env": {}, "output_roots": [str(root)], "artifacts": artifacts,
            })
        self.paths = {name: self.private / value for name, value in self.q.PENDING_RECORD_PATHS.items()}
        self.write_record("parent_plan", self.parent_spec)
        parent_hash = sha(self.paths["parent_plan"])
        self.status = {
            "status": "failed", "plan_sha256": parent_hash,
            "jobs": {name: {"status": "pending" if name in self.q.PENDING_JOB_IDS else "failed"}
                     for name in ALL_IDS},
            "servers": {"old": {"status": "stopped", "cleanup_errors": []}},
        }
        self.freeze = {
            "plan_sha256": parent_hash, "worktree": str(self.parent), "git_head": self.head,
            "original_main_head": self.head, "branch": "codex/synthetic-parent",
            "source_files_sha256": source_pins,
        }
        self.write_record("parent_status", self.status)
        self.write_record("parent_source_freeze", self.freeze)
        self.write_record("authorization", {"user_message": "continue",
                                          "proposal_sha256": self.q.PENDING_PROPOSAL_SHA256})
        self.q.PENDING_PARENT_SHA256 = {name: sha(path) for name, path in self.paths.items()}
        self.spec = copy.deepcopy(self.parent_spec)
        self.spec.update(queue_id=self.q.PENDING_QUEUE_ID, worktree=str(self.new))
        self.spec["continuation"] = {"schema_version": 1, "kind": self.q.PENDING_KIND,
                                    **{name: {"path": str(path), "sha256": sha(path)}
                                       for name, path in self.paths.items()}}
        self.spec["jobs"] = self.spec["jobs"][2:]
        for job in self.spec["jobs"][:2]:
            for artifact in job["artifacts"]:
                artifact["max_json_bytes"] = 512 * 1024 * 1024

    def git(self, *args):
        return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null",
                                        "-C", str(self.parent), *args], text=True,
                                       stderr=subprocess.PIPE,
                                       env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})

    def write_record(self, name, value):
        path = self.paths[name]
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def authorize_fixture_record(self, name, value):
        """Fixture-only trusted bytes to exercise inner contract checks."""
        self.write_record(name, value)
        checksum = sha(self.paths[name])
        self.q.PENDING_PARENT_SHA256[name] = checksum
        self.spec["continuation"][name]["sha256"] = checksum

    def plan(self):
        return self.q.parse_plan(json.dumps(self.spec).encode())

    def binding(self, plan=None, **kwargs):
        return self.q.validate_pending_continuation(plan or self.plan(), **kwargs)

    def test_exact_six_and_original_science_with_only_twenty_capacity_fields(self):
        plan = self.plan()
        binding = self.binding(plan, fresh_jobs=plan.jobs)
        self.assertEqual(binding["pending_job_ids"], list(ALL_IDS[2:]))
        self.assertEqual(len(binding["files"]), 4 + 87 + 86 + 3)
        stripped = copy.deepcopy(plan.spec["jobs"])
        for job in stripped[:2]:
            for artifact in job["artifacts"]:
                self.assertEqual(artifact.pop("max_json_bytes"), 512 * 1024 * 1024)
        self.assertEqual(stripped, self.parent_spec["jobs"][2:])
        self.assertFalse(plan.directory.exists())
        self.assertFalse(any(Path(root).exists() for j in plan.jobs for root in j["output_roots"]))

    def test_reject_omission_reorder_a_b_inclusion_and_duplicate(self):
        good = copy.deepcopy(self.spec["jobs"])
        bad_sets = [good[:-1], list(reversed(good)), good[:5] + [good[0]],
                    [self.parent_spec["jobs"][0], *good[1:]],
                    [self.parent_spec["jobs"][1], *good[1:]], self.parent_spec["jobs"]]
        for jobs in bad_sets:
            with self.subTest(ids=[j["id"] for j in jobs]):
                self.spec["jobs"] = jobs
                with self.assertRaises(self.q.QueueError):
                    self.plan()

    def test_reject_modified_command_env_predicate_output_and_top_level_science(self):
        original = copy.deepcopy(self.spec)
        mutations = [
            lambda s: s["jobs"][0]["command"].__setitem__(-1, "73"),
            lambda s: s["jobs"][0]["env"].update(UNDECLARED_SETTING="1"),
            lambda s: s["jobs"][2]["artifacts"][0]["equals"].update({"/years": 7}),
            lambda s: s.update(input_hashes={}),
            lambda s: s.update(server={"tensor_parallel_size": 1}),
            lambda s: s.update(funding_selection_protocol=None),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.spec = copy.deepcopy(original)
                mutate(self.spec)
                with self.assertRaises(self.q.QueueError):
                    self.binding()

    def test_capacity_must_be_exactly_512_on_all_twenty_and_nowhere_else(self):
        original = copy.deepcopy(self.spec)
        for change in ("missing", "256", "mechanism", "one_extra"):
            with self.subTest(change=change):
                self.spec = copy.deepcopy(original)
                artifact = self.spec["jobs"][0]["artifacts"][0]
                if change == "missing":
                    del artifact["max_json_bytes"]
                elif change == "256":
                    artifact["max_json_bytes"] = 256 * 1024 * 1024
                elif change == "mechanism":
                    self.spec["jobs"][2]["artifacts"][0]["max_json_bytes"] = 512 * 1024 * 1024
                else:
                    artifact["equals"]["/additional"] = True
                with self.assertRaises(self.q.QueueError):
                    self.binding()

    def test_descriptor_rejects_coherently_changed_pin_or_copied_record_path(self):
        good = copy.deepcopy(self.spec["continuation"])
        for field in ("sha256", "path"):
            self.spec["continuation"] = copy.deepcopy(good)
            record = self.spec["continuation"]["parent_status"]
            record[field] = "a" * 64 if field == "sha256" else str(self.paths["parent_status"].with_name("copy.json"))
            with self.assertRaises(self.q.QueueError):
                self.plan()

    def test_parent_must_be_terminal_failed_with_clean_stopped_servers(self):
        for key, value in (("status", "running"), ("cleanup_errors", ["unreaped"]),
                           ("servers", {"old": {"status": "ready", "cleanup_errors": []}})):
            with self.subTest(key=key):
                status = copy.deepcopy(self.status)
                status[key] = value
                self.authorize_fixture_record("parent_status", status)
                with self.assertRaises(self.q.QueueError):
                    self.binding()

    def test_pending_identity_or_any_launch_field_is_rejected_even_null(self):
        for key, value in (("identity", None), ("started_epoch_ns", 1), ("server", "old"),
                           ("exit_code", 0), ("status", "complete")):
            with self.subTest(key=key):
                status = copy.deepcopy(self.status)
                status["jobs"][ALL_IDS[2]][key] = value
                self.authorize_fixture_record("parent_status", status)
                with self.assertRaisesRegex(self.q.QueueError, "pending membership"):
                    self.binding()

    def test_wrong_authorization_or_incomplete_freeze_rejected(self):
        self.authorize_fixture_record("authorization", {"user_message": "ok",
                                                       "proposal_sha256": self.q.PENDING_PROPOSAL_SHA256})
        with self.assertRaisesRegex(self.q.QueueError, "authorization"):
            self.binding()
        self.authorize_fixture_record("authorization", {"user_message": "continue",
                                                       "proposal_sha256": self.q.PENDING_PROPOSAL_SHA256})
        freeze = copy.deepcopy(self.freeze)
        del freeze["source_files_sha256"]["src/scientific_01.py"]
        self.authorize_fixture_record("parent_source_freeze", freeze)
        with self.assertRaisesRegex(self.q.QueueError, "87-source"):
            self.binding()

    def test_parent_and_new_scientific_source_mutation_reject(self):
        for root in (self.parent, self.new):
            path = root / "src/scientific_03.py"
            original = path.read_bytes()
            path.write_bytes(original + b"# changed\n")
            with self.assertRaises(self.q.QueueError):
                self.binding()
            path.write_bytes(original)

    def test_missing_original_source_and_missing_new_required_file_reject(self):
        for path in (self.parent / "src/scientific_04.py",
                     self.new / "src/scientific_04.py",
                     self.new / "docs/frozen_continuation.md"):
            original = path.read_bytes()
            path.unlink()
            with self.assertRaises((self.q.QueueError, OSError)):
                self.binding()
            path.write_bytes(original)

    def test_fresh_roots_empty_or_broken_symlink_rejected(self):
        plan = self.plan()
        root = Path(plan.jobs[0]["output_roots"][0])
        root.mkdir()
        with self.assertRaisesRegex(self.q.QueueError, "Historical or partial"):
            self.binding(plan, fresh_jobs=plan.jobs)
        root.rmdir()
        root.symlink_to(self.root / "absent")
        with self.assertRaisesRegex(self.q.QueueError, "Symlink"):
            self.binding(plan, fresh_jobs=plan.jobs)

    def test_same_bytes_replaced_parent_record_and_alias_rejected_after_startup(self):
        plan = self.plan()
        original = self.binding(plan)
        path = self.paths["authorization"]
        raw = path.read_bytes()
        other = path.with_suffix(".replacement")
        other.write_bytes(raw)
        other.replace(path)
        with self.assertRaisesRegex(self.q.QueueError, "binding changed since startup"):
            self.q.validate_pending_continuation(plan, original)
        original = self.binding(plan)
        link = self.parent / "outputs"
        link.unlink()
        link.symlink_to(self.private / "outputs", target_is_directory=True)
        with self.assertRaisesRegex(self.q.QueueError, "binding changed since startup"):
            self.q.validate_pending_continuation(plan, original)

    def test_main_rejects_bad_parent_before_runtime_write_lock_or_launch(self):
        plan_path = self.root / "synthetic_plan.json"
        plan_path.write_text(json.dumps(self.spec))
        self.paths["parent_status"].write_text("{}")
        with mock.patch.object(self.q, "runtime_guard") as runtime, \
                mock.patch.object(self.q, "private_directory") as mkdir, \
                mock.patch.object(self.q, "Lock") as lock, \
                mock.patch.object(self.q.Queue, "spawn") as spawn:
            self.assertEqual(self.q.main(["--plan", str(plan_path), "--run"]), 2)
            runtime.assert_not_called()
            mkdir.assert_not_called()
            lock.assert_not_called()
            spawn.assert_not_called()

    def test_all_lifecycle_boundaries_reject_mutated_parent_before_admission(self):
        plan = self.plan()
        provenance = {"continuation": self.binding(plan)}
        state = {"jobs": {j["id"]: {"status": "pending"} for j in plan.jobs}}
        queue = self.q.Queue(plan, provenance, state)
        self.paths["parent_status"].write_text("{}")
        actions = [
            lambda: self.q.load_state(plan, provenance, False),
            queue.run,
            lambda: queue.start_job(None, plan.jobs[0]),
            lambda: queue.finish_job(None, 0),
            lambda: self.q.validate_completed(plan, state, provenance, require_all=True),
        ]
        with mock.patch.object(queue, "spawn") as spawn:
            for action in actions:
                with self.subTest(action=action):
                    with self.assertRaisesRegex(self.q.QueueError, "parent hash changed"):
                        action()
            spawn.assert_not_called()
        self.assertFalse(plan.directory.exists())

    def test_fixed_432_hour_outer_budget_no_resume_or_extension(self):
        plan = self.plan()
        provenance = {"continuation": self.binding(plan)}
        self.assertEqual(self.q.queue_timeout_seconds(plan), 432 * 3600)
        self.assertTrue(all(j["command"][-2:] == ["--timeout_hours", "72"] for j in plan.jobs[:2]))
        self.assertTrue(all("--timeout_hours" not in j["command"] for j in plan.jobs[2:]))
        plan.directory.mkdir(parents=True, mode=0o700)
        now = time.time()
        state = self.q.load_state(plan, provenance, False)
        self.assertGreaterEqual(state["deadline_epoch"], now + 432 * 3600)
        self.assertLess(state["deadline_epoch"], time.time() + 432 * 3600 + 1)
        saved = (plan.directory / "status.json").read_bytes()
        for resume in (True, False):
            with self.assertRaises(self.q.QueueError):
                self.q.load_state(plan, provenance, resume)
            self.assertEqual((plan.directory / "status.json").read_bytes(), saved)
        self.assertFalse(state["full_first8_complete"])
        self.assertFalse(state["all12_complete"])
        self.spec["timeout_seconds"] = 999999999
        with self.assertRaises(self.q.QueueError):
            self.plan()

    def overdue_pf_queue(self, *, exit_code=None):
        plan = self.plan()
        provenance = {"continuation": self.binding(plan)}
        job = plan.jobs[2]
        self.assertEqual(job["family"], "mechanisms")
        self.assertNotIn("--timeout_hours", job["command"])
        started = time.time_ns() - (self.q.TIMEOUT_SECONDS + 1) * 1_000_000_000
        state = {"deadline_epoch": time.time() + 432 * 3600,
                 "jobs": {j["id"]: {"status": "pending"} for j in plan.jobs},
                 "servers": {"synthetic_server": {"status": "ready"}}}
        state["jobs"][job["id"]] = {"status": "running", "started_epoch_ns": started}
        queue = self.q.Queue(plan, provenance, state)
        queue.lock_root = self.root / "synthetic_gpu_locks"
        plan.directory.mkdir(parents=True, mode=0o700)
        client = types.SimpleNamespace(identity={"pid": 901, "role": "synthetic_client"},
                                       process=mock.Mock(), log=mock.Mock())
        client.process.poll.return_value = exit_code
        owned = types.SimpleNamespace(identity={"pid": 902, "role": "synthetic_server"},
                                      process=mock.Mock(), log=mock.Mock())
        owned.process.poll.return_value = None
        server = types.SimpleNamespace(name="synthetic_server", job=job, client=client,
                                       owned=owned, locks=[], reservation_token=None, ready=True)
        queue.servers = [server]
        return queue, server, started

    def assert_timeout_cleanup(self, queue, server, started, cleanup):
        record = queue.state["jobs"][server.job["id"]]
        self.assertEqual(queue.state["status"], "failed")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["started_epoch_ns"], started)
        self.assertNotIn("client_exit_observed_epoch_ns", record)
        self.assertIn("72-hour deadline", queue.state["error"])
        self.assertEqual(queue.state["servers"][server.name]["status"], "stopped")
        self.assertEqual(queue.servers, [])
        self.assertEqual([call.args[0] for call in cleanup.call_args_list],
                         [server.client.identity, server.owned.identity])
        self.assertFalse(list(queue.plan.directory.glob("*.receipt.json")))
        saved = json.loads((queue.plan.directory / "status.json").read_text())
        self.assertEqual(saved["jobs"][server.job["id"]]["started_epoch_ns"], started)

    def test_expired_pf_fails_before_poll_and_uses_owned_cleanup_without_reset(self):
        queue, server, started = self.overdue_pf_queue()
        outer = queue.state["deadline_epoch"]
        with mock.patch.object(self.q, "query_gpus", side_effect=AssertionError("GPU query forbidden")) as gpu, \
                mock.patch.object(self.q, "artifact_gate", side_effect=AssertionError("Admission forbidden")) as gate, \
                mock.patch.object(self.q, "stop_owned", return_value=[]) as cleanup:
            self.assertEqual(queue.run(), 1)
            gpu.assert_not_called()
            gate.assert_not_called()
            self.assert_timeout_cleanup(queue, server, started, cleanup)
        self.assertEqual(queue.state["deadline_epoch"], outer)

    def test_overdue_zero_exit_rejected_at_finish_before_admission_and_cleanup(self):
        queue, server, started = self.overdue_pf_queue(exit_code=0)
        with mock.patch.object(queue, "step", side_effect=lambda: queue.finish_job(server, 0)), \
                mock.patch.object(self.q, "artifact_gate", side_effect=AssertionError("Admission forbidden")) as gate, \
                mock.patch.object(self.q, "stop_owned", return_value=[]) as cleanup:
            self.assertEqual(queue.run(), 1)
            gate.assert_not_called()
            self.assert_timeout_cleanup(queue, server, started, cleanup)
        self.assertEqual(queue.state["jobs"][server.job["id"]]["exit_code"], 0)

    def test_client_deadline_exact_boundary_historical_evidence_and_probe_budget(self):
        plan = self.plan()
        now = time.time_ns()
        start = now - self.q.TIMEOUT_SECONDS * 1_000_000_000
        record = {"started_epoch_ns": start}
        self.assertEqual(self.q.check_pending_client_deadline(plan, plan.jobs[2]["id"], record,
                                                             checked_ns=now), now)
        with self.assertRaisesRegex(self.q.QueueError, "72-hour deadline"):
            self.q.check_pending_client_deadline(plan, plan.jobs[2]["id"], record, checked_ns=now + 1)
        self.assertEqual(record, {"started_epoch_ns": start})
        queue, server, _ = self.overdue_pf_queue()
        queue.state["jobs"][server.job["id"]]["started_epoch_ns"] = now - (self.q.TIMEOUT_SECONDS - 2) * 1_000_000_000
        with mock.patch.object(self.q.time, "time_ns", return_value=now):
            self.assertEqual(queue.client_budget_seconds(), 2.0)
        parent = self.q.parse_plan(json.dumps(self.parent_spec).encode())
        self.assertIsNone(self.q.check_pending_client_deadline(parent, ALL_IDS[0], {}, checked_ns=now))

    def test_legacy_eight_retains_72_hour_budget(self):
        parent = self.q.parse_plan(json.dumps(self.parent_spec).encode())
        self.assertEqual(len(parent.jobs), 8)
        self.assertEqual(self.q.queue_timeout_seconds(parent), 72 * 3600)
        self.assertIsNone(self.q.validate_pending_continuation(parent))

    def test_absolute_mechanism_command_keeps_exact_parent_path_and_source_binding(self):
        plan = self.plan()
        self.binding(plan)
        for job in plan.jobs[2:]:
            self.assertEqual(self.q.job_script_root(plan, job),
                             (self.parent / "utopia/experiments/funding_feedback.py", self.parent))
        altered = copy.deepcopy(plan.jobs[2])
        altered["command"][1] = str(self.root / "other/utopia/experiments/funding_feedback.py")
        with self.assertRaisesRegex(self.q.QueueError, "exact original parent mechanism"):
            self.q.job_script_root(plan, altered)
        self.paths["parent_plan"].write_text("{}")
        with self.assertRaisesRegex(self.q.QueueError, "Parent plan changed"):
            self.q.job_script_root(plan, plan.jobs[2])

    def test_preferred_5_6_pair_uses_real_locks_and_live_idle_recheck(self):
        def gpu(index, busy=False):
            return types.SimpleNamespace(index=index, uuid=f"GPU-synthetic-{index}", idle=not busy)
        inventory = [gpu(3), gpu(5), gpu(6)]
        locks = self.root / "locks"
        locks.mkdir(mode=0o700)
        pair, held = self.q.reserve_pair(inventory, locks, query=lambda: inventory, prefer_pending_pair=True)
        self.addCleanup(lambda: [lock.close() for lock in held])
        self.assertEqual([g.index for g in pair], [5, 6])
        self.assertIsNone(self.q.reserve_pair(inventory, locks, query=lambda: inventory,
                                             prefer_pending_pair=True))
        for lock in held:
            lock.close()
        busy = [gpu(3), gpu(5, busy=True), gpu(6)]
        pair2, held2 = self.q.reserve_pair(inventory, locks, query=lambda: busy, prefer_pending_pair=True)
        self.addCleanup(lambda: [lock.close() for lock in held2])
        self.assertEqual({g.index for g in pair2}, {3, 6})

    def test_required_parent_recheck_is_after_journaling_before_spawn(self):
        module = ast.parse(SOURCE.read_bytes())
        queue = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "Queue")
        start = next(n for n in queue.body if isinstance(n, ast.FunctionDef) and n.name == "start_job")
        code = ast.unparse(start)
        journal = code.index("self.save()")
        self.assertLess(journal, code.rindex("recheck_pending("))
        self.assertLess(code.rindex("recheck_pending("), code.index("self.spawn("))
        final = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "validate_completed")
        calls = [n.func.id for n in ast.walk(final) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertGreaterEqual(calls.count("recheck_pending"), 2)
        self.assertIn("artifact_gate", calls)
        self.assertIn("validate_probe_attestation", calls)
        self.assertIn("validate_funding_probe_attestation", calls)
        self.assertIn("source_hashes", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
