"""Mocked queue tests. No CUDA, nvidia-smi, SSH, HTTP, or real child processes."""


import utopia.funding.feedback as feedback

from utopia.utils.paths import project_root
import ast
import copy
import io
import json
import os
import signal
import shutil
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import utopia.runtime.experiment_queue as q


def unit_funding_fixture():
    """Synthetic unit-test input only, never an experiment/probe production fixture."""
    fields = {"application_id": "integer", "applicant_id": "string",
              "rank": "integer", "reason": "string"}
    return {
        "schema_version": 1, "kind": "funding_panel25",
        "messages": [{"role": "system", "content": "Mock unit-test system"},
                     {"role": "user", "content": "Mock unit-test 25 applications"}],
        "json_schema": {
            "type": "object", "required": ["ranked_applications"],
            "$defs": {"RankedApplication": {"type": "object", "required": list(fields),
                      "properties": {key: {"type": value} for key, value in fields.items()}}},
            "properties": {"ranked_applications": {"type": "array",
                          "items": {"$ref": "#/$defs/RankedApplication"}}}},
        "tokenization": {
            "model": q.MODEL, "revision": q.REVISION, "transformers_version": "4.57.3",
            "enable_thinking": True, "add_generation_prompt": True,
            "max_model_len": 32768, "requested_max_tokens": 8192,
            "input_tokens": 4795, "total_requested_tokens": 12987,
            "tokenizer_files_sha256": {"tokenizer.json": "a" * 64, "tokenizer_config.json": "b" * 64},
            "chat_template_sha256": "c" * 64},
        "expected": {
            "application_ids": list(range(25)), "ranks": list(range(1, 26)),
            "rankings_json_pointer": "/ranked_applications", "application_id_key": "application_id",
            "applicant_id_key": "applicant_id", "rank_key": "rank", "application_count": 25,
            "finish_reason": "stop",
            "application_to_applicant": {str(i): f"mock-founder-{i}" for i in range(25)}}}


def funding_payload(fixture):
    payload = probe_payload("unused")
    final = {"ranked_applications": [
        {"application_id": i, "applicant_id": fixture["expected"]["application_to_applicant"][str(i)],
         "rank": 25 - i, "reason": "Mock reason"} for i in range(25)]}
    if fixture.get("output_representation") == "ordered_application_ids_v1":
        final = {"ranked_application_ids": list(reversed(range(25)))}
    payload["choices"][0]["message"]["content"] = json.dumps(final)
    tokens = fixture["tokenization"]["input_tokens"]
    payload["usage"] = {"prompt_tokens": tokens, "completion_tokens": 1000, "total_tokens": tokens + 1000}
    return payload


@pytest.fixture(autouse=True)
def prohibit_external_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Test attempted external process/network/GPU work")
    monkeypatch.setattr(q.subprocess, "Popen", forbidden)
    monkeypatch.setattr(q.subprocess, "check_output", forbidden)
    monkeypatch.setattr(q, "query_gpus", forbidden)
    monkeypatch.setattr(q.urllib.request, "build_opener", forbidden)


@pytest.fixture
def specification(tmp_path):
    private = tmp_path / "private"
    worktree = tmp_path / "worktree"
    (private / "outputs").mkdir(parents=True)
    (private / "data").mkdir()
    (worktree / "src").mkdir(parents=True)
    (worktree / "outputs").symlink_to(private / "outputs", target_is_directory=True)
    (worktree / "data").symlink_to(private / "data", target_is_directory=True)
    (worktree / "src/owner.py").write_text("# inert fixture\n")
    python = str(private / "env/bin/python")
    fixture = private / "unit-funding-fixture.json"
    fixture.write_bytes(q.json_bytes(unit_funding_fixture()))
    fixture.chmod(0o444)
    jobs = []
    for family in ("influx", "mechanisms"):
        for cell in range(4):
            name = f"{family}-{cell}"
            root = private / "outputs/docs" / name
            command = [python, "src/owner.py", "--seed", "42",
                       "--vllm-url", "__SERVER_URL__"]
            if family == "mechanisms":
                command += ["--server-manifest", "__SERVER_MANIFEST__"]
            jobs.append({
                "id": name, "family": family, "seed": 42, "command": command,
                "require_manifest_argument": family == "mechanisms", "env": {},
                "output_roots": [str(root)],
                "artifacts": [{"path": str(root / "done.json"),
                               "equals": {"/status": "complete", "/seed": 42}}],
            })
    return {"schema_version": 1, "seed": 42, "queue_id": "new-eight-cell-cohort",
            "worktree": str(worktree), "private_root": str(private), "jobs": jobs,
            "funding_probe_fixture": {"path": str(fixture), "sha256": q.file_hash(fixture)}}


@pytest.fixture
def plan(specification):
    return q.parse_plan(q.json_bytes(specification))


@pytest.fixture
def queue(plan, monkeypatch):
    return make_test_queue(plan, monkeypatch)


def make_test_queue(plan, monkeypatch):
    provenance = {"sources": {}, "versions": {}, "git_head": "test-head", "branch": "llm",
                  "host": "mock-host", "boot_id": "mock-boot", "runtime_files": {}}
    if "server_profile" in plan.spec:
        provenance.update(q.selected_runtime_binding(plan))
    q.private_directory(plan.directory, plan.private_root)
    state = q.load_state(plan, provenance, False)
    controller = q.Queue(plan, provenance, state)
    controller.lock_root = plan.private_root / "locks"
    controller.lock_root.mkdir(mode=0o700)
    monkeypatch.setattr(q, "source_hashes", lambda _: {})
    monkeypatch.setattr(q, "stop_owned", lambda *a, **k: [])
    monkeypatch.setattr(q, "session_members", lambda _: [])
    return controller


@pytest.fixture
def propensity_specification(specification):
    """Synthetic queue declarations only; production CLI/artifacts come from the driver owner."""
    spec = copy.deepcopy(specification)
    private, worktree = Path(spec["private_root"]), Path(spec["worktree"])
    (worktree / "docs").mkdir()
    (worktree / "docs/switching_propensity_protocol.md").write_text("Unit-test protocol, not a scientific protocol.")
    spec["queue_id"] = "unit-propensity-five-operations-four-worlds"
    init_root = private / "outputs/docs/unit-propensity-initialization"
    cache = str(init_root / "initial_choices.json")
    spec["propensity_initialization"] = {"job_id": "initialize", "artifact": cache}
    jobs = []
    for cell in (None, "LN", "LF", "HN", "HF"):
        root = init_root if cell is None else private / f"outputs/docs/unit-propensity-{cell}"
        command = [str(private / "env/bin/python"), "src/owner.py",
                   "initialize" if cell is None else "run", "--seed", "42",
                   "--vllm-url", "__SERVER_URL__", "--server-manifest", "__SERVER_MANIFEST__"]
        job = {
            "id": "initialize" if cell is None else cell,
            "family": "propensity_init" if cell is None else "propensity",
            "seed": 42, "command": command, "require_manifest_argument": True, "env": {},
            "output_roots": [str(root)],
            "artifacts": [{"path": str(root / ("propensity_manifest.json" if cell
                                              else "initialization_manifest.json")),
                           "equals": {"/status": "complete", "/seed": 42, "/founder_count": 1200,
                                      **({"/kind": "condition", "/cell": cell, "/years_completed": 10} if cell else
                                         {"/kind": "initialization_only", "/choice_count": 1200,
                                          "/scientific_years_completed": 0})}}]}
        audits = {
            "llm_request_audit.summary.json": {
                "/status": "complete", "/n_clients": 1, "/guard_failures": 0, "/pending_requests": 0},
            "funding_validation.summary.json": {
                "/completion_allowed": True, "/installed": False,
                "/output_representation": "ordered_application_ids_v1",
                "/failed_batches": 0, "/audit_failures": 0, "/invalid_final_panels": 0,
                "/processing_failures": 0, "/unprocessed_panels": 0, "/imputed_rankings": 0,
                "/fallback_rankings": 0, "/restore_conflicts": []}}
        if cell is None:
            audits["funding_validation.summary.json"]["/final_panels"] = 0
        job["artifacts"].extend({"path": str(root / name), "equals": predicates}
                                for name, predicates in audits.items())
        job["artifacts"].extend([
            {"path": str(root / "llm_request_audit.jsonl")},
            {"path": str(root / "funding_validation.jsonl"), "min_bytes": 0}])
        if cell:
            job["cell"] = cell
            command += ["--cell", cell, "--initial-choices", cache]
        else:
            job["artifacts"].append({"path": cache, "equals": {"/status": "complete", "/seed": 42}})
        jobs.append(job)
    spec["jobs"] = jobs
    return spec


@pytest.fixture
def propensity_plan(propensity_specification):
    return q.parse_plan(q.json_bytes(propensity_specification))


@pytest.fixture
def propensity_queue(propensity_plan, monkeypatch):
    return make_test_queue(propensity_plan, monkeypatch)


def inventory(free=range(8)):
    return [q.GPU(i, f"GPU-{i:02d}", "Mock GPU", 81920,
                  0 if i in free else 20000, 0 if i in free else 50, "test-driver",
                  frozenset() if i in free else frozenset({9000 + i}))
            for i in range(8)]


def identity(pid, start=None):
    return {"pid": pid, "start_ticks": pid if start is None else start,
            "uid": os.getuid(), "session": pid, "boot_id": "mock-boot", "state": "S"}


class Process:
    def __init__(self, pid):
        self.pid = pid
        self.code = None

    def poll(self):
        return self.code


def owned(pid):
    return q.Owned(Process(pid), identity(pid), io.BytesIO())


def real_probe_builder(seed_components=q.PROBE_SEED_COMPONENTS):
    return q.load_shared_probe_builder(SimpleNamespace(worktree=project_root(q.__file__)),
                                        seed_components)


def probe_payload(nonce, *, finish_reason="stop", reasoning="A short reasoning segment."):
    return {"model": q.MODEL, "choices": [{"index": 0, "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": json.dumps({"probe_nonce": nonce}),
                        "reasoning_content": reasoning}}],
            "usage": {"completion_tokens": 64, "prompt_tokens": 32, "total_tokens": 96}}


def passed_probe_record(root, port):
    builder, metadata = real_probe_builder()
    nonce = "a" * 32
    request = q.probe_request(nonce, builder(q.probe_schema(nonce)))
    payload = probe_payload(nonce)
    raw = q.json_bytes(payload)
    return {
        "status": "passed", "http_status": 200, "process_identity": root,
        "endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
        "request": request, "request_sha256": q.digest_bytes(q.json_bytes(request)),
        "response_body": raw.decode(), "response_sha256": q.digest_bytes(raw),
        "request_seed_base": 42, "request_seed_components": list(q.PROBE_SEED_COMPONENTS),
        "validation": q.verify_probe_response(payload, nonce), "shared_client_builder": metadata,
    }


def passed_funding_record(root, port, fixture, binding):
    builder, metadata = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    request = q.funding_probe_request(fixture, builder(fixture["json_schema"]))
    payload = funding_payload(fixture)
    raw = q.json_bytes(payload)
    return {
        "status": "passed", "http_status": 200, "process_identity": root, "fixture": binding,
        "endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
        "request": request, "request_sha256": q.digest_bytes(q.json_bytes(request)),
        "response_body": raw.decode(), "response_sha256": q.digest_bytes(raw),
        "request_seed_base": 42, "request_seed_components": list(q.FUNDING_SEED_COMPONENTS),
        "validation": q.verify_funding_probe_response(payload, fixture), "shared_client_builder": metadata}


def stub_attestation_probe(monkeypatch):
    real_loader = q.load_shared_probe_builder
    root = SimpleNamespace(worktree=project_root(q.__file__))
    monkeypatch.setattr(q, "load_shared_probe_builder", lambda _, *args: real_loader(root, *args))
    monkeypatch.setattr(q, "constrained_output_probe",
                        lambda root, port, builder, **kwargs: passed_probe_record(root, port))
    monkeypatch.setattr(q, "funding_panel25_probe",
                        lambda root, port, builder, fixture, binding, **kwargs:
                        passed_funding_record(root, port, fixture, binding))
    monkeypatch.setattr(q.Queue, "workers_healthy", lambda *a, **k: None)


def mock_server(queue, number=1, pair=None):
    name = f"server-{number:03d}"
    server = q.Server(name, pair or inventory()[:2], [], 18079 + number,
                      owned(number * 100), ["mock-vllm"], q.time.time(),
                      queue.plan.directory / f"{name}.manifest.json", ready=True)
    server.probe = passed_probe_record(server.owned.identity, server.port)
    server.funding_probe = passed_funding_record(server.owned.identity, server.port,
                                                *q.load_funding_fixture(queue.plan))
    q.atomic_write(server.manifest_path, q.json_bytes(
        {"actual": "mock server", "endpoint": server.url, "structured_output_probe": server.probe,
         "funding_panel25_probe": server.funding_probe}), exclusive=True)
    queue.servers.append(server)
    queue.state["servers"][name] = {
        "status": "ready", "identity": server.owned.identity,
        "manifest": str(server.manifest_path),
        "manifest_sha256": q.file_hash(server.manifest_path),
    }
    return server


def write_artifacts(job, **overrides):
    # Filesystem timestamps may lag time_ns until the next kernel clock tick.
    time.sleep(0.02)
    path = Path(job["artifacts"][0]["path"])
    path.parent.mkdir(parents=True)
    result = {"status": "complete", "seed": 42}
    if q.is_propensity_initializer(job):
        result.update(kind="initialization_only", founder_count=1200, choice_count=1200,
                      scientific_years_completed=0, pre_choice_state_hash="b" * 64, initial_state_hash="a" * 64)
    elif q.is_propensity_world(job):
        cache = Path(job["command"][job["command"].index("--initial-choices") + 1])
        result.update(kind="condition", cell=job["cell"], founder_count=1200, years_completed=10,
                      pre_choice_state_hash="b" * 64,
                      initial_choices_sha256=q.file_hash(cache), initial_state_hash="a" * 64)
    result.update(overrides)
    path.write_bytes(q.json_bytes(result))
    if job["family"] in {"propensity", "propensity_init"}:
        for artifact in job["artifacts"][1:]:
            target = Path(artifact["path"])
            if target.name.endswith(".summary.json"):
                summary = {key[1:]: value for key, value in artifact["equals"].items()}
                if target.name == "llm_request_audit.summary.json":
                    summary["n_requests"] = 1200
                target.write_bytes(q.json_bytes(summary))
            elif target.name == "llm_request_audit.jsonl":
                target.write_text('{"synthetic_unit_test_audit":true}\n')
            elif target.name == "funding_validation.jsonl":
                target.write_bytes(b"")


def mock_client_launch(queue, monkeypatch):
    spawned = []
    monkeypatch.setattr(q, "query_gpus", inventory)
    monkeypatch.setattr(queue, "workers_healthy", lambda *a, **k: None)
    monkeypatch.setattr(q, "endpoint_ready", lambda *a: True)
    def spawn(command, env, logpath):
        result = owned(1000 + len(spawned))
        spawned.append((command, env, result))
        return result
    monkeypatch.setattr(queue, "spawn", spawn)
    return spawned


def test_eight_single_world_jobs_and_seed_derivation(plan):
    assert len(plan.jobs) == 8
    assert q.SERVER_SPEC["request_seed"] == q.SERVER_SPEC["request_seed_base"] == 42
    assert q.SERVER_SPEC["request_seed_policy"] == (
        "derive_seed(run_seed, phase_context, item_index, retry)")
    assert "dependent observations" in q.LIMITATION
    assert "No between-world confidence interval" in q.LIMITATION


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s.update(seed=43), "seed=42"),
    (lambda s: s["jobs"].pop(), "eight"),
    (lambda s: s["jobs"][0].update(seed=1001), "only seed42"),
    (lambda s: s["jobs"][0]["command"].__setitem__(3, "42,43"), "seed arguments"),
    (lambda s: s["jobs"][0]["command"].extend(["--seeds", "42", "43"]), "seed list"),
    (lambda s: s["jobs"][0]["command"].extend(["--request_seed", "12"]), "seed arguments"),
    (lambda s: s["jobs"][0]["command"].extend(["--resume"]), "resume"),
    (lambda s: s["jobs"][0]["command"].__setitem__(0, "/bin/sh"), "no shell"),
    (lambda s: s["jobs"][0]["env"].update(CUDA_VISIBLE_DEVICES="7"), "override"),
    (lambda s: s["jobs"][0]["env"].update(PYTHONHASHSEED="43"), "override"),
    (lambda s: s["jobs"][0]["artifacts"][0].update(equals={}), "completion predicate"),
    (lambda s: s["jobs"][4].update(require_manifest_argument=False), "mechanisms require"),
    (lambda s: s.update(server={"max_model_len": 65536}), "Unsupported server"),
    (lambda s: s.update(server={"gpu_memory_utilization": 0.9}), "Unsupported server"),
])
def test_rejects_protocol_drift(specification, mutation, match):
    mutation(specification)
    with pytest.raises(q.QueueError, match=match):
        q.parse_plan(q.json_bytes(specification))


def test_url_is_standalone_and_manifest_omission_explicit(specification):
    specification["jobs"][0]["command"][-1] = "prefix__SERVER_URL__"
    with pytest.raises(q.QueueError, match="standalone"):
        q.parse_plan(q.json_bytes(specification))
    specification["jobs"][0]["command"][-1] = "__SERVER_URL__"
    del specification["jobs"][0]["require_manifest_argument"]
    with pytest.raises(q.QueueError, match="__SERVER_MANIFEST__"):
        q.parse_plan(q.json_bytes(specification))


def test_rejects_output_overlap_through_worktree_alias(specification):
    first = specification["jobs"][0]
    second = specification["jobs"][1]
    root = str(Path(specification["worktree"]) / "outputs/docs/influx-0")
    second["output_roots"] = [root]
    second["artifacts"][0]["path"] = root + "/done.json"
    with pytest.raises(q.QueueError, match="overlapping"):
        q.parse_plan(q.json_bytes(specification))


def test_duplicate_json_keys_rejected():
    with pytest.raises(q.QueueError, match="Duplicate"):
        q.parse_plan(b'{"seed":42,"seed":43}')


@pytest.mark.parametrize("mode", ["--validate-only", "--dry-run"])
def test_offline_modes_do_not_touch_runtime_or_outputs(plan, tmp_path, monkeypatch, capsys, mode):
    path = tmp_path / "input.json"
    path.write_bytes(plan.raw)
    def forbidden(*args):
        raise AssertionError("Offline mode inspected runtime")
    monkeypatch.setattr(q, "runtime_guard", forbidden)
    assert q.main(["--plan", str(path), mode]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["plan_sha256"] == plan.sha256
    assert not plan.directory.exists()


def test_server_command_pins_settings_and_both_revisions(plan):
    cmd = q.server_command(plan, 18091)
    assert cmd[:3] == [str(plan.python), str(plan.python.parent / "vllm"), "serve"]
    for flag, expected in {
        "--revision": q.REVISION, "--tokenizer-revision": q.REVISION,
        "--dtype": "bfloat16", "--tensor-parallel-size": "2",
        "--max-model-len": "32768", "--max-num-seqs": "128",
        "--gpu-memory-utilization": "0.80", "--seed": "42",
        "--distributed-executor-backend": "mp", "--port": "18091",
    }.items():
        assert cmd[cmd.index(flag) + 1] == expected


def test_known_idle_gpu_set_and_strict_thresholds():
    assert [g.index for g in inventory({0, 3, 5, 6, 7}) if g.idle] == [0, 3, 5, 6, 7]
    gpu = inventory()[0]
    assert replace(gpu, used_mib=1023, utilization=5).idle
    assert not replace(gpu, used_mib=1024).idle
    assert not replace(gpu, utilization=6).idle
    assert not replace(gpu, pids=frozenset({123})).idle


def test_parse_nvidia_smi_and_fail_closed_on_unknown_values():
    text = "\n".join(f"{i}, GPU-{i:02d}, Mock, 81920, 0, 0, 555, 00000000:{i:02x}:00.0"
                     for i in range(8))
    parsed = q.parse_gpus(text, "GPU-01, 999\n")
    assert not parsed[1].idle and parsed[0].idle
    for bad_inventory, bad_processes in [
        (text.replace("81920, 0, 0", "81920, N/A, 0", 1), ""),
        (text, "GPU-00, N/A"),
        (text, "MIG-00, 123"),
        ("\n".join(text.splitlines()[:7]), ""),
    ]:
        with pytest.raises(q.QueueError):
            q.parse_gpus(bad_inventory, bad_processes)


def test_numeric_client_gpu_mapping_must_match_pci_order():
    text = "\n".join(f"{i}, GPU-{i:02d}, Mock, 81920, 0, 0, 555, 00000000:{7-i:02x}:00.0"
                     for i in range(8))
    with pytest.raises(q.QueueError, match="numeric client mask unsafe"):
        q.parse_gpus(text, "")


def test_advisory_locks_block_double_claim_and_release(queue):
    gpus = inventory({0, 3, 5, 6, 7})
    first_pair, first_locks = q.reserve_pair(gpus, queue.lock_root, query=lambda: gpus)
    second_pair, second_locks = q.reserve_pair(gpus, queue.lock_root, query=lambda: gpus)
    assert [g.index for g in first_pair] == [0, 3]
    assert [g.index for g in second_pair] == [5, 6]
    assert q.reserve_pair(gpus, queue.lock_root, query=lambda: gpus) is None
    for lock in first_locks + second_locks:
        lock.close()
    pair, locks = q.reserve_pair(gpus, queue.lock_root, query=lambda: gpus)
    assert [g.index for g in pair] == [0, 3]
    for lock in locks:
        lock.close()


def test_race_recheck_rejects_gpu_that_became_busy(queue):
    before = inventory({0, 3})
    after = inventory({3})
    assert q.reserve_pair(before, queue.lock_root, query=lambda: after) is None
    # The failed recheck must release both advisory locks.
    pair, locks = q.reserve_pair(before, queue.lock_root, query=lambda: before)
    assert [g.index for g in pair] == [0, 3]
    for lock in locks:
        lock.close()


def test_query_failure_releases_all_reservation_locks(queue):
    before = inventory({0, 3})
    def fail():
        raise q.QueueError("telemetry failed")
    with pytest.raises(q.QueueError):
        q.reserve_pair(before, queue.lock_root, query=fail)
    pair, locks = q.reserve_pair(before, queue.lock_root, query=lambda: before)
    assert len(pair) == 2
    for lock in locks:
        lock.close()


def test_last_gpu_recheck_prevents_spawn_and_releases_locks(queue, monkeypatch):
    before = inventory({0, 3})
    pair, locks = q.reserve_pair(before, queue.lock_root, query=lambda: before)
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def bind(self, address):
            pass
    monkeypatch.setattr(q.socket, "socket", Socket)
    monkeypatch.setattr(q, "query_gpus", lambda: inventory({3}))
    with pytest.raises(q.QueueError, match="immediately before"):
        queue.start_server(pair, locks)
    assert not queue.servers and all(lock.fd is None for lock in locks)


def test_no_historical_or_partial_artifact_adoption(queue):
    job = queue.plan.jobs[0]
    write_artifacts(job)
    with pytest.raises(q.QueueError, match="Historical or partial"):
        q.ensure_fresh(job, queue.plan)


def test_artifact_gate_detects_failed_predicate_empty_and_symlink_escape(queue, tmp_path):
    job = queue.plan.jobs[0]
    write_artifacts(job, status="partial")
    with pytest.raises(q.QueueError, match="predicate failed"):
        q.artifact_gate(job, queue.plan, started_ns=1)
    path = Path(job["artifacts"][0]["path"])
    path.write_bytes(b"")
    with pytest.raises(q.QueueError, match="empty"):
        q.artifact_gate(job, queue.plan, started_ns=1)
    path.unlink()
    external = tmp_path / "external.json"
    external.write_bytes(q.json_bytes({"status": "complete", "seed": 42}))
    path.symlink_to(external)
    with pytest.raises(q.QueueError, match="Symlink"):
        q.artifact_gate(job, queue.plan, started_ns=1)


def test_raw_influx_receipt_is_parent_attestation_and_manifest_env(queue, monkeypatch):
    server = mock_server(queue)
    spawned = mock_client_launch(queue, monkeypatch)
    job = queue.plan.jobs[0]
    queue.start_job(server, job)
    command, env, client = spawned[0]
    assert "__SERVER_URL__" not in command
    assert str(server.manifest_path) not in command
    assert env["UTOPIA_SERVER_MANIFEST"] == str(server.manifest_path)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONHASHSEED"] == "42"
    write_artifacts(job)
    queue.finish_job(server, 0)
    receipt = json.loads((queue.plan.directory / f"job-{job['id']}.receipt.json").read_text())
    assert "parent-level attestation only" in receipt["server_provenance_validation"]
    assert receipt["command"] == command
    assert receipt["job_spec_sha256"] == q.digest_bytes(q.json_bytes(job))
    assert receipt["request_seed_base"] == 42


def test_mechanism_manifest_is_passed_in_argv(queue, monkeypatch):
    server = mock_server(queue)
    spawned = mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[4])
    assert str(server.manifest_path) in spawned[0][0]
    assert "__SERVER_MANIFEST__" not in spawned[0][0]
    with pytest.raises(q.QueueError, match="One job"):
        queue.start_job(server, queue.plan.jobs[5])


def test_runtime_environment_is_offline_but_preserves_inherited_home(queue, monkeypatch):
    monkeypatch.setenv("HOME", "/unchanged/operator/home")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    env = queue.environment(inventory()[:2], queue.plan.jobs[0])
    assert env["HOME"] == "/unchanged/operator/home"
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["HF_HOME"] == str(queue.plan.hf_home)


def test_controlled_influx_gpu_placeholders_match_first_physical_client_gpu(
        specification, monkeypatch):
    job = specification["jobs"][0]
    job["require_manifest_argument"] = True
    job["command"] += ["--external_server_manifest", "__SERVER_MANIFEST__",
                       "--server_gpus", "__SERVER_GPUS__", "--client_gpu", "__CLIENT_GPU__"]
    plan = q.parse_plan(q.json_bytes(specification))
    q.private_directory(plan.directory, plan.private_root)
    provenance = {"sources": {}}
    controller = q.Queue(plan, provenance, q.load_state(plan, provenance, False))
    monkeypatch.setattr(q, "source_hashes", lambda _: {})
    server = mock_server(controller, pair=[inventory()[3], inventory()[7]])
    spawned = mock_client_launch(controller, monkeypatch)
    controller.start_job(server, plan.jobs[0])
    command, env, _ = spawned[0]
    assert command[command.index("--server_gpus") + 1] == "3,7"
    assert command[command.index("--client_gpu") + 1] == "3"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert controller.environment(server.pair)["CUDA_VISIBLE_DEVICES"] == "3,7"
    assert not set(command) & q.PLACEHOLDERS


def test_vllm_numeric_device_contract_keeps_reservations_and_manifests_uuid_bound(queue, monkeypatch):
    gpus = inventory()
    monkeypatch.setattr(q, "query_gpus", lambda: gpus)
    stub_attestation_probe(monkeypatch)
    for number, indices in enumerate(((0, 3), (5, 6)), 1):
        pair = [gpus[index] for index in indices]
        env = queue.environment(pair)
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        # vLLM 0.12 Platform.device_id_to_physical_device_id consumes each
        # CUDA_VISIBLE_DEVICES entry with int(); UUID masks raise ValueError.
        physical_ids = [int(env["CUDA_VISIBLE_DEVICES"].split(",")[local_id])
                        for local_id in range(2)]
        assert physical_ids == list(indices)
        reserved_pair, locks = q.reserve_pair(pair, queue.lock_root, query=lambda: gpus)
        try:
            assert [gpu.uuid for gpu in reserved_pair] == [gpu.uuid for gpu in pair]
            assert [lock.path.name for lock in locks] == [f"{gpu.uuid}.lock" for gpu in pair]
        finally:
            for lock in locks:
                lock.close()
        server = mock_server(queue, number=number, pair=pair)
        server.manifest_path.unlink()
        queue.attest(server)
        manifest = json.loads(server.manifest_path.read_text())
        assert manifest["gpu_ids"] == list(indices)
        assert manifest["gpu_uuids"] == [gpu.uuid for gpu in pair]
        assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, indices))


@pytest.mark.parametrize("arguments", [
    ["--server_gpus", "0,1", "--client_gpu", "0"],
    ["--server_gpus", "__SERVER_GPUS__"],
    ["--other", "__SERVER_GPUS__", "--client_gpu", "__CLIENT_GPU__"],
    ["--server_gpus", "__SOME_OTHER_VALUE__"],
])
def test_rejects_uncontrolled_gpu_substitution(specification, arguments):
    specification["jobs"][0]["command"].extend(arguments)
    with pytest.raises(q.QueueError):
        q.parse_plan(q.json_bytes(specification))


def test_job_requires_live_http_and_manifest_before_spawn(queue, monkeypatch):
    server = mock_server(queue)
    spawned = mock_client_launch(queue, monkeypatch)
    monkeypatch.setattr(q, "endpoint_ready", lambda *a: False)
    with pytest.raises(q.QueueError, match="health/model"):
        queue.start_job(server, queue.plan.jobs[0])
    assert not spawned
    monkeypatch.setattr(q, "endpoint_ready", lambda *a: True)
    server.manifest_path.write_text("{}")
    with pytest.raises(q.QueueError, match="manifest changed"):
        queue.start_job(server, queue.plan.jobs[0])
    assert not spawned


def test_exit_zero_without_artifacts_never_completes(queue, monkeypatch):
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    with pytest.raises(FileNotFoundError):
        queue.finish_job(server, 0)
    assert queue.state["jobs"][queue.plan.jobs[0]["id"]]["status"] == "running"
    assert not list(queue.plan.directory.glob("*.receipt.json"))


def test_failed_exit_cannot_be_hidden_by_valid_artifacts(queue, monkeypatch):
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    job = queue.plan.jobs[0]
    queue.start_job(server, job)
    write_artifacts(job)
    with pytest.raises(q.QueueError, match="exited 7"):
        queue.finish_job(server, 7)
    assert not list(queue.plan.directory.glob("*.receipt.json"))


def test_resume_only_exact_plan_and_pending_or_verified_complete(queue, monkeypatch):
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    job = queue.plan.jobs[0]
    queue.start_job(server, job)
    write_artifacts(job)
    queue.finish_job(server, 0)
    assert q.load_state(queue.plan, queue.provenance, True)["jobs"][job["id"]]["status"] == "complete"
    altered = q.parse_plan(queue.plan.raw + b"\n")
    with pytest.raises(q.QueueError, match="exact immutable plan hash"):
        q.load_state(altered, queue.provenance, True)
    path = Path(job["artifacts"][0]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(q.QueueError, match="artifact hash changed"):
        q.load_state(queue.plan, queue.provenance, True)


def test_resume_rejects_interrupted_or_source_changed_job(queue):
    changed = copy.deepcopy(queue.provenance)
    changed["sources"] = {"owner.py": "different"}
    with pytest.raises(q.QueueError, match="source/runtime"):
        q.load_state(queue.plan, changed, True)
    queue.state["jobs"][queue.plan.jobs[0]["id"]]["status"] = "launching"
    queue.save()
    with pytest.raises(q.QueueError, match="Started/incomplete"):
        q.load_state(queue.plan, queue.provenance, True)


def test_pending_only_resume_keeps_original_deadline(queue):
    before = queue.state["deadline_epoch"]
    assert q.load_state(queue.plan, queue.provenance, True)["deadline_epoch"] == before


def test_pid_reuse_cannot_signal_replacement(monkeypatch):
    original = identity(123)
    calls = []
    monkeypatch.setattr(q.os, "pidfd_open", lambda pid: 91, raising=False)
    monkeypatch.setattr(q.os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(q.signal, "pidfd_send_signal",
                        lambda fd, sig: calls.append(("signal", fd, sig)), raising=False)
    monkeypatch.setattr(q, "proc_identity", lambda pid: identity(pid, start=999))
    assert q.signal_verified(original, signal.SIGTERM) is False
    assert calls == [("close", 91)]
    monkeypatch.setattr(q, "proc_identity", lambda pid: original)
    assert q.signal_verified(original, signal.SIGTERM) is True
    assert ("signal", 91, signal.SIGTERM) in calls


def test_health_requires_owned_socket_expected_model_and_live_launch_pid(monkeypatch):
    root = identity(100)
    monkeypatch.setattr(q, "proc_identity", lambda pid: root)
    requests = []
    model = {"id": q.MODEL, "max_model_len": 32768}
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, limit):
            return q.json_bytes({"data": [model]})
    class Opener:
        def open(self, url, timeout):
            requests.append(url)
            return Response()
    monkeypatch.setattr(q.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(q, "owns_listener", lambda *a: False)
    assert not q.endpoint_ready(root, 18080)
    assert requests == []
    monkeypatch.setattr(q, "owns_listener", lambda *a: True)
    assert q.endpoint_ready(root, 18080)
    assert requests == ["http://127.0.0.1:18080/health", "http://127.0.0.1:18080/v1/models"]
    model["id"] = "foreign-model"
    assert not q.endpoint_ready(root, 18080)
    model["id"], model["max_model_len"] = q.MODEL, 65536
    assert not q.endpoint_ready(root, 18080)
    monkeypatch.setattr(q, "proc_identity", lambda pid: identity(pid, start=999))
    with pytest.raises(q.QueueError, match="not alive"):
        q.endpoint_ready(root, 18080)


def test_workers_must_be_live_owned_on_both_gpus_and_foreign_is_never_signaled(queue, monkeypatch):
    server = mock_server(queue)
    monkeypatch.setattr(q, "session_members", lambda root: [root, identity(root["pid"] + 1)])
    gpus = inventory()
    gpus[0] = replace(gpus[0], pids=frozenset({101}))
    with pytest.raises(q.QueueError, match="No live owned"):
        queue.workers_healthy(server, gpus, require_workers=True)
    gpus[1] = replace(gpus[1], pids=frozenset({101}))
    queue.workers_healthy(server, gpus, require_workers=True)
    gpus[1] = replace(gpus[1], pids=frozenset({101, 9001}))
    monkeypatch.setattr(q, "signal_verified", lambda *a: pytest.fail("Foreign signal attempted"))
    with pytest.raises(q.QueueError, match="Foreign compute"):
        queue.workers_healthy(server, gpus, require_workers=True)


def test_manifest_has_actual_launch_identity_hashes_and_conditional_world_limit(queue, monkeypatch):
    server = mock_server(queue)
    server.manifest_path.unlink()
    server.ready = False
    monkeypatch.setattr(q, "query_gpus", inventory)
    stub_attestation_probe(monkeypatch)
    queue.attest(server)
    record = json.loads(server.manifest_path.read_text())
    content_hash = record.pop("manifest_content_sha256")
    assert content_hash == q.digest_bytes(q.json_bytes(record))
    for key, value in q.SERVER_SPEC.items():
        assert record[key] == value
    assert record["endpoint"] == server.url
    assert record["pid"] == server.owned.process.pid
    assert record["launch_command"] == record["command"] == server.command
    assert record["gpu_ids"] == [0, 1]
    assert record["plan_sha256"] == queue.plan.sha256
    assert record["interpretation_limitation"] == q.LIMITATION
    assert server.manifest_path.stat().st_mode & 0o222 == 0


def test_scheduler_caps_four_servers_and_runs_one_client_each(queue, monkeypatch):
    spawned = mock_client_launch(queue, monkeypatch)
    started = []
    def start(pair, locks):
        server = mock_server(queue, len(started) + 1, pair)
        server.locks = locks
        started.append(server)
    monkeypatch.setattr(queue, "start_server", start)
    assert queue.step() is False
    assert len(started) == 4 and not spawned
    assert len({g.index for s in started for g in s.pair}) == 8
    assert queue.step() is False
    assert len(spawned) == 4
    first_jobs = [s.job for s in queue.servers]
    for server in queue.servers:
        write_artifacts(server.job)
        server.client.process.code = 0
    assert queue.step() is False
    assert len(started) == 4 and len(spawned) == 8
    assert not {j["id"] for j in first_jobs} & {s.job["id"] for s in queue.servers}
    for server in queue.servers:
        write_artifacts(server.job)
        server.client.process.code = 0
    assert queue.step() is True
    assert not queue.servers
    assert all(x["status"] == "complete" for x in queue.state["jobs"].values())


def test_unused_still_loading_server_stops_when_other_servers_finished_queue(queue, monkeypatch):
    server = mock_server(queue)
    server.ready = False
    for record in queue.state["jobs"].values():
        record["status"] = "complete"
    monkeypatch.setattr(q, "query_gpus", inventory)
    monkeypatch.setattr(q, "validate_completed", lambda *a, **k: {"test": "separate final-gate tests"})
    assert queue.step() is True
    assert not queue.servers
    assert queue.state["servers"][server.name]["status"] == "stopped"


def test_wait_persists_then_polls_thirty_seconds_and_deadline_stops(queue, monkeypatch):
    monkeypatch.setattr(q, "query_gpus", lambda: inventory(set()))
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        queue.state["deadline_epoch"] = q.time.time() - 1
    monkeypatch.setattr(q.time, "sleep", sleep)
    assert queue.run() == 1
    assert sleeps == [30]
    saved = json.loads((queue.plan.directory / "status.json").read_text())
    assert "72-hour" in saved["error"]
    assert saved["last_gpu_snapshot"][0]["compute_pids"] == [9000]


def test_any_job_failure_stops_own_servers_without_signaling_foreign(queue, monkeypatch):
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    client_pid = server.client.process.pid
    server.client.process.code = 17
    stopped = []
    monkeypatch.setattr(q, "stop_owned",
                        lambda root, **kwargs: stopped.append(root["pid"]) or [])
    assert queue.run() == 1
    assert set(stopped) == {100, client_pid}
    assert 9000 not in stopped
    assert not queue.servers
    assert queue.state["jobs"][queue.plan.jobs[0]["id"]]["status"] == "failed"
    assert queue.state["jobs"][queue.plan.jobs[1]["id"]]["status"] == "pending"


def test_guard_rejects_main_branch_before_any_gpu_work(plan, monkeypatch):
    monkeypatch.setattr(q.sys, "platform", "linux")
    monkeypatch.setattr(q.os, "pidfd_open", lambda _: 1, raising=False)
    monkeypatch.setattr(q.signal, "pidfd_send_signal", lambda *a: None, raising=False)
    def git(command, **kwargs):
        if "--show-toplevel" in command:
            return str(plan.worktree)
        if "symbolic-ref" in command:
            return "main"
        raise AssertionError("Guard progressed past main-branch rejection")
    monkeypatch.setattr(q.subprocess, "check_output", git)
    with pytest.raises(q.QueueError, match="main/master"):
        q.runtime_guard(plan)


def test_guard_rejects_nonlinked_checkout(plan, monkeypatch):
    monkeypatch.setattr(q.sys, "platform", "linux")
    monkeypatch.setattr(q.os, "pidfd_open", lambda _: 1, raising=False)
    monkeypatch.setattr(q.signal, "pidfd_send_signal", lambda *a: None, raising=False)
    def git(command, **kwargs):
        if "--show-toplevel" in command:
            return str(plan.worktree)
        if "symbolic-ref" in command:
            return "llm"
        if "--absolute-git-dir" in command or "--git-common-dir" in command:
            return str(plan.worktree / ".git")
        raise AssertionError("Guard progressed past worktree rejection")
    monkeypatch.setattr(q.subprocess, "check_output", git)
    with pytest.raises(q.QueueError, match="isolated linked worktree"):
        q.runtime_guard(plan)


def test_resume_does_not_adopt_still_live_owned_server(queue, monkeypatch):
    mock_server(queue)
    queue.save()
    monkeypatch.setattr(q, "session_members", lambda root: [root])
    with pytest.raises(q.QueueError, match="still alive"):
        q.load_state(queue.plan, queue.provenance, True)


def test_atomic_write_exclusive_never_overwrites_and_status_replace_is_complete(tmp_path):
    path = tmp_path / "state.json"
    q.atomic_write(path, b'{"state":1}', exclusive=True)
    with pytest.raises(FileExistsError):
        q.atomic_write(path, b'{"state":2}', exclusive=True)
    assert path.read_bytes() == b'{"state":1}'
    q.atomic_write(path, b'{"state":3}')
    assert json.loads(path.read_bytes()) == {"state": 3}
    assert not list(tmp_path.glob(".queue-*"))


def configure_fake_server_launch(queue, monkeypatch):
    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def bind(self, address):
            pass
    monkeypatch.setattr(q.socket, "socket", Socket)
    monkeypatch.setattr(q, "query_gpus", inventory)
    monkeypatch.setattr(q, "boot_id", lambda: "mock-boot")
    monkeypatch.setattr(q, "proc_identity", lambda pid: identity(pid))
    return q.reserve_pair(inventory({0, 1}), queue.lock_root, query=inventory)


def test_durable_intent_precedes_fork_and_blocks_orphan_loading_pair(queue, monkeypatch):
    pair, locks = configure_fake_server_launch(queue, monkeypatch)
    def die_after_fork_before_parent_captures_identity(*args):
        records = [q.read_reservation(q.reservation_path(queue.lock_root, gpu.uuid)) for gpu in pair]
        assert all(record["phase"] == "launch_intent" for record in records)
        assert all(record["server_identity"] is None for record in records)
        assert records[0]["token"] == records[1]["token"]
        assert all(record["controller_identity"]["start_ticks"] > 0 for record in records)
        saved = json.loads((queue.plan.directory / "status.json").read_text())
        assert saved["servers"]["server-001"]["status"] == "launch_intent"
        # This represents a child still loading on CPU, invisible to nvidia-smi.
        raise q.QueueError("controller died after fork before child identity capture")
    monkeypatch.setattr(queue, "spawn", die_after_fork_before_parent_captures_identity)
    with pytest.raises(q.QueueError, match="controller died"):
        queue.start_server(pair, locks)
    assert all(lock.fd is None for lock in locks)
    monkeypatch.setattr(q, "proc_identity", lambda pid: None)  # controller gone
    assert q.reserve_pair(inventory({0, 1}), queue.lock_root, query=inventory) is None
    monkeypatch.setattr(q, "proc_identity", lambda pid: identity(pid, start=999999))  # reused PID
    assert q.reserve_pair(inventory({0, 1}), queue.lock_root, query=inventory) is None
    free_pair, free_locks = q.reserve_pair(inventory(), queue.lock_root, query=inventory)
    assert [gpu.index for gpu in free_pair] == [2, 3]
    for lock in free_locks:
        lock.close()
    with pytest.raises(q.QueueError, match="Uncertain server launch"):
        q.load_state(queue.plan, queue.provenance, True)


def test_known_child_identity_is_persisted_and_failed_cleanup_keeps_quarantine(queue, monkeypatch):
    pair, locks = configure_fake_server_launch(queue, monkeypatch)
    child = owned(8181)
    monkeypatch.setattr(queue, "spawn", lambda *a: child)
    queue.start_server(pair, locks)
    for gpu in pair:
        record = q.read_reservation(q.reservation_path(queue.lock_root, gpu.uuid))
        assert record["phase"] == "active"
        assert record["server_identity"] == child.identity
    monkeypatch.setattr(q, "stop_owned", lambda *a, **k: [8181])
    assert queue.close_server(queue.servers[0])
    assert all(lock.fd is None for lock in locks)
    assert q.reserve_pair(inventory({0, 1}), queue.lock_root, query=inventory) is None


def test_clean_shutdown_releases_durable_intent_while_both_gpu_locks_still_held(queue, monkeypatch):
    pair, locks = configure_fake_server_launch(queue, monkeypatch)
    monkeypatch.setattr(queue, "spawn", lambda *a: owned(8181))
    queue.start_server(pair, locks)
    original_update = queue.update_reservation
    phases = []
    def verify_locked_release(server, phase):
        assert all(lock.fd is not None for lock in locks)
        phases.append(phase)
        original_update(server, phase)
    monkeypatch.setattr(queue, "update_reservation", verify_locked_release)
    assert queue.close_server(queue.servers[0]) == []
    assert phases == ["released"]
    new_pair, new_locks = q.reserve_pair(inventory({0, 1}), queue.lock_root, query=inventory)
    assert [gpu.index for gpu in new_pair] == [0, 1]
    for lock in new_locks:
        lock.close()


def test_different_boot_cannot_have_surviving_orphan_processes(queue, monkeypatch):
    monkeypatch.setattr(q, "boot_id", lambda: "new-boot")
    for gpu in inventory()[:2]:
        q.atomic_write(q.reservation_path(queue.lock_root, gpu.uuid),
                       q.json_bytes({"schema_version": 1, "token": "old-token",
                                     "boot_id": "old-boot", "phase": "launch_intent"}))
    pair, locks = q.reserve_pair(inventory({0, 1}), queue.lock_root, query=inventory)
    assert [gpu.index for gpu in pair] == [0, 1]
    for lock in locks:
        lock.close()


@pytest.mark.parametrize("parent_link", [False, True])
def test_historical_symlink_inside_private_outputs_is_rejected(queue, parent_link):
    job = copy.deepcopy(queue.plan.jobs[0])
    archived = queue.plan.private_root / "outputs/docs/historical"
    archived.mkdir(parents=True)
    (archived / "done.json").write_bytes(q.json_bytes({"status": "complete", "seed": 42}))
    root = Path(job["output_roots"][0])
    root.mkdir()
    if parent_link:
        (root / "linked-parent").symlink_to(archived, target_is_directory=True)
        job["artifacts"][0]["path"] = str(root / "linked-parent/done.json")
    else:
        (root / "done.json").symlink_to(archived / "done.json")
    with pytest.raises(q.QueueError, match="Symlink output component"):
        q.artifact_gate(job, queue.plan, started_ns=1)


def test_artifact_must_belong_to_own_job_root_even_if_inside_private_outputs(queue):
    job = copy.deepcopy(queue.plan.jobs[0])
    other = queue.plan.jobs[1]
    write_artifacts(other)
    job["artifacts"][0]["path"] = other["artifacts"][0]["path"]
    with pytest.raises(q.QueueError, match="this job's fresh output_roots"):
        q.artifact_gate(job, queue.plan, started_ns=1)


def test_stale_regular_file_timestamps_cannot_satisfy_fresh_cohort_gate(queue):
    job = queue.plan.jobs[0]
    write_artifacts(job)
    path = Path(job["artifacts"][0]["path"])
    os.utime(path, ns=(1, 1))
    with pytest.raises(q.QueueError, match="timestamps"):
        q.artifact_gate(job, queue.plan, started_ns=2)


def test_artifact_receipt_binds_file_identity_and_preserves_only_sanctioned_outputs_alias(queue):
    job = copy.deepcopy(queue.plan.jobs[0])
    write_artifacts(job)
    original = Path(job["artifacts"][0]["path"])
    job["output_roots"] = [str(queue.plan.worktree / "outputs/docs/influx-0")]
    job["artifacts"][0]["path"] = str(queue.plan.worktree / "outputs/docs/influx-0/done.json")
    result = q.artifact_gate(job, queue.plan, started_ns=1)[0]
    assert result["canonical_path"] == str(original)
    assert result["inode"] == original.stat().st_ino
    assert result["mtime_ns"] == original.stat().st_mtime_ns
    assert result["ctime_ns"] == original.stat().st_ctime_ns
    assert result["job_started_epoch_ns"] == 1


def complete_eight_mock_jobs(queue, monkeypatch):
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    for job in queue.plan.jobs:
        queue.start_job(server, job)
        write_artifacts(job)
        queue.finish_job(server, 0)
    return server


@pytest.mark.parametrize("mutation,expected", [
    ("artifact", "artifact hash changed"),
    ("manifest", "server manifest changed"),
    ("receipt", "receipt changed"),
    ("sources", "Frozen sources changed"),
    ("same_bytes_new_inode", "file binding changed"),
])
def test_final_success_rechecks_earlier_artifacts_manifests_receipts_and_sources(
        queue, monkeypatch, mutation, expected):
    server = complete_eight_mock_jobs(queue, monkeypatch)
    job = queue.plan.jobs[0]
    artifact = Path(job["artifacts"][0]["path"])
    if mutation == "artifact":
        artifact.write_bytes(artifact.read_bytes() + b" ")
    elif mutation == "manifest":
        server.manifest_path.write_text('{"changed": true}')
    elif mutation == "receipt":
        receipt = queue.plan.directory / f"job-{job['id']}.receipt.json"
        receipt.chmod(0o600)
        receipt.write_bytes(receipt.read_bytes() + b" ")
    elif mutation == "sources":
        monkeypatch.setattr(q, "source_hashes", lambda _: {"changed.py": "different hash"})
    else:
        data = artifact.read_bytes()
        replacement = artifact.with_suffix(".replacement")
        replacement.write_bytes(data)
        os.replace(replacement, artifact)
    assert queue.run() == 1
    assert queue.state["status"] == "failed"
    assert expected in queue.state["error"]
    assert "final_validation" not in queue.state


def test_final_success_attests_all_eight_jobs_after_revalidation(queue, monkeypatch):
    complete_eight_mock_jobs(queue, monkeypatch)
    assert queue.run() == 0
    assert queue.state["status"] == "complete"
    assert queue.state["final_validation"]["completed_jobs"] == 8
    assert queue.state["final_validation"]["server_manifests"] == 1


def probe_wire(monkeypatch, *, mutate=None, status=200, warnings=None, raw=None,
               open_delay=0, read_delay=0, observer=None, funding_fixture=None):
    calls = []
    monkeypatch.setattr(q, "proc_identity", lambda pid: identity(pid))
    monkeypatch.setattr(q, "owns_listener", lambda *a: True)
    class Response:
        def __init__(self, request):
            self.status, self.headers = status, warnings or {}
            self.url = request.full_url
            body = json.loads(request.data)
            schema = body["structured_outputs"]["json"]
            if "probe_nonce" in schema["properties"]:
                nonce = schema["properties"]["probe_nonce"]["const"]
                payload = probe_payload(nonce)
            elif "next_application_id" in schema["properties"]:
                payload = probe_payload("unused")
                payload["choices"][0]["message"]["content"] = json.dumps({
                    "next_application_id": schema["properties"]["next_application_id"]["enum"][-1]})
                tokens = len(json.dumps(body["messages"]))
                payload["usage"] = {"prompt_tokens": tokens, "completion_tokens": 35,
                                    "total_tokens": tokens + 35}
            else:
                assert funding_fixture is not None, "Funding HTTP response requires explicit unit fixture"
                payload = funding_payload(funding_fixture)
            if mutate:
                mutate(payload)
            self.raw = raw if raw is not None else q.json_bytes(payload)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def geturl(self):
            return self.url
        def read(self, limit):
            if read_delay:
                time.sleep(read_delay)
            return self.raw[:limit]
    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            if observer:
                observer(request)
            if open_delay:
                time.sleep(open_delay)
            return Response(request)
    def opener(*handlers):
        assert any(isinstance(handler, q.NoProbeRedirect) for handler in handlers)
        assert any(isinstance(handler, q.urllib.request.ProxyHandler) for handler in handlers)
        return Opener()
    monkeypatch.setattr(q.urllib.request, "build_opener", opener)
    return calls


def test_shared_builder_and_seed_loader_are_dependency_free_and_preserve_production_options():
    builder, metadata = real_probe_builder()
    schema = q.probe_schema("a" * 32)
    fields = builder(schema)
    assert fields["structured_outputs"]["json"] is schema
    assert fields["chat_template_kwargs"] == {"enable_thinking": True}
    assert "guided_json" not in fields
    assert fields["seed"] == metadata["request_seed"]
    assert metadata["request_seed_base"] == 42
    assert metadata["seed_components"] == list(q.PROBE_SEED_COMPONENTS)
    assert metadata["sha256"] == q.file_hash(Path(metadata["path"]))


def test_real_probe_request_accepts_strict_nonce_final_json_with_actual_thinking_protocol(monkeypatch):
    calls = probe_wire(monkeypatch)
    builder, metadata = real_probe_builder()
    records = [q.constrained_output_probe(identity(100), 18080, builder) for _ in range(2)]
    nonces = []
    for record, (req, timeout) in zip(records, calls):
        request = json.loads(req.data)
        nonce = request["structured_outputs"]["json"]["properties"]["probe_nonce"]["const"]
        nonces.append(nonce)
        assert req.get_method() == "POST"
        assert req.full_url == "http://127.0.0.1:18080/v1/chat/completions"
        assert 0 < timeout <= 120
        assert request["model"] == q.MODEL
        assert request["max_tokens"] == 2048 and request["temperature"] == 0.7
        assert request["seed"] == metadata["request_seed"]
        assert request["chat_template_kwargs"] == {"enable_thinking": True}
        assert not request["stream"]
        assert json.dumps(request).count(nonce) == 1
        assert nonce not in json.dumps(request["messages"])
        assert "Do not output JSON" in request["messages"][-1]["content"]
        assert record["status"] == "passed"
        assert record["validation"]["final_json"] == {"probe_nonce": nonce}
        assert record["validation"]["finish_reason"] == "stop"
        assert record["validation"]["reasoning_present"]
        assert record["validation"]["reasoning_characters"] > 0
        q.validate_probe_attestation(record, identity(100), 18080)
    assert len(set(nonces)) == 2


@pytest.mark.parametrize("mutation", [
    lambda p: p["choices"][0]["message"].update(content="TRANSPORT_NOT_JSON"),
    lambda p: p["choices"][0]["message"].update(content=json.dumps({"probe_nonce": "wrong"})),
    lambda p: p["choices"][0]["message"].update(
        content=p["choices"][0]["message"]["content"][:-1] + ', "extra": 1}'),
    lambda p: p["choices"][0]["message"].update(
        content="```json\n" + p["choices"][0]["message"]["content"] + "\n```"),
    lambda p: p["choices"][0]["message"].update(content=None),
    lambda p: p["choices"][0].update(finish_reason="length"),
    lambda p: p["choices"][0].update(finish_reason=None),
    lambda p: p.update(model="wrong-model"),
    lambda p: p.update(warnings=["ignored structured_outputs"]),
    lambda p: p["choices"].append(copy.deepcopy(p["choices"][0])),
    lambda p: p["choices"][0]["message"].update(
        content=p["choices"][0]["message"]["content"][:-1] + ', "probe_nonce":"duplicate"}'),
])
def test_probe_rejects_ignored_constraint_schema_drift_truncation_and_malformed_outputs(
        monkeypatch, mutation):
    calls = probe_wire(monkeypatch, mutate=mutation)
    builder, _ = real_probe_builder()
    with pytest.raises(q.ProbeError) as error:
        q.constrained_output_probe(identity(100), 18080, builder)
    assert len(calls) == 1
    assert error.value.attestation["status"] == "failed"
    assert error.value.attestation["http_status"] == 200
    assert "response_body" in error.value.attestation


@pytest.mark.parametrize("status", [400, 422, 500])
def test_probe_rejects_unsupported_or_failing_http_without_retry(monkeypatch, status):
    calls = probe_wire(monkeypatch, status=status)
    builder, _ = real_probe_builder()
    with pytest.raises(q.ProbeError, match="HTTP status"):
        q.constrained_output_probe(identity(100), 18080, builder)
    assert len(calls) == 1


def test_probe_rejects_ignored_field_warning_even_with_correct_json(monkeypatch):
    probe_wire(monkeypatch, warnings={"Warning": "ignored structured_outputs"})
    builder, _ = real_probe_builder()
    with pytest.raises(q.ProbeError, match="ignored/unsupported"):
        q.constrained_output_probe(identity(100), 18080, builder)


def test_truncated_probe_preserves_finish_reason_and_reasoning_metrics(monkeypatch):
    probe_wire(monkeypatch, mutate=lambda p: p["choices"][0].update(finish_reason="length"))
    builder, _ = real_probe_builder()
    with pytest.raises(q.ProbeError) as error:
        q.constrained_output_probe(identity(100), 18080, builder)
    metrics = error.value.attestation["response_metrics"]
    assert metrics["finish_reason"] == "length"
    assert metrics["reasoning_present"] and metrics["reasoning_characters"] > 0
    assert len(metrics["reasoning_sha256"]) == 64
    assert metrics["usage"]["completion_tokens"] == 64


def test_probe_has_response_size_bound_and_never_follows_redirects(monkeypatch):
    probe_wire(monkeypatch, raw=b"x" * (q.PROBE_MAX_RESPONSE_BYTES + 1))
    builder, _ = real_probe_builder()
    with pytest.raises(q.ProbeError, match="byte limit"):
        q.constrained_output_probe(identity(100), 18080, builder)
    assert q.NoProbeRedirect().redirect_request(None, None, 302, "", {}, "http://foreign") is None


@pytest.mark.parametrize("where", ["open", "read"])
def test_probe_hard_wall_deadline_bounds_slow_headers_and_body_and_restores_alarm(monkeypatch, where):
    calls = probe_wire(monkeypatch, **{where + "_delay": 0.3})
    builder, _ = real_probe_builder()
    original = signal.getsignal(signal.SIGALRM)
    before = time.monotonic()
    with pytest.raises(q.ProbeError, match="wall deadline"):
        q.constrained_output_probe(identity(100), 18080, builder, timeout_seconds=0.03)
    assert time.monotonic() - before < 0.25
    assert len(calls) == 1
    assert signal.getsignal(signal.SIGALRM) is original
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_probe_does_not_replace_existing_alarm(monkeypatch):
    monkeypatch.setattr(q.signal, "getitimer", lambda _: (5.0, 0.0))
    monkeypatch.setattr(q.signal, "signal", lambda *a: pytest.fail("existing alarm changed"))
    with pytest.raises(q.QueueError, match="existing process alarm"):
        with q.probe_deadline(0.1):
            pytest.fail("alarm guard was bypassed")


def test_probe_rejects_wrong_socket_or_pid_reuse_before_or_after_http(monkeypatch):
    calls = probe_wire(monkeypatch)
    builder, _ = real_probe_builder()
    monkeypatch.setattr(q, "owns_listener", lambda *a: False)
    with pytest.raises(q.ProbeError, match="owned API process"):
        q.constrained_output_probe(identity(100), 18080, builder)
    assert not calls
    monkeypatch.setattr(q, "owns_listener", lambda *a: True)
    identities = iter([identity(100), identity(100, start=999999)])
    monkeypatch.setattr(q, "proc_identity", lambda pid: next(identities))
    with pytest.raises(q.ProbeError, match="changed during"):
        q.constrained_output_probe(identity(100), 18080, builder)


def prepare_real_attestation(queue, monkeypatch):
    server = mock_server(queue)
    server.manifest_path.unlink()
    server.ready, server.probe = False, None
    real_loader = q.load_shared_probe_builder
    root = SimpleNamespace(worktree=project_root(q.__file__))
    monkeypatch.setattr(q, "load_shared_probe_builder", lambda _, *args: real_loader(root, *args))
    monkeypatch.setattr(q, "query_gpus", inventory)
    monkeypatch.setattr(queue, "workers_healthy", lambda *a, **k: None)
    monkeypatch.setattr(q, "endpoint_ready", lambda *a: True)
    return server


def test_models_health_cannot_start_jobs_when_real_constrained_output_probe_fails(queue, monkeypatch):
    server = prepare_real_attestation(queue, monkeypatch)
    calls = probe_wire(monkeypatch, mutate=lambda p:
                       p["choices"][0]["message"].update(content="TRANSPORT_NOT_JSON"))
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("client launched before probe passed"))
    assert queue.run() == 1
    assert len(calls) == 1
    assert all(record["status"] == "pending" for record in queue.state["jobs"].values())
    assert not server.ready and server.probe is None
    assert not server.manifest_path.exists()
    failure = json.loads((queue.plan.directory / f"{server.name}.probe.failure.json").read_text())
    assert failure["status"] == "failed"
    assert "TRANSPORT_NOT_JSON" in failure["response_body"]


def test_successful_real_probe_is_immutable_in_manifest_before_any_client(queue, monkeypatch):
    server = prepare_real_attestation(queue, monkeypatch)
    observed = []
    def observe(request):
        assert server.ready is False and server.client is None
        observed.append(queue.state["servers"][server.name]["status"])
    calls = probe_wire(monkeypatch, observer=observe, funding_fixture=q.load_funding_fixture(queue.plan)[0])
    queue.attest(server)
    manifest = json.loads(server.manifest_path.read_text())
    assert observed == ["probing_structured_outputs", "probing_funding_panel25"] and len(calls) == 2
    assert manifest["structured_output_probe"] == server.probe
    assert manifest["structured_output_probe"]["validation"]["finish_reason"] == "stop"
    assert manifest["structured_output_probe"]["shared_client_builder"]["request_seed_base"] == 42
    assert manifest["funding_panel25_probe"] == server.funding_probe
    assert manifest["funding_panel25_probe"]["validation"]["exact_prompt_tokens"] == 4795
    assert server.manifest_path.stat().st_mode & 0o222 == 0
    spawned = mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    assert len(spawned) == 1


@pytest.mark.parametrize("case", ["missing", "wrong_server"])
def test_client_requires_probe_attestation_even_with_healthy_models(queue, monkeypatch, case):
    server = mock_server(queue)
    spawned = mock_client_launch(queue, monkeypatch)
    manifest = json.loads(server.manifest_path.read_text())
    if case == "missing":
        del manifest["structured_output_probe"]
    else:
        manifest["structured_output_probe"]["process_identity"] = identity(9999)
    server.manifest_path.write_bytes(q.json_bytes(manifest))
    queue.state["servers"][server.name]["manifest_sha256"] = q.file_hash(server.manifest_path)
    with pytest.raises(q.QueueError, match="probe"):
        queue.start_job(server, queue.plan.jobs[0])
    assert not spawned


def test_sigterm_during_probe_cannot_launch_client_or_mark_ready(queue, monkeypatch):
    server = prepare_real_attestation(queue, monkeypatch)
    probe_wire(monkeypatch, observer=lambda _: setattr(queue, "interrupted", True))
    with pytest.raises(q.QueueError, match="interrupted during"):
        queue.attest(server)
    assert not server.ready and server.probe is None
    assert all(record["status"] == "pending" for record in queue.state["jobs"].values())


@pytest.mark.parametrize("descriptor", [None, {}, {"path": "/tmp/x", "sha256": "wrong"},
                                      {"path": "relative", "sha256": "a" * 64}])
def test_plan_requires_exact_funding_fixture_descriptor(specification, descriptor):
    specification["funding_probe_fixture"] = descriptor
    with pytest.raises(q.QueueError, match="fixture"):
        q.parse_plan(q.json_bytes(specification))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(kind="another"),
    lambda d: d["expected"]["application_ids"].__setitem__(0, False),
    lambda d: d["expected"]["application_to_applicant"].pop("24"),
    lambda d: d["expected"].update(application_count=24),
    lambda d: d["tokenization"].update(enable_thinking=False),
    lambda d: d["tokenization"].update(requested_max_tokens=2048),
    lambda d: d["tokenization"].update(input_tokens=24577, total_requested_tokens=32769),
    lambda d: d["tokenization"].update(total_requested_tokens=12986),
    lambda d: d["tokenization"].update(revision="wrong"),
    lambda d: d["tokenization"]["tokenizer_files_sha256"].update({"../elsewhere": "a" * 64}),
    lambda d: d["json_schema"]["$defs"]["RankedApplication"]["properties"]["rank"].update(type="number"),
])
def test_funding_fixture_contract_rejects_drift(mutation):
    fixture = unit_funding_fixture()
    mutation(fixture)
    with pytest.raises(q.QueueError):
        q.validate_funding_fixture(fixture)


@pytest.mark.parametrize("case", ["writable", "sha", "symlink", "hardlink", "too_large"])
def test_funding_fixture_is_bound_regular_readonly_input(plan, case):
    path = Path(plan.spec["funding_probe_fixture"]["path"])
    assert q.load_funding_fixture(plan)[1]["sha256"] == q.file_hash(path)
    if case == "writable":
        path.chmod(0o644)
    elif case == "sha":
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)
    elif case == "symlink":
        original = path.with_suffix(".original")
        path.rename(original)
        path.symlink_to(original)
    elif case == "hardlink":
        os.link(path, path.with_suffix(".link"))
    else:
        path.chmod(0o644)
        with path.open("wb") as stream:
            stream.truncate(q.FUNDING_FIXTURE_MAX_BYTES + 1)
        path.chmod(0o444)
    with pytest.raises((q.QueueError, OSError)):
        q.load_funding_fixture(plan)


def test_runtime_checks_provided_tokenizer_files_and_raw_chat_template(plan):
    path = Path(plan.spec["funding_probe_fixture"]["path"])
    fixture = unit_funding_fixture()
    snapshot = plan.hf_home / "hub/models--Qwen--Qwen3-32B/snapshots" / q.REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text('{"mock":true}')
    template = "unit template"
    (snapshot / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    fixture["tokenization"]["tokenizer_files_sha256"] = {
        name: q.file_hash(snapshot / name) for name in ("tokenizer.json", "tokenizer_config.json")}
    fixture["tokenization"]["chat_template_sha256"] = q.digest_bytes(template.encode())
    path.chmod(0o644)
    path.write_bytes(q.json_bytes(fixture))
    path.chmod(0o444)
    plan.spec["funding_probe_fixture"]["sha256"] = q.file_hash(path)
    q.load_funding_fixture(plan, verify_runtime_files=True)
    (snapshot / "tokenizer.json").write_text('{"mock":false}')
    with pytest.raises(q.QueueError, match="tokenizer file changed"):
        q.load_funding_fixture(plan, verify_runtime_files=True)
    # A separate template file takes precedence, as with the production tokenizer.
    (snapshot / "tokenizer.json").write_text('{"mock":true}')
    (snapshot / "chat_template.jinja").write_text("different")
    with pytest.raises(q.QueueError, match="chat-template hash"):
        q.load_funding_fixture(plan, verify_runtime_files=True)


def test_funding_fixture_is_in_frozen_source_set(plan):
    hashes = q.source_hashes(plan)
    descriptor = plan.spec["funding_probe_fixture"]
    assert hashes["funding_probe_fixture:" + descriptor["path"]] == descriptor["sha256"]


def test_funding_probe_sends_exact_fixture_and_8192_with_derived_seed(plan, monkeypatch):
    fixture, binding = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture)
    builder, metadata = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    record = q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
    record["shared_client_builder"] = metadata
    q.validate_funding_probe_attestation(record, identity(100), 18080, fixture, binding)
    assert len(calls) == 1
    request = json.loads(calls[0][0].data)
    assert request["messages"] == fixture["messages"]
    assert request["structured_outputs"]["json"] == fixture["json_schema"]
    assert request["chat_template_kwargs"] == {"enable_thinking": True}
    assert request["max_tokens"] == 8192 and request["temperature"] == 0.7
    assert metadata["seed_components"] == [42, "funding_panel25_preflight", 1, 0, 0]
    assert request["seed"] == metadata["request_seed"]
    assert request["seed"] != real_probe_builder()[1]["request_seed"]
    assert 120 < calls[0][1] <= 600
    assert record["fixture"] == binding and record["validation"]["exact_prompt_tokens"] == 4795
    assert record["validation"]["final_json"]["ranked_applications"][0]["rank"] == 25


def change_funding_rows(payload, mutation):
    final = json.loads(payload["choices"][0]["message"]["content"])
    mutation(final["ranked_applications"])
    payload["choices"][0]["message"]["content"] = json.dumps(final)


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[-1])),
    lambda rows: rows.__setitem__(24, copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(rank=rows[1]["rank"]),
    lambda rows: rows[0].update(rank=0),
    lambda rows: rows[0].update(rank=True),
    lambda rows: rows[0].update(application_id=False),
    lambda rows: rows[0].update(application_id=25),
    lambda rows: rows[0].update(applicant_id=rows[1]["applicant_id"]),
    lambda rows: rows[0].update(reason=12),
    lambda rows: rows[0].pop("reason"),
    lambda rows: rows[0].update(extra="not production"),
])
def test_funding_probe_rejects_incomplete_or_inexact_raw_permutations(plan, monkeypatch, mutation):
    fixture, binding = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture,
                       mutate=lambda p: change_funding_rows(p, mutation))
    builder, _ = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    with pytest.raises(q.ProbeError) as error:
        q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
    assert len(calls) == 1
    assert error.value.attestation["status"] == "failed"
    assert "response_body" in error.value.attestation


@pytest.mark.parametrize("mutation", [
    lambda p: p["usage"].update(prompt_tokens=4796),
    lambda p: p["usage"].update(prompt_tokens="4795"),
    lambda p: p.pop("usage"),
    lambda p: p["choices"][0].update(finish_reason="length"),
    lambda p: p["choices"][0]["message"].update(content="plain text"),
    lambda p: p["choices"][0]["message"].update(content='{"ranked_applications":[],"extra":0}'),
    lambda p: p.update(warning="ignored structured_outputs"),
])
def test_funding_probe_rejects_token_mismatch_truncation_and_ignored_transport(plan, monkeypatch, mutation):
    fixture, binding = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture, mutate=mutation)
    builder, _ = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    with pytest.raises(q.ProbeError) as error:
        q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
    assert len(calls) == 1
    assert error.value.attestation["response_metrics"]["reasoning_present"]


def test_duplicate_applicants_allowed_but_application_rank_permutations_required():
    fixture = unit_funding_fixture()
    fixture["expected"]["application_to_applicant"]["1"] = fixture["expected"]["application_to_applicant"]["0"]
    q.validate_funding_fixture(fixture)
    payload = funding_payload(fixture)
    change_funding_rows(payload, lambda rows: rows.reverse())
    assert q.verify_funding_probe_response(payload, fixture)["complete_application_permutation"]


@pytest.mark.parametrize("where", ["headers", "body"])
def test_funding_600s_maximum_has_hard_bounded_slow_io(plan, monkeypatch, where):
    fixture, binding = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture,
                       open_delay=0.2 if where == "headers" else 0,
                       read_delay=0.2 if where == "body" else 0)
    builder, _ = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(q.ProbeError, match="wall deadline"):
        q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding, timeout_seconds=0.03)
    assert time.monotonic() - started < 0.18 and len(calls) == 1
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) == previous
    with pytest.raises(q.ProbeError, match="time budget"):
        q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding, timeout_seconds=600.1)
    assert len(calls) == 1


def test_funding_response_size_bound(plan, monkeypatch):
    fixture, binding = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture,
                       raw=b"x" * (q.FUNDING_PROBE_MAX_RESPONSE_BYTES + 1))
    builder, _ = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    with pytest.raises(q.ProbeError, match="byte limit"):
        q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
    assert len(calls) == 1


def test_generic_pass_funding_failure_cannot_start_any_client(queue, monkeypatch):
    server = prepare_real_attestation(queue, monkeypatch)
    fixture, _ = q.load_funding_fixture(queue.plan)
    def only_funding_bad(payload):
        if "ranked_applications" in payload["choices"][0]["message"]["content"]:
            payload["usage"]["prompt_tokens"] += 1
    calls = probe_wire(monkeypatch, funding_fixture=fixture, mutate=only_funding_bad)
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("client launched with failing funding probe"))
    assert queue.run() == 1 and len(calls) == 2
    assert not server.ready and not server.manifest_path.exists()
    assert all(r["status"] == "pending" for r in queue.state["jobs"].values())
    failure_path = queue.plan.directory / f"{server.name}.funding-probe.failure.json"
    failure = json.loads(failure_path.read_text())
    assert failure["response_metrics"]["usage"]["prompt_tokens"] == 4796
    assert failure["shared_client_builder"]["seed_components"] == list(q.FUNDING_SEED_COMPONENTS)
    assert failure_path.stat().st_mode & 0o222 == 0
    assert queue.state["servers"][server.name]["structured_output_probe"]["status"] == "passed"


@pytest.mark.parametrize("case", ["missing", "pid", "fixture", "permutation"])
def test_start_job_requires_immutable_same_server_funding_evidence(queue, monkeypatch, case):
    server = mock_server(queue)
    spawned = mock_client_launch(queue, monkeypatch)
    manifest = json.loads(server.manifest_path.read_text())
    if case == "missing":
        manifest.pop("funding_panel25_probe")
    elif case == "pid":
        manifest["funding_panel25_probe"]["process_identity"] = identity(9999)
    elif case == "fixture":
        manifest["funding_panel25_probe"]["fixture"]["sha256"] = "f" * 64
    else:
        manifest["funding_panel25_probe"]["validation"]["complete_rank_permutation"] = False
    server.manifest_path.write_bytes(q.json_bytes(manifest))
    queue.state["servers"][server.name]["manifest_sha256"] = q.file_hash(server.manifest_path)
    with pytest.raises(q.QueueError, match="[Ff]unding"):
        queue.start_job(server, queue.plan.jobs[0])
    assert not spawned


def test_sigterm_immediately_interrupts_long_funding_probe_and_restores_handlers(queue, monkeypatch):
    server = prepare_real_attestation(queue, monkeypatch)
    fixture, _ = q.load_funding_fixture(queue.plan)
    old_term, old_alarm = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGALRM)
    def signal_during_funding(request):
        if json.loads(request.data)["max_tokens"] == 8192:
            os.kill(os.getpid(), signal.SIGTERM)
            pytest.fail("SIGTERM must interrupt the HTTP operation immediately")
    calls = probe_wire(monkeypatch, funding_fixture=fixture, observer=signal_during_funding)
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("interrupted probe launched client"))
    assert queue.run() == 1 and len(calls) == 2
    assert queue.interrupted and not queue.probing and not server.ready
    assert all(r["status"] == "pending" for r in queue.state["jobs"].values())
    assert signal.getsignal(signal.SIGTERM) == old_term
    assert signal.getsignal(signal.SIGALRM) == old_alarm
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_final_gate_revalidates_fixture_and_receipt_probe_binding(queue, monkeypatch):
    server = complete_eight_mock_jobs(queue, monkeypatch)
    job = queue.plan.jobs[0]
    path = queue.plan.directory / f"job-{job['id']}.receipt.json"
    receipt = json.loads(path.read_text())
    assert receipt["funding_panel25_probe_sha256"] == q.digest_bytes(q.json_bytes(server.funding_probe))
    assert receipt["funding_probe_fixture_sha256"] == queue.plan.spec["funding_probe_fixture"]["sha256"]
    receipt["funding_panel25_probe_sha256"] = "d" * 64
    path.chmod(0o600)
    path.write_bytes(q.json_bytes(receipt))
    queue.state["jobs"][job["id"]]["receipt_sha256"] = q.file_hash(path)
    with pytest.raises(q.QueueError, match="readiness probe/fixture binding"):
        q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=True)
    fixture_path = Path(queue.plan.spec["funding_probe_fixture"]["path"])
    fixture_path.chmod(0o644)
    fixture_path.write_bytes(fixture_path.read_bytes() + b" ")
    fixture_path.chmod(0o444)
    with pytest.raises(q.QueueError, match="fixture SHA256"):
        q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=True)


@pytest.fixture
def sequential_plan(specification):
    """Real shared helper/transport sources, synthetic application and tokenizer fixture."""
    spec = copy.deepcopy(specification)
    root = project_root(q.__file__)
    worktree = Path(spec["worktree"])
    for name in ("utopia/funding/sequential.py", "utopia/funding/compact.py",
                 "utopia/models/structured_outputs.py", "utopia/models/request_audit.py",
                 "utopia/utils/seeding.py", "utopia/utils/data_utils.py",
                 "utopia/utils/paths.py", "utopia/runtime/historical.py",
                 "utopia/funding/validation.py", "utopia/constants.py"):
        target = worktree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    protocol = worktree / q.SEQUENTIAL_PROTOCOL_FILE
    protocol.parent.mkdir(parents=True)
    protocol.write_text("Synthetic protocol fixture; never a production plan.\n")
    api, _ = q.load_sequential_api(SimpleNamespace(worktree=worktree))
    compact = api.compact
    original = compact.compact_prompt(
        "Synthetic CPU-only application panel; use the same criteria for all 25.\n"
        + "\n".join(f"Application {i}: mock founder {i}, mock proposal." for i in range(25))
        + '\n## Response Format\n"ranked_applications"', 25)
    fixture = unit_compact_funding_fixture()
    fixture.update(funding_selection_protocol=q.SEQUENTIAL_PROTOCOL, original_compact_prompt=original,
                   program_id="NSF_AI_FOUNDATIONS", panel_index=4,
                   messages=api.build_selection_messages(original))
    path = Path(spec["funding_probe_fixture"]["path"])
    path.chmod(0o644)
    path.write_bytes(q.json_bytes(fixture))
    path.chmod(0o444)
    spec["funding_probe_fixture"]["sha256"] = q.file_hash(path)
    spec["funding_selection_protocol"] = q.SEQUENTIAL_PROTOCOL
    spec["queue_id"] = "new-sequential-v6"
    for job in spec["jobs"]:
        job["id"] = job["id"].replace("-", "_v6_", 1)
    return q.parse_plan(q.json_bytes(spec))


@pytest.fixture
def sequential_queue(sequential_plan, monkeypatch):
    provenance = {"sources": {}, "versions": {}, "git_head": "synthetic-v6", "branch": "llm",
                  "host": "mock-host", "boot_id": "mock-boot", "runtime_files": {}}
    q.private_directory(sequential_plan.directory, sequential_plan.private_root)
    state = q.load_state(sequential_plan, provenance, False)
    result = q.Queue(sequential_plan, provenance, state)
    result.lock_root = sequential_plan.private_root / "locks"
    result.lock_root.mkdir(mode=0o700)
    monkeypatch.setattr(q, "source_hashes", lambda _: {})
    monkeypatch.setattr(q, "stop_owned", lambda *a, **k: [])
    monkeypatch.setattr(q, "session_members", lambda _: [])
    return result


def test_sequential_queue_fixture_has_private_locks(sequential_queue):
    assert sequential_queue.lock_root.stat().st_mode & 0o777 == 0o700
    q.private_directory(sequential_queue.lock_root)


def fake_token_counter(monkeypatch, plan, *, overflow=False):
    fixture, _ = q.load_funding_fixture(plan)
    made = []
    class Counter:
        def __init__(self, *args, **kwargs):
            self.owned_identity = identity(777)
            self.command = ["synthetic-private-python-tokenizer"]
            self.closed = False
            self.requests = []
            self.identity = {
                "protocol": "sequential-tokenizer-v1", "model": q.MODEL, "revision": q.REVISION,
                "request_audit_sha256": q.file_hash(plan.worktree / "utopia/models/request_audit.py"),
                **{key: fixture["tokenization"][key] for key in (
                    "tokenizer_files_sha256", "chat_template_sha256")}}
            made.append(self)
        def count(self, request, **kwargs):
            self.requests.append(copy.deepcopy(request))
            result = {"request_sha256": q.digest_bytes(q.json_bytes(request)),
                      "input_tokens": 32768 if overflow else len(json.dumps(request["messages"])),
                      "rendered_prompt_sha256": "a" * 64}
            q.validate_token_count(result, request)
            return result
        def close(self):
            self.closed = True
    monkeypatch.setattr(q, "ProbeTokenCounter", Counter)
    return made


def completed_sequential_probe(plan, monkeypatch, **wire_options):
    q.private_directory(plan.directory, plan.private_root)
    counters = fake_token_counter(monkeypatch, plan)
    calls = probe_wire(monkeypatch, **wire_options)
    fixture, binding = q.load_funding_fixture(plan)
    record = q.sequential_funding_probe(
        plan, identity(100), 18080, fixture, binding, plan.directory, "server-001")
    return record, calls, counters


def test_sequential_25_exact_shared_requests_and_real_singleton_call(sequential_plan, monkeypatch):
    plan = sequential_plan
    record, calls, counters = completed_sequential_probe(plan, monkeypatch)
    fixture, binding = q.load_funding_fixture(plan)
    q.validate_funding_probe_attestation(record, identity(100), 18080, fixture, binding, plan=plan)
    assert len(calls) == len(record["steps"]) == len(counters[0].requests) == 25
    assert counters[0].closed
    assert record["validation"]["final_json"] == {"ranked_application_ids": list(reversed(range(25)))}
    for i, (step, (req, timeout)) in enumerate(zip(record["steps"], calls)):
        body = json.loads(req.data)
        assert body == counters[0].requests[i] == step["request"]
        assert body["temperature"] == .7 and body["max_tokens"] == 8192
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert step["selected"] == list(reversed(range(25 - i, 25)))
        assert step["remaining"] == list(range(25 - i))
        assert step["seed_context"] == ["phase5_funding_eval", 0, q.SEQUENTIAL_PROTOCOL,
                                       "technical_probe", "NSF_AI_FOUNDATIONS", 4, i]
        assert 0 < timeout <= 600 and step["attempt"] == 0
    assert record["steps"][-1]["schema"]["properties"]["next_application_id"]["enum"] == [0]
    assert len({s["request"]["seed"] for s in record["steps"]}) == 25
    assert len(list(plan.directory.glob("server-001.sequential-step-*.json"))) == 50


@pytest.mark.parametrize("change", [
    lambda r: r["steps"].pop(),
    lambda r: r["steps"][-1].update(selected=list(range(24))),
    lambda r: r["steps"][1].update(remaining=list(range(25))),
    lambda r: r["steps"][0].update(attempt=1),
    lambda r: r["steps"][0]["request"].update(seed=42),
    lambda r: r["steps"][0]["request"].update(max_tokens=2048),
    lambda r: r["steps"][0]["token_count"].update(input_tokens=1),
    lambda r: r.update(attempts_per_step=3),
    lambda r: r.update(timeout_seconds=15001),
    lambda r: r["validation"]["final_json"]["ranked_application_ids"].reverse(),
])
def test_sequential_attestation_rejects_prefix_seed_count_or_budget_drift(
        sequential_plan, monkeypatch, change):
    record, _, _ = completed_sequential_probe(sequential_plan, monkeypatch)
    change(record)
    with pytest.raises(q.QueueError):
        q.validate_funding_probe_attestation(
            record, identity(100), 18080, *q.load_funding_fixture(sequential_plan), plan=sequential_plan)


@pytest.mark.parametrize("mutation", [
    lambda p: p["choices"][0]["message"].update(content='{"next_application_id":true}'),
    lambda p: p["choices"][0]["message"].update(content='{"next_application_id":99}'),
    lambda p: p["choices"][0]["message"].update(
        content='{"next_application_id":0,"next_application_id":1}'),
    lambda p: p["choices"][0].update(finish_reason="length"),
    lambda p: p["usage"].update(prompt_tokens=1),
    lambda p: p["usage"].update(completion_tokens=8193),
])
def test_sequential_invalid_first_selection_aborts_once_without_repair(
        sequential_plan, monkeypatch, mutation):
    q.private_directory(sequential_plan.directory, sequential_plan.private_root)
    counters = fake_token_counter(monkeypatch, sequential_plan)
    calls = probe_wire(monkeypatch, mutate=mutation)
    with pytest.raises(q.ProbeError) as error:
        q.sequential_funding_probe(
            sequential_plan, identity(100), 18080, *q.load_funding_fixture(sequential_plan),
            sequential_plan.directory, "server-001")
    assert len(calls) == 1 and counters[0].closed
    assert error.value.attestation["steps"] == []
    assert error.value.attestation["failed_step"]["status"] == "failed"
    assert "response_body" in error.value.attestation["failed_step"]


def test_sequential_previous_selection_cannot_be_repeated(sequential_plan, monkeypatch):
    q.private_directory(sequential_plan.directory, sequential_plan.private_root)
    fake_token_counter(monkeypatch, sequential_plan)
    seen = []
    def repeat(payload):
        seen.append(1)
        if len(seen) == 2:
            payload["choices"][0]["message"]["content"] = '{"next_application_id":24}'
    calls = probe_wire(monkeypatch, mutate=repeat)
    with pytest.raises(q.ProbeError) as error:
        q.sequential_funding_probe(
            sequential_plan, identity(100), 18080, *q.load_funding_fixture(sequential_plan),
            sequential_plan.directory, "server-001")
    assert len(calls) == 2 and len(error.value.attestation["steps"]) == 1


def test_sequential_context_overflow_stops_before_http(sequential_plan, monkeypatch):
    q.private_directory(sequential_plan.directory, sequential_plan.private_root)
    made = fake_token_counter(monkeypatch, sequential_plan, overflow=True)
    calls = probe_wire(monkeypatch)
    with pytest.raises(q.ProbeError, match="exceeds context"):
        q.sequential_funding_probe(
            sequential_plan, identity(100), 18080, *q.load_funding_fixture(sequential_plan),
            sequential_plan.directory, "server-001")
    assert calls == [] and made[0].closed


@pytest.mark.parametrize("which", ["intent", "result"])
@pytest.mark.parametrize("change", ["writable", "content", "symlink"])
def test_sequential_sidecar_integrity_rechecked(sequential_plan, monkeypatch, which, change):
    record, _, _ = completed_sequential_probe(sequential_plan, monkeypatch)
    path = Path(record["steps"][7]["persistence"][which]["path"])
    if change == "writable":
        path.chmod(0o644)
    elif change == "content":
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)
    else:
        path.rename(path.with_suffix(".original"))
        path.symlink_to(path.with_suffix(".original"))
    with pytest.raises((q.QueueError, OSError)):
        q.validate_funding_probe_attestation(
            record, identity(100), 18080, *q.load_funding_fixture(sequential_plan), plan=sequential_plan)


def test_new_selection_scope_rejects_old_fixture_and_v5_ids(sequential_plan):
    spec = copy.deepcopy(sequential_plan.spec)
    spec.pop("funding_selection_protocol")
    with pytest.raises(q.QueueError, match="v6"):
        q.parse_plan(q.json_bytes(spec))
    spec = copy.deepcopy(sequential_plan.spec)
    spec["jobs"][0]["id"] = "influx_v5_A"
    with pytest.raises(q.QueueError, match="fresh v6"):
        q.parse_plan(q.json_bytes(spec))
    spec = copy.deepcopy(sequential_plan.spec)
    fixture = Path(spec["funding_probe_fixture"]["path"])
    fixture.chmod(0o644)
    fixture.write_bytes(q.json_bytes(unit_compact_funding_fixture()))
    fixture.chmod(0o444)
    spec["funding_probe_fixture"]["sha256"] = q.file_hash(fixture)
    plan = q.parse_plan(q.json_bytes(spec))
    with pytest.raises(q.QueueError, match="Plan/fixture"):
        q.load_funding_fixture(plan)


def test_sequential_protocol_markdown_in_frozen_source_set(sequential_plan):
    before = q.source_hashes(sequential_plan)
    assert q.SEQUENTIAL_PROTOCOL_FILE in before
    (sequential_plan.worktree / q.SEQUENTIAL_PROTOCOL_FILE).write_text("Changed protocol")
    assert q.source_hashes(sequential_plan) != before


def test_old_compact_proof_cannot_authorize_v6(sequential_plan):
    fixture, binding = q.load_funding_fixture(sequential_plan)
    old = unit_compact_funding_fixture()
    stale = passed_funding_record(identity(100), 18080, old, binding)
    with pytest.raises(q.QueueError, match="legacy"):
        q.validate_funding_probe_attestation(stale, identity(100), 18080, fixture, binding,
                                            plan=sequential_plan)


def test_sequential_readiness_needs_nonce_plus_25_and_final_rechecks(
        sequential_queue, monkeypatch):
    queue = sequential_queue
    server = prepare_real_attestation(queue, monkeypatch)
    # prepare_real_attestation overrides only ownership/runtime methods.
    fake_token_counter(monkeypatch, queue.plan)
    calls = probe_wire(monkeypatch)
    queue.attest(server)
    assert server.ready and len(calls) == 26
    manifest = json.loads(server.manifest_path.read_text())
    assert manifest["funding_selection_protocol"] == q.SEQUENTIAL_PROTOCOL
    assert manifest["funding_panel25_probe"]["validation"]["model_selections"] == 25
    mock_client_launch(queue, monkeypatch)
    job = queue.plan.jobs[0]
    queue.start_job(server, job)
    write_artifacts(job)
    queue.finish_job(server, 0)
    q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=False)
    path = Path(server.funding_probe["steps"][0]["persistence"]["result"]["path"])
    path.chmod(0o644)
    with pytest.raises(q.QueueError, match="writable"):
        q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=False)


def test_sequential_waits_for_shared_uuid_reservations(sequential_queue, monkeypatch):
    queue = sequential_queue
    monkeypatch.setattr(q, "query_gpus", inventory)
    held = [q.reserve_pair(inventory(), queue.lock_root, query=inventory) for _ in range(4)]
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("Must not adopt another queue's pair"))
    try:
        assert queue.step() is False
        assert queue.state["status"] == "waiting_for_idle_pair"
        assert not queue.servers
        assert all(v["status"] == "pending" for v in queue.state["jobs"].values())
    finally:
        for _, locks in held:
            for lock in locks:
                lock.close()


@pytest.mark.parametrize("reply,match", [(b"", "closed"), (b"x" * 65537 + b"\n", "too large")])
def test_tokenizer_sidecar_eof_and_oversized_reply_fail_bounded(monkeypatch, reply, match):
    counter = object.__new__(q.ProbeTokenCounter)
    counter.cancelled = lambda: False
    counter.buffer = b""
    counter.process = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 123))
    monkeypatch.setattr(q.select, "select", lambda *a: ([counter.process.stdout], [], []))
    monkeypatch.setattr(q.os, "read", lambda *a: reply)
    with pytest.raises(q.QueueError, match=match):
        counter._read_reply(time.monotonic() + 1)


def test_tokenizer_sidecar_timeout_and_cancellation_do_not_send_http(monkeypatch):
    counter = object.__new__(q.ProbeTokenCounter)
    counter.cancelled = lambda: False
    counter.buffer = b""
    counter.process = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 123))
    monkeypatch.setattr(q.select, "select", lambda *a: ([], [], []))
    with pytest.raises(q.QueueError, match="time budget"):
        counter._read_reply(time.monotonic() + .01)
    counter.cancelled = lambda: True
    with pytest.raises(q.QueueError, match="interrupted"):
        counter._read_reply(time.monotonic() + 1)


@pytest.mark.parametrize("failure", [None, "startup", "identity"])
def test_tokenizer_private_cpu_process_is_bounded_and_only_owned_child_cleaned(
        sequential_plan, monkeypatch, tmp_path, failure):
    launches, stops, killed = [], [], []
    class Pipe(io.BytesIO):
        def fileno(self):
            return 120
    child = SimpleNamespace(pid=777, stdin=Pipe(), stdout=Pipe(), poll=lambda: None,
                            kill=lambda: killed.append(777), wait=lambda **k: 0)
    monkeypatch.setattr(q.subprocess, "Popen",
                        lambda command, **kwargs: launches.append((command, kwargs)) or child)
    monkeypatch.setattr(q, "proc_identity",
                        lambda pid: None if failure == "identity" else identity(pid))
    monkeypatch.setattr(q, "stop_owned", lambda root, **k: stops.append(root) or [])
    monkeypatch.setattr(q.os, "set_blocking", lambda *a: None)
    def reply(self, deadline):
        assert 0 < deadline - time.monotonic() <= 120
        if failure == "startup":
            raise q.QueueError("CPU tokenizer startup failed")
        return {"synthetic": True}
    monkeypatch.setattr(q.ProbeTokenCounter, "_read_reply", reply)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3")
    monkeypatch.setenv("PYTHONPATH", "/unrelated-source")
    log = tmp_path / "tokenizer.log"
    if failure:
        with pytest.raises(q.QueueError):
            q.ProbeTokenCounter(sequential_plan, log)
    else:
        counter = q.ProbeTokenCounter(sequential_plan, log)
        assert counter.identity == {"synthetic": True}
        counter.close()
    command, kwargs = launches[0]
    assert command[:3] == [str(sequential_plan.python), "-u", "-B"]
    assert command[-3:] == ["--tokenizer-sidecar", str(sequential_plan.worktree),
                           str(sequential_plan.hf_home)]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["env"]["PYTHONHASHSEED"] == "42" and "PYTHONPATH" not in kwargs["env"]
    assert kwargs["start_new_session"] and kwargs["close_fds"]
    assert (killed == [777] and not stops) if failure == "identity" else (
        not killed and stops == [identity(777)])
    assert child.stdin.closed and child.stdout.closed


def test_tokenizer_aggregate_caps_each_wait_and_cancel_prevents_spawn(
        sequential_plan, monkeypatch, tmp_path):
    counter = object.__new__(q.ProbeTokenCounter)
    counter.cancelled = lambda: False
    counter.aggregate_deadline = time.monotonic() + .01
    assert 0 < counter._remaining(time.monotonic() + 120) <= .01
    counter.aggregate_deadline = time.monotonic() - .01
    with pytest.raises(q.QueueError, match="fixed time budget"):
        counter._remaining(time.monotonic() + 120)
    monkeypatch.setattr(q.subprocess, "Popen", lambda *a, **k: pytest.fail("cancelled CPU spawn"))
    with pytest.raises(q.QueueError, match="interrupted"):
        q.ProbeTokenCounter(sequential_plan, tmp_path / "cancelled.log", cancelled=lambda: True)


def test_sequential_late_failure_preserves_prefix_and_never_starts_client(
        sequential_queue, monkeypatch):
    queue = sequential_queue
    server = prepare_real_attestation(queue, monkeypatch)
    counters = fake_token_counter(monkeypatch, queue.plan)
    selected_calls = []
    def corrupt_third_choice(payload):
        if "next_application_id" in payload["choices"][0]["message"]["content"]:
            selected_calls.append(payload)
            if len(selected_calls) == 3:
                payload["choices"][0]["message"]["content"] = '{"next_application_id":24}'
    calls = probe_wire(monkeypatch, mutate=corrupt_third_choice)
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("failed probe started a client"))
    assert queue.run() == 1 and len(calls) == 4  # Nonce, two valid choices, failed choice.
    assert counters[0].closed and not server.ready and not server.manifest_path.exists()
    failure = json.loads((queue.plan.directory / f"{server.name}.funding-probe.failure.json").read_text())
    assert len(failure["steps"]) == 2
    assert failure["pending_step"]["selected"] == [24, 23]
    assert failure["failed_step"]["response_metrics"]["finish_reason"] == "stop"
    assert "validation" not in failure["failed_step"]
    assert '{"next_application_id":24}' in failure["failed_step"]["response_body"].replace('\\"', '"')
    assert all(v["status"] == "pending" for v in queue.state["jobs"].values())


@pytest.mark.parametrize("family", ["influx", "mechanisms"])
def test_only_declared_empty_compact_raw_with_clean_summary_can_receive_receipt(
        sequential_plan, family):
    spec = copy.deepcopy(sequential_plan.spec)
    job = next(j for j in spec["jobs"] if j["family"] == family)
    root = Path(job["output_roots"][0])
    raw_name, summary_name = (("funding_validation_audit.jsonl", "funding_validation_summary.json")
                             if family == "influx"
                             else ("funding_validation.jsonl", "funding_validation.summary.json"))
    clean = {"/completion_allowed": True, "/installed": False, "/invalid_final_panels": 0,
             "/failed_batches": 0, "/audit_failures": 0,
             "/output_representation": "ordered_application_ids_v1"}
    job["artifacts"].extend([
        {"path": str(root / raw_name), "min_bytes": 0},
        {"path": str(root / summary_name), "min_bytes": 1, "equals": clean}])
    plan = q.parse_plan(q.json_bytes(spec))
    started = time.time_ns()
    write_artifacts(job)
    (root / raw_name).write_bytes(b"")
    (root / summary_name).write_bytes(q.json_bytes({key[1:]: value for key, value in clean.items()}))
    receipts = q.artifact_gate(job, plan, started_ns=started)
    receipt = next(r for r in receipts if r["path"] == str(root / raw_name))
    assert receipt["size"] == 0 and receipt["sha256"] == q.digest_bytes(b"")
    (root / raw_name).unlink()
    with pytest.raises((q.QueueError, OSError)):
        q.artifact_gate(job, plan, started_ns=started)


@pytest.mark.parametrize("case", ["sequential", "unknown", "summary_missing", "dirty", "empty_summary"])
def test_empty_artifact_exception_cannot_weaken_other_audits(sequential_plan, case):
    spec = copy.deepcopy(sequential_plan.spec)
    job = spec["jobs"][0]
    root = Path(job["output_roots"][0])
    raw = {"path": str(root / "funding_validation_audit.jsonl"), "min_bytes": 0}
    summary = {
        "path": str(root / "funding_validation_summary.json"), "min_bytes": 1,
        "equals": {"/completion_allowed": True, "/installed": False, "/invalid_final_panels": 0,
                   "/failed_batches": 0, "/audit_failures": 0,
                   "/output_representation": "ordered_application_ids_v1"}}
    if case in ("sequential", "unknown"):
        raw["path"] = str(root / ("funding_sequential.jsonl" if case == "sequential" else "other.jsonl"))
    elif case == "dirty":
        summary["equals"]["/invalid_final_panels"] = 1
    elif case == "empty_summary":
        summary["min_bytes"] = 0
    job["artifacts"].append(raw)
    if case != "summary_missing":
        job["artifacts"].append(summary)
    with pytest.raises(q.QueueError, match="(min_bytes|clean summary)"):
        q.parse_plan(q.json_bytes(spec))


def test_cancellation_during_launch_journaling_cannot_spawn(queue, monkeypatch):
    pair, locks = configure_fake_server_launch(queue, monkeypatch)
    original_save = queue.save
    def cancel_after_journal():
        original_save()
        queue.interrupted = True
    monkeypatch.setattr(queue, "save", cancel_after_journal)
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("spawn after journal cancellation"))
    with pytest.raises(q.QueueError, match="interrupted before server process creation"):
        queue.start_server(pair, locks)
    assert not queue.servers and all(lock.fd is None for lock in locks)
    assert all(q.reservation_blocks(q.reservation_path(queue.lock_root, gpu.uuid)) for gpu in pair)
    # The final Popen boundary also guards cancellation after call-site argument preparation.
    with pytest.raises(q.QueueError, match="interrupted before process creation"):
        q.Queue.spawn(queue, ["never-execute"], {}, queue.plan.directory / "cancelled-spawn.log")


def test_cancellation_during_postprobe_gpu_query_cannot_commit_readiness(queue, monkeypatch):
    server = prepare_real_attestation(queue, monkeypatch)
    fixture, _ = q.load_funding_fixture(queue.plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture)
    def cancel_during_gpu_query():
        queue.interrupted = True
        return inventory()
    monkeypatch.setattr(q, "query_gpus", cancel_during_gpu_query)
    with pytest.raises(q.QueueError, match="interrupted before readiness commit"):
        queue.attest(server)
    assert len(calls) == 2
    assert not server.ready and server.probe is None and server.funding_probe is None
    assert not server.manifest_path.exists()
    assert all(record["status"] == "pending" for record in queue.state["jobs"].values())


def test_funding_completion_usage_requires_bounded_integer_and_coherent_total(plan, monkeypatch):
    fixture, binding = q.load_funding_fixture(plan)
    builder, _ = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    invalid_usage = [
        {"prompt_tokens": 4795, "total_tokens": 4795},
        {"prompt_tokens": 4795, "completion_tokens": 8193, "total_tokens": 12988},
        {"prompt_tokens": 4795, "completion_tokens": -1, "total_tokens": 4794},
        {"prompt_tokens": 4795, "completion_tokens": True, "total_tokens": 4796},
        {"prompt_tokens": 4795, "completion_tokens": 1000},
        {"prompt_tokens": 4795, "completion_tokens": 1000, "total_tokens": 5796},
    ]
    for usage in invalid_usage:
        calls = probe_wire(monkeypatch, funding_fixture=fixture,
                           mutate=lambda payload: payload.update(usage=usage))
        with pytest.raises(q.ProbeError, match="Funding (completion_tokens|total_tokens)"):
            q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
        assert len(calls) == 1
    for completion in (0, 8192):
        payload = funding_payload(fixture)
        payload["usage"].update(completion_tokens=completion, total_tokens=4795 + completion)
        q.verify_funding_probe_response(payload, fixture)


def write_initialization_artifacts(queue, *, readonly=True):
    # Filesystem timestamps may lag time_ns until the next kernel clock tick.
    time.sleep(0.02)
    job = next(job for job in queue.plan.jobs if q.is_propensity_initializer(job))
    write_artifacts(job)
    path = Path(queue.plan.initialization["artifact"])
    path.write_bytes(q.json_bytes({"status": "complete", "seed": 42,
                                  "choices": {"mock-founder": {"topic": "mock-topic"}}}))
    if readonly:
        path.chmod(0o444)


def complete_mock_initialization(queue, monkeypatch):
    server = mock_server(queue)
    spawned = mock_client_launch(queue, monkeypatch)
    job = next(job for job in queue.plan.jobs if q.is_propensity_initializer(job))
    queue.start_job(server, job)
    write_initialization_artifacts(queue)
    queue.finish_job(server, 0)
    return server, spawned


def test_propensity_is_one_cache_operation_and_exactly_four_seed42_worlds(propensity_plan):
    assert len(propensity_plan.jobs) == 5
    assert sum(job["family"] == "propensity_init" for job in propensity_plan.jobs) == 1
    assert {job["cell"] for job in propensity_plan.jobs if job["family"] == "propensity"} == {
        "LN", "LF", "HN", "HF"}
    assert {job["seed"] for job in propensity_plan.jobs} == {42}
    assert q.source_hashes(propensity_plan)["docs/switching_propensity_protocol.md"] == q.file_hash(
        propensity_plan.worktree / "docs/switching_propensity_protocol.md")


@pytest.mark.parametrize("mutation", [
    lambda s: s["jobs"].pop(),
    lambda s: s["jobs"].append(copy.deepcopy(s["jobs"][-1])),
    lambda s: s["jobs"][1].update(cell="LF"),
    lambda s: s["jobs"][1].update(seed=43),
    lambda s: s["jobs"][1].update(family="mechanisms"),
    lambda s: s["jobs"][1].update(require_manifest_argument=False),
    lambda s: s["propensity_initialization"].update(job_id="LN"),
    lambda s: s["jobs"][0]["artifacts"].pop(),
    lambda s: s["jobs"][1]["command"].__setitem__(-1, "/other/initial_choices.json"),
    lambda s: s["jobs"][0]["command"].extend(["--initial-choices", s["propensity_initialization"]["artifact"]]),
    lambda s: s["propensity_initialization"].update(artifact="/outside/initial_choices.json"),
])
def test_propensity_plan_rejects_extra_worlds_bad_cells_or_dependency_drift(propensity_specification, mutation):
    mutation(propensity_specification)
    with pytest.raises(q.QueueError):
        q.parse_plan(q.json_bytes(propensity_specification))


def test_worlds_cannot_launch_from_pending_or_running_initialization(propensity_queue, monkeypatch):
    queue = propensity_queue
    server = mock_server(queue)
    spawned = mock_client_launch(queue, monkeypatch)
    world = next(job for job in queue.plan.jobs if job["family"] == "propensity")
    for status in ("pending", "running"):
        queue.state["jobs"]["initialize"]["status"] = status
        with pytest.raises(q.QueueError, match="not runnable"):
            queue.start_job(server, world)
        assert queue.runnable() == ([queue.plan.jobs[0]] if status == "pending" else [])
        assert queue.server_limit() == 1
    assert not spawned


def test_initializer_priority_is_independent_of_plan_job_order(propensity_specification, monkeypatch):
    propensity_specification["jobs"].reverse()
    queue = make_test_queue(q.parse_plan(q.json_bytes(propensity_specification)), monkeypatch)
    assert [job["id"] for job in queue.runnable()] == ["initialize"]


def test_initialization_exit_zero_requires_complete_readonly_cache(propensity_queue, monkeypatch):
    queue = propensity_queue
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    write_initialization_artifacts(queue, readonly=False)
    with pytest.raises(q.QueueError, match="immutable regular"):
        queue.finish_job(server, 0)
    assert queue.state["jobs"]["initialize"]["status"] == "running"
    assert not (queue.plan.directory / "job-initialize.receipt.json").exists()
    assert queue.runnable() == []


def test_initialization_failure_never_unblocks_worlds(propensity_queue, monkeypatch):
    queue = propensity_queue
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    write_initialization_artifacts(queue)
    with pytest.raises(q.QueueError, match="exited 1"):
        queue.finish_job(server, 1)
    assert not (queue.plan.directory / "job-initialize.receipt.json").exists()
    assert not queue.runnable()


def test_verified_initialization_binds_each_world_and_four_world_final_count(propensity_queue, monkeypatch):
    queue = propensity_queue
    server, spawned = complete_mock_initialization(queue, monkeypatch)
    dependency = q.initialization_binding(queue.plan, queue.state, queue.provenance)
    initial_receipt = json.loads((queue.plan.directory / "job-initialize.receipt.json").read_text())
    assert initial_receipt["scientific_world"] is False
    assert initial_receipt["initial_choices"] == dependency["initial_choices"]
    assert queue.server_limit() == 4
    for job in queue.plan.jobs[1:]:
        queue.start_job(server, job)
        assert queue.state["jobs"][job["id"]]["initialization_dependency"] == dependency
        assert spawned[-1][0][-1] == queue.plan.initialization["artifact"]
        write_artifacts(job)
        queue.finish_job(server, 0)
        receipt = json.loads((queue.plan.directory / f"job-{job['id']}.receipt.json").read_text())
        assert receipt["scientific_world"] is True and receipt["cell"] == job["cell"]
        assert receipt["initialization_dependency"] == dependency
    result = q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=True)
    assert result["completed_jobs"] == 5 and result["completed_scientific_worlds"] == 4
    assert result["initialization_is_scientific_world"] is False
    assert result["initialization_dependency"] == dependency


@pytest.mark.parametrize("tamper", ["writable", "content", "inode", "symlink", "receipt", "sources", "predicate"])
def test_world_launch_rechecks_cache_receipt_source_and_initialization_artifacts(
        propensity_queue, monkeypatch, tamper):
    queue = propensity_queue
    server, spawned = complete_mock_initialization(queue, monkeypatch)
    cache = Path(queue.plan.initialization["artifact"])
    if tamper == "writable":
        cache.chmod(0o644)
    elif tamper == "content":
        cache.chmod(0o644)
        cache.write_bytes(cache.read_bytes() + b" ")
        cache.chmod(0o444)
    elif tamper == "inode":
        replacement = cache.with_suffix(".replacement")
        replacement.write_bytes(cache.read_bytes())
        replacement.chmod(0o444)
        os.replace(replacement, cache)
    elif tamper == "symlink":
        original = cache.with_suffix(".original")
        cache.rename(original)
        cache.symlink_to(original)
    elif tamper == "receipt":
        receipt = queue.plan.directory / "job-initialize.receipt.json"
        receipt.chmod(0o644)
        receipt.write_bytes(receipt.read_bytes() + b" ")
    elif tamper == "sources":
        monkeypatch.setattr(q, "source_hashes", lambda _: {"modified.py": "bad"})
    else:
        Path(queue.plan.jobs[0]["artifacts"][0]["path"]).write_text('{"status":"failed","seed":42}')
    with pytest.raises(q.QueueError):
        queue.start_job(server, queue.plan.jobs[1])
    assert len(spawned) == 1
    assert queue.state["jobs"]["LN"]["status"] == "pending"


def test_dependency_rechecked_after_world_launch_journaling(propensity_queue, monkeypatch):
    queue = propensity_queue
    server, spawned = complete_mock_initialization(queue, monkeypatch)
    original_save = queue.save
    def mutate_after_journal():
        original_save()
        Path(queue.plan.initialization["artifact"]).chmod(0o644)
    monkeypatch.setattr(queue, "save", mutate_after_journal)
    with pytest.raises(q.QueueError, match="immutable regular"):
        queue.start_job(server, queue.plan.jobs[1])
    assert len(spawned) == 1


def test_cache_change_during_world_prevents_scientific_completion(propensity_queue, monkeypatch):
    queue = propensity_queue
    server, _ = complete_mock_initialization(queue, monkeypatch)
    job = queue.plan.jobs[1]
    queue.start_job(server, job)
    write_artifacts(job)
    Path(queue.plan.initialization["artifact"]).chmod(0o644)
    with pytest.raises(q.QueueError, match="immutable regular"):
        queue.finish_job(server, 0)
    assert not (queue.plan.directory / f"job-{job['id']}.receipt.json").exists()


def test_propensity_resume_retains_completed_cache_without_rerunning_init(propensity_queue, monkeypatch):
    queue = propensity_queue
    server, _ = complete_mock_initialization(queue, monkeypatch)
    queue.close_server(server)
    queue.save()
    state = q.load_state(queue.plan, queue.provenance, True)
    resumed = q.Queue(queue.plan, queue.provenance, state)
    assert state["jobs"]["initialize"]["status"] == "complete"
    assert {job["id"] for job in resumed.runnable()} == {"LN", "LF", "HN", "HF"}
    Path(queue.plan.initialization["artifact"]).chmod(0o644)
    with pytest.raises(q.QueueError, match="immutable regular"):
        q.load_state(queue.plan, queue.provenance, True)


def test_scheduler_uses_one_server_for_init_then_expands_to_four(propensity_queue, monkeypatch):
    queue = propensity_queue
    spawned = mock_client_launch(queue, monkeypatch)
    started = []
    def start(pair, locks):
        server = mock_server(queue, len(started) + 1, pair)
        server.locks = locks
        started.append(server)
    monkeypatch.setattr(queue, "start_server", start)
    assert queue.step() is False
    assert len(started) == 1 and not spawned
    assert queue.step() is False
    assert len(started) == len(spawned) == 1
    assert started[0].job["family"] == "propensity_init"
    assert queue.step() is False
    assert len(started) == len(spawned) == 1
    write_initialization_artifacts(queue)
    started[0].client.process.code = 0
    assert queue.step() is False
    assert len(started) == 4 and len(spawned) == 2
    assert queue.step() is False
    assert len(spawned) == 5
    assert {server.job["cell"] for server in started} == {"LN", "LF", "HN", "HF"}
    for server in started:
        write_artifacts(server.job)
        server.client.process.code = 0
    assert queue.step() is True
    assert queue.state["final_validation"]["completed_scientific_worlds"] == 4


def test_propensity_scheduler_shares_uuid_locks_with_other_queue(propensity_queue, monkeypatch):
    queue = propensity_queue
    mock_client_launch(queue, monkeypatch)
    external_pairs = [q.reserve_pair(inventory(), queue.lock_root, query=inventory) for _ in range(2)]
    assert [[gpu.index for gpu in pair] for pair, _ in external_pairs] == [[0, 1], [2, 3]]
    def start(pair, locks):
        server = mock_server(queue, pair=pair)
        server.locks = locks
    monkeypatch.setattr(queue, "start_server", start)
    try:
        queue.step()
        assert len(queue.servers) == 1
        assert [gpu.index for gpu in queue.servers[0].pair] == [4, 5]
        # Every production instance derives the same host/user lock root.
        assert q.Queue(queue.plan, {}, {}).lock_root == Path("/tmp") / f"scienceutopia-gpu-locks-{os.getuid()}"
    finally:
        for server in list(queue.servers):
            queue.close_server(server)
        for _, locks in external_pairs:
            for lock in locks:
                lock.close()


def unit_compact_funding_fixture():
    fixture = unit_funding_fixture()
    fixture["output_representation"] = "ordered_application_ids_v1"
    fixture["json_schema"] = {
        "type": "object", "properties": {"ranked_application_ids": {
            "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 24},
            "minItems": 1, "maxItems": 25}},
        "required": ["ranked_application_ids"], "additionalProperties": False}
    fixture["expected"]["rankings_json_pointer"] = "/ranked_application_ids"
    # Deliberately distinct synthetic count; the production fixture is authored/tokenized elsewhere.
    fixture["tokenization"].update(input_tokens=4300, total_requested_tokens=12492)
    return fixture



def install_unit_compact_fixture(specification):
    spec = copy.deepcopy(specification)
    path = Path(spec["funding_probe_fixture"]["path"])
    path.chmod(0o644)
    path.write_bytes(q.json_bytes(unit_compact_funding_fixture()))
    path.chmod(0o444)
    spec["funding_probe_fixture"]["sha256"] = q.file_hash(path)
    return q.parse_plan(q.json_bytes(spec))



def test_compact_funding_exact_wire_and_trusted_implicit_rank_mapping(specification, monkeypatch):
    plan = install_unit_compact_fixture(specification)
    fixture, binding = q.load_funding_fixture(plan)
    q.validate_funding_fixture(fixture)
    calls = probe_wire(monkeypatch, funding_fixture=fixture)
    builder, metadata = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    record = q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
    record["shared_client_builder"] = metadata
    q.validate_funding_probe_attestation(record, identity(100), 18080, fixture, binding)
    assert len(calls) == 1
    request = json.loads(calls[0][0].data)
    assert request["messages"] == fixture["messages"]
    assert request["structured_outputs"]["json"] == fixture["json_schema"]
    assert request["seed"] == 670339654 and request["max_tokens"] == 8192
    assert request["temperature"] == .7 and request["chat_template_kwargs"]["enable_thinking"] is True
    assert binding["output_representation"] == "ordered_application_ids_v1"
    validation = record["validation"]
    assert validation["exact_prompt_tokens"] == 4300
    assert validation["normalized_application_ranks"][0] == {
        "application_id": 24, "applicant_id": "mock-founder-24", "rank": 1}
    assert validation["normalized_application_ranks"][-1]["rank"] == 25
    assert "not generated" in validation["applicant_mapping_source"]



@pytest.mark.parametrize("mutation", [
    lambda f: f.update(output_representation="unknown"),
    lambda f: f.pop("output_representation"),
    lambda f: f["expected"].update(rankings_json_pointer="/ranked_applications"),
    lambda f: f["json_schema"].update(additionalProperties=True),
    lambda f: f["json_schema"]["properties"]["ranked_application_ids"].update(minItems=25),
    lambda f: f["json_schema"]["properties"]["ranked_application_ids"]["items"].update(minimum=False),
    lambda f: f["json_schema"]["properties"]["ranked_application_ids"]["items"].update(maximum=25),
])
def test_compact_funding_fixture_refuses_schema_or_representation_drift(mutation):
    fixture = unit_compact_funding_fixture()
    mutation(fixture)
    with pytest.raises(q.QueueError):
        q.validate_funding_fixture(fixture)



@pytest.mark.parametrize("result", [
    {"ranked_application_ids": list(range(24))},
    {"ranked_application_ids": list(range(26))},
    {"ranked_application_ids": [5 if i == 12 else i for i in range(25)]},
    {"ranked_application_ids": [-1] + list(range(1, 25))},
    {"ranked_application_ids": [25] + list(range(1, 25))},
    {"ranked_application_ids": [False] + list(range(1, 25))},
    {"ranked_application_ids": [0.0] + list(range(1, 25))},
    {"ranked_application_ids": ["0"] + list(range(1, 25))},
    {"ranked_application_ids": list(range(25)), "ranked_applications": []},
    {"ranked_applications": []},
])
def test_compact_funding_rejects_incomplete_duplicate_or_nonstrict_ids_once(specification, monkeypatch, result):
    plan = install_unit_compact_fixture(specification)
    fixture, binding = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture, mutate=lambda p:
                       p["choices"][0]["message"].update(content=json.dumps(result)))
    builder, _ = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    with pytest.raises(q.ProbeError):
        q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
    assert len(calls) == 1



@pytest.mark.parametrize("mutation", [
    lambda p: p["usage"].update(prompt_tokens=4301),
    lambda p: p["usage"].pop("completion_tokens"),
    lambda p: p["usage"].update(completion_tokens=8193, total_tokens=12493),
    lambda p: p["usage"].update(total_tokens=0),
    lambda p: p["choices"][0].update(finish_reason="length"),
])
def test_compact_funding_keeps_exact_usage_and_stop_guards(specification, monkeypatch, mutation):
    plan = install_unit_compact_fixture(specification)
    fixture, binding = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture, mutate=mutation)
    builder, _ = real_probe_builder(q.FUNDING_SEED_COMPONENTS)
    with pytest.raises(q.ProbeError):
        q.funding_panel25_probe(identity(100), 18080, builder, fixture, binding)
    assert len(calls) == 1



def test_compact_fixture_probes_gate_readiness_and_persist_both_receipts(specification, monkeypatch):
    plan = install_unit_compact_fixture(specification)
    provenance = {"sources": {}, "versions": {}, "git_head": "test-head", "branch": "llm",
                  "host": "mock-host", "boot_id": "mock-boot", "runtime_files": {}}
    q.private_directory(plan.directory, plan.private_root)
    controller = q.Queue(plan, provenance, q.load_state(plan, provenance, False))
    monkeypatch.setattr(q, "source_hashes", lambda _: {})
    monkeypatch.setattr(q, "stop_owned", lambda *a, **k: [])
    monkeypatch.setattr(q, "session_members", lambda _: [])
    server = prepare_real_attestation(controller, monkeypatch)
    fixture, _ = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture)
    controller.attest(server)
    assert server.ready and len(calls) == 2
    manifest = json.loads(server.manifest_path.read_text())
    assert manifest["funding_panel25_probe"]["validation"]["output_representation"] == "ordered_application_ids_v1"
    assert manifest["structured_output_probe"]["validation"]["finish_reason"] == "stop"
    mock_client_launch(controller, monkeypatch)
    controller.start_job(server, plan.jobs[0])
    write_artifacts(plan.jobs[0])
    controller.finish_job(server, 0)
    receipt = json.loads((plan.directory / f"job-{plan.jobs[0]['id']}.receipt.json").read_text())
    assert receipt["funding_probe_fixture_sha256"] == plan.spec["funding_probe_fixture"]["sha256"]
    assert receipt["funding_panel25_probe_sha256"] == q.digest_bytes(q.json_bytes(server.funding_probe))


@pytest.mark.parametrize("mutation", [
    lambda s: s["jobs"][0]["command"].__setitem__(2, "run"),
    lambda s: s["jobs"][1]["command"].__setitem__(2, "initialize"),
    lambda s: s["jobs"][1]["command"].__setitem__(-3, "HF"),
    lambda s: s["jobs"][1]["command"].extend(["--cell", "LN"]),
    lambda s: s["jobs"][0]["artifacts"][0]["equals"].pop("/scientific_years_completed"),
    lambda s: s["jobs"][1]["artifacts"][0]["equals"].pop("/kind"),
    lambda s: s["jobs"][0]["artifacts"][0]["equals"].update({"/choice_count": 1199}),
    lambda s: s["jobs"][1]["artifacts"][0]["equals"].update({"/years_completed": 9}),
    lambda s: s["jobs"][0]["artifacts"][1]["equals"].pop("/pending_requests"),
    lambda s: s["jobs"][0]["artifacts"][2]["equals"].update({"/installed": True}),
    lambda s: s["jobs"][0]["artifacts"][2]["equals"].pop("/final_panels"),
    lambda s: s["jobs"][1]["artifacts"][3].update(min_bytes=0),
])
def test_propensity_plan_enforces_exact_driver_roles_and_completion_contracts(propensity_specification, mutation):
    mutation(propensity_specification)
    with pytest.raises(q.QueueError):
        q.parse_plan(q.json_bytes(propensity_specification))


@pytest.mark.parametrize("count", [None, 0, -1, True, "1200"])
def test_propensity_initializer_requires_positive_integer_request_count(propensity_queue, monkeypatch, count):
    queue = propensity_queue
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    write_initialization_artifacts(queue)
    path = Path(queue.plan.initialization["artifact"]).with_name("llm_request_audit.summary.json")
    summary = json.loads(path.read_text())
    if count is None:
        summary.pop("n_requests")
    else:
        summary["n_requests"] = count
    path.write_bytes(q.json_bytes(summary))
    with pytest.raises(q.QueueError, match="n_requests>0"):
        queue.finish_job(server, 0)
    assert not (queue.plan.directory / "job-initialize.receipt.json").exists()
    assert not queue.runnable()


@pytest.mark.parametrize("name,value", [
    ("pre_choice_state_hash", None), ("pre_choice_state_hash", "not-a-hash"),
    ("initial_state_hash", ""), ("initial_state_hash", 123),
])
def test_initializer_hash_contract_fails_before_receipt(propensity_queue, monkeypatch, name, value):
    queue = propensity_queue
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    write_initialization_artifacts(queue)
    path = Path(queue.plan.jobs[0]["artifacts"][0]["path"])
    manifest = json.loads(path.read_text())
    manifest[name] = value
    path.write_bytes(q.json_bytes(manifest))
    with pytest.raises(q.QueueError, match="state hashes must be SHA256"):
        queue.finish_job(server, 0)
    assert not (queue.plan.directory / "job-initialize.receipt.json").exists()


@pytest.mark.parametrize("field", ["initial_choices_sha256", "pre_choice_state_hash", "initial_state_hash"])
def test_world_must_match_cache_and_both_producer_state_hashes(propensity_queue, monkeypatch, field):
    queue = propensity_queue
    server, _ = complete_mock_initialization(queue, monkeypatch)
    job = queue.plan.jobs[1]
    queue.start_job(server, job)
    write_artifacts(job, **{field: "f" * 64})
    with pytest.raises(q.QueueError, match="different initial choices|state hashes differ"):
        queue.finish_job(server, 0)
    assert not (queue.plan.directory / f"job-{job['id']}.receipt.json").exists()


def test_empty_funding_jsonl_is_bound_but_missing_audit_cannot_complete(propensity_queue, monkeypatch):
    queue = propensity_queue
    server, _ = complete_mock_initialization(queue, monkeypatch)
    initial_artifacts = queue.state["jobs"]["initialize"]["artifacts"]
    proof = next(a for a in initial_artifacts if Path(a["path"]).name == "funding_validation.jsonl")
    assert proof["size"] == 0 and proof["sha256"] == q.digest_bytes(b"")
    job = queue.plan.jobs[1]
    queue.start_job(server, job)
    write_artifacts(job)
    docs = Path(job["artifacts"][0]["path"]).parent
    (docs / "funding_validation.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        queue.finish_job(server, 0)
    assert not (queue.plan.directory / f"job-{job['id']}.receipt.json").exists()


def test_funding_summary_failure_blocks_even_zero_application_completion(propensity_queue, monkeypatch):
    queue = propensity_queue
    server = mock_server(queue)
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, queue.plan.jobs[0])
    write_initialization_artifacts(queue)
    path = Path(queue.plan.initialization["artifact"]).with_name("funding_validation.summary.json")
    summary = json.loads(path.read_text())
    summary["restore_conflicts"] = ["injected unit-test conflict"]
    path.write_bytes(q.json_bytes(summary))
    with pytest.raises(q.QueueError, match="predicate failed"):
        queue.finish_job(server, 0)
    assert not queue.runnable()


def test_propensity_dependency_and_compact_live_probes_share_manifest_receipts(
        propensity_specification, monkeypatch):
    plan = install_unit_compact_fixture(propensity_specification)
    queue = make_test_queue(plan, monkeypatch)
    server = prepare_real_attestation(queue, monkeypatch)
    fixture, _ = q.load_funding_fixture(plan)
    calls = probe_wire(monkeypatch, funding_fixture=fixture)
    queue.attest(server)
    assert len(calls) == 2
    mock_client_launch(queue, monkeypatch)
    queue.start_job(server, plan.jobs[0])
    write_initialization_artifacts(queue)
    queue.finish_job(server, 0)
    world = plan.jobs[1]
    queue.start_job(server, world)
    write_artifacts(world)
    queue.finish_job(server, 0)
    result = q.validate_completed(plan, queue.state, queue.provenance, require_all=False)
    assert result["completed_jobs"] == 2 and result["completed_scientific_worlds"] == 1
    assert result["initial_state_hash"] == "a" * 64
    dependency = result["initialization_dependency"]
    assert dependency["initial_state_hash"] == "a" * 64
    assert dependency["pre_choice_state_hash"] == "b" * 64
    receipt = json.loads((plan.directory / f"job-{world['id']}.receipt.json").read_text())
    assert receipt["initialization_dependency"] == dependency
    assert receipt["world_input_binding"]["initial_choices_sha256"] == dependency["initial_choices"]["sha256"]
    assert receipt["funding_panel25_probe_sha256"] == q.digest_bytes(q.json_bytes(server.funding_probe))


def native_role_specification(spec):
    result = copy.deepcopy(spec)
    init_id = result["propensity_initialization"]["job_id"]
    for job in result["jobs"]:
        initial = job["id"] == init_id
        job.update(family="propensity", role="initialize" if initial else "condition",
                   depends_on=[] if initial else [init_id])
        if initial:
            job["cell"] = None
    return result


def test_native_owner_roles_and_depends_on_keep_exactly_four_scientific_worlds(
        propensity_specification, monkeypatch):
    plan = q.parse_plan(q.json_bytes(native_role_specification(propensity_specification)))
    queue = make_test_queue(plan, monkeypatch)
    assert [job["id"] for job in queue.runnable()] == ["initialize"]
    assert sum(q.is_propensity_initializer(job) for job in plan.jobs) == 1
    assert sum(q.is_propensity_world(job) for job in plan.jobs) == 4
    server, _ = complete_mock_initialization(queue, monkeypatch)
    for job in plan.jobs[1:]:
        queue.start_job(server, job)
        write_artifacts(job)
        queue.finish_job(server, 0)
    result = q.validate_completed(plan, queue.state, queue.provenance, require_all=True)
    assert result["completed_jobs"] == 5 and result["completed_scientific_worlds"] == 4
    assert result["initialization_is_scientific_world"] is False
    initial_receipt = json.loads((plan.directory / "job-initialize.receipt.json").read_text())
    assert initial_receipt["scientific_world"] is False


@pytest.mark.parametrize("mutation", [
    lambda s: s["jobs"][0].update(depends_on=["LN"]),
    lambda s: s["jobs"][1].update(depends_on=[]),
    lambda s: s["jobs"][1].update(depends_on=["LF"]),
    lambda s: s["jobs"][1].update(depends_on=["initialize", "initialize"]),
    lambda s: s["jobs"][1].update(role="initialize"),
    lambda s: s["jobs"][0].update(role="condition"),
    lambda s: s["jobs"][0].update(cell="LN"),
    lambda s: s["jobs"][1].update(role="other"),
])
def test_native_owner_schema_rejects_extra_edges_cycles_or_role_drift(propensity_specification, mutation):
    spec = native_role_specification(propensity_specification)
    mutation(spec)
    with pytest.raises(q.QueueError):
        q.parse_plan(q.json_bytes(spec))


def test_pathfix_actual_owner_queue_jobs_builds_the_fixed_dependency_plan(pathfix_plan):
    """Execute the actual pure declarative emitter, never a simulator or model."""
    root = project_root(q.__file__)
    with q.propensity_admission_driver(SimpleNamespace(worktree=root)) as driver:
        from utopia.experiments.switching_propensity import queue_jobs
        jobs = queue_jobs(pathfix_plan.worktree, pathfix_plan.private_root)
    initial = next(job for job in jobs if job["role"] == "initialize")
    cache = next(a["path"] for a in initial["artifacts"] if Path(a["path"]).name == "initial_choices.json")
    spec = copy.deepcopy(pathfix_plan.spec)
    spec.update(jobs=jobs, propensity_initialization={"job_id": initial["id"], "artifact": cache})
    plan = q.parse_plan(q.json_bytes(spec))
    assert initial["id"] == "switching_propensity_v3tp1p1_n1200_y10_initial_seed42"
    assert Path(cache) == pathfix_plan.worktree / "outputs/docs" / initial["id"] / "initial_choices.json"
    assert len(plan.jobs) == 5 and sum(q.is_propensity_world(job) for job in plan.jobs) == 4
    for job in plan.jobs:
        initial_job = q.is_propensity_initializer(job)
        assert job["depends_on"] == ([] if initial_job else [initial["id"]])
        assert len(job["output_roots"]) == 4
        assert all(Path(path).is_relative_to(pathfix_plan.worktree / "outputs")
                   for path in job["output_roots"])
        if not initial_job:
            assert job["command"][job["command"].index("--initial-choices") + 1] == cache
        assert q.propensity_audit_artifacts(job)["funding_validation.jsonl"]["min_bytes"] == 0
        assert q.propensity_sequential_artifacts(job)["funding_sequential.jsonl"]["min_bytes"] == 1
        names = {Path(a["path"]).name for a in job["artifacts"]}
        assert "run_manifest.json" in names
        if not initial_job:
            assert {f"year_{year:02d}.json" for year in range(1, 11)} <= names
            assert {f"mechanisms_year_{year}.json" for year in range(1, 11)} <= names
            assert "direction_embedding_norms.json" in names


@pytest.fixture
def propensity_sequential_plan(propensity_specification):
    """Synthetic five-operation plan with real shared sequential request builders."""
    spec = native_role_specification(propensity_specification)
    worktree, private = Path(spec["worktree"]), Path(spec["private_root"])
    source = project_root(q.__file__)
    names = ("utopia/funding/sequential.py", "utopia/funding/compact.py",
             "utopia/models/structured_outputs.py", "utopia/models/request_audit.py",
             "utopia/utils/seeding.py", "utopia/utils/data_utils.py",
                 "utopia/utils/paths.py", "utopia/runtime/historical.py",
                 "utopia/funding/validation.py", "utopia/constants.py")
    for name in names:
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    (worktree / q.SEQUENTIAL_PROTOCOL_FILE).write_text("Synthetic technical-probe protocol.\n")
    api, _ = q.load_sequential_api(SimpleNamespace(worktree=worktree))
    original = api.compact.compact_prompt(
        "Synthetic 25-application panel for controller CPU tests.\n## Response Format\n"
        '"ranked_applications"', 25)
    fixture = unit_compact_funding_fixture()
    fixture.update(funding_selection_protocol=q.SEQUENTIAL_PROTOCOL, original_compact_prompt=original,
                   program_id="NSF_THEORY", panel_index=0, messages=api.build_selection_messages(original))
    path = Path(spec["funding_probe_fixture"]["path"])
    path.chmod(0o644)
    path.write_bytes(q.json_bytes(fixture))
    path.chmod(0o444)
    spec["funding_probe_fixture"]["sha256"] = q.file_hash(path)
    spec.update(queue_id="unit-propensity-v3-sequential", funding_selection_protocol=q.SEQUENTIAL_PROTOCOL)
    sources = {name: q.file_hash(worktree / name) for name in q.PROPENSITY_SOURCE_FILES}
    init_id = "switching_propensity_v3_n1200_y10_initial_seed42"
    cache = str(private / "outputs/docs" / init_id / "initial_choices.json")
    spec["propensity_initialization"] = {"job_id": init_id, "artifact": cache}
    for job in spec["jobs"]:
        initial = q.is_propensity_initializer(job)
        cell = "initial" if initial else job["cell"]
        name = f"switching_propensity_v3_n1200_y10_{cell}_seed42"
        old_root = job["output_roots"][0]
        docs = private / "outputs/docs" / name
        job.update(id=name, depends_on=[] if initial else [init_id],
                   output_roots=[str(private / "outputs" / kind / name)
                                 for kind in ("docs", "logs", "checkpoints", "visual")])
        for artifact in job["artifacts"]:
            artifact["path"] = str(docs) + artifact["path"][len(old_root):]
        if not initial:
            job["command"][-1] = cache
        gates = job["artifacts"][0]["equals"]
        gates.update({"/protocol": q.PROPENSITY_SEQUENTIAL_PROTOCOL,
                      "/funding_selection_protocol": q.SEQUENTIAL_PROTOCOL,
                      "/sequential_audit_valid": True, "/funding_application_evidence_valid": True,
                      "/inputs/source_files_sha256": sources,
                      "/sequential_source_binding": {key: sources[key] for key in q.PROPENSITY_SOURCE_FILES[:2]}})
        years = 0 if initial else 10
        summary = {
            "funding_selection_protocol": q.SEQUENTIAL_PROTOCOL,
            "output_representation": "ordered_application_ids_v1",
            "installed": False, "status": "complete", "completion_allowed": True,
            "failed_panels": 0, "cancelled_panels": 0, "unprocessed_panels": 0,
            "fatal_errors": 0, "guard_failures": 0, "processing_failures": 0,
            "audit_failures": 0, "restore_conflicts": [],
            "years_started": years, "years_completed": years, "agency_registrations": 2 * years}
        if initial:
            summary.update(registered_panels=0, processed_panels=0, accepted_steps=0, sdk_calls=0)
        job["artifacts"].extend([
            {"path": str(docs / "funding_sequential.jsonl"), "min_bytes": 1},
            {"path": str(docs / "funding_sequential.summary.json"), "min_bytes": 1,
             "equals": {"/" + key: value for key, value in summary.items()}}])
    return q.parse_plan(q.json_bytes(spec))


@pytest.fixture
def propensity_sequential_queue(propensity_sequential_plan, monkeypatch):
    return make_propensity_sequential_queue(propensity_sequential_plan, monkeypatch)


def make_propensity_sequential_queue(plan, monkeypatch):
    queue = make_test_queue(plan, monkeypatch)
    sources = {name: q.file_hash(plan.worktree / name) for name in q.PROPENSITY_SOURCE_FILES}
    if "server_profile" in plan.spec:
        sources["utopia/runtime/server_profiles.py"] = q.file_hash(plan.worktree / "utopia/runtime/server_profiles.py")
    queue.provenance["sources"] = sources
    queue.state["provenance"]["sources"] = sources
    monkeypatch.setattr(q, "source_hashes", lambda _: sources)
    fake_token_counter(monkeypatch, plan)
    queue.admission_calls = []
    def validate(directory, manifest, *, expected_source):
        assert expected_source == sources  # Comes from queue provenance, never manifest claims.
        queue.admission_calls.append((manifest["kind"], expected_source.copy()))
        years = 0 if manifest["kind"] == "initialization_only" else 10
        checkpoint = Path(manifest["args"]["output_dir"])
        present = {path.name: q.file_hash(path) for path in checkpoint.glob("funding_applications_year_*.jsonl")}
        return {"funding_selection_protocol": q.SEQUENTIAL_PROTOCOL, "scientific_years": years,
                "funding_application_evidence": {
                    "present_sha256": present, "absent_years": [
                        year for year in range(1, years + 1)
                        if f"funding_applications_year_{year}.jsonl" not in present]}}
    @q.contextlib.contextmanager
    def driver(_):
        yield SimpleNamespace(source_identity=lambda: sources, validate_funding_evidence=validate)
    monkeypatch.setattr(q, "propensity_admission_driver", driver)
    return queue


def write_propensity_sequential_artifacts(queue, job, **overrides):
    # Filesystem timestamps may lag time_ns until the next kernel clock tick.
    time.sleep(0.02)
    def assign(target, pointer, value):
        parts = pointer.lstrip("/").split("/")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    checkpoint = queue.plan.private_root / "outputs/checkpoints" / job["id"]
    checkpoint.mkdir(parents=True)
    for artifact in job["artifacts"]:
        path = Path(artifact["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            path.write_bytes(b"" if path.name == "funding_validation.jsonl"
                             else b'{"synthetic_controller_test":true}\n')
            continue
        value = {}
        for key, expected in artifact.get("equals", {}).items():
            assign(value, key, expected)
        if path.name == "llm_request_audit.summary.json":
            value["n_requests"] = 1200
        if path.name in ("initialization_manifest.json", "propensity_manifest.json"):
            value.update(pre_choice_state_hash="b" * 64, initial_state_hash="a" * 64,
                         args={"output_dir": str(checkpoint)})
            if q.is_propensity_world(job):
                initial_manifest = Path(queue.plan.initialization["artifact"]).with_name("initialization_manifest.json")
                value.update(initial_choices_sha256=q.file_hash(Path(queue.plan.initialization["artifact"])),
                             initialization_manifest_path=str(initial_manifest),
                             initialization_manifest_sha256=q.file_hash(initial_manifest))
            value.update(overrides)
        path.write_bytes(q.json_bytes(value))
        if path.name == "initial_choices.json":
            path.chmod(0o444)


def ready_propensity_sequential(queue, monkeypatch):
    server = prepare_real_attestation(queue, monkeypatch)
    if "server_profile" in queue.plan.spec:
        server.pair = [inventory()[7]]
        server.command = q.server_command(queue.plan, server.port)
        queue.state["servers"][server.name].update(
            **q.selected_runtime_binding(queue.plan), command=server.command,
            gpu_ids=[7], gpu_uuids=[server.pair[0].uuid])
    calls = probe_wire(monkeypatch)
    queue.attest(server)
    assert len(calls) == 26
    return server, mock_client_launch(queue, monkeypatch)


@pytest.fixture
def tp1_plan(propensity_sequential_plan):
    spec = json.loads(json.dumps(propensity_sequential_plan.spec).replace("_v3_", "_v3tp1_"))
    spec.update(server_profile=q.TP1_PLAN_PROFILE, queue_id="unit-propensity-tp1")
    path = propensity_sequential_plan.worktree / "utopia/runtime/server_profiles.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((project_root(q.__file__) / 'utopia/runtime/server_profiles.py').read_bytes())
    for job in spec["jobs"]:
        gates = q.propensity_manifest_artifact(job)["equals"]
        gates["/inputs/source_files_sha256"]["utopia/runtime/server_profiles.py"] = q.file_hash(path)
        gates["/inputs/server_runtime_profile"] = q.TP1_PROFILE_NAME
    return q.parse_plan(q.json_bytes(spec))


@pytest.fixture
def tp1_queue(tp1_plan, monkeypatch):
    return make_propensity_sequential_queue(tp1_plan, monkeypatch)


@pytest.fixture
def pathfix_plan(tp1_plan):
    spec = json.loads(json.dumps(tp1_plan.spec).replace("_v3tp1_", "_v3tp1p1_"))
    spec["queue_id"] = "unit-propensity-v3tp1p1"
    def producer(value):
        return str(tp1_plan.worktree / Path(value).relative_to(tp1_plan.private_root))
    spec["propensity_initialization"]["artifact"] = producer(spec["propensity_initialization"]["artifact"])
    for job in spec["jobs"]:
        job["output_roots"] = [producer(path) for path in job["output_roots"]]
        for artifact in job["artifacts"]:
            artifact["path"] = producer(artifact["path"])
        if "--initial-choices" in job["command"]:
            job["command"][job["command"].index("--initial-choices") + 1] = (
                spec["propensity_initialization"]["artifact"])
    return q.parse_plan(q.json_bytes(spec))


@pytest.mark.parametrize("change", ["mixed_ids", "unknown_suffix", "no_tp1", "no_sequential"])
def test_pathfix_requires_one_complete_named_profile_cohort(pathfix_plan, change):
    spec = copy.deepcopy(pathfix_plan.spec)
    if change == "mixed_ids":
        spec["jobs"][1]["id"] = spec["jobs"][1]["id"].replace("_v3tp1p1_", "_v3tp1_")
    elif change == "unknown_suffix":
        spec = json.loads(json.dumps(spec).replace("_v3tp1p1_", "_v3tp1p2_"))
    elif change == "no_tp1":
        spec.pop("server_profile")
    else:
        spec.pop("funding_selection_protocol")
    with pytest.raises(q.QueueError):
        q.parse_plan(q.json_bytes(spec))


@pytest.fixture(params=[0, 10], ids=["initializer-zero-years", "condition-ten-years"])
def real_pathfix_evidence(pathfix_plan, request, monkeypatch):
    """Real raw writers/validators; only tokenizer and SDK responses are scripted.

    The ten-year fixture includes one native two-application panel plus explicit
    empty agency/year coverage. It is a validator fixture, not a simulated world.
    """
    import tests.support.sequential_funding as native
    source = project_root(q.__file__)
    with q.propensity_admission_driver(SimpleNamespace(worktree=source)) as driver:
        names = tuple(driver.source_identity())
    for name in names:
        path = pathfix_plan.worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    with q.propensity_admission_driver(pathfix_plan) as driver:
        sources = driver.source_identity()
    spec = copy.deepcopy(pathfix_plan.spec)
    for job in spec["jobs"]:
        gates = q.propensity_manifest_artifact(job)["equals"]
        gates["/inputs/source_files_sha256"] = sources
        gates["/sequential_source_binding"] = {key: sources[key] for key in q.PROPENSITY_SOURCE_FILES[:2]}
    plan = q.parse_plan(q.json_bytes(spec))
    years = request.param
    job = plan.jobs[0 if years == 0 else 1]
    started = time.time_ns()
    docs = plan.worktree / "outputs/docs" / job["id"]
    checkpoint = plan.worktree / "outputs/checkpoints" / job["id"]
    docs.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    seed_module = ModuleType("utopia.utils.seeding")
    seed_module.derive_seed = native.native_seed()
    monkeypatch.setitem(q.sys.modules, seed_module.__name__, seed_module)
    with q.propensity_admission_driver(plan) as driver:
        from utopia.funding.validation import install_funding_validation
        from utopia.funding.sequential import install_sequential_funding
        funding, model_type = native.funding_class(), native.model_class()
        compact = install_funding_validation(funding, docs / "funding_validation.jsonl", compact=True)
        sequential = install_sequential_funding(funding, model_type, docs / "funding_sequential.jsonl")
        audit = native.audit_module.RequestAudit(docs / "llm_request_audit.jsonl",
                                                _tokenizer=native.TinyTokenizer())
        model = model_type()
        model.model_name, model.run_seed, model.enable_thinking = q.MODEL, 42, True
        model.max_concurrent_requests, model.request_audit = 1, audit
        audit.bind(q.MODEL)
        model.sdk = native.SDK()
        model.client = SimpleNamespace(max_retries=5, chat=SimpleNamespace(completions=model.sdk))
        model.call_stats = {key: 0 for key in (
            "n_prompts", "n_first_attempt_success", "n_retries", "n_failures",
            "elapsed_seconds", "prompt_tokens", "completion_tokens")}
        try:
            model.generate_batch(["Unrelated initialization request for CPU validation"],
                                 seed_ctx=("phase1_fixture", 1, 0))
            for year in range(1, years + 1):
                sequential.begin_year(year, ["NSF", "DARPA"])
                for agency_id, program, rate in (
                        ("NSF", "NSF_THEORY", .23), ("DARPA", "DARPA_SECURITY", .10)):
                    agency = funding()
                    agency.id = agency_id
                    agency.funding_programs = {program: SimpleNamespace(
                        name=program, research_directions=[native.Direction("algorithms")],
                        funding_rate=rate)}
                    n = 2 if year == 1 and agency_id == "NSF" else 0
                    applications = [{program: {
                        "submit": True, "research_proposal": f"CPU fixture proposal {i}",
                        "author": SimpleNamespace(id=f"fixture-{i}",
                            expertise=[native.Direction("algorithms")], university_name="Fixture"),
                        "relevant_projects": [],
                    }} for i in range(n)]
                    prompts, schema, metadata = agency.get_funding_evaluation_prompts(
                        applications, {}, panel_max_apps=25,
                        panel_seed=native.native_seed()(42, "funding_panels", year))
                    if prompts:
                        result = model.generate_batch(prompts, response_format=schema, max_tokens=8192,
                                                      seed_ctx=("phase5_funding_eval", year, 0))
                        ledger = []
                        funding.process_funding_evaluation_results(
                            result, metadata, agency.funding_programs, application_log=ledger)
                        (checkpoint / f"funding_applications_year_{year}.jsonl").write_text(
                            "".join(json.dumps({**row, "year": year}) + "\n" for row in ledger))
                sequential.end_year(year)
            audit.close(True)
        finally:
            if not audit._closed:
                audit.close(False)
            sequential.restore()
            compact.restore()
        compact_summary, sequential_summary = compact.summary(), sequential.summary()
        (docs / "funding_validation.summary.json").write_bytes(q.json_bytes(compact_summary))
        endpoint = "http://127.0.0.1:18081/v1"
        server_path = plan.private_root / "outputs" / f"{job['id']}-server.manifest.json"
        server_path.write_bytes(q.json_bytes({
            **q.tp1_server_spec(), "server_profile": q.TP1_PLAN_PROFILE,
            "endpoint": endpoint, "gpu_ids": [7], "host": "cpu-fixture", "pid": 777,
            "launch_command": q.server_command(plan, 18081), "started_utc": "cpu-fixture",
        }))
        server = driver.provenance.server_provenance(server_path, endpoint, runtime_profile=q.TP1_PROFILE_NAME)
        manifest = {
            "protocol": driver.PROTOCOL, "status": "complete",
            "kind": "initialization_only" if years == 0 else "condition",
            "seed": 42, "founder_count": 1200, "choice_count": 1200,
            "scientific_years_completed": years, "years_completed": years,
            "cell": None if years == 0 else job["cell"],
            "inputs": {"source_files_sha256": sources, "server_runtime_profile": q.TP1_PROFILE_NAME,
                       "runtime_amendment": driver.RUNTIME_AMENDMENT,
                       "server_runtime_hash": server["runtime_hash"]},
            "server_provenance": server,
            "args": {"output_dir": str(checkpoint), "vllm_url": endpoint},
            "funding_selection_protocol": q.SEQUENTIAL_PROTOCOL,
            "funding_output_representation": driver.OUTPUT_REPRESENTATION,
            "funding_baseline": driver.feedback.FUNDING_BASELINE,
            "sequential_source_binding": {key: sources[key] for key in q.PROPENSITY_SOURCE_FILES[:2]},
            "funding_validation": compact_summary, "request_audit": audit.summary(),
            "funding_sequential": sequential_summary,
            "funding_sequential_audits": {name: q.file_hash(docs / name) for name in (
                "funding_sequential.jsonl", "funding_sequential.summary.json")},
            "sequential_audit_valid": True,
            "funding_application_evidence": driver.feedback.funding_ledger_evidence(checkpoint, years),
            "funding_application_evidence_valid": True,
        }
        (docs / "run_manifest.json").write_bytes(q.json_bytes({
            "status": "initialization_only_complete" if years == 0 else "complete",
            "years_completed": years, "scientific_years_completed": years}))
        manifest_path = Path(q.propensity_manifest_artifact(job)["path"])
        manifest_path.write_bytes(q.json_bytes(manifest))
        # Other declared artifacts are outside this funding-admission API's scope.
        for artifact in job["artifacts"]:
            path = Path(artifact["path"])
            if path.exists():
                continue
            value = {}
            for pointer, expected in artifact.get("equals", {}).items():
                parts, target = pointer.lstrip("/").split("/"), value
                for key in parts[:-1]:
                    target = target.setdefault(key, {})
                target[parts[-1]] = expected
            path.write_bytes(q.json_bytes(value))
    artifacts = q.artifact_gate(job, plan, started_ns=started)
    return SimpleNamespace(plan=plan, job=job, artifacts=artifacts,
                           provenance={"sources": sources}, started=started, docs=docs,
                           manifest=manifest, years=years, scripted_calls=len(model.sdk.calls))


def test_pathfix_real_raw_admission_accepts_producer_alias_and_reproduces_old_failure(real_pathfix_evidence):
    evidence = real_pathfix_evidence
    plan, job = evidence.plan, evidence.job
    canonical = q.output_path(plan, evidence.docs)
    assert canonical != evidence.docs and canonical.samefile(evidence.docs)
    with q.propensity_admission_driver(plan) as driver:
        from utopia.funding.sequential import SequentialFundingError
        with pytest.raises(SequentialFundingError) as error:
            driver.validate_funding_evidence(
                canonical, evidence.manifest, expected_source=evidence.provenance["sources"])
        assert error.value.diagnostics["detail"] == "audit path mismatch"
    report = q.propensity_funding_admission(
        plan, job, evidence.artifacts, evidence.provenance, evidence.started)
    assert report["validation"]["scientific_years"] == evidence.years
    summary = report["validation"]["funding_sequential"]
    assert summary["years_completed"] == evidence.years
    assert summary["agency_registrations"] == 2 * evidence.years
    assert summary["sdk_calls"] == (2 if evidence.years else 0)
    assert evidence.scripted_calls == (3 if evidence.years else 1)
    assert len(report["native_ledger_files"]) == bool(evidence.years)


@pytest.mark.parametrize("change", ["outside_alias", "other_job", "extra_alias", "directory_link"])
def test_pathfix_rejects_untrusted_or_redirected_docs_before_validator(real_pathfix_evidence, monkeypatch, change):
    evidence = real_pathfix_evidence
    plan, job = evidence.plan, copy.deepcopy(evidence.job)
    canonical = q.output_path(plan, evidence.docs)
    if change == "outside_alias":
        outside = plan.worktree.parent / "outside"
        outside.mkdir()
        (plan.worktree / "outputs").unlink()
        (plan.worktree / "outputs").symlink_to(outside, target_is_directory=True)
    elif change == "other_job":
        q.propensity_manifest_artifact(job)["path"] = str(
            plan.private_root / "outputs/docs/another-job" / Path(q.propensity_manifest_artifact(job)["path"]).name)
    elif change == "extra_alias":
        link = plan.worktree.parent / "unapproved-alias"
        link.symlink_to(canonical, target_is_directory=True)
        q.propensity_manifest_artifact(job)["path"] = str(
            link / Path(q.propensity_manifest_artifact(job)["path"]).name)
    else:
        old = canonical.with_name(canonical.name + "-other")
        canonical.rename(old)
        canonical.symlink_to(old, target_is_directory=True)
    monkeypatch.setattr(q, "propensity_admission_driver",
                        lambda *a: pytest.fail("Invalid directory reached the driver"))
    with pytest.raises((q.QueueError, FileNotFoundError)):
        q.propensity_funding_admission(
            plan, job, evidence.artifacts, evidence.provenance, evidence.started)


def test_pathfix_rechecks_alias_mapping_after_unchanged_raw_validator(real_pathfix_evidence, monkeypatch):
    evidence = real_pathfix_evidence
    original = q.propensity_admission_driver
    @q.contextlib.contextmanager
    def moved_alias(plan):
        with original(plan) as driver:
            yield driver  # Actual unmodified validation executes before the redirect.
        outside = plan.worktree.parent / "post-validation-outside"
        outside.mkdir()
        (plan.worktree / "outputs").unlink()
        (plan.worktree / "outputs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(q, "propensity_admission_driver", moved_alias)
    with pytest.raises(q.QueueError, match="outputs alias changed"):
        q.propensity_funding_admission(evidence.plan, evidence.job, evidence.artifacts,
                                      evidence.provenance, evidence.started)


def test_pathfix_rejects_alias_retarget_and_restore_during_real_validator(real_pathfix_evidence, monkeypatch):
    evidence = real_pathfix_evidence
    plan = evidence.plan
    canonical = q.output_path(plan, evidence.docs)
    before_directory = q.stat_binding(canonical.stat())
    alias = plan.worktree / "outputs"
    before_link, target = alias.lstat(), os.readlink(alias)
    alternate = plan.worktree.parent / "copied-evidence-outputs"
    shutil.copytree(plan.private_root / "outputs", alternate)
    saved_link = plan.worktree / "saved-original-outputs-link"
    original = q.propensity_admission_driver
    validator_completed = []

    @q.contextlib.contextmanager
    def retarget_and_restore(plan):
        with original(plan) as driver:
            alias.rename(saved_link)
            alias.symlink_to(alternate, target_is_directory=True)
            try:
                yield driver  # The unchanged full validator reads the copied raw evidence.
                validator_completed.append(True)
            finally:
                alias.unlink()
                saved_link.rename(alias)  # Restore even the original symlink inode/target.

    monkeypatch.setattr(q, "propensity_admission_driver", retarget_and_restore)
    with pytest.raises(q.QueueError, match="producer directory changed while validating"):
        q.propensity_funding_admission(plan, evidence.job, evidence.artifacts,
                                      evidence.provenance, evidence.started)
    assert validator_completed == [True]
    assert os.readlink(alias) == target and alias.lstat().st_ino == before_link.st_ino
    assert q.stat_binding(canonical.stat()) == before_directory  # Old directory-only proof would miss this.
    assert q.artifact_gate(evidence.job, plan, started_ns=evidence.started) == evidence.artifacts


def test_pathfix_rejects_alias_change_inside_identity_capture(real_pathfix_evidence, monkeypatch):
    evidence = real_pathfix_evidence
    alias = evidence.plan.worktree / "outputs"
    target = os.readlink(alias)
    original = os.readlink
    changed = []

    def replaced_while_reading(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if Path(path) == alias and not changed:
            changed.append(True)
            alias.unlink()
            alias.symlink_to(target, target_is_directory=True)
        return result

    monkeypatch.setattr(q.os, "readlink", replaced_while_reading)
    monkeypatch.setattr(q, "propensity_admission_driver",
                        lambda *args: pytest.fail("Inconsistent capture reached the driver"))
    with pytest.raises(q.QueueError, match="outputs alias changed during identity capture"):
        q.propensity_funding_admission(evidence.plan, evidence.job, evidence.artifacts,
                                      evidence.provenance, evidence.started)
    assert changed == [True]


def test_pathfix_file_analyzer_preserves_producer_spelling(real_pathfix_evidence):
    """Exercise the analyzer's unchanged funding-file adapter, without endpoint analysis."""
    evidence = real_pathfix_evidence
    with q.propensity_admission_driver(evidence.plan):
        path = (project_root(q.__file__) / 'utopia/analysis/switching_propensity.py')
        spec = q.importlib.util.spec_from_file_location("pathfix_analysis_fixture", path)
        analyzer = q.importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer)
        report = analyzer._validate_funding_files(
            evidence.docs, evidence.manifest, evidence.provenance["sources"])
        assert report["scientific_years"] == evidence.years
        with pytest.raises(analyzer.AnalysisError) as error:
            analyzer._validate_funding_files(
                q.output_path(evidence.plan, evidence.docs), evidence.manifest, evidence.provenance["sources"])
        assert error.value.__cause__.diagnostics["detail"] == "audit path mismatch"


def test_pathfix_never_takes_validator_directory_from_manifest_alias(real_pathfix_evidence):
    evidence = real_pathfix_evidence
    other = evidence.plan.worktree.parent / "same-file-but-untrusted-alias"
    other.symlink_to(q.output_path(evidence.plan, evidence.docs), target_is_directory=True)
    summary = copy.deepcopy(evidence.manifest["funding_sequential"])
    summary["audit_path"] = str(other / "funding_sequential.jsonl")
    summary_path = evidence.docs / "funding_sequential.summary.json"
    summary_path.write_bytes(q.json_bytes(summary))
    manifest = copy.deepcopy(evidence.manifest)
    manifest["funding_sequential"] = summary
    manifest["funding_sequential_audits"]["funding_sequential.summary.json"] = q.file_hash(summary_path)
    Path(q.propensity_manifest_artifact(evidence.job)["path"]).write_bytes(q.json_bytes(manifest))
    artifacts = q.artifact_gate(evidence.job, evidence.plan, started_ns=evidence.started)
    with q.propensity_admission_driver(evidence.plan):
        from utopia.funding.sequential import SequentialFundingError
        # The admission loader reloads the exception class, so inspect the fatal
        # error's concrete diagnostics without replacing any validation logic.
        with pytest.raises(BaseException) as error:
            q.propensity_funding_admission(evidence.plan, evidence.job, artifacts,
                                          evidence.provenance, evidence.started)
    assert type(error.value).__name__ == SequentialFundingError.__name__
    assert error.value.diagnostics["detail"] == "audit path mismatch"


def test_tp1_exact_profile_does_not_mutate_default_or_other_accessors(tp1_plan, plan):
    from utopia.runtime.server_profiles import TP1_SERVER_SPEC
    baseline = copy.deepcopy(q.SERVER_SPEC)
    selected = q.selected_server_spec(tp1_plan)
    assert selected["runtime_profile"] == q.TP1_PROFILE_NAME
    assert {key: selected[key] for key in (
        "tensor_parallel_size", "max_num_seqs", "gpu_memory_utilization",
        "enforce_eager", "enable_chunked_prefill", "max_num_batched_tokens")} == {
            "tensor_parallel_size": 1, "max_num_seqs": 16, "gpu_memory_utilization": 0.94,
            "enforce_eager": True, "enable_chunked_prefill": True, "max_num_batched_tokens": 2048}
    for key in ("model", "revision", "dtype", "max_model_len", "seed", "request_seed_policy"):
        assert selected[key] == baseline[key]
    selected["seed"] = 0
    assert q.tp1_server_spec()["seed"] == 42
    assert q.selected_server_spec(plan) == q.SERVER_SPEC == baseline
    with pytest.raises(TypeError):
        TP1_SERVER_SPEC["seed"] = 0
    command = q.server_command(tp1_plan, 18080)
    for flag, value in (("--tensor-parallel-size", "1"), ("--max-num-seqs", "16"),
                        ("--gpu-memory-utilization", "0.94"), ("--max-num-batched-tokens", "2048"),
                        ("--max-model-len", "32768"), ("--seed", "42")):
        assert command[command.index(flag) + 1] == value
    assert command.count("--enforce-eager") == command.count("--enable-chunked-prefill") == 1
    assert "--max-num-batched-tokens" not in q.server_command(plan, 18080)
    assert "--enforce-eager" not in q.server_command(plan, 18080)


@pytest.mark.parametrize("change", [
    "unknown", "canonical_as_plan", "null", "no_profile", "old_ids", "no_sequential",
    "no_initialization", "arbitrary_setting", "bool_tp", "no_chunking", "old_roots",
    "missing_source", "wrong_manifest_profile",
])
def test_tp1_plan_rejects_profile_scope_or_provenance_escape(tp1_plan, change):
    spec = copy.deepcopy(tp1_plan.spec)
    if change == "unknown":
        spec["server_profile"] = "custom"
    elif change == "canonical_as_plan":
        spec["server_profile"] = q.TP1_PROFILE_NAME
    elif change == "null":
        spec["server_profile"] = None
    elif change == "no_profile":
        spec.pop("server_profile")
    elif change == "old_ids":
        spec = json.loads(json.dumps(spec).replace("_v3tp1_", "_v3_"))
    elif change == "no_sequential":
        spec.pop("funding_selection_protocol")
    elif change == "no_initialization":
        spec.pop("propensity_initialization")
    elif change == "arbitrary_setting":
        spec["server"] = {"max_num_seqs": 128}
    elif change == "bool_tp":
        spec["server"] = {"tensor_parallel_size": True}
    elif change == "no_chunking":
        spec["server"] = {"enable_chunked_prefill": False}
    elif change == "old_roots":
        spec["jobs"][0]["output_roots"][1] = spec["jobs"][0]["output_roots"][1].replace("_v3tp1_", "_v3_")
    elif change == "missing_source":
        spec["jobs"][0]["artifacts"][0]["equals"]["/inputs/source_files_sha256"].pop(
            "utopia/runtime/server_profiles.py")
    elif change == "wrong_manifest_profile":
        spec["jobs"][0]["artifacts"][0]["equals"]["/inputs/server_runtime_profile"] = "tp2"
    with pytest.raises(q.QueueError):
        q.parse_plan(q.json_bytes(spec))


def test_tp1_is_not_an_eight_job_profile(specification):
    specification["server_profile"] = q.TP1_PLAN_PROFILE
    with pytest.raises(q.QueueError, match="restricted to sequential propensity"):
        q.parse_plan(q.json_bytes(specification))


def test_tp1_single_allocator_uses_shared_pair_locks_and_cannot_double_assign(tp1_queue, monkeypatch):
    queue = tp1_queue
    shared = q.reserve_pair(inventory({5, 6, 7}), queue.lock_root, query=inventory)
    one = second = None
    try:
        assert [gpu.index for gpu in shared[0]] == [5, 6]
        one = q.reserve_pair(inventory({5, 6, 7}), queue.lock_root, query=inventory, device_count=1)
        assert [gpu.index for gpu in one[0]] == [7]
        assert q.reserve_pair(inventory({5, 6, 7}), queue.lock_root, query=inventory, device_count=1) is None
        assert q.reserve_pair(inventory({6, 7}), queue.lock_root, query=inventory) is None
    finally:
        for reservation in (one, second, shared):
            if reservation:
                for lock in reservation[1]:
                    lock.close()
    # An idle snapshot cannot authorize a device that becomes busy at the recheck.
    assert q.reserve_pair(inventory({7}), queue.lock_root,
                          query=lambda: inventory(set()), device_count=1) is None
    lock = q.Lock(queue.lock_root / f"{inventory()[7].uuid}.lock")
    assert lock.acquire()
    lock.close()


def configure_tp1_launch(queue, monkeypatch):
    _, old_locks = configure_fake_server_launch(queue, monkeypatch)
    for lock in old_locks:
        lock.close()
    monkeypatch.setattr(q, "query_gpus", lambda: inventory({7}))
    return q.reserve_pair(inventory({7}), queue.lock_root,
                          query=lambda: inventory({7}), device_count=1)


@pytest.mark.parametrize("outcome", ["complete_cleanup", "uncertain_fork", "became_busy"])
def test_tp1_launch_intent_and_cleanup_preserve_single_uuid_quarantine(tp1_queue, monkeypatch, outcome):
    queue = tp1_queue
    pair, locks = configure_tp1_launch(queue, monkeypatch)
    calls = []
    def spawn(command, env, path):
        calls.append(command)
        record = q.read_reservation(q.reservation_path(queue.lock_root, pair[0].uuid))
        assert record["phase"] == "launch_intent" and record["server_identity"] is None
        assert record["gpu_ids"] == [7] and record["gpu_uuids"] == [pair[0].uuid]
        assert record["runtime_hash"] == queue.provenance["runtime_hash"]
        assert record["selected_server_spec"] == q.tp1_server_spec()
        assert env["CUDA_VISIBLE_DEVICES"] == "7" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        if outcome == "uncertain_fork":
            raise q.QueueError("uncertain fork")
        return owned(8181)
    monkeypatch.setattr(queue, "spawn", spawn)
    if outcome == "became_busy":
        snapshots = iter([inventory({7}), inventory(set())])
        monkeypatch.setattr(q, "query_gpus", lambda: next(snapshots))
    if outcome != "complete_cleanup":
        with pytest.raises(q.QueueError, match="uncertain fork|became busy after reservation"):
            queue.start_server(pair, locks)
        assert len(calls) == (outcome == "uncertain_fork")
        assert all(lock.fd is None for lock in locks)
        assert q.reserve_pair(inventory({7}), queue.lock_root,
                              query=lambda: inventory({7}), device_count=1) is None
    else:
        queue.start_server(pair, locks)
        assert len(calls) == 1 and len(queue.servers[0].pair) == 1
        assert queue.state["servers"]["server-001"]["server_profile"] == q.TP1_PLAN_PROFILE
        assert queue.close_server(queue.servers[0]) == []
        record = q.read_reservation(q.reservation_path(queue.lock_root, pair[0].uuid))
        assert record["phase"] == "released" and all(lock.fd is None for lock in locks)
    assert not any(q.reservation_path(queue.lock_root, inventory()[index].uuid).exists()
                   for index in range(7))


def test_tp1_step_allocates_lone_free_card_and_respects_initializer_limit(tp1_queue, monkeypatch):
    queue = tp1_queue
    pair, locks = configure_tp1_launch(queue, monkeypatch)
    for lock in locks:
        lock.close()
    monkeypatch.setattr(queue, "spawn", lambda *a: owned(8181))
    assert queue.step() is False
    assert len(queue.servers) == queue.server_limit() == 1
    assert [gpu.index for gpu in queue.servers[0].pair] == [7]
    assert queue.state["servers"]["server-001"]["status"] == "starting"
    assert all(record["status"] == "pending" for record in queue.state["jobs"].values())
    assert queue.close_server(queue.servers[0]) == []


def test_tp1_wrong_pair_cannot_start_or_probe(tp1_queue, monkeypatch):
    queue = tp1_queue
    pair, locks = configure_fake_server_launch(queue, monkeypatch)
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("wrong count spawned"))
    with pytest.raises(q.QueueError, match="exactly one reserved GPU"):
        queue.start_server(pair, locks)
    assert all(lock.fd is None for lock in locks) and not queue.servers
    server = prepare_real_attestation(queue, monkeypatch)
    calls = probe_wire(monkeypatch)
    with pytest.raises(q.QueueError, match="device count/command"):
        queue.attest(server)
    assert not calls and not server.manifest_path.exists()


def test_tp1_worker_must_belong_to_owned_server_and_foreign_is_never_signaled(tp1_queue, monkeypatch):
    queue = tp1_queue
    server = mock_server(queue, pair=[inventory()[7]])
    server.command = q.server_command(queue.plan, server.port)
    monkeypatch.setattr(q, "session_members", lambda root: [root, identity(root["pid"] + 1)])
    gpus = inventory({7})
    with pytest.raises(q.QueueError, match="No live owned"):
        queue.workers_healthy(server, gpus, require_workers=True)
    gpus[7] = replace(gpus[7], pids=frozenset({101}))
    queue.workers_healthy(server, gpus, require_workers=True)
    gpus[7] = replace(gpus[7], pids=frozenset({101, 9999}))
    monkeypatch.setattr(q, "signal_verified", lambda *a: pytest.fail("Foreign signal attempted"))
    with pytest.raises(q.QueueError, match="Foreign compute"):
        queue.workers_healthy(server, gpus, require_workers=True)


def test_tp1_all_five_jobs_require_both_probes_and_profile_bound_receipts(tp1_queue, monkeypatch):
    queue = tp1_queue
    server, spawned = ready_propensity_sequential(queue, monkeypatch)
    manifest = json.loads(server.manifest_path.read_text())
    q.validate_selected_runtime(queue.plan, queue.provenance, manifest)
    assert manifest["gpu_ids"] == [7] and manifest["gpu_uuids"] == [inventory()[7].uuid]
    assert len(manifest["gpus"]) == 1 and manifest["runtime_profile"] == q.TP1_PROFILE_NAME
    assert len(server.funding_probe["steps"]) == 25
    assert queue.server_limit() == 1
    with pytest.raises(q.QueueError, match="not runnable"):
        queue.start_job(server, queue.plan.jobs[1])
    for index, job in enumerate(queue.plan.jobs):
        queue.start_job(server, job)
        write_propensity_sequential_artifacts(queue, job)
        queue.finish_job(server, 0)
        receipt = json.loads((queue.plan.directory / f"job-{job['id']}.receipt.json").read_text())
        assert all(receipt[key] == value for key, value in q.selected_runtime_binding(queue.plan).items())
        if index == 0:
            assert queue.server_limit() == 4 and len(queue.runnable()) == 4
    assert len(spawned) == 5 and all(env["CUDA_VISIBLE_DEVICES"] == "7" for _, env, _ in spawned)
    result = q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=True)
    assert result["completed_jobs"] == 5 and result["completed_scientific_worlds"] == 4
    assert result["server_manifests"] == 1
    queue.close_server(server)
    queue.save()
    assert q.load_state(queue.plan, queue.provenance, True)["jobs"] == queue.state["jobs"]


@pytest.mark.parametrize("tamper", ["tp2", "gpu_count", "gpu_uuid", "command", "profile", "runtime_hash"])
def test_tp1_manifest_tampering_blocks_job_and_final_validation(tp1_queue, monkeypatch, tamper):
    queue = tp1_queue
    server, spawned = ready_propensity_sequential(queue, monkeypatch)
    manifest = json.loads(server.manifest_path.read_text())
    if tamper == "tp2":
        manifest["tensor_parallel_size"] = 2
    elif tamper == "gpu_count":
        manifest["gpu_ids"] = [7, 0]
    elif tamper == "gpu_uuid":
        manifest["gpu_uuids"] = ["GPU-other"]
    elif tamper == "command":
        manifest["command"] = manifest["launch_command"] = ["mock-vllm"]
    elif tamper == "profile":
        manifest["server_profile"] = "tp2"
    else:
        manifest["runtime_hash"] = "0" * 64
    server.manifest_path.chmod(0o600)
    server.manifest_path.write_bytes(q.json_bytes(manifest))
    # Recompute the byte receipt to isolate semantic profile validation.
    queue.state["servers"][server.name]["manifest_sha256"] = q.file_hash(server.manifest_path)
    with pytest.raises(q.QueueError):
        queue.start_job(server, queue.plan.jobs[0])
    assert not spawned
    with pytest.raises(q.QueueError):
        q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=False)


def test_tp1_failed_sequential_probe_cannot_authorize_initializer(tp1_queue, monkeypatch):
    queue = tp1_queue
    server = prepare_real_attestation(queue, monkeypatch)
    server.pair = [inventory()[7]]
    server.command = q.server_command(queue.plan, server.port)
    def fail(payload):
        if "next_application_id" in payload["choices"][0]["message"]["content"]:
            payload["choices"][0]["finish_reason"] = "length"
    calls = probe_wire(monkeypatch, mutate=fail)
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("failed probe spawned"))
    with pytest.raises(q.ProbeError):
        queue.attest(server)
    assert len(calls) == 2 and not server.ready
    assert not server.manifest_path.exists()
    assert all(row["status"] == "pending" for row in queue.state["jobs"].values())


def test_tp1_resume_rejects_changed_frozen_profile_before_any_process_work(tp1_queue, monkeypatch):
    queue = tp1_queue
    changed = copy.deepcopy(queue.provenance)
    changed["selected_server_spec"]["max_num_seqs"] = 128
    monkeypatch.setattr(q, "session_members", lambda *a: pytest.fail("checked processes before profile"))
    with pytest.raises(q.QueueError, match="frozen provenance"):
        q.load_state(queue.plan, changed, True)
    queue.state["provenance"]["runtime_hash"] = "0" * 64
    queue.save()
    with pytest.raises(q.QueueError, match="frozen provenance"):
        q.load_state(queue.plan, q.selected_runtime_binding(queue.plan), True)


def test_propensity_v3_requires_both_live_proofs_then_initializer_then_four_worlds(
        propensity_sequential_queue, monkeypatch):
    queue = propensity_sequential_queue
    server, spawned = ready_propensity_sequential(queue, monkeypatch)
    assert queue.server_limit() == 1 and queue.runnable() == queue.plan.jobs[:1]
    with pytest.raises(q.QueueError, match="not runnable"):
        queue.start_job(server, queue.plan.jobs[1])
    assert not spawned
    for index, job in enumerate(queue.plan.jobs):
        queue.start_job(server, job)
        write_propensity_sequential_artifacts(queue, job)
        queue.finish_job(server, 0)
        if index == 0:
            assert queue.server_limit() == 4 and len(queue.runnable()) == 4
    result = q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=True)
    assert result["completed_jobs"] == 5 and result["completed_scientific_worlds"] == 4
    assert not result["initialization_is_scientific_world"]
    assert len(spawned) == 5
    assert {kind for kind, _ in queue.admission_calls} == {"initialization_only", "condition"}
    receipt = json.loads((queue.plan.directory / f"job-{queue.plan.jobs[1]['id']}.receipt.json").read_text())
    assert receipt["world_input_binding"]["initialization_manifest_sha256"] == (
        result["initialization_dependency"]["initialization_manifest_sha256"])
    assert receipt["propensity_funding_admission"]["validation"]["scientific_years"] == 10


@pytest.mark.parametrize("change", ["no_protocol", "v2_id", "empty_seq", "wrong_years", "missing_md", "false_gate"])
def test_propensity_v3_plan_cannot_weaken_sequential_or_fresh_scope(propensity_sequential_plan, change):
    spec = copy.deepcopy(propensity_sequential_plan.spec)
    if change == "no_protocol":
        spec.pop("funding_selection_protocol")
    elif change == "v2_id":
        spec["jobs"][1]["id"] = spec["jobs"][1]["id"].replace("_v3_", "_v2_")
    elif change == "empty_seq":
        spec["jobs"][0]["artifacts"][-2]["min_bytes"] = 0
    elif change == "wrong_years":
        spec["jobs"][0]["artifacts"][-1]["equals"]["/years_completed"] = 10
    elif change == "missing_md":
        spec["jobs"][0]["artifacts"][0]["equals"]["/inputs/source_files_sha256"].pop(
            "docs/switching_propensity_protocol.md")
    else:
        spec["jobs"][0]["artifacts"][0]["equals"]["/sequential_audit_valid"] = False
    with pytest.raises(q.QueueError):
        q.parse_plan(q.json_bytes(spec))


def test_propensity_v3_failing_sequential_probe_cannot_create_initializer(
        propensity_sequential_queue, monkeypatch):
    queue = propensity_sequential_queue
    server = prepare_real_attestation(queue, monkeypatch)
    def fail_selection(payload):
        if "next_application_id" in payload["choices"][0]["message"]["content"]:
            payload["choices"][0]["finish_reason"] = "length"
    calls = probe_wire(monkeypatch, mutate=fail_selection)
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("failed probe launched a job"))
    with pytest.raises(q.ProbeError):
        queue.attest(server)
    assert len(calls) == 2 and not server.ready
    assert not Path(queue.plan.initialization["artifact"]).exists()
    assert all(row["status"] == "pending" for row in queue.state["jobs"].values())


@pytest.mark.parametrize("field", ["initialization_manifest_sha256", "initialization_manifest_path"])
def test_propensity_v3_world_cannot_adopt_another_initializer_manifest(
        propensity_sequential_queue, monkeypatch, field):
    queue = propensity_sequential_queue
    server, _ = ready_propensity_sequential(queue, monkeypatch)
    initial, world = queue.plan.jobs[:2]
    queue.start_job(server, initial)
    write_propensity_sequential_artifacts(queue, initial)
    queue.finish_job(server, 0)
    queue.start_job(server, world)
    wrong = "d" * 64 if field.endswith("sha256") else str(
        queue.plan.private_root / "outputs/docs/old-v2/initialization_manifest.json")
    write_propensity_sequential_artifacts(queue, world, **{field: wrong})
    with pytest.raises(q.QueueError, match="another initialization manifest"):
        queue.finish_job(server, 0)
    assert not (queue.plan.directory / f"job-{world['id']}.receipt.json").exists()


def test_propensity_v3_readonly_admission_failure_blocks_cache_receipt(
        propensity_sequential_queue, monkeypatch):
    queue = propensity_sequential_queue
    server, _ = ready_propensity_sequential(queue, monkeypatch)
    job = queue.plan.jobs[0]
    queue.start_job(server, job)
    write_propensity_sequential_artifacts(queue, job)
    @q.contextlib.contextmanager
    def rejected(_):
        def validate(*args, **kwargs):
            raise ValueError("invalid restored sequential/request audit")
        yield SimpleNamespace(source_identity=lambda: queue.provenance["sources"],
                              validate_funding_evidence=validate)
    monkeypatch.setattr(q, "propensity_admission_driver", rejected)
    with pytest.raises(ValueError, match="invalid restored"):
        queue.finish_job(server, 0)
    assert not queue.runnable()
    assert not (queue.plan.directory / f"job-{job['id']}.receipt.json").exists()


def test_propensity_v3_retains_foreign_pair_locks_without_probing_or_adoption(
        propensity_sequential_queue, monkeypatch):
    queue = propensity_sequential_queue
    monkeypatch.setattr(q, "query_gpus", inventory)
    held = [q.reserve_pair(inventory(), queue.lock_root, query=inventory) for _ in range(4)]
    monkeypatch.setattr(queue, "spawn", lambda *a: pytest.fail("must not use another queue's server"))
    monkeypatch.setattr(queue, "attest", lambda *a: pytest.fail("must not probe another server"))
    try:
        assert queue.step() is False
        assert queue.state["status"] == "waiting_for_idle_pair" and not queue.servers
    finally:
        for _, locks in held:
            for lock in locks:
                lock.close()


def test_propensity_v3_hashes_both_protocol_documents(propensity_sequential_plan):
    sources = q.source_hashes(propensity_sequential_plan)
    assert all(name in sources for name in q.PROPENSITY_SOURCE_FILES)


def test_propensity_v3_freezes_executable_data_modules_for_exact_driver_source_map(propensity_sequential_plan):
    path = propensity_sequential_plan.worktree / "utopia/data/tracker.py"
    path.parent.mkdir(parents=True)
    path.write_text("native_tracker_version = 1\n")
    before = q.source_hashes(propensity_sequential_plan)
    assert before["utopia/data/tracker.py"] == q.file_hash(path)
    path.write_text("native_tracker_version = 2\n")
    assert q.source_hashes(propensity_sequential_plan)["utopia/data/tracker.py"] != before["utopia/data/tracker.py"]


def test_propensity_v3_completed_initializer_admission_rechecked_before_world_and_resume(
        propensity_sequential_queue, monkeypatch):
    queue = propensity_sequential_queue
    server, spawned = ready_propensity_sequential(queue, monkeypatch)
    initial = queue.plan.jobs[0]
    queue.start_job(server, initial)
    write_propensity_sequential_artifacts(queue, initial)
    queue.finish_job(server, 0)
    @q.contextlib.contextmanager
    def rejected(_):
        def validate(*args, **kwargs):
            raise ValueError("initializer admission no longer valid")
        yield SimpleNamespace(source_identity=lambda: queue.provenance["sources"],
                              validate_funding_evidence=validate)
    monkeypatch.setattr(q, "propensity_admission_driver", rejected)
    with pytest.raises(ValueError, match="no longer valid"):
        queue.start_job(server, queue.plan.jobs[1])
    assert len(spawned) == 1
    queue.close_server(server)
    queue.save()
    with pytest.raises(ValueError, match="no longer valid"):
        q.load_state(queue.plan, queue.provenance, True)


def test_propensity_v3_dynamic_native_ledger_bytes_are_receipted_and_rechecked(
        propensity_sequential_queue, monkeypatch):
    queue = propensity_sequential_queue
    server, _ = ready_propensity_sequential(queue, monkeypatch)
    initial, world = queue.plan.jobs[:2]
    queue.start_job(server, initial)
    write_propensity_sequential_artifacts(queue, initial)
    queue.finish_job(server, 0)
    queue.start_job(server, world)
    write_propensity_sequential_artifacts(queue, world)
    ledger = queue.plan.private_root / "outputs/checkpoints" / world["id"] / "funding_applications_year_1.jsonl"
    ledger.write_text('{"synthetic_controller_ledger":1}\n')
    queue.finish_job(server, 0)
    receipt = json.loads((queue.plan.directory / f"job-{world['id']}.receipt.json").read_text())
    proof = receipt["propensity_funding_admission"]["native_ledger_files"]
    assert len(proof) == 1 and proof[0]["sha256"] == q.file_hash(ledger)
    ledger.write_text('{"synthetic_controller_ledger":2}\n')
    with pytest.raises(q.QueueError, match="admission changed"):
        q.validate_completed(queue.plan, queue.state, queue.provenance, require_all=False)


def test_propensity_v3_validator_cannot_substitute_unfrozen_expected_sources(
        propensity_sequential_queue, monkeypatch):
    queue = propensity_sequential_queue
    server, _ = ready_propensity_sequential(queue, monkeypatch)
    initial = queue.plan.jobs[0]
    queue.start_job(server, initial)
    write_propensity_sequential_artifacts(queue, initial)
    changed = {**queue.provenance["sources"], "utopia/funding/sequential.py": "f" * 64}
    @q.contextlib.contextmanager
    def rejected(_):
        yield SimpleNamespace(source_identity=lambda: changed,
                              validate_funding_evidence=lambda *a, **k: pytest.fail("unfrozen validator called"))
    monkeypatch.setattr(q, "propensity_admission_driver", rejected)
    with pytest.raises(q.QueueError, match="source map differs"):
        queue.finish_job(server, 0)
    assert not (queue.plan.directory / f"job-{initial['id']}.receipt.json").exists()


def test_continuation_cannot_also_declare_propensity_initialization(propensity_specification):
    spec = copy.deepcopy(propensity_specification)
    assert q.parse_plan(q.json_bytes(spec)).initialization is not None
    spec["continuation"] = {
        "schema_version": 1, "kind": q.PENDING_KIND,
        **{name: {"path": str(Path(spec["private_root"]) / q.PENDING_RECORD_PATHS[name]),
                  "sha256": checksum}
           for name, checksum in q.PENDING_PARENT_SHA256.items()},
    }
    with pytest.raises(q.QueueError, match="Pending continuation cannot include propensity initialization"):
        q.parse_plan(q.json_bytes(spec))


def test_pending_pairs_and_single_gpu_profiles_share_locks(queue):
    gpus = inventory({3, 5, 6})
    held = []
    try:
        pair, locks = q.reserve_pair(
            gpus, queue.lock_root, query=lambda: gpus, prefer_pending_pair=True)
        held.extend(locks)
        assert [gpu.index for gpu in pair] == [5, 6]
        single, locks = q.reserve_pair(
            gpus, queue.lock_root, query=lambda: gpus, device_count=1)
        held.extend(locks)
        assert [gpu.index for gpu in single] == [3]
        assert q.reserve_pair(gpus, queue.lock_root, query=lambda: gpus, device_count=1) is None
        assert q.reserve_pair(
            gpus, queue.lock_root, query=lambda: gpus, prefer_pending_pair=True) is None
    finally:
        for lock in held:
            lock.close()
