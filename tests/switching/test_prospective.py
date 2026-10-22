"""Actual imported simulator/checkpoint methods, with CPU-only scripted transport."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from openai.types.chat import ChatCompletion

import utopia.experiments.switching_propensity as study
import utopia.experiments.funding_feedback as shared
import utopia.simulation as native
import utopia.agents.research_direction as directions
import utopia.metrics.embedding_tracker as embeddings
from utopia.agents.conference import ConferenceSystem, select_conferences_for_simulation
from utopia.config import SIMULATION_CONFIG
from utopia.data.paper_tracker import ArchivedPaper
from utopia.models.models import VLLMServerModel
from utopia.models.response_cache import CompletedResponseCache
from utopia.runtime.checkpointing import PortableCheckpointError, PortableStop, pack_state, unpack_state


def options(root, cell="INIT", initial=None, pilot=True):
    return SimpleNamespace(
        cell=cell, seed=42, founders=5 if pilot else 5000, years=2 if pilot else 6,
        run_id="cpu_switching", vllm_url="http://127.0.0.1:1/v1",
        server_manifest="unused", fulltext_index="unused", deadline_epoch=9999999999.,
        batch_size=1, model="scripted", model_revision="cpu-fixture",
        context_tokens=131072, max_tokens=8192, resume=False, pilot=pilot, s3_prefix=None,
        initialize_only=cell == "INIT", initial_choices=initial or root / "common.json")


class VectorModel:
    def encode(self, texts, **kwargs):
        return np.stack([np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8)
                         .astype(np.float32) + 1 for text in texts])


class MemoryRAG:
    def __init__(self, **kwargs):
        self.start_year = kwargs.get("start_year", 2016)
        self.current_sim_year = 0
        self.document_status = pd.DataFrame(columns=["id", "status", "publication_year"])
        self.fulltext_corpus = None

    def to_dict(self):
        return dict(start_year=self.start_year, current_sim_year=self.current_sim_year,
                    document_status=self.document_status.to_dict("records"))


class RecordingAudit:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def before(self, payload, *, seed_ctx, item_index, attempt):
        ticket = dict(seed_ctx=list(seed_ctx), item_index=item_index, attempt=attempt,
                      request_seed=payload["extra_body"]["seed"])
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(event="request_started", **ticket)) + "\n")
        return ticket

    def response(self, ticket, response):
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(event="response", **ticket)) + "\n")

    def transport_error(self, ticket, error):
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(event="error", error=str(error), **ticket)) + "\n")

    def create(self, callback, *, seed_ctx, item_index, attempt, **payload):
        ticket = self.before(payload, seed_ctx=seed_ctx, item_index=item_index, attempt=attempt)
        response = callback(**payload)
        self.response(ticket, response)
        return response


def scripted_model(root, binding, calls):
    model = VLLMServerModel.__new__(VLLMServerModel)
    model.model_name, model.model_revision, model.run_seed = "scripted", "cpu-fixture", 42
    model.enable_thinking = True
    model.max_tokens_override, model.max_concurrent_requests = 8192, 1
    model.call_stats = dict(n_prompts=0, n_first_attempt_success=0, n_retries=0, n_failures=0,
                            elapsed_seconds=0., prompt_tokens=0, completion_tokens=0)
    model.response_cache = CompletedResponseCache(root / "responses.sqlite", binding=binding)
    model.request_audit = RecordingAudit(root / "audit.jsonl")

    def create(**payload):
        calls.append(deepcopy(payload))
        topics = re.findall(r"### Direction \d+: ([^\n]+)", payload["messages"][-1]["content"])
        assert topics
        response = {"topic": topics[0], "detailed_focus": "Scripted CPU project.",
                    "reason": "Explicit fixture choice."}
        return ChatCompletion.model_validate(dict(
            id="cpu", object="chat.completion", created=0, model="scripted",
            choices=[dict(index=0, message=dict(role="assistant", content=json.dumps(response)),
                          finish_reason="stop")],
            usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8)))
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return model


@pytest.fixture(autouse=True)
def cpu_boundaries(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setitem(SIMULATION_CONFIG["wandb"], "enabled", False)
    monkeypatch.setattr(native, "RAG", MemoryRAG)
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda *a, **k: VectorModel())
    monkeypatch.setattr(study, "resolve_embedding_snapshot", lambda: (tmp_path / "fake-embedding", {}))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def make_sim(root, opts, monkeypatch, calls, *, binding_override=None):
    monkeypatch.setattr(shared, "ROOT", root)
    args = shared.configure_prospective(opts)
    extra = study.configure_prospective_switching(args, opts)
    binding = dict(protocol=study.PROSPECTIVE_PROTOCOL, cell=opts.cell, seed=42,
                   model="scripted", model_revision="cpu-fixture", corpus_sha256="scripted-corpus",
                   science_config={"config": deepcopy(SIMULATION_CONFIG),
                                   "switching": study.prospective_spec(opts), "extra": extra})
    if binding_override:
        binding.update(binding_override)
    args.checkpoint_binding = binding
    model = scripted_model(root / "model", binding, calls)
    sim = study.prospective_simulation_class(native.Simulation)(
        llm=model, args=args, propensity_options=opts,
        mechanisms=study.feedback.Mechanisms(), num_years=opts.years,
        output_dir=args.output_dir, funding_allocation_mode="fixed",
        experiment_name="default_prospective", start_year=2016)
    return sim


def initialize(sim):
    sim.initialize_agents()
    founders = [a for a in sim.ecosystem.agent_population.values() if a.get_type() == "university"]
    sim.conference_system = ConferenceSystem(deepcopy(select_conferences_for_simulation(
        agents=founders, num_conferences=6)))
    # Match the native run() initialization before its first durable checkpoint.
    for agent in sim.ecosystem.agent_population.values():
        if getattr(agent, "resources", None) is not None:
            sim.agent_tracker.record_agent_resources(agent.id, agent.resources, 0)
    sim._save_checkpoint(0, phase=0)


def prepare_common(root, monkeypatch):
    opts = options(root)
    calls = []
    with study.one_year_context(native, directions):
        sim = make_sim(root, opts, monkeypatch, calls)
        with pytest.raises(PortableStop, match="prospective_initial_choices_complete"):
            sim.run()
        assert len(calls) == opts.founders
        assert not sim.yearly_results and not sim.funding_feedback_years
        assert not sim.paper_tracker.papers_by_id
        common = json.loads(opts.initial_choices.read_text())
        assert common["scientific_years_completed"] == 0
        assert common["status"] == "initial_choices_ready"
        sim.llm.response_cache.close()
    return opts.initial_choices, common


def test_bound_formal_design_and_unchanged_historical_defaults(tmp_path):
    for cell, probability in (("LN", .25), ("HN", .75)):
        opts = options(tmp_path, cell, pilot=False)
        spec = study.prospective_spec(opts)
        assert (spec["founders"], spec["years"], spec["switch_probability"]) == (5000, 6, probability)
        assert spec["distance_policy"] == "near-history" and spec["resubmissions"] == "native_enabled"
    opts.years = 10
    with pytest.raises(BaseException, match="formal_switching_requires"):
        study.prospective_spec(opts)
    assert (study.FOUNDERS, study.YEARS, study.CELLS) == (1200, 10, ("LN", "LF", "HN", "HF"))


def test_fresh_common_initializer_replays_both_cells_without_requests(tmp_path, monkeypatch):
    path, common = prepare_common(tmp_path / "init", monkeypatch)
    for cell in study.PROSPECTIVE_CELLS:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        opts = options(tmp_path / cell, cell, path)
        calls = []
        with study.one_year_context(native, directions):
            sim = make_sim(tmp_path / cell, opts, monkeypatch, calls)
            initialize(sim)
            sim._setup_year(1)
            sim._run_phase_1_research_directions(1, {})
            assert not calls
            assert sim.initial_state_hash == common["initial_state_hash"]
            assert sim.initial_records == common["records"]
            assert all(e["adjacent_direction_distance"] is None for e in sim.direction_events)
            sim.llm.response_cache.close()


def test_changed_initial_response_or_corpus_is_rejected(tmp_path, monkeypatch):
    path, common = prepare_common(tmp_path / "init", monkeypatch)
    opts = options(tmp_path / "LN", "LN", path)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    with study.one_year_context(native, directions), pytest.raises(BaseException, match="binding_mismatch"):
        make_sim(tmp_path / "LN", opts, monkeypatch, [], binding_override={"corpus_sha256": "different"})
    common["records"][0]["raw_response"]["topic"] = "tampered"
    path.write_text(json.dumps(common))
    with study.one_year_context(native, directions), pytest.raises(BaseException, match="record_hash_mismatch"):
        make_sim(tmp_path / "HN", options(tmp_path / "HN", "HN", path), monkeypatch, [])


def test_distances_are_distinct_and_invalid_response_is_fatal():
    sim = study.ProspectivePropensityMixin.__new__(study.ProspectivePropensityMixin)
    sim.direction_lookup = {k: SimpleNamespace(topic=k, keywords=[]) for k in ("old", "new")}
    sim.embedding_tracker = SimpleNamespace(_encode_cached=lambda texts: np.array([[1., 0.], [0., 1.]]))
    sim.neutral_history = study.NeutralHistory
    event = dict(fallback=False, conditional_switch_distance=.2, is_initial_choice=False,
                 previous_topic="old", chosen_topic="new")
    observed = sim.direction_observation(event)
    assert observed["adjacent_direction_distance"] == 1.0
    assert observed["distance_to_history"] == .2
    observed["chosen_topic"] = "old"
    observed["conditional_switch_distance"] = None
    assert sim.direction_observation(observed)["adjacent_direction_distance"] == 0.0
    observed["fallback"] = True
    with pytest.raises(PortableCheckpointError, match="invalid direction"):
        sim.direction_observation(observed)


def test_all_subclass_state_survives_actual_checkpoint_relocation(tmp_path, monkeypatch):
    path, _ = prepare_common(tmp_path / "init", monkeypatch)
    old = tmp_path / "old"
    opts = options(old, "LN", path)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    with study.one_year_context(native, directions):
        sim = make_sim(old, opts, monkeypatch, [])
        initialize(sim)
        sim._setup_year(1)
        sim._run_phase_1_research_directions(1, {})
        author = sim.founder_ids[0]
        paper = ArchivedPaper("cpu-paper", "CPU paper", "CPU abstract", author, "university",
                              1, sim.conference_system.conferences[0].conference_id, "reject")
        sim.paper_tracker.papers_by_id[paper.id] = paper
        sim.paper_tracker.papers_by_author[author] = [paper]
        sim.neutral_history.ingest(sim.paper_tracker.papers_by_id, 1)
        sim.funding_feedback_years[1] = {"year": 1, "fixture_only": True}
        sim.yearly_results = [{"year": 1, "fixture_only": True}]
        for agent in sim.ecosystem.get_available_authors():
            sim.agent_tracker.record_agent_resources(agent.id, agent.resources, 1)
        sim._save_checkpoint(1, phase=5)
        expected = pack_state(sim.extra_checkpoint_state())
        assert unpack_state(expected)["prospective_switching"]["registry"]
        sim.llm.response_cache.close()
    new = tmp_path / "new"
    shutil.copytree(old, new)
    copied_initial = new / "relocated-common.json"
    shutil.copyfile(path, copied_initial)
    resumed_options = options(new, "LN", copied_initial)
    resumed_options.resume = True
    with study.one_year_context(native, directions):
        resumed = make_sim(new, resumed_options, monkeypatch, [])
        assert resumed.load_checkpoint(1)
        assert pack_state(resumed.extra_checkpoint_state()) == expected
        assert resumed.neutral_history.founders[author] is resumed.ecosystem.agent_population[author]
        assert resumed.args.output_dir.startswith(str(new))
        saved = resumed.extra_checkpoint_state()
        saved["prospective_switching"]["specification"]["cell"] = "HN"
        with pytest.raises(PortableCheckpointError, match="policy differs"):
            resumed.restore_extra_checkpoint_state(saved)
        resumed.llm.response_cache.close()


def test_near_history_index_matches_native_centroids_and_repeat_policy(tmp_path, monkeypatch):
    path, _ = prepare_common(tmp_path / "init", monkeypatch)
    observations = {}
    for cell in study.PROSPECTIVE_CELLS:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        root = tmp_path / cell
        with study.one_year_context(native, directions):
            calls = []
            sim = make_sim(root, options(root, cell, path), monkeypatch, calls)
            initialize(sim)
            sim._setup_year(1)
            sim._run_phase_1_research_directions(1, {})
            author = sim.founder_ids[0]
            paper = ArchivedPaper("prior", "Title", "Prior work", author, "university", 1,
                                  sim.conference_system.conferences[0].conference_id, "reject")
            sim.paper_tracker.papers_by_id[paper.id] = paper
            sim.neutral_history.ingest(sim.paper_tracker.papers_by_id, 1)
            sim.neutral_history.prepare_year(2)
            complete_metadata = sim.embedding_tracker.paper_metadata
            for agent_id in sim.founder_ids:
                expected, expected_info = study.NeutralHistory.reference(
                    sim.neutral_history, agent_id, 2, sim.paper_tracker.papers_by_id)
                actual, actual_info = sim.neutral_history.reference(
                    agent_id, 2, sim.paper_tracker.papers_by_id)
                np.testing.assert_array_equal(actual, expected)
                assert actual_info == expected_info
                assert sim.embedding_tracker.paper_metadata is complete_metadata
            sim._setup_year(2)
            sim._run_phase_1_research_directions(2, {})
            rows = [e for e in sim.direction_events if e["year"] == 2]
            assert len(calls) == len(rows) == 5
            for row in rows:
                assert row["project_start_year"] == row["project_end_year"] == 2
                assert row["adjacent_direction_distance"] is not None
                assert row["realized_switch"] == (row["previous_topic"] != row["chosen_topic"])
                assert len(row["candidate_topics"]) == (17 if row["realized_switch"] else 1)
                if row["realized_switch"]:
                    assert row["previous_topic"] not in row["candidate_topics"]
                    assert row["distance_to_history"] is not None
            observations[cell] = {row["agent_id"]: row for row in rows}
            sim.llm.response_cache.close()
    for agent_id, low in observations["LN"].items():
        high = observations["HN"][agent_id]
        assert low["u"] == high["u"]
        assert low["realized_switch"] == (low["u"] < .25)
        assert high["realized_switch"] == (high["u"] < .75)


def test_common_file_relocation_is_bound_by_contents_only(tmp_path, monkeypatch):
    path, _ = prepare_common(tmp_path / "init", monkeypatch)
    other = tmp_path / "another-machine" / "choices.json"
    other.parent.mkdir()
    shutil.copyfile(path, other)
    bindings = []
    def inspect_runner(opts, *, configure, simulation_factory, run_context):
        args = SimpleNamespace(num_years=opts.years, start_year=2016, papers_per_project=1,
                               annual_influx_rate=0., disable_resubmission=False, initial_funding=100,
                               docs_dir=str(tmp_path / "runtime-metrics"))
        before = vars(args).copy()
        bindings.append(configure(args))
        assert {k: v for k, v in vars(args).items() if k != "checkpoint_replay_paths"} == before
        assert "switching/direction_events.jsonl" in args.checkpoint_replay_paths
        assert not any(isinstance(value, Path) for value in bindings[-1].values())
        assert isinstance(simulation_factory(native.Simulation).func, type)
        original = native.AVAILABLE_DIRECTIONS
        with run_context():
            assert all(d.years == 1 for d in native.AVAILABLE_DIRECTIONS)
        assert native.AVAILABLE_DIRECTIONS is original
        return bindings[-1]
    monkeypatch.setattr(shared, "run_prospective", inspect_runner)
    study.run_prospective(options(tmp_path / "first", "LN", path))
    study.run_prospective(options(tmp_path / "second", "LN", other))
    assert bindings[0] == bindings[1]
    assert bindings[0]["initial_choices_sha256"] == study.file_sha256(path)


def test_initializer_year_zero_resume_keeps_common_choices_and_uses_cached_replies(tmp_path, monkeypatch):
    root = tmp_path / "init"
    opts = options(root)
    calls = []
    with study.one_year_context(native, directions):
        sim = make_sim(root, opts, monkeypatch, calls)
        initialize(sim)
        sim._setup_year(1)
        with pytest.raises(PortableStop, match="prospective_initial_choices_complete"):
            sim._run_phase_1_research_directions(1, {})
        initial_bytes = opts.initial_choices.read_bytes()
        assert len(calls) == 5
        sim.llm.response_cache.close()
        opts.resume = True
        resumed_calls = []
        resumed = make_sim(root, opts, monkeypatch, resumed_calls)
        assert resumed.load_checkpoint(0)
        resumed._setup_year(1)
        with pytest.raises(PortableStop, match="prospective_initial_choices_complete"):
            resumed._run_phase_1_research_directions(1, {})
        assert not resumed_calls
        assert opts.initial_choices.read_bytes() == initial_bytes
        assert not resumed.funding_feedback_years and not resumed.yearly_results
        resumed.llm.response_cache.close()

class FakePublicationS3:
    """S3 object/checksum semantics at the CLI boundary; no network or credentials."""
    def __init__(self):
        self.objects, self.calls = {}, []
        self.fail_on = None
        self.before_put = lambda key: None

    def run(self, command, **kwargs):
        import base64
        import subprocess
        assert command[:2] == ["aws", "s3api"]
        assert "--profile" not in command
        assert kwargs["check"] and kwargs["env"]["AWS_MAX_ATTEMPTS"] == "1"
        action = command[2]
        bucket = command[command.index("--bucket") + 1]
        key = command[command.index("--key") + 1]
        self.calls.append((action, bucket, key))
        if key.endswith(self.fail_on or "not-a-real-suffix"):
            raise subprocess.CalledProcessError(1, command, stderr="injected transfer failure")
        identity = (bucket, key)
        if action == "put-object":
            self.before_put(key)
            if "--if-none-match" in command and identity in self.objects:
                raise subprocess.CalledProcessError(1, command, stderr="PreconditionFailed (412)")
            data = Path(command[command.index("--body") + 1]).read_bytes()
            checksum = base64.b64encode(hashlib.sha256(data).digest()).decode()
            assert command[command.index("--checksum-sha256") + 1] == checksum
            self.objects[identity] = data
        else:
            assert action == "head-object"
            data = self.objects[identity]
            checksum = base64.b64encode(hashlib.sha256(data).digest()).decode()
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "ChecksumSHA256": checksum, "ContentLength": len(data)}), "")


def publication_ready(root, monkeypatch):
    """Real native initializer plus the parent runner's stopped-manifest handoff."""
    monkeypatch.setattr(study, "ROOT", root)
    opts = options(root)
    opts.s3_prefix = "s3://cpu-backups/study"
    with study.one_year_context(native, directions):
        sim = make_sim(root, opts, monkeypatch, [])
        with pytest.raises(PortableStop, match="prospective_initial_choices_complete"):
            sim.run()
        binding = deepcopy(sim.args.checkpoint_binding)
        sim.llm.response_cache.close()
    docs = root / "outputs/docs" / f"{opts.run_id}_INIT"
    manifest = docs / "experiment.json"
    study.provenance.write_json(manifest, {
        "status": "paused", "reason": "prospective_initial_choices_complete",
        "years_completed": 0, "binding": binding})
    checkpoint = root / "outputs/checkpoints" / f"{opts.run_id}_INIT" / "portable_checkpoint_year_0.json.gz"
    return opts, manifest, checkpoint


def test_publication_completes_only_after_content_and_pointer_upload(tmp_path, monkeypatch):
    import subprocess
    opts, manifest_path, checkpoint = publication_ready(tmp_path, monkeypatch)
    payload, checkpoint_bytes = opts.initial_choices.read_bytes(), checkpoint.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    s3 = FakePublicationS3()
    def check_pending(key):
        assert json.loads(manifest_path.read_text())["status"] == "paused"
        if key.endswith("/initial_choices.json"):
            assert s3.objects[("cpu-backups", f"study/shared/initial_choices_{digest}.json")] == payload
    s3.before_put = check_pending
    monkeypatch.setattr(subprocess, "run", s3.run)
    result = study.publish_prospective_initial_choices(opts)
    assert result["status"] == "initialization_only_complete"
    assert result["initial_choices_uri"] == f"s3://cpu-backups/study/shared/initial_choices_{digest}.json"
    pointer = json.loads(s3.objects[("cpu-backups", "study/shared/initial_choices.json")])
    assert pointer["initial_choices_sha256"] == digest
    assert pointer["scientific_years_completed"] == 0
    assert json.loads(manifest_path.read_text())["status"] == "initialization_only_complete"
    assert opts.initial_choices.read_bytes() == payload and checkpoint.read_bytes() == checkpoint_bytes
    # The immutable data object describes local choices readiness, not upload success.
    assert json.loads(payload)["status"] == "initial_choices_ready"


@pytest.mark.parametrize("stage", ["data", "pointer"])
def test_publication_failure_preserves_state_and_resume_never_enters_runner(tmp_path, monkeypatch, stage):
    import subprocess
    opts, manifest_path, checkpoint = publication_ready(tmp_path, monkeypatch)
    payload, checkpoint_bytes = opts.initial_choices.read_bytes(), checkpoint.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    s3 = FakePublicationS3()
    s3.fail_on = f"initial_choices_{digest}.json" if stage == "data" else "/initial_choices.json"
    monkeypatch.setattr(subprocess, "run", s3.run)
    with pytest.raises(PortableStop, match="initial_choices_publication_failed"):
        study.publish_prospective_initial_choices(opts)
    failed = json.loads(manifest_path.read_text())
    assert failed["status"] == "paused" and failed["initial_choices_sha256"] == digest
    assert opts.initial_choices.read_bytes() == payload and checkpoint.read_bytes() == checkpoint_bytes
    assert ("cpu-backups", "study/shared/initial_choices.json") not in s3.objects
    def no_runner(*args, **kwargs):
        pytest.fail("publication-only resume must not construct a model or replay phase 1")
    monkeypatch.setattr(shared, "run_prospective", no_runner)
    s3.fail_on = None
    with pytest.raises(FileExistsError, match="explicit --resume"):
        study.run_prospective(opts)
    relocated = tmp_path / "relocated-common.json"
    shutil.copyfile(opts.initial_choices, relocated)
    opts.initial_choices, opts.vllm_url, opts.resume = relocated, "http://new-machine:8000/v1", True
    result = study.run_prospective(opts)
    assert result["status"] == "initialization_only_complete"
    assert relocated.read_bytes() == payload and checkpoint.read_bytes() == checkpoint_bytes
    if stage == "pointer":
        assert any(action == "head-object" for action, _, _ in s3.calls)
    # Repeating publication is safe, too: identical immutable bytes are verified.
    assert study.run_prospective(opts) == result


def test_publication_does_not_overwrite_conflicting_immutable_content(tmp_path, monkeypatch):
    import subprocess
    opts, manifest, checkpoint = publication_ready(tmp_path, monkeypatch)
    digest = study.file_sha256(opts.initial_choices)
    key = ("cpu-backups", f"study/shared/initial_choices_{digest}.json")
    s3 = FakePublicationS3()
    s3.objects[key] = b"conflicting bytes"
    monkeypatch.setattr(subprocess, "run", s3.run)
    with pytest.raises(PortableStop, match="publication_failed"):
        study.publish_prospective_initial_choices(opts)
    assert s3.objects[key] == b"conflicting bytes"
    assert ("cpu-backups", "study/shared/initial_choices.json") not in s3.objects
    assert json.loads(manifest.read_text())["status"] == "paused" and checkpoint.exists()


@pytest.mark.parametrize("damage", ["payload", "binding", "receipt"])
def test_publication_rejects_invalid_choices_before_s3(tmp_path, monkeypatch, damage):
    import subprocess
    opts, manifest_path, _ = publication_ready(tmp_path, monkeypatch)
    record = json.loads(opts.initial_choices.read_text())
    manifest = json.loads(manifest_path.read_text())
    if damage == "payload":
        record["records"][0]["raw_response"]["topic"] = "changed"
        opts.initial_choices.write_text(json.dumps(record))
    elif damage == "binding":
        manifest["binding"]["science_config"]["extra"]["switching"]["years"] = 10
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest["initial_choices_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    s3 = FakePublicationS3()
    monkeypatch.setattr(subprocess, "run", s3.run)
    with pytest.raises(BaseException, match="initial_choices_.*mismatch"):
        study.publish_prospective_initial_choices(opts)
    assert not s3.calls


def test_initializer_stop_signal_publishes_via_new_entrypoint(tmp_path, monkeypatch):
    import subprocess
    opts, manifest, _ = publication_ready(tmp_path, monkeypatch)
    payload = opts.initial_choices.read_bytes()
    opts.initial_choices.unlink()
    calls = []
    def stopped_parent(options, **kwargs):
        calls.append(options)
        Path(options.initial_choices).write_bytes(payload)
        raise PortableStop("prospective_initial_choices_complete")
    monkeypatch.setattr(shared, "run_prospective", stopped_parent)
    s3 = FakePublicationS3()
    monkeypatch.setattr(subprocess, "run", s3.run)
    assert study.run_prospective(opts)["status"] == "initialization_only_complete"
    assert calls == [opts] and json.loads(manifest.read_text())["status"] == "initialization_only_complete"
