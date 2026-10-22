"""CPU fixtures exercise the real assignment/review methods, never inference."""
import copy
import json
import os
import random
from types import SimpleNamespace

import pytest

from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.conference import CONFERENCES, TOPIC_TO_CATEGORY_MAPPING, ConferenceSystem
from utopia.experiments import reviewer_matching as experiment
from utopia.funding.feedback import FundingFeedbackMixin, Mechanisms
from utopia.runtime.checkpointing import PortableCheckpointError, read_portable_checkpoint, write_portable_checkpoint
from utopia.simulation import Simulation


class Reviewer:
    def __init__(self, identifier, topic):
        self.id = identifier
        self.expertise = [SimpleNamespace(topic=topic)]
        self.is_active = self.can_review = True
        self.conflict_of_interest = set()
        self.received = []

    def get_review_prompt(self, paper, **kwargs):
        return f"{self.id}:{paper['id']}", {}

    def process_received_reviews(self, reviews, **kwargs):
        self.received.extend(copy.deepcopy(reviews))


class ReviewModel:
    def generate_batch(self, prompts, **kwargs):
        return [(dict(overall_score=(i % 3) * 2 + 2, justification="CPU fixture"), [])
                for i, _ in enumerate(prompts)]


@pytest.fixture
def world():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("PYTHONHASHSEED") == "42"
    conf = next(conf for conf in CONFERENCES.values()
                if conf.category in TOPIC_TO_CATEGORY_MAPPING.values() and conf.topics)
    topic = next(topic for topic, category in TOPIC_TO_CATEGORY_MAPPING.items() if category == conf.category)
    other = next(topic for topic, category in TOPIC_TO_CATEGORY_MAPPING.items() if category != conf.category)
    agents = [Reviewer("author", topic), Reviewer("coauthor", topic), Reviewer("coi", topic),
              Reviewer("match1", topic), Reviewer("match2", topic),
              Reviewer("other1", other), Reviewer("other2", other), Reviewer("other3", other)]
    agents[1].conflict_of_interest.add("coi")
    by_id = {agent.id: agent for agent in agents}
    sim = object.__new__(experiment.simulation_class(Simulation))
    sim.args = SimpleNamespace(seed=42, portable_checkpoint=True)
    sim.reviewer_capacity, sim.reviews_per_paper, sim.reviewer_matching = None, 3, "topic"
    sim.review_policy, sim.review_score_mode = "persona", "int"
    sim.experiment_name = "default_prospective"
    sim._capacity_mode_on = False
    sim.mechanisms = Mechanisms.from_cell("P1F1")
    sim.llm = ReviewModel()
    sim.funding_feedback_years = {}
    sim.initial_world = {"fixture": True}
    sim.initial_world_hash, sim.corpus_hash = "world", "corpus"
    sim.founder_ids = tuple(by_id)
    sim.yearly_results = []
    sim.ecosystem = MultiAgentEcosystem(agent_configs={})
    sim.ecosystem.agent_population = by_id
    conference = copy.deepcopy(conf)
    conference.reset_for_next_year(1)
    sim.conference_system = ConferenceSystem([conference])
    sim.paper = {"id": "paper", "author_id": ["author", "coauthor"],
                 "title": "CPU fixture", "tags": [conf.topics[0]],
                 "conference": conference.conference_id, "year": 1}
    conference.submit_paper(sim.paper, 1)
    return sim


@pytest.mark.parametrize("matching", ["random", "topic"])
def test_seeded_assignment_excludes_authors_and_coi_without_global_rng(world, matching):
    world.reviewer_matching = matching
    world.ecosystem.agent_population["other3"].is_active = False
    before = random.getstate()
    first = world._assign_reviewers_with_capacity(1, {})
    assert random.getstate() == before
    second = world._assign_reviewers_with_capacity(1, {})
    assert random.getstate() == before
    assert [r.id for r in first["paper"]] == [r.id for r in second["paper"]]
    assert len(first["paper"]) == len({r.id for r in first["paper"]}) == 3
    assert not {r.id for r in first["paper"]} & {"author", "coauthor", "coi"}
    if matching == "topic":
        assert {"match1", "match2"} <= {r.id for r in first["paper"]}


def test_topic_priority_fallback_and_zero_load_denominator(world):
    result = {}
    world._assign_reviewers_with_capacity(1, result)
    metrics = result["reviewer_matching"]
    assert metrics["n_eligible_reviewers"] == 8
    assert metrics["assigned_reviews"] == 3
    assert metrics["matched_reviews"] == 2
    assert metrics["matched_fraction"] == pytest.approx(2 / 3)
    assert metrics["fallback_reviews"] == 1
    assert metrics["fallback_fraction"] == pytest.approx(1 / 3)
    assert metrics["assignment_workload"]["zero_load_reviewers"] == 5
    assert metrics["assignment_workload"]["mean_load"] == 3 / 8
    assert metrics["assignment_workload"]["top_decile_share"] == 1 / 3
    world.ecosystem.agent_population["other1"].expertise = world.ecosystem.agent_population["match1"].expertise
    world._assign_reviewers_with_capacity(1, result)
    assert result["reviewer_matching"]["matched_reviews"] == 3
    assert result["reviewer_matching"]["fallback_reviews"] == 0


@pytest.mark.parametrize("eligible", [0, 1, 2])
def test_shortfall_is_available_eligible_or_native_zero_failure(world, eligible):
    keep = ["match1", "match2"][:eligible]
    # The native can_review roster includes inactive researchers.
    world.ecosystem.agent_population["match1"].is_active = False
    for rid, reviewer in world.ecosystem.agent_population.items():
        reviewer.can_review = rid in keep
    result = {}
    if not eligible:
        with pytest.raises(RuntimeError, match="no eligible reviewer"):
            world._assign_reviewers_with_capacity(1, result)
    else:
        chosen = world._assign_reviewers_with_capacity(1, result)["paper"]
        assert len(chosen) == eligible
        assert result["reviewer_matching"]["papers_below_target"] == 1


def test_missing_tags_are_unmatched_not_missing_observations(world):
    world.paper["tags"] = ["unmapped-test-tag"]
    result = {}
    world._assign_reviewers_with_capacity(1, result)
    assert result["reviewer_matching"]["matched_reviews"] == 0
    assert result["reviewer_matching"]["fallback_reviews"] == 3


@pytest.mark.parametrize("matching", ["random", "topic"])
def test_native_phase_route_dispersion_and_attrition_history(world, matching, monkeypatch):
    world.reviewer_matching = matching
    # Any legacy global random.sample path would fail this test.
    monkeypatch.setattr(random, "sample", lambda *a, **k: pytest.fail("legacy assignment path"))
    state = random.getstate()
    result = {"year": 1}
    averages = world._run_phase_3_peer_review(1, result)
    assert random.getstate() == state
    metrics = result["reviewer_matching"]
    assert world._capacity_mode_on is True
    assert metrics["completed_reviews"] == 3
    assert metrics["completed_workload"]["mean_load"] == 3 / 8
    assert metrics["paper_rows"][0]["score_variance"] == pytest.approx(8 / 3)
    assert metrics["mean_within_paper_variance"] == pytest.approx(8 / 3)
    assert metrics["paper_rows"][0]["score_range"] == 4
    assert averages[world.paper["conference"]]["paper"] == 4
    if matching == "random":
        assert metrics["fallback_reviews"] is None
        assert metrics["fallback_fraction"] is None
    world.yearly_results.append(result)
    before = copy.deepcopy(result)
    for reviewer in world.ecosystem.agent_population.values():
        reviewer.is_active = False
    # A later phase/year cannot shrink the historical assignment denominator.
    assert world.yearly_results[0] == before
    assert len(metrics["reviewer_rows"]) == 8
    assert len(world.ecosystem.get_available_reviewers()) == 8
    next_year = {}
    world.conference_system.conferences[0].reset_for_next_year(2)
    world._run_phase_3_peer_review(2, next_year)
    assert next_year["reviewer_matching"]["n_eligible_reviewers"] == 8
    assert next_year["reviewer_matching"]["assignment_workload"]["zero_load_reviewers"] == 8
    assert world.yearly_results[0] == before


def test_empty_year_and_single_review_do_not_claim_zero_disagreement(world):
    for rid, reviewer in world.ecosystem.agent_population.items():
        reviewer.can_review = rid == "match1"
    result = {}
    world._run_phase_3_peer_review(1, result)
    assert result["reviewer_matching"]["papers_with_dispersion"] == 0
    assert result["reviewer_matching"]["mean_within_paper_variance"] is None
    conference = world.conference_system.conferences[0]
    conference.reset_for_next_year(2)
    empty = {}
    world._run_phase_3_peer_review(2, empty)
    metrics = empty["reviewer_matching"]
    assert metrics["n_eligible_reviewers"] == 1
    assert metrics["n_submissions"] == metrics["completed_reviews"] == 0
    assert metrics["matched_fraction"] is None
    assert metrics["completed_workload"]["top_decile_share"] is None


def test_portable_serializer_chains_funding_and_preserves_annual_records(world, tmp_path):
    result = {"year": 1}
    world._run_phase_3_peer_review(1, result)
    world.yearly_results.append(result)
    world.funding_feedback_years = {1: {"fixture_funding": 7}}
    path = tmp_path / "portable_checkpoint_year_1.json.gz"
    write_portable_checkpoint(path, {"yearly_results": world.yearly_results,
                                     "extra": world.extra_checkpoint_state()})
    saved = read_portable_checkpoint(path)
    assert saved["yearly_results"] == world.yearly_results
    world.funding_feedback_years = {}
    world.restore_extra_checkpoint_state(saved["extra"])
    assert world.funding_feedback_years == {1: {"fixture_funding": 7}}
    assert world._capacity_mode_on is True
    world.reviewer_matching = "random"
    with pytest.raises(PortableCheckpointError, match="Reviewer treatment"):
        world.restore_extra_checkpoint_state(saved["extra"])
    world.reviewer_matching = "topic"
    world.mechanisms = Mechanisms.from_cell("P1F0")
    with pytest.raises(PortableCheckpointError, match="Funding treatment"):
        world.restore_extra_checkpoint_state(saved["extra"])
    world.mechanisms = Mechanisms.from_cell("P1F1")
    without_treatment = dict(saved["extra"])
    without_treatment.pop("reviewer_matching")
    with pytest.raises(PortableCheckpointError, match="Reviewer treatment"):
        world.restore_extra_checkpoint_state(without_treatment)


def test_constructor_composes_native_funding_mixin_and_forces_random_route():
    class Base:
        def __init__(self, **kwargs):
            self.reviewer_matching, self.reviewer_capacity, self.reviews_per_paper = "random", None, 3
            self._capacity_mode_on = False
    sim = experiment.simulation_class(Base)()
    assert isinstance(sim, FundingFeedbackMixin)
    assert sim._capacity_mode_on is True
    assert sim.mechanisms == Mechanisms.from_cell("P1F1")
    with pytest.raises(ValueError, match="P1F1"):
        experiment.simulation_class(Base)(mechanisms=Mechanisms.from_cell("P0F1"))


@pytest.mark.parametrize("cell", ["RANDOM", "TOPIC"])
def test_entry_uses_shared_runner_and_binds_resolved_rules(cell, monkeypatch):
    from utopia.experiments import funding_feedback
    seen = {}
    def run(options, simulation_factory, configure):
        args = SimpleNamespace(annual_influx_rate=0.0, disable_resubmission=False,
                               papers_per_project=1, population_mode="university_only",
                               num_years=6, expected_founders=5000, strategy_mix="balanced",
                               num_conferences=6)
        seen.update(options=options, args=args, extra=configure(args), factory=simulation_factory)
        return "fixture result"
    monkeypatch.setattr(funding_feedback, "run_prospective", run)
    argv = ["run-fulltext", "--cell", cell, "--run-id", "cpu_fixture",
            "--vllm-url", "http://unused", "--server-manifest", "unused.json",
            "--fulltext-index", "unused-index.json", "--deadline-epoch", "1791927446",
            "--s3-prefix", "s3://unused/fixture", "--no-thinking"]
    assert experiment.main(argv) == "fixture result"
    assert seen["options"].protocol == experiment.PROTOCOL
    assert seen["options"].funding_cell == "P1F1"
    assert seen["options"].thinking is False
    assert seen["options"].max_tokens == 32768
    assert seen["options"].context_tokens == 196608
    assert seen["options"].deadline_epoch == 1791927446
    assert seen["args"].reviewer_capacity is None
    assert seen["args"].reviewer_matching == cell.lower()
    assert seen["factory"] is experiment.simulation_class
    assert json.loads(json.dumps(seen["extra"]))["reviewer_matching"]["matching"] == cell.lower()
    with pytest.raises(ValueError, match="thinking=False"):
        experiment.main(argv + ["--thinking"])
    with pytest.raises(ValueError, match="founders=5000"):
        experiment.main(argv + ["--founders", "10000"])


# End-to-end replay uses the existing fake HTTP/corpus fixture and real six-year
# simulation methods. This does not run a scientific pilot or contact a model.
from tests.runtime.test_portable_checkpoint_integration import fixed_corpus


@pytest.mark.parametrize("matching", ["random", "topic"])
@pytest.mark.parametrize("failure", ["midbatch", "after-effects"])
def test_real_six_year_reviewer_world_survives_relocation(tmp_path, monkeypatch, fixed_corpus,
                                                         matching, failure):
    import shutil
    from tests.runtime import test_portable_checkpoint_integration as native
    from utopia.runtime.checkpointing import PortableStop
    binding = copy.deepcopy(native.BINDING)
    binding["protocol"] = "reviewer-six-year-cpu-fixture-v1"
    binding["cell"] = matching.upper()
    binding["science_config"].update(reviewer_matching=matching, thinking_override=False,
                                   max_tokens=32768, context_tokens=196608)
    monkeypatch.setattr(native, "BINDING", binding)

    class SmallReviewerWorld(experiment.ReviewerMatchingMixin, native.SmallRealSimulation):
        def __init__(self, root, model, **kwargs):
            model.enable_thinking = model.thinking_override = False
            model.max_tokens_override = 32768
            super().__init__(root, model, **kwargs)
            self.reviewer_matching = self.args.reviewer_matching = matching

    reference_requests = []
    root = tmp_path / "reference"
    reference = SmallReviewerWorld(root, native.FakeHTTPModel(root, reference_requests))
    reference.run()
    expected = native.science_snapshot(reference)
    assert reference.uploaded_years == list(range(7))
    assert all("reviewer_matching" in year for year in reference.yearly_results)
    reference.llm.response_cache.close()
    native.seed()
    old = tmp_path / "old-host"
    sent = []
    interrupted = SmallReviewerWorld(old, native.FakeHTTPModel(old, sent, interrupt=failure == "midbatch"),
                                    after_effects=failure == "after-effects")
    with pytest.raises(PortableStop):
        interrupted.run()
    assert interrupted.uploaded_years == [0, 1, 2]
    interrupted.llm.response_cache.close()
    relocated = tmp_path / "new-host"
    shutil.copytree(old, relocated)
    resumed = SmallReviewerWorld(relocated, native.FakeHTTPModel(relocated, sent))
    resumed.run()
    assert resumed.llm.reused > 0
    assert resumed.uploaded_years == [2, 3, 4, 5, 6]
    assert len(sent) == len(reference_requests)
    native.assert_same_state(native.science_snapshot(resumed), expected)
    resumed.llm.response_cache.close()
