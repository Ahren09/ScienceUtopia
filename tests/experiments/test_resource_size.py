"""Preflight + regression tests for the initial-resource x institution-size
factorial (seed 8301). Pure-logic tests (no LLM / no Modal): blueprint design
invariants, determinism, the CORRECT treatment invariant (assignment mutates
ONLY the resource field), treatment-orthogonality of the candidate menu, the
exploration-flag behavioral-equivalence facts, and ecosystem serialization
round-trip.

Run:  pytest tests/experiments/test_resource_size.py -q
"""

from utopia.utils.paths import project_root
import os
import sys
from collections import Counter

import pytest

REPO_ROOT = str(project_root(__file__))
import utopia.experiments.resource_size as rf
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS, get_strategy_filtered_candidate_directions
from utopia.agents.researcher_agents import UniversityResearcher
from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.config import SIMULATION_CONFIG


def test_sizes_and_total():
    bp = rf.build_factorial_blueprint(8301)
    assert len(bp) == 112
    tier_of_inst = {r["institution"]: r["tier"] for r in bp}
    assert len(tier_of_inst) == 24
    assert Counter(tier_of_inst.values()) == {"small": 8, "medium": 8, "large": 8}
    # sizes per tier
    size_by_inst = Counter(r["institution"] for r in bp)
    for inst, tier in tier_of_inst.items():
        assert size_by_inst[inst] == rf.TIER_SIZES[tier]


def test_within_institution_half_half_and_totals():
    bp = rf.build_factorial_blueprint(8301)
    by_inst = {}
    for r in bp:
        by_inst.setdefault(r["institution"], []).append(r)
    for inst, members in by_inst.items():
        n_low = sum(m["treatment"] == "LOW" for m in members)
        n_high = sum(m["treatment"] == "HIGH" for m in members)
        assert n_low == n_high == len(members) // 2
    assert sum(r["treatment"] == "LOW" for r in bp) == 56
    assert sum(r["treatment"] == "HIGH" for r in bp) == 56
    rf._assert_design_invariants(bp)  # should not raise


def test_determinism_and_stable_hash():
    bp1 = rf.build_factorial_blueprint(8301)
    bp2 = rf.build_factorial_blueprint(8301)
    assert bp1 == bp2
    assert rf._assignment_hash(bp1) == rf._assignment_hash(bp2)
    # a different seed changes the assignment
    bp3 = rf.build_factorial_blueprint(9999)
    assert rf._assignment_hash(bp3) != rf._assignment_hash(bp1)


def test_treatment_mutates_only_resource_field(monkeypatch):
    """The correct invariant: changing the LOW/HIGH resource VALUES changes only
    each researcher's funding_level, never expertise/strategy/tier/identity or
    which slot is LOW vs HIGH."""
    bp = rf.build_factorial_blueprint(8301)
    # funding_level is exactly the treatment mapping
    for r in bp:
        want = rf.LOW_RESOURCES if r["treatment"] == "LOW" else rf.HIGH_RESOURCES
        assert r["funding_level"] == want

    # Rebuild with different resource values; everything except funding_level
    # (and it alone) must be byte-identical, and treatment slots must be unchanged.
    monkeypatch.setattr(rf, "LOW_RESOURCES", 11)
    monkeypatch.setattr(rf, "HIGH_RESOURCES", 22)
    bp_alt = rf.build_factorial_blueprint(8301)
    assert len(bp) == len(bp_alt)
    for a, b in zip(bp, bp_alt):
        assert a["researcher_name"] == b["researcher_name"]
        assert a["institution"] == b["institution"]
        assert a["tier"] == b["tier"]
        assert a["strategy"] == b["strategy"] == "balanced"
        assert a["expertise_indices"] == b["expertise_indices"]
        assert a["treatment"] == b["treatment"]           # slot unchanged
        # only the resource value differs
        assert b["funding_level"] == (11 if b["treatment"] == "LOW" else 22)


def test_candidate_menu_is_resource_independent():
    """LOW(60) and HIGH(140) researchers with identical expertise get an
    IDENTICAL candidate menu -> the enumerated-candidate mechanism cannot
    confound the treatment contrast."""
    expertise = list(AVAILABLE_DIRECTIONS[:3])
    exp_cfg = SIMULATION_CONFIG["exploration_experiment"]
    low = UniversityResearcher("r_low", "institution_0000", funding_level=60,
                               expertise=expertise, llm=None,
                               exploration_strategy="balanced")
    high = UniversityResearcher("r_high", "institution_0000", funding_level=140,
                                expertise=expertise, llm=None,
                                exploration_strategy="balanced")
    cand_low = get_strategy_filtered_candidate_directions(
        low, AVAILABLE_DIRECTIONS, None, 1, exp_cfg)
    cand_high = get_strategy_filtered_candidate_directions(
        high, AVAILABLE_DIRECTIONS, None, 1, exp_cfg)
    assert [d.topic for d in cand_low] == [d.topic for d in cand_high]
    # for balanced the menu is exactly the researcher's expertise
    assert [d.topic for d in cand_low] == [d.topic for d in expertise]


def test_balanced_strategy_adds_no_exploration_prompt():
    """`balanced` neutralizes the exploration-strategy prompt branch, so the
    direction prompt is not biased toward exploration/exploitation."""
    r = UniversityResearcher("r0", "institution_0000", funding_level=100,
                             expertise=list(AVAILABLE_DIRECTIONS[:3]), llm=None,
                             exploration_strategy="balanced")
    assert r._get_exploration_strategy_prompt().strip() == ""


def test_flag_and_panel_neutralization():
    """is_exploration_experiment is forced True (metrics/exporters + enumerated
    menu) and the driver neutralizes funding panelization (panel_max_apps=0)."""
    # property override present on the subclass and returns True
    prop = rf.FactorialSimulation.__dict__["is_exploration_experiment"]
    assert isinstance(prop, property)
    assert prop.fget(object()) is True
    # driver freezes funding_panel_max_apps=0 in resolved config
    import argparse
    cfg = rf._resolved_config(argparse.Namespace(vllm_url=None))
    assert cfg["funding_panel_max_apps"] == 0
    assert cfg["all_strategies"] == "balanced"
    assert cfg["low_resources"] == 60 and cfg["high_resources"] == 140


def test_ecosystem_serialization_preserves_resources():
    """to_dict() carries the treated resource field correctly. (Full from_dict
    round-trip requires a post-Phase-1 newest_direction, so it is validated by
    the real checkpoint/resume after year 1, not on a fresh population.)"""
    eco = MultiAgentEcosystem(agent_configs={})
    for name, fl in [("institution_0000_researcher_0", 60),
                     ("institution_0000_researcher_1", 140)]:
        eco.add_agent(UniversityResearcher(
            name, "institution_0000", funding_level=fl,
            expertise=list(AVAILABLE_DIRECTIONS[:3]), llm=None,
            exploration_strategy="balanced"))
    agents = {a["id"]: a for a in eco.to_dict()["agents"]}
    assert agents["institution_0000_researcher_0"]["resources"] == 60
    assert agents["institution_0000_researcher_1"]["resources"] == 140
    assert agents["institution_0000_researcher_0"]["exploration_strategy"] == "balanced"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
