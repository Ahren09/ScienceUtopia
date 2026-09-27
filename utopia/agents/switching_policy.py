"""Pure gate/menu policy for the reviewed Funding feedback propensity protocol v0.2.

    decision = build_decision(
        cell, agent_id, year, previous_topic, canonical_topics53, distance_map52,
        seed=42,
    )
    candidate_map[agent_id] = [one_year_directions[t]
                               for t in decision["candidate_topics"]]
    event = finalize_choice(decision, selected_topic, fallback_kind=None)

Call only for native phase-1 eligible REPEAT choices, in years 2..10. The
driver owns activity/exposure logs, common initialization, one-year timelines,
native centroid computation, finite nonzero embeddings, and complete history
ingestion. In particular, distances alone cannot establish those properties.
The supplied distances must come from the native direction-to-career-centroid
calculation using max_years=3, current_year=year-1 (inclusive years t-4..t-1).

No model, tracker, encoder, HTTP call, file write, or global RNG draw is used.
The seed function is the original utopia.utils.seeding.derive_seed,
resolved lazily; a caller already holding that function may supply it as
derive_seed_fn. Lightweight tests inject that exact source function to avoid
importing the simulator's optional Torch dependencies.

Input order never determines menus or fallback position. Rank by (distance,
topic); take 17 from each end of 52, then present alphabetically in BOTH arms.
Distances are retained without clipping or rounding, including values above 1.
Only a fixed eight-float32-epsilon boundary allowance accommodates the native
float32 cosine computation. This tolerance is NOT applied to separation:
every encountered eligible state must satisfy the exact strict gap > 0.

Policy errors inherit BaseException so native broad Exception handlers cannot
turn an invalid/domain-failed state into an undocumented fallback. The driver
must persist error.diagnostics and terminate that cell. A zero geometric gap
is out of the declared policy domain, not evidence against the hypothesis.
"""

from utopia.runtime.historical import SWITCH_GATE_SEED_NAMESPACE

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Real
import random
import sys


SEED = 42
POLICY_VERSION = "switching_propensity_distance_v1"
GATE_NAMESPACE = SWITCH_GATE_SEED_NAMESPACE
MENU_SIZE = 17
DISTANCE_BOUNDARY_TOLERANCE = 8 * 2.0 ** -23
CELLS = {
    "LN": (0.25, "near-history"),
    "LF": (0.25, "far-history"),
    "HN": (0.75, "near-history"),
    "HF": (0.75, "far-history"),
}
CANONICAL_TOPICS = (
    "3d_vision", "ai_ethics", "algorithmic_game_theory", "algorithms",
    "artificial_intelligence", "audio_computing", "complexity_theory",
    "computational_science", "computational_topology", "computer_architecture",
    "computer_graphics", "computer_networks", "computing_education",
    "control_systems", "cryptography", "cybersecurity", "data_structures",
    "database_systems", "deep_learning", "dialogue_systems", "digital_libraries",
    "discrete_mathematics", "distributed_systems", "evolutionary_computation",
    "formal_methods", "generative_models", "geometric_algorithms",
    "human_computer_interaction", "image_recognition", "information_retrieval",
    "information_theory", "interdisciplinary_cs", "interpretable_ml",
    "logic_in_cs", "mathematical_software", "meta-learning", "multiagent_systems",
    "multimedia_systems", "natural_language_processing", "neural_symbolic_ai",
    "numerical_analysis", "operating_systems", "parallel_computing",
    "performance_analysis", "planning_and_scheduling", "programming_languages",
    "quantum_computing", "reinforcement_learning", "robotics", "social_networks",
    "software_engineering", "symbolic_computation", "video_understanding",
)


class PolicyError(BaseException):
    """Fatal per-cell policy failure with detached, JSON-ready diagnostics."""

    status = "invalid"

    def __init__(self, reason, **details):
        self.diagnostics = {
            **deepcopy(details), "status": self.status, "reason": reason,
            "policy_version": POLICY_VERSION,
        }
        super().__init__(reason)


class PolicyInputError(PolicyError):
    """Missing/malformed inputs or an implementation invariant violation."""


class PolicyDomainError(PolicyError):
    """Valid geometry has no strictly separated near/far menus."""

    status = "out_of_domain"


def _identity(agent_id, year, seed):
    if type(seed) is not int or seed != SEED:
        raise PolicyInputError("only_seed42_is_permitted")
    if type(agent_id) is not str or not agent_id.strip() or agent_id != agent_id.strip():
        raise PolicyInputError("invalid_agent_id")
    if type(year) is not int or not 2 <= year <= 10:
        raise PolicyInputError("repeat_choice_requires_calendar_year_2_through_10")


def _gate(agent_id, year, seed, derive_seed_fn):
    _identity(agent_id, year, seed)
    if derive_seed_fn is None:
        from utopia.utils.seeding import derive_seed
        derive_seed_fn = derive_seed
    if not callable(derive_seed_fn):
        raise PolicyInputError("derive_seed_fn_must_be_the_native_callable")
    gate_seed = derive_seed_fn(SEED, GATE_NAMESPACE, agent_id, year)
    if type(gate_seed) is not int or not 0 <= gate_seed < 2 ** 31:
        raise PolicyInputError("native_derive_seed_returned_invalid_seed")
    return gate_seed, random.Random(gate_seed).random()


def switch_uniform(agent_id, year, seed=SEED, *, derive_seed_fn=None):
    """One calendar-keyed U; cell, menu, retries and mutable RNG state are absent."""
    return _gate(agent_id, year, seed, derive_seed_fn)[1]


def requested_switch(u, p):
    """Exact u < p, including the 0.25/0.75 boundaries; no gate resampling."""
    if (isinstance(u, bool) or not isinstance(u, Real) or not math.isfinite(u)
            or not 0 <= u < 1):
        raise PolicyInputError("uniform_must_be_finite_in_half_open_unit_interval")
    if isinstance(p, bool) or not isinstance(p, Real) or p not in (0.25, 0.75):
        raise PolicyInputError("probability_must_be_0_25_or_0_75")
    return bool(u < p)


def build_decision(cell, agent_id, year, previous_topic, canonical_topics53,
                   distance_map52, seed=SEED, *, derive_seed_fn=None):
    """Return both menus and the canonical strict candidate menu for one repeat.

    canonical_topics53 accepts the actual 53 stock labels in any input order.
    distance_map52 must contain exactly those labels minus previous_topic.
    No default distances, heuristics, expertise substitutions or input mutation.
    """
    _identity(agent_id, year, seed)
    if type(cell) is not str or cell not in CELLS:
        raise PolicyInputError("unknown_cell")
    context = {"cell": cell, "agent_id": agent_id, "year": year, "seed": seed}
    if (isinstance(canonical_topics53, (str, bytes))
            or not isinstance(canonical_topics53, Sequence)):
        raise PolicyInputError("canonical_topics_must_be_a_sequence", **context)
    topics = list(canonical_topics53)
    if (len(topics) != 53 or any(type(topic) is not str for topic in topics)
            or len(set(topics)) != 53 or set(topics) != set(CANONICAL_TOPICS)):
        raise PolicyInputError("canonical_topics_must_be_exactly_the_53_stock_topics", **context)
    if type(previous_topic) is not str or previous_topic not in CANONICAL_TOPICS:
        raise PolicyInputError("repeat_choice_requires_a_valid_previous_topic", **context)
    context["previous_topic"] = previous_topic
    eligible = set(CANONICAL_TOPICS) - {previous_topic}
    if not isinstance(distance_map52, Mapping):
        raise PolicyInputError("distance_map_must_be_a_mapping", **context)
    if (any(type(topic) is not str for topic in distance_map52)
            or set(distance_map52) != eligible):
        raise PolicyInputError("distance_map_must_cover_exactly_52_nonprevious_topics", **context)
    distances = {}
    for topic in sorted(eligible):
        value = distance_map52[topic]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise PolicyInputError("distance_is_not_a_real_number", topic=topic, **context)
        value = float(value)
        if not math.isfinite(value):
            raise PolicyInputError("distance_is_not_finite", topic=topic, **context)
        if not -DISTANCE_BOUNDARY_TOLERANCE <= value <= 2 + DISTANCE_BOUNDARY_TOLERANCE:
            raise PolicyInputError("distance_is_outside_cosine_range", topic=topic, **context)
        distances[topic] = value

    gate_seed, u = _gate(agent_id, year, seed, derive_seed_fn)
    p, distance_mode = CELLS[cell]
    switch = requested_switch(u, p)
    ranked = sorted(eligible, key=lambda topic: (distances[topic], topic))
    near_ranked, far_ranked = ranked[:MENU_SIZE], ranked[-MENU_SIZE:]
    near, far = sorted(near_ranked), sorted(far_ranked)
    near_min, near_max = distances[near_ranked[0]], distances[near_ranked[-1]]
    far_min, far_max = distances[far_ranked[0]], distances[far_ranked[-1]]
    gap = far_min - near_max
    event = {
        **context, "policy_version": POLICY_VERSION, "status": "planned",
        "event_type": "eligible_repeat_choice", "eligible_repeat": True,
        "is_initial_choice": False, "p": p, "u": u,
        "gate_seed": gate_seed, "gate_namespace": GATE_NAMESPACE,
        "gate_rng": "random.Random", "python_version": sys.version.split()[0],
        "requested_switch": switch, "requested_action": "switch" if switch else "stay",
        "distance_mode": distance_mode,
        "distance_definition": "direction_to_historical_paper_centroid_cosine",
        "history_window_start": year - 4, "history_window_end": year - 1,
        "native_centroid_max_years": 3,
        "n_canonical_topics": 53, "n_eligible_topics": 52,
        "ranked_eligible_topics": ranked, "near_topics": near, "far_topics": far,
        "n_near": len(near), "n_far": len(far), "n_middle_unused": 18,
        "near_min": near_min, "near_max": near_max,
        "far_min": far_min, "far_max": far_max, "separation_gap": gap,
        "distance_boundary_tolerance": DISTANCE_BOUNDARY_TOLERANCE,
        "distance_map": distances,
    }
    if len(near) != 17 or len(far) != 17 or set(near) & set(far):
        raise PolicyInputError("menus_are_not_disjoint_17_topic_sets", **event)
    if gap < 0:
        raise PolicyInputError("negative_separation_after_sorting", **event)
    if gap == 0:
        raise PolicyDomainError("absent_strict_separation", **event)
    event["candidate_topics"] = (
        near if distance_mode == "near-history" else far) if switch else [previous_topic]
    return deepcopy(event)


def finalize_choice(decision, chosen_topic, fallback_kind=None):
    """Validate the native result/fallback; return a detached analysis-ready event.

    fallback_kind is None, "parse", or "validation". The latter two require
    the native first-displayed-candidate fallback. This function performs no
    selection, fallback, gate draw, memory update, or project-state update.
    The distance of a stay is not supplied in distance_map52; its conditional
    switch distance is therefore explicitly None rather than a fabricated zero.
    """
    if not isinstance(decision, Mapping) or decision.get("status") != "planned":
        raise PolicyInputError("finalization_requires_a_planned_decision")
    if fallback_kind not in (None, "parse", "validation"):
        raise PolicyInputError("unknown_fallback_kind")
    candidates = decision.get("candidate_topics")
    if (type(candidates) is not list or not candidates
            or type(chosen_topic) is not str or chosen_topic not in candidates):
        raise PolicyInputError("chosen_topic_is_not_in_explicit_menu")
    if fallback_kind is not None and chosen_topic != candidates[0]:
        raise PolicyInputError("fallback_must_be_first_canonical_candidate")
    switched = chosen_topic != decision["previous_topic"]
    if (type(decision.get("requested_switch")) is not bool
            or switched != decision["requested_switch"]
            or switched != requested_switch(decision["u"], decision["p"])):
        raise PolicyInputError("realized_switch_disagrees_with_gate")
    event = deepcopy(dict(decision))
    event.update({
        "status": "complete", "chosen_topic": chosen_topic,
        "realized_switch": switched,
        "conditional_switch_distance": decision["distance_map"][chosen_topic] if switched else None,
        "fallback": fallback_kind is not None, "fallback_kind": fallback_kind,
        "eligible_repeat_count": 1, "realized_switch_count": int(switched),
        "new_project_choice_count": 1, "initial_choice_count": 0,
    })
    return event
