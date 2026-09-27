"""Read-only aliases for pre-package source records and deterministic seed salts.

Aliases resolve names only. They never replace or bypass recorded content hashes.
Frozen execution must continue in its original source worktree.
"""

SOURCE_PATHS = {
    "utopia/simulation.py": "run_simulation.py",
    "utopia/constants.py": "const.py",
    "utopia/utils/general_utils.py": "utopia/utils/generail_utils.py",
    "utopia/utils/seeding.py": "utopia/utils/generail_utils.py",
    "utopia/analysis/network_disclosure.py": "src/analyze_james_network.py",
    "utopia/analysis/research_strategy.py": "src/analyze_james_strategy.py",
    "utopia/analysis/project_cost.py": "src/analyze_project_cost.py",
    "utopia/analysis/switching_propensity.py": "src/james_propensity_analysis.py",
    "utopia/funding/compact.py": "src/james_funding_compact.py",
    "utopia/funding/sequential.py": "src/james_funding_sequential.py",
    "utopia/funding/validation.py": "src/james_funding_validation.py",
    "utopia/funding/accounting.py": "src/project_cost_accounting.py",
    "utopia/runtime/server_profiles.py": "src/james_server_profiles.py",
    "utopia/experiments/funding_feedback.py": "src/james_mechanisms.py",
    "utopia/experiments/switching_propensity.py": "src/james_propensity.py",
    "utopia/agents/switching_policy.py": "src/james_propensity_policy.py",
    "utopia/experiments/exploration.py": "src/run_experiments.py",
    "utopia/experiments/scale_expansion.py": "src/run_scale_expansion.py",
    "utopia/experiments/influx_factorial.py": "src/run_influx_factorial.py",
    "utopia/experiments/resource_size.py": "src/run_factorial_resource_size.py",
    "utopia/experiments/funding_cutoff.py": "src/run_matthew_rd.py",
    "utopia/experiments/review_replay.py": "src/run_review_replay.py",
    "utopia/experiments/project_cost.py": "src/run_project_cost_campaign.py",
    "utopia/analysis/funding_cutoff.py": "utopia/analysis/matthew_rd_analysis.py",
    "utopia/analysis/resource_size.py": "utopia/analysis/factorial_resource_size_analysis.py",
    "docs/switching_propensity_protocol.md": "docs/james_propensity_protocol.md",
    "docs/sequential_funding_protocol.md": "docs/james_sequential_funding_protocol.md",
    "tests/funding/test_compact.py": "src/test_james_funding_compact.py",
    "tests/funding/test_sequential.py": "src/test_james_funding_sequential.py",
    "tests/funding/test_validation.py": "src/test_james_funding_validation.py",
    "tests/funding/test_accounting.py": "src/test_project_cost_accounting.py",
    "tests/experiments/test_influx_runtime.py": "src/test_james_influx.py",
    "tests/experiments/test_funding_feedback.py": "src/test_james_mechanisms.py",
    "tests/experiments/test_switching_propensity.py": "src/test_james_propensity.py",
    "tests/experiments/test_switching_policy.py": "src/test_james_propensity_policy.py",
    "tests/experiments/test_project_cost.py": "src/test_project_cost_campaign.py",
    "tests/analysis/test_network_disclosure.py": "src/test_james_network.py",
    "tests/analysis/test_research_strategy.py": "src/test_james_strategy.py",
    "tests/analysis/test_switching_propensity.py": "src/test_james_propensity_analysis.py",
    "tests/models/test_request_audit.py": "src/test_james_request_audit.py",
    "tests/models/test_schema_transport.py": "src/test_vllm_schema_transport.py",
    "tests/experiments/test_exploration.py": "tests/test_exploration_experiment.py",
    "tests/experiments/test_scale_expansion.py": "tests/test_scale_expansion.py",
    "tests/experiments/test_influx_factorial.py": "tests/test_influx_factorial.py",
    "tests/experiments/test_resource_size.py": "tests/test_factorial_resource_size.py",
    "tests/experiments/test_funding_cutoff.py": "tests/test_matthew_rd.py",
    "tests/experiments/test_review_replay.py": "tests/test_review_replay.py",
}

SWITCH_GATE_SEED_NAMESPACE = "james_propensity_distance_v1_switch"


def historical_source_path(current_path):
    """Locate the corresponding file in historical Git trees."""
    return SOURCE_PATHS.get(current_path, current_path)
















IDENTIFIER_ALIASES = {
    "funding_feedback": "james_mechanisms",
    "funding-feedback": "james-mechanisms",
    "FundingFeedbackSimulation": "JamesMechanismSimulation",
    "switching_propensity": "james_propensity",
    "switching-propensity": "james-propensity",
    "SwitchingPropensitySimulation": "JamesPropensitySimulation",
    "initial_world_hash": "james_initial_world_hash",
    "initial_world": "james_initial_world",
    "founder_ids": "james_founders",
    "corpus_hash": "james_corpus_hash",
    "cache_provenance": "james_cache_provenance",
    "influx1202": "james1202",
    "influx_result": "james_result",
    "influx_protocol": "james_protocol",
    "build_fixed_cohort": "build_james_cohort",
    "UTOPIA_": "JAMES_",
    "research_strategy": "james_revision_existing",
    "original_N200_network_disclosure_reanalysis": "original_N200_exploratory_reanalysis_for_James",
    "funding_cutoff": "matthew_rd",
    "experiment_queue": "james_ec2_queue",
    "funding-feedback-args-": "james-args-",
    "funding_seed42": "james_seed42",
}


def legacy_identifier(value):
    import re

    pattern = "|".join(
        re.escape(name) for name in sorted(IDENTIFIER_ALIASES, key=len, reverse=True)
    )
    return re.sub(pattern, lambda match: IDENTIFIER_ALIASES[match.group()], value)


def identifier_matches(value, current):
    return value == current or value == legacy_identifier(current)


def historical_value(record, key, default=None):
    return record.get(key, record.get(legacy_identifier(key), default))


def historical_artifact(path):
    """Prefer a new artifact; resolve an existing historical name read-only."""
    from pathlib import Path

    path = Path(path)
    if path.exists():
        return path
    return Path(legacy_identifier(str(path)))


def source_binding_value(binding, current_path):
    return next((binding[name] for name in source_path_candidates(current_path)
                 if name in binding), None)


def source_path_candidates(current_path):
    """Resolve supported source layouts without changing their recorded contents."""
    previous = ("utopia/utils/general_utils.py",) if current_path == "utopia/utils/seeding.py" else ()
    return tuple(dict.fromkeys((current_path, *previous, historical_source_path(current_path))))


def source_binding_subset(trusted_sources, current_paths):
    """Select required pins with their original keys and exact trusted hashes."""
    result = {}
    for current in current_paths:
        key = next((name for name in source_path_candidates(current) if name in trusted_sources), current)
        result[key] = trusted_sources[key]
    return result


def trusted_audit_sources(recorded_sources, trusted_sources):
    """Require every recorded executable source to match the external source pin."""
    current = {
        "utopia/funding/sequential.py",
        "utopia/funding/compact.py",
        "utopia/funding/validation.py",
        "utopia/models/models.py",
        "utopia/models/request_audit.py",
        "utopia/agents/funding_agents.py",
        "utopia/constants.py",
        "utopia/utils/seeding.py",
        "utopia/utils/data_utils.py",
        "utopia/utils/paths.py",
        "utopia/runtime/historical.py",
    }
    legacy = {
        historical_source_path(name)
        for name in current
        if name
        not in {
            "utopia/utils/seeding.py",
            "utopia/utils/data_utils.py",
            "utopia/utils/paths.py",
            "utopia/runtime/historical.py",
        }
    }
    is_legacy = (
        historical_source_path("utopia/funding/sequential.py") in recorded_sources
    )
    previous_layout = (current - {"utopia/utils/seeding.py"}) | {"utopia/utils/general_utils.py"}
    if is_legacy:
        required = legacy
    elif "utopia/utils/general_utils.py" in recorded_sources:
        required = previous_layout
    else:
        required = current
    protocol = "docs/sequential_funding_protocol.md"
    protocol = historical_source_path(protocol) if is_legacy else protocol
    if protocol in trusted_sources:
        required = required | {protocol}
    if set(recorded_sources) != required or any(
        trusted_sources.get(name) != checksum
        for name, checksum in recorded_sources.items()
    ):
        raise ValueError("Historical audit differs from trusted source files")
    return dict(recorded_sources)


INFLUX_POPULATION_SEED_NAMESPACE = "james1202_v2"

HISTORICAL_PROFILE_SHA256 = (
    "8011e0aac664dcec036fc19820fd5c732beb74dfcbf7b21220a1c3dbf1c0091e"
)

