"""Initial-resource x institution-size one-world randomized factorial driver.

Experiment: one shared academic ecosystem, 24 institutions (8 x 2, 8 x 4, 8 x 8
researchers = 112 total), 8 years, Qwen3-32B (thinking on), ONE seed = 8301.
Two designed factors:
  * initial-resource endowment  -- RANDOMIZED, within-institution half/half,
    LOW=60 vs HIGH=140 (the primary causal manipulation);
  * institution size            -- small/medium/large (a designed
    finite-population composition contrast; the mechanism audit found NO direct
    first-decision pathway, so it is exploratory/descriptive, not causal).

E4 SAFETY: this driver imports and subclasses the stock `Simulation` in-process
and NEVER edits utopia/simulation.py / arguments.py. It reuses the whole inherited
phase loop, batching, checkpoint/resume, and Parquet metric export. The only
overrides are (1) `is_exploration_experiment -> True` (deliberately enable the
exploration metric/tracker exporters AND the enumerated-candidate direction
mechanism that supplies the horizon menu -- see mechanism audit), and (2)
`_initialize_university_only_population` (heterogeneous sizes + within-institution
LOW/HIGH treatment). Funding panelization -- the only other scientific mechanism
the exploration path would change -- is neutralized with --funding_panel_max_apps 0
(global funding ranking = neutral default). All researchers are `balanced`
(no exploration-strategy prior), so no strategy confound.

Output layout (canonical, NEVER explore_*):
  outputs/{logs,checkpoints,visual,docs}/<EXP_ID>/
  data/initial_resource_x_institution_size/cache/
where EXP_ID = initial_resource_x_institution_size_qwen3_32b_i24_n112_y8_seed8301.

Run:
  # preflight (no LLM): build population, write manifests + balance table, assert
  python -m utopia.experiments.resource_size --init_only
  # engineering connectivity check (one batched Modal request):
  python -m utopia.experiments.resource_size --connectivity_check \
      --vllm_url https://continual-learning--science-utopia-irxs-vllm-serve.modal.run/v1
  # full seed-8301 run (checkpointed, resumable via the same command):
  python -m utopia.experiments.resource_size \
      --vllm_url https://continual-learning--science-utopia-irxs-vllm-serve.modal.run/v1
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.paths import project_root
import argparse
import hashlib
import json
import os
import sys

REPO_ROOT = str(project_root(__file__))
import pandas as pd

from utopia.simulation import Simulation
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS, DIRECTIONS_DICT
from utopia.agents.researcher_agents import UniversityResearcher
from utopia.arguments import parse_arguments
from utopia.models.models import VLLMServerModel
from utopia.utils.seeding import derive_seed, set_seed
from utopia.runtime.setup import project_setup
from utopia.runtime.provenance import write_run_manifest

# ----------------------------------------------------------------------------
# Frozen experiment constants (prereg). Do NOT tune after seeing any outcome.
# ----------------------------------------------------------------------------
SEED = 8301
NUM_YEARS = 8
NUM_CONFERENCES = 6                      # config exploration_experiment default (neutral)
MODEL = "Qwen/Qwen3-32B"
EXP_ID = "initial_resource_x_institution_size_qwen3_32b_i24_n112_y8_seed8301"
DATA_DIR = os.path.join("data", "initial_resource_x_institution_size")

# Institution-size roster: 8 institutions per tier -> 8*2 + 8*4 + 8*8 = 112.
TIER_SIZES = {"small": 2, "medium": 4, "large": 8}
INSTITUTIONS_PER_TIER = 8
NUM_INSTITUTIONS = INSTITUTIONS_PER_TIER * len(TIER_SIZES)   # 24
NUM_EXPERTISE = 3

# Within-institution randomized resource treatment (only the resource field).
LOW_RESOURCES = 60
HIGH_RESOURCES = 140

# Frozen serving environment (matches the deployed Modal server / paper freeze).
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
VLLM_VERSION = "0.12.0"
DEFAULT_VLLM_URL = "https://continual-learning--science-utopia-irxs-vllm-serve.modal.run/v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Deterministic heterogeneous-size population blueprint + LOW/HIGH treatment.
# Row schema is a SUPERSET of build_population_blueprint()'s schema (adds
# `tier`, `treatment`, `funding_level`), so the existing construction / CoI /
# CSV-persist path is reused verbatim.
# ----------------------------------------------------------------------------
def build_factorial_blueprint(seed: int = SEED) -> list:
    """Build the frozen 24-institution / 112-researcher blueprint.

    Determinism (all from `seed`):
      * institution size tiers  -- Random(derive_seed(seed,'institution_size'))
      * expertise (3 each)      -- Random(derive_seed(seed,'population'))
      * LOW/HIGH within each institution -- Random(derive_seed(seed,'resource_treatment',inst_id))

    Every researcher is `balanced` (no exploration-strategy prior). Treatment
    sets ONLY `funding_level`; nothing else depends on it.
    """
    import random

    # 1. Assign size tiers to the 24 institutions (shuffled so tier does not
    #    track the institution index). 8 small / 8 medium / 8 large.
    tier_pool = []
    for tier in ("small", "medium", "large"):
        tier_pool += [tier] * INSTITUTIONS_PER_TIER
    random.Random(derive_seed(seed, "institution_size")).shuffle(tier_pool)

    # 2. Expertise draws share one deterministic RNG stream, consumed in a fixed
    #    institution-then-researcher order (same convention as
    #    build_population_blueprint).
    rng_pop = random.Random(derive_seed(seed, "population"))

    rows = []
    for inst_idx in range(NUM_INSTITUTIONS):
        institution = f"institution_{inst_idx:04d}"
        tier = tier_pool[inst_idx]
        size = TIER_SIZES[tier]

        inst_rows = []
        for r_idx in range(size):
            expertise_indices = rng_pop.sample(
                range(len(AVAILABLE_DIRECTIONS)),
                min(NUM_EXPERTISE, len(AVAILABLE_DIRECTIONS)))
            inst_rows.append({
                "institution": institution,
                "tier": tier,
                "researcher_name": f"{institution}_researcher_{r_idx}",
                "strategy": "balanced",
                "expertise_indices": expertise_indices,
                "expertise_topics": [AVAILABLE_DIRECTIONS[i].topic for i in expertise_indices],
            })

        # 3. Exact within-institution half LOW / half HIGH (sizes are all even:
        #    1/1, 2/2, 4/4). Only `funding_level`/`treatment` are set here.
        order = list(range(size))
        random.Random(derive_seed(seed, "resource_treatment", institution)).shuffle(order)
        half = size // 2
        low_slots = set(order[:half])
        for r_idx, row in enumerate(inst_rows):
            is_low = r_idx in low_slots
            row["treatment"] = "LOW" if is_low else "HIGH"
            row["funding_level"] = LOW_RESOURCES if is_low else HIGH_RESOURCES
        rows.extend(inst_rows)

    return rows


def _assignment_hash(rows: list) -> str:
    """Stable hash of the (researcher -> treatment,resources) assignment."""
    payload = ";".join(
        f"{r['researcher_name']}:{r['tier']}:{r['treatment']}:{r['funding_level']}"
        for r in sorted(rows, key=lambda r: r["researcher_name"]))
    return _sha256(payload)


def _write_manifests_and_balance(rows: list, docs_dir: str) -> dict:
    """Write institution + researcher manifests and the pre-treatment balance
    table; return the assignment hashes / summary."""
    os.makedirs(docs_dir, exist_ok=True)
    overall_hash = _assignment_hash(rows)

    # --- institution manifest ---
    inst_rows = {}
    for r in rows:
        inst = r["institution"]
        inst_rows.setdefault(inst, []).append(r)
    institution_manifest = []
    for inst in sorted(inst_rows):
        members = inst_rows[inst]
        institution_manifest.append({
            "institution_id": inst,
            "size_tier": members[0]["tier"],
            "researcher_count": len(members),
            # All institutions identical at init except headcount. No
            # institution-level shared budget/capacity exists in the simulator
            # (resources are strictly per-researcher), so there are NO
            # headcount-scaled institution-level quantities to equalize.
            "initial_reputation_per_researcher": 5,
            "headcount_scaled_quantities": "none",
            "n_low": sum(m["treatment"] == "LOW" for m in members),
            "n_high": sum(m["treatment"] == "HIGH" for m in members),
            "assignment_hash": _assignment_hash(members),
        })
    pd.DataFrame(institution_manifest).to_csv(
        os.path.join(docs_dir, "institution_manifest.csv"), index=False)
    write_json_file({'overall_assignment_hash': overall_hash, 'institutions': institution_manifest}, os.path.join(docs_dir, 'institution_manifest.json'), indent=2)

    # --- researcher manifest ---
    researcher_manifest = [{
        "researcher_id": r["researcher_name"],
        "institution_id": r["institution"],
        "size_tier": r["tier"],
        "expertise_topics": "|".join(r["expertise_topics"]),
        "strategy": r["strategy"],
        "treatment": r["treatment"],
        "initial_resources": r["funding_level"],
    } for r in rows]
    pd.DataFrame(researcher_manifest).to_csv(
        os.path.join(docs_dir, "researcher_manifest.csv"), index=False)

    # --- pre-treatment balance table (non-resource attributes) ---
    # Covariate: expertise-menu durations (direction.years), a fixed pre-treatment
    # attribute that could in principle bias the horizon endpoint. Reported across
    # LOW/HIGH (balanced by within-institution randomization) and across tiers.
    def _menu_years(r):
        return [DIRECTIONS_DICT[t].years for t in r["expertise_topics"]]

    def _group_stats(group_rows):
        import numpy as np
        menu = [_menu_years(r) for r in group_rows]
        mean_years = [float(np.mean(m)) for m in menu]
        min_years = [float(np.min(m)) for m in menu]
        max_years = [float(np.max(m)) for m in menu]
        n = len(group_rows)
        return {
            "n": n,
            "strategy_all_balanced": all(r["strategy"] == "balanced" for r in group_rows),
            "mean_expertise_years": round(float(np.mean(mean_years)), 4) if n else float("nan"),
            "mean_min_expertise_years": round(float(np.mean(min_years)), 4) if n else float("nan"),
            "mean_max_expertise_years": round(float(np.mean(max_years)), 4) if n else float("nan"),
            "frac_small": round(sum(r["tier"] == "small" for r in group_rows) / n, 4) if n else float("nan"),
            "frac_medium": round(sum(r["tier"] == "medium" for r in group_rows) / n, 4) if n else float("nan"),
            "frac_large": round(sum(r["tier"] == "large" for r in group_rows) / n, 4) if n else float("nan"),
        }

    balance = []
    for label, sel in [("LOW", [r for r in rows if r["treatment"] == "LOW"]),
                       ("HIGH", [r for r in rows if r["treatment"] == "HIGH"]),
                       ("tier:small", [r for r in rows if r["tier"] == "small"]),
                       ("tier:medium", [r for r in rows if r["tier"] == "medium"]),
                       ("tier:large", [r for r in rows if r["tier"] == "large"]),
                       ("ALL", rows)]:
        stats = _group_stats(sel)
        stats["group"] = label
        balance.append(stats)
    pd.DataFrame(balance).set_index("group").to_csv(
        os.path.join(docs_dir, "balance_table.csv"))

    return {"overall_assignment_hash": overall_hash,
            "n_low": sum(r["treatment"] == "LOW" for r in rows),
            "n_high": sum(r["treatment"] == "HIGH" for r in rows),
            "n_institutions": len(inst_rows),
            "n_researchers": len(rows)}


def _assert_design_invariants(rows: list):
    """Assert the frozen design + the CORRECT treatment invariant.

    LOW and HIGH are DIFFERENT randomized researchers (different expertise /
    identity), NOT identical paired agents. The provable invariant is that
    treatment assignment sets ONLY the resource field; institution blocking
    guarantees EXACT within-institution half/half counts.
    """
    assert len(rows) == 112, f"expected 112 researchers, got {len(rows)}"
    by_inst = {}
    for r in rows:
        by_inst.setdefault(r["institution"], []).append(r)
    assert len(by_inst) == NUM_INSTITUTIONS, f"expected {NUM_INSTITUTIONS} institutions"

    tier_counts = {"small": 0, "medium": 0, "large": 0}
    for inst, members in by_inst.items():
        tier = members[0]["tier"]
        assert all(m["tier"] == tier for m in members), f"{inst} mixed tiers"
        assert len(members) == TIER_SIZES[tier], f"{inst} wrong size"
        tier_counts[tier] += 1
        n_low = sum(m["treatment"] == "LOW" for m in members)
        n_high = sum(m["treatment"] == "HIGH" for m in members)
        assert n_low == n_high == len(members) // 2, \
            f"{inst} not half/half: LOW={n_low} HIGH={n_high}"
    assert all(c == INSTITUTIONS_PER_TIER for c in tier_counts.values()), \
        f"tier counts != {INSTITUTIONS_PER_TIER} each: {tier_counts}"

    # Treatment mutates ONLY the resource field: LOW<->60, HIGH<->140, nothing else.
    for r in rows:
        expected = LOW_RESOURCES if r["treatment"] == "LOW" else HIGH_RESOURCES
        assert r["funding_level"] == expected, \
            f"{r['researcher_name']} treatment/resource mismatch"
    assert sum(r["treatment"] == "LOW" for r in rows) == 56
    assert sum(r["treatment"] == "HIGH" for r in rows) == 56


def initialize_factorial_population(sim, blueprint: list):
    """Construct the heterogeneous population and apply the LOW/HIGH treatment,
    reusing the stock construction / conflict-of-interest / blueprint-persist
    path. REUSED verbatim: UniversityResearcher.__init__ (per-row funding_level
    -> self.resources), ecosystem.add_agent, add_conflict_of_interest,
    _add_funding_agencies, blueprint CSV persist. OVERRIDDEN only: the blueprint
    generation (variable sizes + per-row treatment)."""
    from collections import defaultdict

    _assert_design_invariants(blueprint)

    for row in blueprint:
        researcher = UniversityResearcher(
            researcher_name=row["researcher_name"],
            university_name=row["institution"],
            funding_level=row["funding_level"],   # LOW=60 / HIGH=140 (only treated field)
            expertise=[AVAILABLE_DIRECTIONS[i] for i in row["expertise_indices"]],
            llm=sim.llm,
            generate_research_proposal=True,
            exploration_strategy="balanced",
        )
        sim.ecosystem.add_agent(researcher)

    # Within-institution conflicts of interest (same as stock university_only).
    by_institution = defaultdict(set)
    for row in blueprint:
        by_institution[row["institution"]].add(row["researcher_name"])
    for institution, names in by_institution.items():
        for name in names:
            sim.ecosystem.get_agent_by_id(name).add_conflict_of_interest(names - {name})

    # Persist blueprint (audit; same filename/schema-superset as stock path).
    os.makedirs(sim.output_dir, exist_ok=True)
    pd.DataFrame(blueprint).to_csv(
        os.path.join(sim.output_dir, "population_blueprint.csv"), index=False)

    # Manifests + balance table + invariant record.
    summary = _write_manifests_and_balance(blueprint, sim.args.docs_dir)

    # Treatment/tier lookup for the decision log.
    sim._treatment_by_agent = {r["researcher_name"]: r["treatment"] for r in blueprint}
    sim._tier_by_agent = {r["researcher_name"]: r["tier"] for r in blueprint}

    sim._add_funding_agencies()
    print(f"[factorial] population: {summary['n_researchers']} researchers @ "
          f"{summary['n_institutions']} institutions "
          f"(LOW={summary['n_low']}, HIGH={summary['n_high']}); "
          f"assignment_hash={summary['overall_assignment_hash'][:12]}...", flush=True)


class FactorialSimulation(Simulation):
    """Stock Simulation with (1) exploration metrics/exporters + enumerated-
    candidate direction mechanism enabled, and (2) the heterogeneous-size +
    LOW/HIGH population. Everything else is inherited unchanged."""

    @property
    def is_exploration_experiment(self) -> bool:
        # Deliberately True: enables the Parquet metric exporters + embedding/
        # novelty/CD trackers AND the enumerated-candidate direction mechanism
        # (the horizon-choice menu). Per the mechanism audit, with all-`balanced`
        # + --funding_panel_max_apps 0 this alters no treatment-relevant
        # scientific mechanism vs the neutral default (candidate menus are
        # resource-independent; funding panelization is neutralized).
        return True

    def _initialize_university_only_population(self):
        blueprint = build_factorial_blueprint(getattr(self.args, "seed", SEED))
        initialize_factorial_population(self, blueprint)

    def _run_phase_1_research_directions(self, year, year_results):
        # Capture who receives a NEW direction this year BEFORE the stock method
        # updates project_end_year (same filter as the parent, line ~1802). No
        # extra LLM call; the decision log is written from the results the parent
        # already produced (agent.newest_direction) + the resource-independent
        # candidate menu (= the researcher's expertise for `balanced`).
        authors = self.ecosystem.get_available_authors()
        pending_ids = [a.id for a in authors if a.project_end_year < year and a.is_active]
        result = super()._run_phase_1_research_directions(year, year_results)
        self._log_project_decisions(year, pending_ids)
        return result

    def _log_project_decisions(self, year, pending_ids):
        path = os.path.join(self.args.docs_dir, "project_decisions.jsonl")
        with open(path, "a") as f:
            for aid in pending_ids:
                agent = self.ecosystem.get_agent_by_id(aid)
                nd = getattr(agent, "newest_direction", None)
                if not nd:
                    continue
                # Candidate menu for a `balanced` agent = its expertise directions
                # (get_strategy_filtered_candidate_directions returns expertise;
                # resource-independent). Durations are direction.years.
                cand_topics = [d.topic for d in agent.expertise]
                cand_durations = [d.years for d in agent.expertise]
                sel = nd["direction"]
                n_awards = sum(len(v) for v in getattr(agent, "funding_success_history", {}).values())
                rec = {
                    "year": year,
                    "researcher_id": aid,
                    "institution_id": getattr(agent, "university_name", None),
                    "size_tier": self._tier_by_agent.get(aid),
                    "treatment": self._treatment_by_agent.get(aid),
                    "current_resources": getattr(agent, "resources", None),
                    "candidate_topics": cand_topics,
                    "candidate_durations": cand_durations,
                    "selected_topic": sel.topic,
                    "selected_duration": sel.years,
                    "rationale": nd.get("reason"),
                    "project_start_year": getattr(agent, "project_start_year", None),
                    "project_end_year": getattr(agent, "project_end_year", None),
                    "prior_funding_awards": n_awards,
                    "is_active": getattr(agent, "is_active", None),
                }
                f.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------
# Argument / path wiring (in-process; canonical non-explore_* layout).
# ----------------------------------------------------------------------------
def _build_args(vllm_url: str):
    sim_argv = [
        "--experiment_name", "default_irxs",   # neutral, passes the whitelist
        "--seed", str(SEED),
        "--population_mode", "university_only",
        "--num_institutions", str(NUM_INSTITUTIONS),
        "--num_years", str(NUM_YEARS),
        "--num_conferences", str(NUM_CONFERENCES),
        "--model", MODEL,
        "--batch_size", "128",
        "--industry_funding_mode", "performance",
        "--funding_allocation_mode", "fixed",
        "--funding_panel_max_apps", "0",       # neutralize funding panelization
    ]
    if vllm_url:
        sim_argv += ["--vllm_url", vllm_url]
    args = parse_arguments(sim_argv)

    # Override to the canonical factorial layout (NEVER explore_*).
    base = "outputs"
    args.experiment_id = EXP_ID
    args.log_dir = os.path.join(base, "logs", EXP_ID)
    args.checkpoint_dir = os.path.join(base, "checkpoints", EXP_ID)
    args.visual_dir = os.path.join(base, "visual", EXP_ID)
    args.docs_dir = os.path.join(base, "docs", EXP_ID)
    args.output_dir = args.checkpoint_dir
    args.data_cache_dir = os.path.join(DATA_DIR, "cache")
    args.strategy_mix_list = None
    for d in (args.log_dir, args.checkpoint_dir, args.visual_dir, args.docs_dir,
              args.data_cache_dir):
        os.makedirs(d, exist_ok=True)
    return args


def _make_sim(args, llm):
    return FactorialSimulation(
        llm=llm,
        verbose=args.verbose,
        num_years=args.num_years,
        output_dir=args.output_dir,
        debug=args.debug,
        industry_funding_mode=args.industry_funding_mode,
        funding_allocation_mode=args.funding_allocation_mode,
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain,
        max_retries=args.max_retries,
        weighted_funding_assignment=args.weighted_funding_assignment,
        experiment_name=args.experiment_name,
        collaboration_mode=args.collaboration_mode,
        collaboration_network_growth_mode=args.collaboration_network_growth_mode,
        start_year=args.start_year,
        args=args,
    )


def _resolved_config(args) -> dict:
    return {
        "experiment_id": EXP_ID,
        "seed": SEED,
        "num_years": NUM_YEARS,
        "num_institutions": NUM_INSTITUTIONS,
        "num_researchers": 112,
        "institution_size_roster": {t: INSTITUTIONS_PER_TIER for t in TIER_SIZES},
        "tier_sizes": TIER_SIZES,
        "low_resources": LOW_RESOURCES,
        "high_resources": HIGH_RESOURCES,
        "all_strategies": "balanced",
        "funding_panel_max_apps": 0,
        "num_conferences": NUM_CONFERENCES,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "vllm_version": VLLM_VERSION,
        "vllm_url": getattr(args, "vllm_url", None),
        "server_seed": SEED,
        "temperature": 0.7,
        "max_tokens": 2048,
        "thinking_mode": True,
        "collaboration": "disabled (neutral pathway)",
    }


def run_init_only():
    """Preflight: build population via the real init path, write manifests +
    balance table, assert invariants, and round-trip the ecosystem
    serialization. NO scientific LLM decisions; effects NOT inspected."""
    args = _build_args(vllm_url=None)
    project_setup()
    set_seed(SEED)
    sim = _make_sim(args, llm=None)   # offline: researchers created with llm=None
    sim.initialize_agents()

    # Ecosystem serialization check: to_dict() must carry the treated resource
    # field for every researcher. (A full from_dict round-trip needs a
    # post-Phase-1 newest_direction, so it is exercised by the real
    # checkpoint/resume after year 1, not on a fresh population.)
    agents = {a["id"]: a for a in sim.ecosystem.to_dict()["agents"]}
    for aid, treat in sim._treatment_by_agent.items():
        want = LOW_RESOURCES if treat == "LOW" else HIGH_RESOURCES
        got = agents[aid]["resources"]
        assert got == want, f"serialization resource mismatch {aid}: {got} != {want}"

    manifest = {"experiment_id": EXP_ID, "n_researchers": len(sim._treatment_by_agent),
                "resolved_config": _resolved_config(args)}
    write_json_file(manifest, os.path.join(args.docs_dir, 'init_only_summary.json'), indent=2)
    print(f"[factorial] init_only OK -> manifests + balance_table in {args.docs_dir}; "
          f"ecosystem round-trip preserved resources.", flush=True)


def run_connectivity_check(vllm_url: str):
    """Engineering-only: one batched structured-output request to the Modal
    server + a freeze record. Does NOT enter analysis."""
    project_setup()
    set_seed(SEED)
    llm = VLLMServerModel(model_name=MODEL, base_url=vllm_url,
                          max_concurrent_requests=8, run_seed=SEED)
    schema = {"type": "object",
              "properties": {"answer": {"type": "integer"}},
              "required": ["answer"]}
    prompts = [f"Return JSON with field 'answer' = {a}+{b}." for a, b in [(2, 3), (10, 4), (7, 8)]]
    outputs = llm.generate_batch(
        prompts, response_format={"type": "json_schema", "json_object": {"schema": schema}},
        seed_ctx=("connectivity", 0))
    print("[factorial] connectivity outputs:", outputs, flush=True)
    print("[factorial] call_stats:", getattr(llm, "call_stats", None), flush=True)
    freeze = _resolved_config(argparse.Namespace(vllm_url=vllm_url))
    freeze["endpoint"] = vllm_url
    write_json_file(freeze, 'outputs/docs/%s/connectivity_freeze.json' % EXP_ID, indent=2)
    print("[factorial] freeze record written.", flush=True)


def run_full(vllm_url: str):
    """Full seed-8301 simulation (checkpointed, resumable via the same command)."""
    assert vllm_url, "--vllm_url is required for the full run"
    args = _build_args(vllm_url)
    project_setup()
    set_seed(SEED)
    print(f"[factorial] run seed={SEED} -> {args.checkpoint_dir}", flush=True)

    llm = VLLMServerModel(model_name=MODEL, base_url=vllm_url,
                          max_concurrent_requests=args.batch_size, run_seed=SEED)
    # Record the resolved/frozen config up front (run() then merges status).
    write_run_manifest(args.docs_dir, status="initializing", args=args,
                       resolved_config=_resolved_config(args))
    with open(os.path.join(args.log_dir, "command.txt"), "w") as f:
        f.write(" ".join([sys.executable] + sys.argv) + "\n")

    sim = _make_sim(args, llm)
    sim.run()
    print(f"[factorial] run complete -> tables in {args.docs_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm_url", default=None,
                   help=f"Modal endpoint (default {DEFAULT_VLLM_URL} for the full run)")
    p.add_argument("--init_only", action="store_true",
                   help="Preflight: build population + manifests + asserts, then exit")
    p.add_argument("--connectivity_check", action="store_true",
                   help="Engineering-only batched request to the Modal server, then exit")
    args = p.parse_args()

    if args.init_only:
        run_init_only()
    elif args.connectivity_check:
        run_connectivity_check(args.vllm_url or DEFAULT_VLLM_URL)
    else:
        run_full(args.vllm_url or DEFAULT_VLLM_URL)


if __name__ == "__main__":
    main()
