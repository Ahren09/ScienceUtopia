"""Output-cost policy and passive, checkpointed resource accounting.

Boundary: after Phase 2 removes claimed coauthors, each active lead whose project
ends this year enters production, BEFORE intentions, retrieval or paper selection.
``per_project`` collects project_production_cost once at that boundary. An omitted
override uses the existing funding_cost_per_paper. An explicit 0 charges nothing.
The fee is never divided by k or refunded for no candidates, refusal, partial
selection, parse failure or fallback. An unaffordable project cannot enter.
Already paid projects have no further paper affordability gate.

BASELINE DISCREPANCY (2bb7e69): funding_cost_per_paper and min_funding_threshold
exist in config but are unused by ALL actual submission paths. ``per_paper``
preserves that flow, including its ZERO direct submission debit. It does not
silently activate the unused constant. Annual research costs are unchanged.

Equation for each researcher/year, including inactive founders:
opening + grant_income + industry_income - annual_research_cost
        - production_submission_cost - resubmission_cost
        - funding_application_cost = closing.
Costs in this equation are COLLECTED amounts. requested_delta and uncollected_cost
also expose the native zero-clipping rule. There is no residual balancing entry.
No function in this module consumes RNG.
"""
from __future__ import annotations

from utopia.utils.data_utils import write_jsonl_atomic

from utopia.utils.data_utils import json_sha256

from copy import deepcopy
import json
import math
from pathlib import Path


CATEGORIES = (
    "grant_income", "industry_income", "annual_research_cost",
    "production_submission_cost", "resubmission_cost", "funding_application_cost",
)
INCOME_CATEGORIES = frozenset(("grant_income", "industry_income"))
SCHEMA_VERSION = 1


def stable_id(*parts):
    """Unambiguous identity without wall time, RNG, or Python's salted hash."""
    return json.dumps(parts, separators=(",", ":"), ensure_ascii=True)


def project_id(agent):
    # Native flow permits at most one project per researcher/start year.
    return stable_id("project", agent.id, agent.project_start_year, agent.project_end_year)


def resolve_cost_policy(args, config):
    mode = getattr(args, "production_cost_mode", "per_paper")
    override = getattr(args, "resubmission_cost", None)
    production_override = getattr(args, "project_production_cost", None)
    if mode not in ("per_paper", "per_project"):
        raise ValueError(f"Unknown production_cost_mode: {mode}")
    if override is not None and (not math.isfinite(override) or override < 0):
        raise ValueError("--resubmission_cost must be finite and nonnegative")
    if production_override is not None and (
            not math.isfinite(production_override) or production_override < 0):
        raise ValueError("--project_production_cost must be finite and nonnegative")
    fee = (config["paper"]["funding_cost_per_paper"]
           if production_override is None else production_override)
    resubmission = config["conference"]["resubmission_cost"] if override is None else override
    if any(not math.isfinite(value) or value < 0 for value in (fee, resubmission)):
        raise ValueError("Production and resubmission costs must be finite and nonnegative")
    return {
        "schema_version": SCHEMA_VERSION,
        "production_cost_mode": mode,
        "project_production_cost_override": production_override,
        "project_production_cost": fee,
        "legacy_effective_per_paper_cost": 0,
        "production_boundary": "phase2_completed_lead_before_selection",
        "production_admission_minimum": fee if mode == "per_project" else None,
        "resubmission_cost_override": override,
        "resubmission_cost": resubmission,
        "log_resource_ledger": bool(getattr(args, "log_resource_ledger", False)),
        "papers_per_project": int(getattr(args, "papers_per_project", 1)),
        "resubmission_cell": "zero_fee" if resubmission == 0 else "positive_fee_sensitivity",
    }


def cost_policy_enabled(policy):
    return (policy["production_cost_mode"] != "per_paper"
            or policy["project_production_cost_override"] is not None
            or policy["resubmission_cost_override"] is not None
            or policy["log_resource_ledger"])


def cost_policy_tag(policy):
    """Default CLI retains historical paths. Every opt-in gets a policy identity."""
    if not cost_policy_enabled(policy):
        return ""
    digest = json_sha256(policy, sort_keys=True)[:12]
    fee = format(policy["resubmission_cost"], ".17g").replace(".", "p")
    return (f"_costv{SCHEMA_VERSION}_{policy['production_cost_mode']}"
            f"_resub{fee}_ledger{int(policy['log_resource_ledger'])}_{digest}")


def apply_resource_change(agent, delta, *, category, year, project=None, paper=None,
                          program=None, reason=None):
    """Use native update_resources, preserving its old call signature when off."""
    if getattr(agent, "resource_ledger", None) is None:
        agent.update_resources(delta)
        return True
    return agent.update_resources(
        delta=delta, category=category, year=year, project=project,
        paper=paper, program=program, reason=reason,
    )


def enter_project_production(agent, year, policy, funding_tracker):
    """Admit and charge once, including repeated calls and checkpoint restores."""
    if policy["production_cost_mode"] == "per_paper":
        return True
    if agent.project_end_year != year or agent.project_start_year > year:
        raise ValueError(f"Project is not completed in production year: {project_id(agent)}")
    pid = project_id(agent)
    entries = getattr(agent, "production_cost_entries", {})
    if pid in entries:
        return entries[pid]["admitted"]
    fee = policy["project_production_cost"]
    admitted = agent.resources >= fee
    applied = apply_resource_change(
        agent, -fee if admitted else 0, category="production_submission_cost",
        year=year, project=pid, reason="production_entry" if admitted else "unaffordable",
    )
    entries[pid] = {
        "project_id": pid, "year": year, "admitted": admitted,
        "fee": fee, "charged": fee if admitted else 0,
    }
    agent.production_cost_entries = entries
    if admitted and applied:
        funding_tracker.record_funding_consumption(year, fee, agent.get_type())
    return admitted


class ResourceLedger:
    """Native resource events plus yearly balances. Checkpoint is authoritative."""

    def __init__(self, state=None):
        state = deepcopy(state) if state is not None else {}
        if state and state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported resource ledger schema")
        self.transactions = state.get("transactions", [])
        self.years = state.get("years", {})
        self.entry_years = state.get("entry_years", {})
        self._events = {}
        for event in self.transactions:
            if event["event_id"] in self._events:
                raise ValueError("Duplicate event in resource ledger checkpoint")
            self._events[event["event_id"]] = event

    @staticmethod
    def researchers(population):
        return {a.id: a for a in population.values()
                if getattr(a, "can_author", False) and getattr(a, "resources", None) is not None}

    def attach(self, population):
        for agent in self.researchers(population).values():
            agent.resource_ledger = self

    def event(self, event_id):
        return self._events.get(event_id)

    def start_year(self, year, population):
        agents = self.researchers(population)
        self.attach(population)
        if str(year) in self.years:
            if set(self.years[str(year)]) != set(agents):
                raise ValueError("Researcher roster changed within an accounting year")
            return
        if self.years:
            previous = max(int(y) for y in self.years)
            if year != previous + 1:
                raise ValueError("Resource ledger years must be consecutive")
            self.reconcile(previous, population, require_closed=True)
        elif year != 1:
            raise ValueError("Audit requires opening balances from year 1")
        rows = {}
        for aid, agent in agents.items():
            if not math.isfinite(agent.resources) or agent.resources < 0:
                raise ValueError(f"Invalid opening resources for {aid}")
            self.entry_years.setdefault(aid, year)
            rows[aid] = {
                "year": year, "researcher_id": aid, "researcher_type": agent.get_type(),
                "entry_year": self.entry_years[aid], "founder": self.entry_years[aid] == 1,
                "opening_balance": agent.resources, "opening_active": agent.is_active,
                "closing_balance": None,
            }
        self.years[str(year)] = rows

    def prepare(self, agent, delta, *, category, year, project=None, paper=None,
                program=None, reason=None):
        if category not in CATEGORIES or year is None or not math.isfinite(delta):
            raise ValueError(f"Unclassified or invalid resource change for {agent.id}: {category}")
        if (category in INCOME_CATEGORIES and delta < 0
                or category not in INCOME_CATEGORIES and delta > 0):
            raise ValueError(f"Wrong resource change sign for {category}: {delta}")
        event = {
            "event_id": stable_id(year, agent.id, category, project, paper, program, reason),
            "year": year, "researcher_id": agent.id, "category": category,
            "project_id": project, "paper_id": paper, "program_id": program,
            "reason": reason, "requested_delta": delta,
        }
        prior = self._events.get(event["event_id"])
        if prior is not None:
            if any(prior[k] != v for k, v in event.items()):
                raise ValueError(f"Resource event replay changed: {event['event_id']}")
            return None
        row = self.years.get(str(year), {}).get(agent.id)
        if row is None or row["closing_balance"] is not None:
            raise ValueError(f"Resource change outside open accounting year: {year}/{agent.id}")
        event["balance_before"] = agent.resources
        return event

    def record(self, agent, event):
        event["balance_after"] = agent.resources
        event["delta"] = agent.resources - event["balance_before"]
        event["uncollected_cost"] = max(0, event["delta"] - event["requested_delta"])
        event["active_after"] = agent.is_active
        self.transactions.append(event)
        self._events[event["event_id"]] = event

    def reconcile(self, year, population, *, require_closed=False):
        """Reconstruct from events, validate their chain, and reject any residual."""
        agents = self.researchers(population)
        rows = self.years[str(year)]
        if set(rows) - set(agents):
            raise ValueError("Researchers disappeared from the resource ledger")
        events_by_agent = {aid: [] for aid in rows}
        for event in self.transactions:
            if event["year"] == year:
                events_by_agent[event["researcher_id"]].append(event)
        results = []
        for aid, row in rows.items():
            running = row["opening_balance"]
            amounts = {category: [] for category in CATEGORIES}
            uncollected = []
            for event in events_by_agent[aid]:
                if not math.isclose(running, event["balance_before"], rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Unexplained resource change before {event['event_id']}")
                expected = max(0, running + event["requested_delta"])
                if not math.isclose(expected, event["balance_after"], rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Invalid native resource transition: {event['event_id']}")
                if not math.isclose(event["delta"], event["balance_after"] - running,
                                    rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Invalid resource delta: {event['event_id']}")
                running = event["balance_after"]
                amounts[event["category"]].append(event["delta"])
                uncollected.append(event["uncollected_cost"])
            reconstructed = row["opening_balance"] + math.fsum(
                math.fsum(values) for values in amounts.values())
            closing = agents[aid].resources
            residual = closing - reconstructed
            if not math.isfinite(residual) or abs(residual) > 1e-9:
                raise ValueError(f"Unexplained resource residual year={year} researcher={aid}: {residual}")
            if require_closed and (row["closing_balance"] is None
                                   or row["closing_balance"] != closing):
                raise ValueError(f"Checkpoint closing balance mismatch: {year}/{aid}")
            result = dict(row, closing_balance=closing, closing_active=agents[aid].is_active,
                          residual=residual, uncollected_cost=math.fsum(uncollected))
            result.update({key: math.fsum(values) * (1 if key in INCOME_CATEGORIES else -1)
                           for key, values in amounts.items()})
            results.append(result)
        return results

    def close_year(self, year, population):
        rows = self.reconcile(year, population)
        self.years[str(year)] = {row["researcher_id"]: row for row in rows}
        return rows

    def to_dict(self):
        return deepcopy({
            "schema_version": SCHEMA_VERSION, "transactions": self.transactions,
            "years": self.years, "entry_years": self.entry_years,
        })

    def write(self, directory):
        """Replace complete exports, never append, so rollback/resume cannot duplicate."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        exports = {
            "resource_ledger.jsonl": self.transactions,
            "resource_balances.jsonl": [row for year in sorted(self.years, key=int)
                                       for row in self.years[year].values()],
        }
        for name, rows in exports.items():
            target = directory / name
            write_jsonl_atomic(target, rows, sort_keys=True, allow_nan=False)
