"""Prospective RANDOM versus TOPIC full-text review, with P1F1 funding fixed.

Both arms use the native seeded assignment helper without a reviewer capacity
cap. TOPIC means coarse category overlap, not a new learned expertise score.
Observations live in each yearly_results entry under "reviewer_matching" and therefore
in native portable checkpoints. No extra replayable files or inference calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
from math import ceil
from statistics import mean, pvariance

from utopia.funding.feedback import FundingFeedbackMixin, Mechanisms
from utopia.runtime.checkpointing import PortableCheckpointError

PROTOCOL = "reviewer-matching-fulltext-portable-v1"
CELLS = {"RANDOM": "random", "TOPIC": "topic"}


def protocol_rules(matching):
    if matching not in CELLS.values():
        raise ValueError("Reviewer matching must be random or topic")
    return {
        "protocol": PROTOCOL, "matching": matching, "funding_cell": "P1F1",
        "assignment": "Simulation._assign_reviewers_with_capacity in both arms",
        "seed_namespace": "derive_seed(seed, 'phase3_assign', year)",
        "reviewer_capacity": None, "reviews_per_paper": 3,
        "eligibility": "native can_review roster minus all authors and their COIs",
        "shortfall": "three or available eligible reviewers; zero eligible aborts",
        "expertise": "native expertise-topic and paper-arxiv-tag category overlap",
        "topic_fallback": "random unmatched eligible reviewers fill remaining slots",
        "random_fallback": "not applicable to uniform random assignment",
        "project_durations": "native mixed durations", "annual_influx_rate": 0.0,
        "workload_denominator": "all can_review reviewers at assignment, including zero load",
        "dispersion": "population variance per paper with at least two completed reviews",
        "history": "immutable annual scalar/ID records, including later inactive reviewers",
        "uncertainty": "one seed and one world per condition; no world-level significance",
    }


def workload_summary(loads):
    """Include zero-load can_review reviewers; undefined shares remain null."""
    total = sum(loads)
    top_n = ceil(len(loads) / 10)
    return {
        "n_reviewers": len(loads), "reviews": total,
        "zero_load_reviewers": loads.count(0),
        "mean_load": mean(loads) if loads else None,
        "max_load": max(loads) if loads else None,
        "top_decile_reviewers": top_n,
        "top_decile_share": sum(sorted(loads, reverse=True)[:top_n]) / total if total else None,
    }


class ReviewerMatchingMixin:
    """Add one treatment and observations, inheriting native reviews and funding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_reviewer_rules()
        self._capacity_mode_on = True

    def check_reviewer_rules(self):
        protocol_rules(self.reviewer_matching)
        if self.reviewer_capacity is not None or self.reviews_per_paper != 3:
            raise ValueError("Reviewer comparison requires capacity=None and three reviews")
        if self.mechanisms != Mechanisms.from_cell("P1F1"):
            raise ValueError("Reviewer comparison fixes funding at P1F1")

    def extra_checkpoint_state(self):
        state = super().extra_checkpoint_state()
        state["reviewer_matching"] = protocol_rules(self.reviewer_matching)
        return state

    def restore_extra_checkpoint_state(self, state):
        state = dict(state)
        if state.pop("reviewer_matching", None) != protocol_rules(self.reviewer_matching):
            raise PortableCheckpointError("Reviewer treatment/protocol differs from checkpoint")
        super().restore_extra_checkpoint_state(state)
        self.check_reviewer_rules()
        self._capacity_mode_on = True

    def _assign_reviewers_with_capacity(self, year, year_results):
        assignment = super()._assign_reviewers_with_capacity(year, year_results)
        # Observe the same native category definition in either arm. No sampling.
        from utopia.agents.conference import CONFERENCES, TOPIC_TO_CATEGORY_MAPPING
        tag_categories = defaultdict(set)
        for conference in CONFERENCES.values():
            for tag in conference.topics:
                tag_categories[tag.lower()].add(conference.category)
        available = self.ecosystem.get_available_reviewers()
        categories = {
            reviewer.id: {TOPIC_TO_CATEGORY_MAPPING.get(direction.topic, "Interdisciplinary")
                          for direction in getattr(reviewer, "expertise", [])}
            for reviewer in available
        }
        reviewers = {reviewer.id: {"reviewer_id": reviewer.id, "assigned": 0,
                                  "matched": 0, "completed": 0}
                     for reviewer in available}
        papers = []
        for conference in self.conference_system.conferences:
            for paper in conference.submitted_papers:
                paper_categories = set()
                for tag in paper.get("tags", []):
                    paper_categories.update(tag_categories.get(str(tag).lower(), ()))
                assigned = [reviewer.id for reviewer in assignment[paper["id"]]]
                matched = [rid for rid in assigned if categories[rid] & paper_categories]
                for rid in assigned:
                    reviewers[rid]["assigned"] += 1
                    reviewers[rid]["matched"] += int(rid in matched)
                papers.append({
                    "paper_id": paper["id"], "conference_id": conference.conference_id,
                    "assigned_reviewer_ids": assigned, "matched_reviewer_ids": matched,
                    "unmatched_reviewer_ids": [rid for rid in assigned if rid not in matched],
                })
        assigned = sum(row["assigned"] for row in reviewers.values())
        matched = sum(row["matched"] for row in reviewers.values())
        year_results["reviewer_matching"] = {
            "protocol": PROTOCOL, "year": year, "matching": self.reviewer_matching,
            "n_submissions": len(papers), "n_eligible_reviewers": len(available),
            "assigned_reviews": assigned, "matched_reviews": matched,
            "unmatched_reviews": assigned - matched,
            "matched_fraction": matched / assigned if assigned else None,
            "fallback_reviews": assigned - matched if self.reviewer_matching == "topic" else None,
            "fallback_fraction": ((assigned - matched) / assigned
                                  if self.reviewer_matching == "topic" and assigned else None),
            "papers_below_target": sum(len(row["assigned_reviewer_ids"]) < 3 for row in papers),
            "assignment_workload": workload_summary([row["assigned"] for row in reviewers.values()]),
            "reviewer_rows": list(reviewers.values()), "paper_rows": papers,
        }
        return assignment

    def _run_phase_3_peer_review(self, year, year_results):
        self.check_reviewer_rules()
        if not hasattr(self.llm, "generate_batch"):
            raise ValueError("Reviewer comparison requires the native batched review path")
        # The stock random arm otherwise bypasses the seeded assignment helper.
        self._capacity_mode_on = True
        averages = super()._run_phase_3_peer_review(year, year_results)
        observations = year_results["reviewer_matching"]
        reviewers = {row["reviewer_id"]: row for row in observations["reviewer_rows"]}
        conferences = {conf.conference_id: conf for conf in self.conference_system.conferences}
        variances = []
        for paper in observations["paper_rows"]:
            reviews = conferences[paper["conference_id"]].reviews.get(paper["paper_id"], [])
            scores = [review["overall_score"] for review in reviews]
            for review in reviews:
                reviewers[review["reviewer_id"]]["completed"] += 1
            variance = pvariance(scores) if len(scores) >= 2 else None
            if variance is not None:
                variances.append(variance)
            paper.update(
                completed_reviews=len(scores),
                reviews=[{"reviewer_id": review["reviewer_id"], "score": review["overall_score"]}
                         for review in reviews],
                score_mean=mean(scores) if scores else None,
                score_variance=variance,
                score_range=max(scores) - min(scores) if len(scores) >= 2 else None,
            )
        observations["completed_workload"] = workload_summary(
            [row["completed"] for row in observations["reviewer_rows"]])
        observations["completed_reviews"] = observations["completed_workload"]["reviews"]
        observations["papers_with_dispersion"] = len(variances)
        observations["mean_within_paper_variance"] = mean(variances) if variances else None
        return averages


def simulation_class(base=None):
    if base is None:
        from utopia.simulation import Simulation
        base = Simulation
    return type("ReviewerMatchingSimulation", (ReviewerMatchingMixin, FundingFeedbackMixin, base), {})


def configure_reviewers(args, *, cell):
    """Called by the shared runner before the scientific binding is serialized."""
    if cell not in CELLS:
        raise ValueError("Reviewer cell must be RANDOM or TOPIC")
    for name, value in {"annual_influx_rate": 0.0, "disable_resubmission": False,
                        "papers_per_project": 1, "population_mode": "university_only",
                        "num_years": 6, "expected_founders": 5000,
                        "strategy_mix": "balanced", "num_conferences": 6}.items():
        if getattr(args, name) != value:
            raise ValueError(f"Reviewer comparison differs from the fixed {name}")
    args.reviewer_matching = CELLS[cell]
    args.reviewer_capacity = None
    args.reviews_per_paper = 3
    return {"reviewer_matching": protocol_rules(args.reviewer_matching)}


def run_prospective(options):
    from utopia.experiments import funding_feedback
    for name, value in {"founders": 5000, "years": 6, "seed": 42, "thinking": False,
                        "max_tokens": 32768, "context_tokens": 196608}.items():
        if getattr(options, name) != value:
            raise ValueError(f"Reviewer comparison requires {name}={value}")
    if options.cell not in CELLS:
        raise ValueError("Reviewer cell must be RANDOM or TOPIC")
    options.protocol, options.funding_cell = PROTOCOL, "P1F1"
    return funding_feedback.run_prospective(
        options, simulation_factory=simulation_class,
        configure=partial(configure_reviewers, cell=options.cell))


def main(argv=None):
    from utopia.experiments.funding_feedback import prospective_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run-fulltext")
    prospective_arguments(command)
    command.set_defaults(founders=5000, years=6, thinking=False,
                         max_tokens=32768, context_tokens=196608)
    for action in command._actions:
        if action.dest == "cell":
            action.choices = tuple(CELLS)
    return run_prospective(parser.parse_args(argv))


if __name__ == "__main__":
    main()
