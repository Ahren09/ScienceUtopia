# Research and publication costs for multi-agent simulation
RESEARCH_COST = 5  # Cost deducted from agent resources for generating research ideas
PAPER_SUBMISSION_COST = 15  # Cost deducted from agent resources for writing and submitting papers

# Agent resource limits
MAX_FUNDING_LEVEL = 100  # Maximum funding level for any agent
MIN_FUNDING_LEVEL = 0    # Minimum funding level (agents cannot go below this)

# Simulation parameters
MIN_RESOURCES_FOR_RESEARCH = 10  # Minimum resources required to conduct research
MIN_RESOURCES_FOR_SUBMISSION = 20  # Minimum resources required to submit a paper

REVIEW_CRITERIA = """
- 5: Award-worthy (top 2.5%)
- 4: Strong accept (top 10%)
- 3: Borderline accept (top 30%)
- 2: Needs major revisions (top 50%)
- 1: Reject

NOTE: 
* This is a VERY rigorous top-tier conference. >60% papers receive a score of 2.0-2.5 or lower. Use this to calibrate your scores. Be careful about giving high scores.
* Scores in decimals like 1.5 is allowed.
"""

IMPORTANT_NOTES = """
## Important Notes
* You MUST keep your thinking concise, meaningfuls and to the point. DO NOT generate excessive but meaningless reasoning.
"""

COI_NOTE = "(NOTE: This paper is from your own institution ({institution}). Potential conflict of interest: they may be competing for the same funding sources as you.)"
# Frozen standardized reviewer policy (identical to review-replay prereg_v1
# prompt_construction.policy_text_A). Used by --review_policy standardized: it
# replaces the reviewer's persona/memory block; paper block, scale and JSON
# schema are unchanged.
STANDARDIZED_REVIEW_POLICY = (
    "You apply the conference's standard review policy: weigh methodological soundness, "
    "novelty, clarity, significance, and feasibility evenly. Base your score only on the "
    "paper's content as presented, applying the scoring rubric consistently and impartially."
)
