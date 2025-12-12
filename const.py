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
* You MUST keep your thinking concise and to the point. DO NOT generate excessive reasoning.
"""
