"""
Simplified Agent Review System

Internal implementation of paper review pipeline for multi-agent ecosystem.
Replaces external AgentReview framework dependency.
"""

from .review_pipeline import AgentReviewPipeline
from .prompts import ReviewPrompts

__all__ = ['AgentReviewPipeline', 'ReviewPrompts']