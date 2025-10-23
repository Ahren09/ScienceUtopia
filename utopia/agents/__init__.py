"""
Multi-Agent Scientific Ecosystem Framework

This module extends the AI-Scientist-v2 infrastructure with multi-agent capabilities
WITHOUT modifying any original files. All classes extend existing AI-Scientist components.
"""

from .base_agent import MultiAgentEcosystem, SimulationAgent
from .researcher_agents import UniversityResearcher, IndustryResearcher, FreelancerAgent
from .funding_agents import FundingAgency, FundingProgram, IndustryFundingSystem

__all__ = [
    'MultiAgentEcosystem',
    'SimulationAgent',
    'UniversityResearcher',
    'IndustryResearcher',
    'FreelancerAgent',
    'FundingAgency',
    'FundingProgram',
    'IndustryFundingSystem'
]