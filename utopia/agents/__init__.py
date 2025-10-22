"""
Multi-Agent Scientific Ecosystem Framework

This module extends the AI-Scientist-v2 infrastructure with multi-agent capabilities
WITHOUT modifying any original files. All classes extend existing AI-Scientist components.
"""

from .base_agent import MultiAgentEcosystem, SimulationAgent
from .specialized_agents import UniversityResearcher, IndustryResearcher, FreelancerAgent, FundingAgent

__all__ = [
    'MultiAgentEcosystem',
    'SimulationAgent', 
    'UniversityResearcher',
    'IndustryResearcher',
    'FreelancerAgent',
    'FundingAgent'
]