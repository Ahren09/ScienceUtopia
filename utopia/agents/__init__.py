"""Agent types, loaded on demand so policy utilities stay lightweight."""

from importlib import import_module

_EXPORTS = {'MultiAgentEcosystem': 'utopia.agents.base_agent', 'SimulationAgent': 'utopia.agents.base_agent', 'UniversityResearcher': 'utopia.agents.researcher_agents', 'IndustryResearcher': 'utopia.agents.researcher_agents', 'FreelancerAgent': 'utopia.agents.researcher_agents', 'FundingAgency': 'utopia.agents.funding_agents', 'FundingProgram': 'utopia.agents.funding_agents', 'IndustryFundingSystem': 'utopia.agents.funding_agents'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
