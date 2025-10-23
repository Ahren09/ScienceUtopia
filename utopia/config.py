"""
Global Configuration for Science Utopia Simulation

This module contains default parameters for the simulation that can be overridden
by specific agents, conferences, or funding programs.

Usage:
    from utopia.config import SIMULATION_CONFIG

    # Get default conference acceptance rate
    acceptance_rate = SIMULATION_CONFIG['conference']['default_acceptance_rate']

    # Get funding rate
    funding_rate = SIMULATION_CONFIG['funding']['default_funding_rate']
"""

SIMULATION_CONFIG = {
    # Conference settings
    'conference': {
        'acceptance_rate': 0.3,  # 25% acceptance rate if not specified
        'use_default_acceptance_rate': True,
    },

    # Funding agency settings
    'funding': {
        # Fixed-rate mode: percentage of applications to fund
        'default_funding_rate': 0.3,     # 25% funding rate if not specified
        'nsf_funding_rate': 0.3,         # NSF: more generous (foundational research)
        'darpa_funding_rate': 0.20,       # DARPA: more selective (applied research)
        'academic_base_budget': 50, 
        'industry_base_budget': 50, 
    },

    # Agent resource settings
    'agent': {
        # Initial resources
        'university_initial_funding': 70,
        'industry_initial_funding': 90,
        'freelancer_initial_funding': 30,

        # Resource bounds
        'min_funding': 0,
        'max_funding': 100,

        # Initial reputation
        'university_initial_reputation': 5,
        'industry_initial_reputation': 6,
        'freelancer_initial_reputation': 4,
        'funding_agent_initial_reputation': 8,
    },
    
    # Paper generation settings
    'paper': {
        'min_funding_threshold': 20,       # Min funding to write a paper
        'funding_cost_per_paper': 15,      # Funding consumed per paper
    },

    # Review settings
    'review': {
        'base_score_range': (1, 10),       # Score range for reviews
        'confidence_range': (1, 5),        # Confidence range for reviewers
    },

    # Success history tracking
    'learning': {
        'success_history_alpha': 0.3,      # Exponential moving average weight
        'initial_success_rate': 0.5,       # Initial assumed success rate
    },

    # Simulation settings
    'simulation': {
        'default_num_years': 5,            # Default simulation duration
        'papers_per_year_per_agent': 1,    # Target papers per agent per year
    }
}

