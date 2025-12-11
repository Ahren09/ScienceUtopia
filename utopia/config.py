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
        'resubmission_cost': 5, # Fixed cost for resubmitting a paper
        'annual_cost': 10, # Fixed cost for each year of research
    },

    # Funding agency settings
    'funding': {
        # Fixed-rate mode: percentage of applications to fund
        'default_funding_rate': 0.3,     # 25% funding rate if not specified
        'nsf_funding_rate': 0.3,         # NSF: more generous (foundational research)
        'darpa_funding_rate': 0.20,       # DARPA: more selective (applied research)
        'academic_base_budget': 50, 
        'industry_base_budget': 50, 
        'funding_growth_rate': 0.1,
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

    # 
    'citation': {
        'expected_citation_per_round': 2,      # Expected numbers of citations generated per round for each paper. If lower than this number, we try to cite rejected papers.
        'num_candidate_papers': 20,      # Number of candidate papers to consider for citations.
        'initial_success_rate': 0.5,       # Initial assumed success rate
    },

    # Simulation settings
    'simulation': {
        'default_num_years': 5,            # Default simulation duration
        'papers_per_year_per_agent': 1,    # Target papers per agent per year
    },

    # Exploration vs Exploitation Experiment
    'exploration_experiment': {
        'strategies': ['explorer', 'exploiter', 'cautious_explorer'],
        'embedding_model': 'all-MiniLM-L6-v2',
        'num_keywords': 10,
        'min_citations_for_cd': 5,  # Minimum citations to calculate CD index
    }
}

