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
        'academic_base_budget': 20, 
        'industry_base_budget': 20, 
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

    # LLM generation settings
    'llm': {
        'max_tokens': 2048,            # Default max tokens for generate_batch()
        'enable_thinking': True,      # Enable thinking mode for Qwen3 models
    },

    # Simulation settings
    'simulation': {
        'default_num_years': 5,            # Default simulation duration
        'papers_per_year_per_agent': 1,    # Target papers per agent per year
    },

    # Exploration vs Exploitation Experiment
    'exploration_experiment': {
        'strategies': ['explorer', 'exploiter', 'cautious_explorer'],
        'strategy_mix': {'explorer': 0.2, 'exploiter': 0.6, 'cautious_explorer': 0.2},
        'embedding_model': 'all-MiniLM-L6-v2',
        'num_keywords': 10,
        'near_quantile': 0.33,   # empirical percentile of direction-pair distances (see EmbeddingTracker.compute_direction_thresholds)
        'far_quantile': 0.66,
        'history_window_years': 3,

        # Balanced university-only population (neutral main condition)
        'population_mode': 'university_only',
        'num_institutions': 12,
        'researchers_per_institution': 5,
        'institution_strategy_mix': ['explorer', 'exploiter', 'exploiter', 'exploiter', 'cautious_explorer'],
        'initial_funding': 100,
        'num_conferences': 6,

        # Mechanism toggles. All OFF = neutral main condition: strategy affects
        # direction choice only; no direct score/funding/citation adjustment.
        'review_strategy_adjustment': False,  # synthetic stress test only, never part of main results
        'funding_intervention': False,
        'citation_intervention': False,
        'lambda_citation': 0.0,               # weight of log(citation potential) in candidate reranking
        'citation_potential_clip': (0.5, 3.0),

        # Legacy mechanism parameters (apply ONLY when the toggles above are on)
        'explorer_success_bias': -0.05,
        'explorer_variance': 0.25,
        'exploiter_success_bias': 0.05,
        'exploiter_variance': 0.08,
        'cautious_success_bias': 0.0,
        'cautious_variance': 0.15,
        'novelty_penalty_in_funding': 0.15,   # lambda_funding for the funding-conservatism intervention
        'novelty_reward_in_long_term_citations': 0.30,
        'breakthrough_multiplier_mean': 1.8,
        'breakthrough_probability_threshold': 0.7,

        # Outcome measurement
        'hit_paper_percentile': 0.95,
        'citation_age_windows': [0, 1, 2, 3, 5],
        'cd_age_windows': [3, 5],
        'cd_min_denominator': 3,
        'min_citations_for_cd': 5,  # legacy current-year CD gate (kept for backward compat)

        # Scale controls (deterministic; may change between E3 and E4)
        'funding_panel_max_apps': 25,   # max applications ranked in one funding prompt

        # Prompt versions: any post-freeze edit requires a new version string
        'prompt_versions': {
            'direction_selection': 'v1',
            'paper_selection': 'v1',
            'citation_selection': 'v1',
            'review': 'v1',
            'funding_evaluation': 'v1',
            'keyword_extraction': 'v1',
        },
    },

    # Preferential Attachment Experiment
    'preferential_attachment_experiment': {
        'max_new_collaborations_per_year': 2,
        'min_similarity_threshold': 0.1,
        'top_k_candidates': 3,
        'collaboration_mode': 'cross_institute',  # 'intra_institute' or 'cross_institute'
        'max_collaborations_per_agent': 1,  # per round
    },

    # Weights & Biases logging
    'wandb': {
        'enabled': False,          # Optional scalar logging, enabled explicitly.
        'project': 'science-utopia',
        'run_name': None,           # Auto-generated from experiment_name + model if None
    },
}
