"""
Research Direction Assignment System for Multi-Agent Scientific Ecosystem

Supports two modes for assigning research directions:
1. Batch mode: Assign research directions to all agents simultaneously
2. Individual mode: Assign research directions one agent at a time

This enables flexible simulation of different academic scenarios.
"""

from typing import List, Dict, Optional, Callable
import random
from dataclasses import dataclass


@dataclass
class ResearchDirection:
    """Represents a research direction with topic and focus areas

    All research directions have the same amortized gain per year,
    but differ in years (1-4 years).
    """

    topic: str
    focus_areas: List[str]
    years: int = 2  # Years needed to complete research (1-4)


class ResearchDirectionAssigner:
    """Assigns research directions to agents in the ecosystem

    Supports both batch and individual assignment modes to enable
    different simulation scenarios.
    """

    def __init__(
        self,
        mode: str = "batch",
        available_directions: Optional[List[ResearchDirection]] = None,
        assignment_strategy: Optional[Callable] = None
    ):
        """Initialize the research direction assigner

        Args:
            mode: Assignment mode - "batch" or "individual"
            available_directions: List of research directions to assign from
            assignment_strategy: Optional custom strategy function for assignment
        """
        if mode not in ["batch", "individual"]:
            raise ValueError(f"mode must be 'batch' or 'individual', got: {mode}")
        self.mode = mode
        self.available_directions = available_directions or self._create_default_directions()
        self.assignment_strategy = assignment_strategy or self._default_assignment_strategy
        self.assignment_history: List[Dict] = []

    def _create_default_directions(self) -> List[ResearchDirection]:
        """Create default research directions based on arXiv CS taxonomy

        Returns:
            List of ResearchDirection objects covering major CS topics
        """
        return [
            # cs.AI - Artificial Intelligence (hot topic - multiple directions)
            # Fast-moving applied AI: 1-2 years
            ResearchDirection(
                topic="knowledge_representation",
                focus_areas=["ontologies", "semantic_networks", "reasoning_systems"],
                years=2
            ),
            ResearchDirection(
                topic="planning_and_scheduling",
                focus_areas=["automated_planning", "constraint_satisfaction", "search_algorithms"],
                years=2
            ),
            ResearchDirection(
                topic="neural_symbolic_ai",
                focus_areas=["hybrid_systems", "neuro_symbolic_integration", "logic_based_learning"],
                years=2
            ),

            # cs.AR - Hardware Architecture (requires fabrication time: 3 years)
            ResearchDirection(
                topic="computer_architecture",
                focus_areas=["processor_design", "memory_hierarchies", "hardware_accelerators"],
                years=3
            ),

            # cs.CC - Computational Complexity (deep theory: 4 years)
            ResearchDirection(
                topic="complexity_theory",
                focus_areas=["complexity_classes", "lower_bounds", "circuit_complexity"],
                years=4
            ),

            # cs.CE - Computational Engineering (standard engineering pace: 2 years)
            ResearchDirection(
                topic="computational_science",
                focus_areas=["scientific_computing", "numerical_methods", "simulation"],
                years=2
            ),

            # cs.CG - Computational Geometry (hot topic - multiple directions)
            # Algorithm-focused: 2-3 years
            ResearchDirection(
                topic="geometric_algorithms",
                focus_areas=["convex_hulls", "voronoi_diagrams", "mesh_generation"],
                years=2
            ),
            ResearchDirection(
                topic="computational_topology",
                focus_areas=["persistent_homology", "shape_analysis", "topological_data_analysis"],
                years=3
            ),

            # cs.CL - Computation and Language (fast-moving: 1 year)
            ResearchDirection(
                topic="natural_language_processing",
                focus_areas=["language_modeling", "machine_translation", "question_answering"],
                years=1
            ),
            ResearchDirection(
                topic="dialogue_systems",
                focus_areas=["conversational_ai", "task_oriented_dialogue", "response_generation"],
                years=1
            ),

            # cs.CR - Cryptography and Security
            # Cryptography theory: 3 years, Applied security: 1 year
            ResearchDirection(
                topic="cryptography",
                focus_areas=["public_key_systems", "zero_knowledge_proofs", "post_quantum_crypto"],
                years=3
            ),
            ResearchDirection(
                topic="cybersecurity",
                focus_areas=["threat_detection", "network_security", "privacy_protection"],
                years=1
            ),

            # cs.CV - Computer Vision (hot topic - multiple directions)
            # Fast-moving applied vision: 1 year
            ResearchDirection(
                topic="image_recognition",
                focus_areas=["object_detection", "semantic_segmentation", "instance_segmentation"],
                years=1
            ),
            ResearchDirection(
                topic="3d_vision",
                focus_areas=["depth_estimation", "3d_reconstruction", "point_cloud_processing"],
                years=1
            ),
            ResearchDirection(
                topic="video_understanding",
                focus_areas=["action_recognition", "video_captioning", "temporal_modeling"],
                years=1
            ),

            # cs.CY - Computers and Society (empirical studies: 2 years)
            ResearchDirection(
                topic="ai_ethics",
                focus_areas=["fairness", "accountability", "transparency"],
                years=2
            ),

            # cs.DB - Databases (systems work: 2 years)
            ResearchDirection(
                topic="database_systems",
                focus_areas=["query_optimization", "distributed_databases", "data_management"],
                years=2
            ),

            # cs.DC - Distributed Computing (systems work: 2 years)
            ResearchDirection(
                topic="distributed_systems",
                focus_areas=["consensus_algorithms", "fault_tolerance", "distributed_coordination"],
                years=2
            ),
            ResearchDirection(
                topic="parallel_computing",
                focus_areas=["parallel_algorithms", "gpu_computing", "high_performance_computing"],
                years=2
            ),

            # cs.DL - Digital Libraries (applied systems: 2 years)
            ResearchDirection(
                topic="digital_libraries",
                focus_areas=["information_retrieval", "document_management", "metadata_systems"],
                years=2
            ),

            # cs.DM - Discrete Mathematics (pure theory: 3 years)
            ResearchDirection(
                topic="discrete_mathematics",
                focus_areas=["graph_theory", "combinatorics", "discrete_optimization"],
                years=3
            ),

            # cs.DS - Data Structures and Algorithms (theory with proofs: 3 years)
            ResearchDirection(
                topic="algorithms",
                focus_areas=["approximation_algorithms", "randomized_algorithms", "online_algorithms"],
                years=3
            ),
            ResearchDirection(
                topic="data_structures",
                focus_areas=["advanced_data_structures", "succinct_data_structures", "cache_efficient_structures"],
                years=3
            ),

            # cs.ET - Emerging Technologies (quantum: very long-term: 4 years)
            ResearchDirection(
                topic="quantum_computing",
                focus_areas=["quantum_algorithms", "quantum_error_correction", "quantum_simulation"],
                years=4
            ),

            # cs.FL - Formal Languages and Automata (formal verification: 4 years)
            ResearchDirection(
                topic="formal_methods",
                focus_areas=["program_verification", "model_checking", "theorem_proving"],
                years=4
            ),

            # cs.GL - General Literature (education research: 2 years)
            ResearchDirection(
                topic="computing_education",
                focus_areas=["cs_pedagogy", "educational_technology", "learning_analytics"],
                years=2
            ),

            # cs.GR - Graphics (applied systems: 2 years)
            ResearchDirection(
                topic="computer_graphics",
                focus_areas=["rendering", "animation", "geometry_processing"],
                years=2
            ),

            # cs.GT - Computer Science and Game Theory (theory: 3 years)
            ResearchDirection(
                topic="algorithmic_game_theory",
                focus_areas=["mechanism_design", "auction_theory", "equilibrium_computation"],
                years=3
            ),

            # cs.HC - Human-Computer Interaction (empirical: 2 years)
            ResearchDirection(
                topic="human_computer_interaction",
                focus_areas=["user_interface_design", "interaction_techniques", "usability_studies"],
                years=2
            ),

            # cs.IR - Information Retrieval (applied: 1 year)
            ResearchDirection(
                topic="information_retrieval",
                focus_areas=["search_engines", "ranking_algorithms", "query_understanding"],
                years=1
            ),

            # cs.IT - Information Theory (deep theory: 3 years)
            ResearchDirection(
                topic="information_theory",
                focus_areas=["coding_theory", "compression", "channel_capacity"],
                years=3
            ),

            # cs.LG - Machine Learning (hot topic - multiple directions)
            # Fast-moving applied ML: 1 year
            ResearchDirection(
                topic="deep_learning",
                focus_areas=["neural_architecture_search", "efficient_training", "transfer_learning"],
                years=1
            ),
            ResearchDirection(
                topic="reinforcement_learning",
                focus_areas=["policy_optimization", "multi_agent_rl", "model_based_rl"],
                years=1
            ),
            ResearchDirection(
                topic="generative_models",
                focus_areas=["diffusion_models", "gans", "variational_autoencoders"],
                years=1
            ),
            ResearchDirection(
                topic="meta_learning",
                focus_areas=["few_shot_learning", "learning_to_learn", "task_adaptation"],
                years=1
            ),
            ResearchDirection(
                topic="interpretable_ml",
                focus_areas=["explainable_ai", "model_interpretability", "feature_attribution"],
                years=1
            ),

            # cs.LO - Logic in Computer Science (deep theory: 4 years)
            ResearchDirection(
                topic="logic_in_cs",
                focus_areas=["modal_logic", "temporal_logic", "constraint_logic"],
                years=4
            ),

            # cs.MA - Multiagent Systems (systems work: 2 years)
            ResearchDirection(
                topic="multiagent_systems",
                focus_areas=["coordination", "cooperation", "agent_communication"],
                years=2
            ),

            # cs.MM - Multimedia (applied systems: 2 years)
            ResearchDirection(
                topic="multimedia_systems",
                focus_areas=["video_processing", "audio_processing", "multimodal_learning"],
                years=2
            ),

            # cs.MS - Mathematical Software (systems: 2 years)
            ResearchDirection(
                topic="mathematical_software",
                focus_areas=["symbolic_computation", "numerical_software", "computational_algebra"],
                years=2
            ),

            # cs.NA - Numerical Analysis (applied math: 3 years)
            ResearchDirection(
                topic="numerical_analysis",
                focus_areas=["numerical_linear_algebra", "optimization_methods", "differential_equations"],
                years=3
            ),

            # cs.NE - Neural and Evolutionary Computing (applied: 2 years)
            ResearchDirection(
                topic="evolutionary_computation",
                focus_areas=["genetic_algorithms", "evolutionary_strategies", "neuroevolution"],
                years=2
            ),

            # cs.NI - Networking and Internet Architecture (systems: 2 years)
            ResearchDirection(
                topic="computer_networks",
                focus_areas=["network_protocols", "wireless_networks", "network_optimization"],
                years=2
            ),

            # cs.OH - Other Computer Science (interdisciplinary: 2 years)
            ResearchDirection(
                topic="interdisciplinary_cs",
                focus_areas=["computational_biology", "computational_social_science", "digital_humanities"],
                years=2
            ),

            # cs.OS - Operating Systems (systems: 2 years)
            ResearchDirection(
                topic="operating_systems",
                focus_areas=["kernel_design", "resource_management", "virtualization"],
                years=2
            ),

            # cs.PF - Performance (systems: 2 years)
            ResearchDirection(
                topic="performance_analysis",
                focus_areas=["performance_modeling", "benchmarking", "profiling"],
                years=2
            ),

            # cs.PL - Programming Languages (language design: 3 years)
            ResearchDirection(
                topic="programming_languages",
                focus_areas=["language_design", "compilers", "type_systems"],
                years=3
            ),

            # cs.RO - Robotics (hardware integration: 3 years)
            ResearchDirection(
                topic="robotics",
                focus_areas=["robot_perception", "motion_planning", "robot_learning"],
                years=3
            ),

            # cs.SC - Symbolic Computation (systems: 3 years)
            ResearchDirection(
                topic="symbolic_computation",
                focus_areas=["computer_algebra", "automated_reasoning", "symbolic_integration"],
                years=3
            ),

            # cs.SD - Sound (applied signal processing: 2 years)
            ResearchDirection(
                topic="audio_computing",
                focus_areas=["speech_processing", "music_information_retrieval", "sound_synthesis"],
                years=2
            ),

            # cs.SE - Software Engineering (empirical systems: 2 years)
            ResearchDirection(
                topic="software_engineering",
                focus_areas=["software_testing", "program_analysis", "software_maintenance"],
                years=2
            ),

            # cs.SI - Social and Information Networks (data science: 2 years)
            ResearchDirection(
                topic="social_networks",
                focus_areas=["network_analysis", "community_detection", "influence_propagation"],
                years=2
            ),

            # cs.SY - Systems and Control (control theory: 2 years)
            ResearchDirection(
                topic="control_systems",
                focus_areas=["optimal_control", "adaptive_control", "cyber_physical_systems"],
                years=2
            ),
        ]

    def assign_directions_batch(
        self,
        agents: List,
        year: int = 0
    ) -> Dict[str, ResearchDirection]:
        """Assign research directions to all agents simultaneously (batch mode)

        All agents receive their research directions at the same time,
        simulating a synchronized research cycle (e.g., all labs starting
        projects after a funding cycle).

        Args:
            agents: List of agent objects to assign directions to
            year: Current simulation year

        Returns:
            Dictionary mapping agent_id to assigned ResearchDirection
        """
        assignments = {}

        for agent in agents:
            # Apply assignment strategy
            direction = self.assignment_strategy(agent, self.available_directions)

            assignments[agent.id] = direction

            # Record assignment
            self._record_assignment(agent.id, direction, year, "batch")

        return assignments

    def assign_direction_individual(
        self,
        agent,
        year: int = 0,
        context: Optional[Dict] = None
    ) -> ResearchDirection:
        """Assign research direction to a single agent (individual mode)

        Agents receive directions one at a time, potentially allowing
        for sequential decision-making (e.g., later agents can see what
        earlier agents chose).

        Args:
            agent: Agent object to assign direction to
            year: Current simulation year
            context: Optional context dictionary (e.g., previous assignments)

        Returns:
            Assigned ResearchDirection
        """
        # Apply assignment strategy with context
        direction = self.assignment_strategy(
            agent,
            self.available_directions,
            context=context
        )

        # Record assignment
        self._record_assignment(agent.id, direction, year, "individual")

        return direction

    def assign_directions(
        self,
        agents: List,
        year: int = 0,
        context: Optional[Dict] = None
    ) -> Dict[str, ResearchDirection]:
        """Assign research directions using the configured mode

        Args:
            agents: List of agent objects
            year: Current simulation year
            context: Optional context for individual mode

        Returns:
            Dictionary mapping agent_id to assigned ResearchDirection
        """
        if self.mode == "batch":
            return self.assign_directions_batch(agents, year)
        else:
            # Individual mode: assign one at a time
            assignments = {}
            individual_context = context or {}

            for agent in agents:
                # Update context with previous assignments
                individual_context['previous_assignments'] = assignments.copy()

                direction = self.assign_direction_individual(agent, year, individual_context)
                assignments[agent.id] = direction

            return assignments

    def _default_assignment_strategy(
        self,
        agent,
        directions: List[ResearchDirection],
        context: Optional[Dict] = None
    ) -> ResearchDirection:
        """Default strategy for assigning research directions

        Takes into account agent specializations and context.

        Args:
            agent: Agent object
            directions: Available research directions
            context: Optional context (used in individual mode)

        Returns:
            Selected ResearchDirection
        """
        # Match agent specializations to directions
        agent_specializations = set(
            spec.lower().replace('_', ' ')
            for spec in getattr(agent, 'specializations', [])
        )

        # Score each direction based on match
        scored_directions = []
        for direction in directions:
            score = 0.0

            # Check topic match
            topic_match = any(
                spec in direction.topic.lower() or direction.topic.lower() in spec
                for spec in agent_specializations
            )
            if topic_match:
                score += 1.0

            # Check focus area match
            for focus in direction.focus_areas:
                focus_match = any(
                    spec in focus.lower() or focus.lower() in spec
                    for spec in agent_specializations
                )
                if focus_match:
                    score += 0.5

            # In individual mode, penalize overcrowded topics
            if context and 'previous_assignments' in context:
                previous = context['previous_assignments'].values()
                topic_count = sum(1 for d in previous if d.topic == direction.topic)
                # Penalty increases with overcrowding
                score -= topic_count * 0.3

            scored_directions.append((direction, score))

        # Select direction with highest score (with some randomness)
        scored_directions.sort(key=lambda x: x[1], reverse=True)

        # Top 3 candidates - introduce some randomness
        top_candidates = scored_directions[:3]
        if top_candidates:
            # Weighted random selection from top candidates
            weights = [score for _, score in top_candidates]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(top_candidates)] * len(top_candidates)

            selected_direction = random.choices(
                [d for d, _ in top_candidates],
                weights=weights,
                k=1
            )[0]
        else:
            # Fallback: random selection
            selected_direction = random.choice(directions)

        return selected_direction

    def _record_assignment(
        self,
        agent_id: str,
        direction: ResearchDirection,
        year: int,
        mode: str
    ):
        """Record an assignment for tracking purposes

        Args:
            agent_id: ID of the agent
            direction: Assigned research direction
            year: Simulation year
            mode: Assignment mode used
        """
        self.assignment_history.append({
            'agent_id': agent_id,
            'direction': direction,
            'year': year,
            'mode': mode,
            'timestamp': __import__('time').time()
        })

    def get_assignment_statistics(self) -> Dict:
        """Get statistics about research direction assignments

        Returns:
            Dictionary with assignment statistics
        """
        if not self.assignment_history:
            return {}

        topic_counts = {}
        for record in self.assignment_history:
            topic = record['direction'].topic
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return {
            'total_assignments': len(self.assignment_history),
            'unique_agents': len(set(r['agent_id'] for r in self.assignment_history)),
            'topic_distribution': topic_counts,
            'mode_used': self.mode
        }

    def clear_history(self):
        """Clear assignment history"""
        self.assignment_history = []


def create_batch_assigner(
    available_directions: Optional[List[ResearchDirection]] = None
) -> ResearchDirectionAssigner:
    """Create a research direction assigner in batch mode

    Args:
        available_directions: Optional list of research directions

    Returns:
        ResearchDirectionAssigner configured for batch mode
    """
    return ResearchDirectionAssigner(
        mode="batch",
        available_directions=available_directions
    )


def create_individual_assigner(
    available_directions: Optional[List[ResearchDirection]] = None
) -> ResearchDirectionAssigner:
    """Create a research direction assigner in individual mode

    Args:
        available_directions: Optional list of research directions

    Returns:
        ResearchDirectionAssigner configured for individual mode
    """
    return ResearchDirectionAssigner(
        mode="individual",
        available_directions=available_directions
    )
