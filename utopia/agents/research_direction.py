"""
Research Direction Assignment System for Multi-Agent Scientific Ecosystem

Each agent has inherent expertise (3 research directions sampled at initialization).
When selecting research directions, agents randomly sample from their expertise.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import traceback

from pydantic import BaseModel

from utopia.data.paper_tracker import PaperTracker

logger = logging.getLogger(__name__)


class ResearchDirectionSelection(BaseModel):
    """Pydantic model for structured output of research direction selection"""
    topic: str
    detailed_focus: str
    reason: str


@dataclass
class ResearchDirection:
    """Represents a research direction with topic and focus areas

    All research directions have the same amortized gain per year,
    but differ in years (1-4 years).
    """
    topic: str
    focus_areas: List[str]
    years: int = 2  # Years needed to complete research (1-4)
    keywords: List[str] = None  # 10-15 keywords describing the research area
    related_topics: List[str] = None  # 3 related research topics

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'topic': self.topic,
            'focus_areas': self.focus_areas,
            'years': self.years,
            'keywords': self.keywords,
            'related_topics': self.related_topics
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ResearchDirection':
        """Deserialize from dictionary"""
        return cls(
            topic=data['topic'],
            focus_areas=data['focus_areas'],
            years=data.get('years', 2),
            keywords=data.get('keywords'),
            related_topics=data.get('related_topics')
        )


def initialize_agent_expertise(
        agents: List,
        directions: Optional[List[ResearchDirection]] = None,
        num_expertise: int = 3
) -> None:
    """Initialize each agent's inherent expertise by randomly sampling research directions

    Each agent receives `num_expertise` randomly sampled ResearchDirection objects
    that represent their inherent areas of expertise. These are stored in the
    agent's `expertise_directions` attribute.

    Args:
        agents: List of agent objects to initialize
        directions: Available research directions (defaults to AVAILABLE_DIRECTIONS)
        num_expertise: Number of expertise areas per agent (default: 3)
    """
    if directions is None:
        directions = AVAILABLE_DIRECTIONS

    for agent in agents:
        # Randomly sample num_expertise directions for this agent
        if hasattr(agent, 'expertise_directions'):
            agent.expertise_directions = random.sample(directions, min(num_expertise, len(directions)))


def build_direction_prompt(
        agent,  # SimulationAgent
        year: int,
        paper_tracker: PaperTracker,
        candidate_directions: List[ResearchDirection] = None,
        average_funding_level: float = None
) -> Tuple[str, Dict, List[str], List[str]]:
    """Build prompt for research direction selection (without calling LLM)

    Args:
        agent: Agent to build prompt for
        year: Current simulation year
        paper_tracker: PaperTracker for retrieving past papers
        candidate_directions: Optional list of candidate directions
        average_funding_level: Average funding level of all agents

    Returns:
        Tuple of (prompt, response_format, core_expertise, related_expertise)
    """
    if not candidate_directions:
        # Use filtered candidates from exploration experiment if available
        candidate_directions = getattr(agent, '_filtered_candidates', None) or agent.expertise
        
    core_expertise = []
    related_expertise = set()
    for direction in agent.expertise:
        core_expertise.append(direction.topic)
        related_expertise.update(direction.related_topics)

    related_expertise = list(related_expertise)

    past_accepted_papers, past_rejected_papers = paper_tracker.get_papers_by_decision(agent.id)

    prompt = f"""
{agent._get_personality_prompt(include_exploration_strategy=True)}

Now, you are trying to decide the research direction of your next project.
"""

    if len(past_accepted_papers) > 0:

        prompt += f"""## Your Past Publications (Accepted Papers)
The following are your past accepted papers.
You can use them to inspire your next project, such as deciding on research directions.
"""
        for idx_paper, paper in enumerate(past_accepted_papers):
            prompt += f"  - Paper {idx_paper + 1}: \"{paper.title}\" (Accepted in Conference {paper.conference}). Topic: {paper.topics}. Average score: {paper.review_history[-1]['score']:.2f}. Abstract: {paper.abstract[:300]} ...\n"

    if len(past_rejected_papers) > 0:
        prompt += f"""
## Your Past Rejected Papers
The following are your past rejected papers.
You can use them to learn from your mistakes and improve your next project, such as deciding on research directions or topics.
"""
    for idx_paper, paper in enumerate(past_rejected_papers):
        prompt += f"  - Paper {idx_paper + 1}: \"{paper.title}\". Topic: {paper.topics}. Average score: {paper.review_history[-1]['score']:.2f}. Abstract: {paper.abstract[:300]} ...\n"

    prompt += f"""
## Your Funding Situation
"""
    if agent.resources > average_funding_level:
        funding_comparison = "higher than"

    elif agent.resources < average_funding_level:
        funding_comparison = "lower than"

    else:
        funding_comparison = "equal to"

    prompt += f"Funding level: {agent.resources} ({funding_comparison} average funding level of {average_funding_level:.2f})"

    if agent.funding_success_history:
        funding_history = ""
        for funding_program_id, records in agent.funding_success_history.items():
            for record in records:
                funding_history += f"  - You received {record['amount']} units of funding from {funding_program_id} in year {record['year']}\n"

    else:
        funding_history = "No funding received yet."

    prompt += f"""
### Funding history
{funding_history}
"""

    # Add memory context
    if agent.memory_bank:
        prompt += "\n\n## Past Experiences\n\n"
        for memory in agent.memory_bank:
            if "reviews_received" in memory.get('type', ''):
                prompt += f"""### Year: {memory.get('year', '')} - Event: {memory.get('type')}
Thought: {memory.get('thought', '')}
"""
            elif memory.get('type', "") == "select_research_direction":
                prompt += f"""### Year: {memory.get('year', '')} - Event: I selected {memory.get('direction')} as my research direction. Thought: {memory.get('thought', '')}
"""

            else:
                raise NotImplementedError(f"Unknown memory type: {memory.get('type')}")

    prompt += f"""

## Directions You Can Choose From

Below are the candidate directions you can choose from. Please note that projects in all directions are expected to complete in either 1, 2, or 3 years.
"""
    # List candidate directions
    for i, direction in enumerate(candidate_directions):
        if isinstance(direction, ResearchDirection):
            prompt += f"""### Direction {i + 1}: {direction.topic}
- Focus Areas: {', '.join(direction.focus_areas[:3])}
- Expected Project Duration: {direction.years} year(s) 
- Keywords: {', '.join(direction.keywords[:8])}

"""
        else:
            prompt += f"* Direction {i + 1}: {direction}\n\n"

    strategy = getattr(agent, 'exploration_strategy', 'balanced')
    if strategy != 'balanced':
        # Strategy-filtered candidates: the ONLY valid choices are the listed
        # directions. Do not re-anchor the agent to its core expertise here —
        # that would contradict the strategy manipulation (plan 4.1/8.4).
        prompt += f"""## Your Task
Select exactly ONE research direction from the candidate directions listed above. Then, give a VERY DETAILED description of the research project using around 5 sentences. Mention the main techniques you will use.

## Strategy Constraint
Your research strategy is: {strategy.upper()}.
You must justify the distance of your chosen direction from your prior work: explain whether this direction is CLOSE, MODERATE, or FAR from your past research, and why it fits your strategy.

## Considerations
- Longer projects (3-4 years) require stable funding
- Only the candidate directions listed above are valid choices
"""
    else:
        prompt += f"""## Your Task
First, select one research direction that best aligns with your expertise, current situation, and long-term goals. Then, give a VERY DETAILED description of the research project using around 5 sentences. Mention the main techniques you will use.

Propose the directions based on your expertise:
- Your Core Expertise (most preferred): {', '.join(core_expertise)}
- Your Relevant Expertise (less preferred but still doable): {', '.join(related_expertise)}

## Considerations
- Longer projects (3-4 years) require stable funding
- Past successes suggest continuing in related areas
- Past failures might indicate need for pivoting or building stronger foundation
"""

    prompt += """
## Example Response
The response must contain 3 fields: "topic", "detailed_focus", and "reason".
```json
{
    "topic": "natural_language_processing", <- MUST be one of the candidate directions
    "detailed_focus": "I will develop methods that leverage reinforcement learning from human feedback (RLHF) to align large language models with clinical reasoning, ethical standards, and domain-specific decision-making in healthcare." <- 4-5 sentence detailed description of the research direction. mention the main techniques you will use.
    "reason": "This direction aligns with my recent publications, such as the paper titled 'Towards LLM Alignment using Reinforcement Learning from Human Feedback'." <- 2 sentence justification
}
```
"""

    response_format = {
        'type': 'json_schema',
        'json_object': {
            'name': 'research_direction_selection',
            'strict': True,
            'schema': ResearchDirectionSelection.model_json_schema()
        },
    }

    return prompt, response_format, core_expertise, related_expertise


def create_research_directions_batch(
        agents: List,  # List of SimulationAgent
        year: int,
        paper_tracker: PaperTracker,
        llm,  # LLM instance with generate_batch method
        average_funding_level: float = None,
        candidate_map: Optional[Dict[str, List[ResearchDirection]]] = None,
        fallback_counter: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Assign research directions for all agents at once using batched LLM calls

    This function builds all prompts first, then makes a single batched call
    to the LLM for maximum efficiency.

    Args:
        agents: List of agent objects to assign directions to
        year: Current simulation year
        paper_tracker: PaperTracker for retrieving past papers
        llm: LLM instance (must have generate_batch method)
        average_funding_level: Average funding level of all agents
        candidate_map: agent_id -> explicit candidate ResearchDirection list
            (exploration experiment). The prompt shows EXACTLY these candidates
            and the response is validated against EXACTLY their topics (plan 4.1).
            When None, legacy behavior (agent.expertise + related topics) applies.
        fallback_counter: optional dict updated in place with
            n_agents / n_parse_fallback / n_validation_fallback.

    Returns:
        Dictionary mapping agent_id to direction dict with 'direction', 'detailed_focus', 'reason'
    """
    if not agents:
        return {}

    # Step 1: Build all prompts (NO LLM calls yet)
    batch_prompts = []
    batch_metadata = []
    direction_response_format = None

    for agent in agents:
        candidates = (candidate_map or {}).get(agent.id)
        strict_candidates = candidates is not None
        prompt, response_format, core_expertise, related_expertise = build_direction_prompt(
            agent=agent,
            year=year,
            paper_tracker=paper_tracker,
            candidate_directions=candidates,
            average_funding_level=average_funding_level
        )
        if candidates is None:
            candidates = getattr(agent, '_filtered_candidates', None) or agent.expertise
        # Exactly the topics shown in the prompt
        candidate_topics = [d.topic if isinstance(d, ResearchDirection) else str(d)
                            for d in candidates]
        # Legacy (non-strict) mode additionally allows related-expertise topics
        allowed_topics = set(candidate_topics)
        if not strict_candidates:
            allowed_topics.update(core_expertise)
            allowed_topics.update(related_expertise)
        batch_prompts.append(prompt)
        batch_metadata.append({
            'agent': agent,
            'core_expertise': core_expertise,
            'related_expertise': related_expertise,
            'candidate_topics': candidate_topics,
            'allowed_topics': allowed_topics,
        })

        if direction_response_format is None:
            direction_response_format = response_format

    # Step 2: Single batch call for ALL prompts
    logger.info(f"[Direction] Generating {len(batch_prompts)} research directions in batch...")
    batch_results = llm.generate_batch(
        batch_prompts,
        temperature=0.7,
        response_format=direction_response_format,
        desc=f"{len(batch_prompts)} research directions",
        seed_ctx=('phase1_directions', year),
    )

    # Step 3: Process results
    if fallback_counter is not None:
        fallback_counter.setdefault('n_agents', 0)
        fallback_counter.setdefault('n_parse_fallback', 0)
        fallback_counter.setdefault('n_validation_fallback', 0)
    directions = {}
    for i, (selection, message_history) in enumerate(batch_results):
        meta = batch_metadata[i]
        agent = meta['agent']
        candidate_topics = meta['candidate_topics']
        allowed_topics = meta['allowed_topics']
        if fallback_counter is not None:
            fallback_counter['n_agents'] += 1
        # Fallback stays within the strategy-valid candidate set (plan 8.4)
        fallback_topic = candidate_topics[0] if candidate_topics else 'artificial_intelligence'

        # Handle failed JSON parse
        if selection is None:
            logger.warning(f"Failed to parse direction response for {agent.id}, using fallback")
            if fallback_counter is not None:
                fallback_counter['n_parse_fallback'] += 1
            selection = {
                'topic': fallback_topic,
                'detailed_focus': f"Research in {fallback_topic}",
                'reason': "Fallback due to parsing error"
            }

        # Validate response against exactly the topics shown to the model
        try:
            assert all(field in selection for field in {'topic', 'detailed_focus', 'reason'}), \
                f"Missing fields in response for {agent.id}: {list(selection.keys())}"

            assert selection['topic'] in allowed_topics, \
                f"Invalid topic {selection['topic']} for {agent.id}; allowed: {sorted(allowed_topics)}"

        except AssertionError as e:
            logger.warning(f"Invalid direction from {agent.id}: {e}, using fallback")
            if fallback_counter is not None:
                fallback_counter['n_validation_fallback'] += 1
            selection = {
                'topic': fallback_topic,
                'detailed_focus': f"Research in {fallback_topic}",
                'reason': "Fallback due to validation error"
            }

        # Update agent memory
        agent.memory_bank.append({
            "type": "select_research_direction",
            "direction": selection['topic'],
            "detailed_focus": selection['detailed_focus'],
            "thought": selection['reason'],
            "year": year,
            "timestamp": time.time(),
        })

        directions[agent.id] = {
            "direction": DIRECTIONS_DICT[selection['topic']],
            "detailed_focus": selection['detailed_focus'],
            "reason": selection['reason'],
        }

    logger.info(f"[Direction] Batch assignment complete: {len(directions)} directions assigned")
    return directions


def assign_directions_individual(
        agent,  # SimulationAgent
        year: int,
        paper_tracker: PaperTracker,
        candidate_directions: List[ResearchDirection] = None,
        average_funding_level: float = None
) -> Dict[str, ResearchDirection]:
    """Assign research directions one agent at a time using LLM-based selection

    Each agent uses LLM to select from their inherent expertise based on:
    - Past publication record (accepted/rejected papers)
    - Current funding situation
    - Recent review experiences
    - Strategic career considerations

    Args:
        agent: Agent object to assign direction to
        year: Current simulation year
        paper_tracker: PaperTracker for retrieving past papers
        candidate_directions: Optional list of candidate directions
        average_funding_level: Average funding level of all agents

    Returns:
        Dictionary with 'direction', 'detailed_focus', 'reason'
    """
    prompt, response_format, core_expertise, related_expertise = build_direction_prompt(
        agent=agent,
        year=year,
        paper_tracker=paper_tracker,
        candidate_directions=candidate_directions,
        average_funding_level=average_funding_level
    )

    candidate_directions_list = core_expertise + related_expertise

    for _ in range(agent.max_retries):

        try:
            selection, message_history = agent.llm.generate(prompt=prompt, response_format=response_format)
            assert all(field in selection for field in {'topic', 'detailed_focus',
                                                        'reason'}), f"Some fields are missing in the response of research direction selection. Existing fields: {str(list(selection.keys()))}"

            assert selection['topic'] in candidate_directions_list or selection[
                'topic'] in related_expertise, f"The selected topic {selection['topic']} is not in the candidate directions or related expertise. Candidate directions: {str(candidate_directions_list)}. Related expertise: {str(related_expertise)}"

            break

        except:
            traceback.print_exc()
            if _ >= agent.max_retries - 1:
                raise Exception("Failed to select research direction after retries")
            time.sleep(1)

    agent.memory_bank.append(
        {
            "type": "select_research_direction",
            "direction": selection['topic'],
            "detailed_focus": selection['detailed_focus'],
            "thought": selection['reason'],
            "year": year,
            "timestamp": time.time(),
        }
    )

    return {
        "direction": DIRECTIONS_DICT[selection['topic']],
        "detailed_focus": selection['detailed_focus'],
        "reason": selection['reason'],
    }


def get_strategy_filtered_candidate_directions(
    agent, all_directions, embedding_tracker, year, config
):
    """Filter candidate directions based on agent's exploration strategy.

    Uses embedding distances from career centroid to partition directions
    into near/mid/far buckets, then selects appropriate candidates.

    Args:
        agent: Agent with exploration_strategy attribute
        all_directions: List of all ResearchDirection objects
        embedding_tracker: EmbeddingTracker instance (may be None)
        year: Current simulation year
        config: exploration_experiment config dict

    Returns:
        List of ResearchDirection objects appropriate for the agent's strategy
    """
    strategy = getattr(agent, 'exploration_strategy', 'balanced')
    if strategy == 'balanced':
        return agent.expertise

    near_q = config.get('near_quantile', 0.33)
    far_q = config.get('far_quantile', 0.66)
    history_window = config.get('history_window_years', 3)

    # Try embedding-based filtering if tracker is available
    centroid = None
    if embedding_tracker is not None:
        centroid = embedding_tracker.compute_career_centroid(
            agent.id, max_years=history_window, current_year=year - 1
        )

    if centroid is not None:
        # Compute distances from centroid to all directions
        dist_map = embedding_tracker.compute_direction_distances(centroid, all_directions)
        if dist_map:
            sorted_dirs = sorted(all_directions, key=lambda d: dist_map.get(d.topic, 0.5))
            n = len(sorted_dirs)
            near_idx = max(1, int(n * near_q))
            far_idx = max(near_idx + 1, int(n * far_q))

            near = sorted_dirs[:near_idx]
            mid = sorted_dirs[near_idx:far_idx]
            far = sorted_dirs[far_idx:]

            if strategy == 'exploiter':
                candidates = near if len(near) >= 2 else near + mid
            elif strategy == 'cautious_explorer':
                candidates = mid if len(mid) >= 2 else near + mid
            else:  # explorer
                candidates = far if len(far) >= 2 else mid + far

            return candidates if candidates else agent.expertise

    # Heuristic fallback: use expertise topic names
    expertise_topics = {d.topic for d in agent.expertise}
    all_topics = {d.topic for d in all_directions}

    if strategy == 'exploiter':
        # Stay within expertise
        return agent.expertise
    elif strategy == 'cautious_explorer':
        # Expertise + related topics
        related_topics = set()
        for d in agent.expertise:
            related_topics.update(d.related_topics or [])
        candidate_topics = expertise_topics | related_topics
        candidates = [d for d in all_directions if d.topic in candidate_topics]
        return candidates if len(candidates) >= 2 else agent.expertise
    else:  # explorer
        # Everything outside expertise
        candidates = [d for d in all_directions if d.topic not in expertise_topics]
        return candidates if len(candidates) >= 2 else all_directions


AVAILABLE_DIRECTIONS = [
    # cs.AI - Artificial Intelligence (hot topic - multiple directions)
    # Fast-moving applied AI: 1-2 years
    ResearchDirection(
        topic="artificial_intelligence",
        focus_areas=["ontologies", "semantic networks", "reasoning systems"],
        years=1,
        keywords=["knowledge graphs", "semantic web", "ontology engineering", "description logic",
                  "reasoning engines", "inference mechanisms", "knowledge bases", "linked data",
                  "conceptual modeling", "formal semantics", "knowledge extraction", "taxonomies"],
        related_topics=["neural_symbolic_ai", "planning_and_scheduling", "multiagent_systems"]
    ),
    ResearchDirection(
        topic="planning_and_scheduling",
        focus_areas=["automated planning", "constraint satisfaction", "search algorithms"],
        years=2,
        keywords=["task planning", "resource allocation", "heuristic search", "optimization algorithms",
                  "scheduling problems", "constraint programming", "temporal planning", "partial order planning",
                  "hierarchical planning", "planners", "goal reasoning", "action selection", "decision making"],
        related_topics=["artificial_intelligence", "algorithmic_game_theory", "robotics"]
    ),
    ResearchDirection(
        topic="neural_symbolic_ai",
        focus_areas=["hybrid systems", "neuro symbolic integration", "logic based learning"],
        years=1,
        keywords=["symbolic reasoning", "neural networks", "differentiable logic", "inductive logic programming",
                  "rule extraction", "knowledge integration", "interpretable AI", "logical neural networks",
                  "symbolic grounding", "concept learning", "structured representations", "reasoning with neural nets"],
        related_topics=["artificial_intelligence", "deep_learning", "logic_in_cs"]
    ),

    # cs.AR - Hardware Architecture (requires fabrication time: 3 years)
    ResearchDirection(
        topic="computer_architecture",
        focus_areas=["processor design", "memory hierarchies", "hardware accelerators"],
        years=3,
        keywords=["chip design", "instruction set architecture", "cache systems", "pipeline optimization",
                  "parallel processing", "GPU architecture", "ASIC design", "system on chip",
                  "memory bandwidth", "performance modeling", "power efficiency", "hardware security", "FPGA"],
        related_topics=["operating_systems", "performance_analysis", "parallel_computing"]
    ),

    # cs.CC - Computational Complexity (deep theory: 4 years)
    ResearchDirection(
        topic="complexity_theory",
        focus_areas=["complexity classes", "lower bounds", "circuit complexity"],
        years=3,
        keywords=["P versus NP", "hardness of approximation", "derandomization", "proof complexity",
                  "computational hardness", "time complexity", "space complexity", "parameterized complexity",
                  "fine grained complexity", "boolean circuits", "complexity hierarchies", "reduction techniques"],
        related_topics=["algorithms", "discrete_mathematics", "logic_in_cs"]
    ),

    # cs.CE - Computational Engineering (standard engineering pace: 2 years)
    ResearchDirection(
        topic="computational_science",
        focus_areas=["scientific computing", "numerical methods", "simulation"],
        years=2,
        keywords=["high performance computing", "parallel algorithms", "multiscale modeling", "computational physics",
                  "finite element methods", "computational fluid dynamics", "molecular dynamics", "climate modeling",
                  "scientific visualization", "solver optimization", "computational chemistry",
                  "engineering simulation"],
        related_topics=["numerical_analysis", "parallel_computing", "performance_analysis"]
    ),

    # cs.CG - Computational Geometry (hot topic - multiple directions)
    # Algorithm-focused: 2-3 years
    ResearchDirection(
        topic="geometric_algorithms",
        focus_areas=["convex hulls", "voronoi diagrams", "mesh generation"],
        years=2,
        keywords=["computational geometry", "polygon triangulation", "spatial data structures",
                  "geometric optimization",
                  "proximity problems", "intersection algorithms", "point location", "delaunay triangulation",
                  "geometric search", "arrangement algorithms", "visibility problems", "motion planning"],
        related_topics=["computational_topology", "discrete_mathematics", "computer_graphics"]
    ),
    ResearchDirection(
        topic="computational_topology",
        focus_areas=["persistent homology", "shape analysis", "topological data analysis"],
        years=2,
        keywords=["topological features", "homology computation", "morse theory", "shape matching",
                  "topological signatures", "mapper algorithm", "filtration", "betti numbers",
                  "topological persistence", "shape descriptors", "surface reconstruction", "manifold learning"],
        related_topics=["geometric_algorithms", "discrete_mathematics", "data_structures"]
    ),

    # cs.CL - Computation and Language (fast-moving: 1 year)
    ResearchDirection(
        topic="natural_language_processing",
        focus_areas=["language modeling", "machine translation", "question answering"],
        years=1,
        keywords=["transformers", "large language models", "neural machine translation", "text generation",
                  "semantic parsing", "information extraction", "named entity recognition", "sentiment analysis",
                  "text classification", "language understanding", "contextual embeddings", "attention mechanisms",
                  "tokenization"],
        related_topics=["dialogue_systems", "deep_learning", "information_retrieval"]
    ),
    ResearchDirection(
        topic="dialogue_systems",
        focus_areas=["conversational ai", "task oriented dialogue", "response generation"],
        years=1,
        keywords=["chatbots", "dialogue management", "intent recognition", "slot filling", "conversation flow",
                  "dialogue state tracking", "response selection", "natural language generation", "context modeling",
                  "user modeling", "chitchat systems", "multi turn dialogue", "dialogue evaluation"],
        related_topics=["natural_language_processing", "human_computer_interaction", "multiagent_systems"]
    ),

    # cs.CR - Cryptography and Security
    # Cryptography theory: 3 years, Applied security: 1 year
    ResearchDirection(
        topic="cryptography",
        focus_areas=["public key systems", "zero knowledge proofs", "post quantum crypto"],
        years=3,
        keywords=["encryption algorithms", "digital signatures", "key exchange", "lattice cryptography",
                  "homomorphic encryption", "secure multiparty computation", "cryptographic protocols",
                  "blockchain security", "quantum resistant cryptography", "elliptic curve cryptography",
                  "hash functions"],
        related_topics=["cybersecurity", "information_theory", "formal_methods"]
    ),
    ResearchDirection(
        topic="cybersecurity",
        focus_areas=["threat detection", "network security", "privacy protection"],
        years=2,
        keywords=["intrusion detection", "malware analysis", "vulnerability assessment", "penetration testing",
                  "security monitoring", "incident response", "access control", "authentication systems",
                  "data privacy", "secure communication", "firewall systems", "security policies",
                  "threat intelligence"],
        related_topics=["cryptography", "computer_networks", "operating_systems"]
    ),

    # cs.CV - Computer Vision (hot topic - multiple directions)
    # Fast-moving applied vision: 1 year
    ResearchDirection(
        topic="image_recognition",
        focus_areas=["object detection", "semantic segmentation", "instance segmentation"],
        years=1,
        keywords=["convolutional neural networks", "image classification", "visual recognition", "feature extraction",
                  "bounding boxes", "region proposals", "mask generation", "panoptic segmentation",
                  "visual attention", "multi scale detection", "image understanding", "transfer learning"],
        related_topics=["deep_learning", "3d_vision", "computer_graphics"]
    ),
    ResearchDirection(
        topic="3d_vision",
        focus_areas=["depth estimation", "3d reconstruction", "point cloud processing"],
        years=1,
        keywords=["stereo vision", "structure from motion", "SLAM", "neural radiance fields",
                  "3D scene understanding", "volumetric reconstruction", "mesh generation", "depth sensors",
                  "point cloud segmentation", "3D object detection", "multiview geometry", "camera calibration"],
        related_topics=["image_recognition", "video_understanding", "robotics"]
    ),
    ResearchDirection(
        topic="video_understanding",
        focus_areas=["action recognition", "video captioning", "temporal modeling"],
        years=1,
        keywords=["video analysis", "temporal convolution", "optical flow", "activity recognition",
                  "video generation", "motion detection", "spatiotemporal features", "video summarization",
                  "event detection", "video segmentation", "frame interpolation", "video prediction",
                  "action localization"],
        related_topics=["image_recognition", "3d_vision", "multimedia_systems"]
    ),

    # cs.CY - Computers and Society (empirical studies: 2 years)
    ResearchDirection(
        topic="ai_ethics",
        focus_areas=["fairness", "accountability", "transparency"],
        years=2,
        keywords=["algorithmic bias", "ethical AI", "responsible AI", "AI governance", "fairness metrics",
                  "explainable AI", "algorithmic accountability", "discrimination detection", "bias mitigation",
                  "ethical frameworks", "AI policy", "social impact", "trustworthy AI", "value alignment"],
        related_topics=["interpretable_ml", "artificial_intelligence", "human_computer_interaction"]
    ),

    # cs.DB - Databases (systems work: 2 years)
    ResearchDirection(
        topic="database_systems",
        focus_areas=["query optimization", "distributed databases", "data management"],
        years=2,
        keywords=["SQL optimization", "indexing strategies", "transaction processing", "NoSQL databases",
                  "database architecture", "data warehousing", "OLAP", "query execution", "storage engines",
                  "database security", "data replication", "sharding", "consistency models"],
        related_topics=["distributed_systems", "information_retrieval", "data_structures"]
    ),

    # cs.DC - Distributed Computing (systems work: 2 years)
    ResearchDirection(
        topic="distributed_systems",
        focus_areas=["consensus algorithms", "fault tolerance", "distributed coordination"],
        years=2,
        keywords=["distributed protocols", "replicated state machines", "byzantine fault tolerance", "leader election",
                  "distributed transactions", "Paxos", "Raft", "eventual consistency", "distributed locks",
                  "clock synchronization", "failure detection", "distributed debugging", "microservices"],
        related_topics=["parallel_computing", "computer_networks", "database_systems"]
    ),
    ResearchDirection(
        topic="parallel_computing",
        focus_areas=["parallel algorithms", "gpu computing", "high performance computing"],
        years=2,
        keywords=["CUDA programming", "parallel programming models", "MPI", "OpenMP", "task parallelism",
                  "data parallelism", "load balancing", "scalability", "performance tuning", "heterogeneous computing",
                  "parallel numerical methods", "supercomputing", "cluster computing"],
        related_topics=["distributed_systems", "computer_architecture", "performance_analysis"]
    ),

    # cs.DL - Digital Libraries (applied systems: 2 years)
    ResearchDirection(
        topic="digital_libraries",
        focus_areas=["information retrieval", "document management", "metadata systems"],
        years=2,
        keywords=["digital archives", "content management", "bibliographic systems", "document indexing",
                  "metadata standards", "digital preservation", "repository systems", "scholarly communication",
                  "open access", "citation analysis", "full text search", "Dublin Core", "institutional repositories"],
        related_topics=["information_retrieval", "database_systems", "natural_language_processing"]
    ),

    # cs.DM - Discrete Mathematics (pure theory: 3 years)
    ResearchDirection(
        topic="discrete_mathematics",
        focus_areas=["graph theory", "combinatorics", "discrete optimization"],
        years=3,
        keywords=["graph algorithms", "network flows", "matching theory", "coloring problems",
                  "combinatorial optimization",
                  "enumeration", "extremal combinatorics", "ramsey theory", "algebraic graph theory",
                  "matroid theory", "integer programming", "polyhedral combinatorics", "probabilistic methods"],
        related_topics=["algorithms", "complexity_theory", "geometric_algorithms"]
    ),

    # cs.DS - Data Structures and Algorithms (theory with proofs: 3 years)
    ResearchDirection(
        topic="algorithms",
        focus_areas=["approximation algorithms", "randomized algorithms", "online algorithms"],
        years=3,
        keywords=["algorithm design", "algorithm analysis", "NP hard problems", "greedy algorithms",
                  "dynamic programming", "streaming algorithms", "sublinear algorithms", "competitive analysis",
                  "primal dual methods", "linear programming", "algorithm complexity", "hardness results"],
        related_topics=["complexity_theory", "discrete_mathematics", "data_structures"]
    ),
    ResearchDirection(
        topic="data_structures",
        focus_areas=["advanced data structures", "succinct data structures", "cache efficient structures"],
        years=3,
        keywords=["tree structures", "hash tables", "priority queues", "spatial data structures",
                  "persistent data structures",
                  "compressed data structures", "memory efficient structures", "I/O efficient algorithms",
                  "external memory algorithms", "dynamic data structures", "range queries", "dictionary structures"],
        related_topics=["algorithms", "database_systems", "computational_topology"]
    ),

    # cs.ET - Emerging Technologies (quantum: very long-term: 4 years)
    ResearchDirection(
        topic="quantum_computing",
        focus_areas=["quantum algorithms", "quantum error correction", "quantum simulation"],
        years=3,
        keywords=["quantum gates", "quantum circuits", "superconducting qubits", "quantum supremacy",
                  "variational quantum algorithms", "quantum annealing", "topological quantum computing",
                  "quantum machine learning", "quantum cryptography", "entanglement", "quantum noise",
                  "NISQ algorithms"],
        related_topics=["computer_architecture", "complexity_theory", "cryptography"]
    ),

    # cs.FL - Formal Languages and Automata (formal verification: 4 years)
    ResearchDirection(
        topic="formal_methods",
        focus_areas=["program verification", "model checking", "theorem proving"],
        years=3,
        keywords=["formal specification", "software verification", "temporal logic", "automated reasoning",
                  "proof assistants", "bounded model checking", "SAT solvers", "SMT solvers", "invariant generation",
                  "abstract interpretation", "symbolic execution", "program synthesis", "correctness proofs"],
        related_topics=["logic_in_cs", "programming_languages", "software_engineering"]
    ),

    # cs.GL - General Literature (education research: 2 years)
    ResearchDirection(
        topic="computing_education",
        focus_areas=["cs pedagogy", "educational technology", "learning analytics"],
        years=2,
        keywords=["programming education", "computational thinking", "coding bootcamps", "online learning",
                  "student assessment", "curriculum design", "learning outcomes", "educational games",
                  "teaching methods", "CS1 courses", "active learning", "peer instruction", "MOOCs"],
        related_topics=["human_computer_interaction", "software_engineering", "interdisciplinary_cs"]
    ),

    # cs.GR - Graphics (applied systems: 2 years)
    ResearchDirection(
        topic="computer_graphics",
        focus_areas=["rendering", "animation", "geometry processing"],
        years=2,
        keywords=["ray tracing", "rasterization", "shader programming", "physically based rendering",
                  "real time rendering", "character animation", "procedural generation", "texture mapping",
                  "lighting models", "mesh processing", "surface modeling", "visual effects", "GPU graphics"],
        related_topics=["image_recognition", "3d_vision", "geometric_algorithms"]
    ),

    # cs.GT - Computer Science and Game Theory (theory: 3 years)
    ResearchDirection(
        topic="algorithmic_game_theory",
        focus_areas=["mechanism design", "auction theory", "equilibrium computation"],
        years=3,
        keywords=["Nash equilibrium", "incentive compatibility", "price of anarchy", "voting theory",
                  "social choice", "market design", "combinatorial auctions", "fair division",
                  "strategic behavior", "payoff matrices", "cooperative games", "game tree search"],
        related_topics=["multiagent_systems", "planning_and_scheduling", "artificial_intelligence"]
    ),

    # cs.HC - Human-Computer Interaction (empirical: 2 years)
    ResearchDirection(
        topic="human_computer_interaction",
        focus_areas=["user interface design", "interaction techniques", "usability studies"],
        years=2,
        keywords=["user experience", "interface design", "usability testing", "accessibility", "gesture interaction",
                  "touchscreen interfaces", "virtual reality interfaces", "augmented reality", "user centered design",
                  "interaction design", "mobile interfaces", "information visualization", "human factors"],
        related_topics=["dialogue_systems", "computer_graphics", "ai_ethics"]
    ),

    # cs.IR - Information Retrieval (applied: 1 year)
    ResearchDirection(
        topic="information_retrieval",
        focus_areas=["search engines", "ranking algorithms", "query understanding"],
        years=1,
        keywords=["web search", "relevance ranking", "document retrieval", "indexing methods", "search quality",
                  "query expansion", "learning to rank", "semantic search", "personalized search",
                  "recommendation systems", "text mining", "entity retrieval", "BM25"],
        related_topics=["natural_language_processing", "database_systems", "digital_libraries"]
    ),

    # cs.IT - Information Theory (deep theory: 3 years)
    ResearchDirection(
        topic="information_theory",
        focus_areas=["coding theory", "compression", "channel capacity"],
        years=3,
        keywords=["Shannon entropy", "mutual information", "error correcting codes", "data compression",
                  "source coding", "channel coding", "rate distortion theory", "network information theory",
                  "LDPC codes", "turbo codes", "lossless compression", "lossy compression", "capacity theorems"],
        related_topics=["cryptography", "data_structures", "multimedia_systems"]
    ),

    # cs.LG - Machine Learning (hot topic - multiple directions)
    # Fast-moving applied ML: 1 year
    ResearchDirection(
        topic="deep_learning",
        focus_areas=["neural architecture search", "efficient training", "transfer learning"],
        years=1,
        keywords=["neural networks", "backpropagation", "optimization", "batch normalization", "dropout",
                  "residual networks", "attention mechanisms", "model compression", "knowledge distillation",
                  "pre training", "fine tuning", "multi task learning", "domain adaptation"],
        related_topics=["reinforcement_learning", "generative_models", "image_recognition"]
    ),
    ResearchDirection(
        topic="reinforcement_learning",
        focus_areas=["policy optimization", "multi agent rl", "model based rl"],
        years=1,
        keywords=["Q learning", "policy gradients", "actor critic", "deep RL", "reward shaping",
                  "exploration strategies", "temporal difference learning", "value functions",
                  "Markov decision processes",
                  "model free RL", "sample efficiency", "off policy learning", "experience replay"],
        related_topics=["deep_learning", "robotics", "planning_and_scheduling"]
    ),
    ResearchDirection(
        topic="generative_models",
        focus_areas=["diffusion models", "gans", "variational autoencoders"],
        years=1,
        keywords=["image generation", "text generation", "score based models", "denoising diffusion",
                  "adversarial training", "latent variable models", "energy based models", "flow models",
                  "conditional generation", "data augmentation", "synthetic data", "likelihood estimation"],
        related_topics=["deep_learning", "image_recognition", "computer_graphics"]
    ),
    ResearchDirection(
        topic="meta-learning",
        focus_areas=["few-shot learning", "meta-learning", "learning to learn", "task adaptation"],
        years=1,
        keywords=["rapid adaptation", "MAML", "prototypical networks", "metric learning", "transfer learning",
                  "multi task learning", "continual learning", "neural architecture search",
                  "hyperparameter optimization",
                  "zero-shot learning", "cross domain learning", "model agnostic methods"],
        related_topics=["deep_learning", "reinforcement_learning", "interpretable_ml"]
    ),
    ResearchDirection(
        topic="interpretable_ml",
        focus_areas=["explainable ai", "model interpretability", "feature attribution"],
        years=1,
        keywords=["model explanations", "feature importance", "attention visualization", "saliency maps",
                  "LIME", "SHAP", "counterfactual explanations", "model transparency", "decision rules",
                  "interpretable representations", "concept learning", "neural network visualization", "trust in AI"],
        related_topics=["deep_learning", "ai_ethics", "artificial_intelligence"]
    ),

    # cs.LO - Logic in Computer Science (deep theory: 4 years)
    ResearchDirection(
        topic="logic_in_cs",
        focus_areas=["modal logic", "temporal logic", "constraint logic"],
        years=3,
        keywords=["propositional logic", "predicate logic", "logical inference", "satisfiability",
                  "proof theory", "model theory", "automated theorem proving", "logic programming",
                  "description logics", "epistemic logic", "deontic logic", "linear temporal logic"],
        related_topics=["formal_methods", "neural_symbolic_ai", "complexity_theory"]
    ),

    # cs.MA - Multiagent Systems (systems work: 2 years)
    ResearchDirection(
        topic="multiagent_systems",
        focus_areas=["coordination", "cooperation", "agent communication"],
        years=2,
        keywords=["agent architectures", "multi agent planning", "distributed AI", "agent negotiation",
                  "coalition formation", "agent protocols", "emergent behavior", "swarm intelligence",
                  "agent based modeling", "team formation", "communication protocols", "agent cooperation"],
        related_topics=["artificial_intelligence", "algorithmic_game_theory", "distributed_systems"]
    ),

    # cs.MM - Multimedia (applied systems: 2 years)
    ResearchDirection(
        topic="multimedia_systems",
        focus_areas=["video processing", "audio processing", "multimodal learning"],
        years=2,
        keywords=["video encoding", "audio signal processing", "multimedia databases", "streaming systems",
                  "multimedia retrieval", "video compression", "audio video synchronization", "media formats",
                  "cross modal learning", "audio visual fusion", "multimedia indexing", "content delivery"],
        related_topics=["video_understanding", "audio_computing", "computer_graphics"]
    ),

    # cs.MS - Mathematical Software (systems: 2 years)
    ResearchDirection(
        topic="mathematical_software",
        focus_areas=["symbolic computation", "numerical software", "computational algebra"],
        years=2,
        keywords=["computer algebra systems", "mathematical libraries", "numerical solvers", "symbolic differentiation",
                  "polynomial computation", "algebraic algorithms", "Groebner bases", "mathematical optimization",
                  "linear algebra packages", "differential equation solvers", "interval arithmetic",
                  "exact arithmetic"],
        related_topics=["symbolic_computation", "numerical_analysis", "computational_science"]
    ),

    # cs.NA - Numerical Analysis (applied math: 3 years)
    ResearchDirection(
        topic="numerical_analysis",
        focus_areas=["numerical linear algebra", "optimization methods", "differential equations"],
        years=3,
        keywords=["iterative methods", "matrix decomposition", "eigenvalue problems", "convex optimization",
                  "nonlinear optimization", "gradient methods", "finite difference methods", "spectral methods",
                  "convergence analysis", "numerical stability", "condition numbers", "preconditioning"],
        related_topics=["computational_science", "mathematical_software", "algorithms"]
    ),

    # cs.NE - Neural and Evolutionary Computing (applied: 2 years)
    ResearchDirection(
        topic="evolutionary_computation",
        focus_areas=["genetic algorithms", "evolutionary strategies", "neuroevolution"],
        years=2,
        keywords=["population based optimization", "fitness functions", "crossover operators", "mutation operators",
                  "selection mechanisms", "multi objective optimization", "coevolution", "genetic programming",
                  "differential evolution", "evolution of neural networks", "artificial life", "swarm optimization"],
        related_topics=["deep_learning", "algorithmic_game_theory", "meta-learning"]
    ),

    # cs.NI - Networking and Internet Architecture (systems: 2 years)
    ResearchDirection(
        topic="computer_networks",
        focus_areas=["network protocols", "wireless networks", "network optimization"],
        years=2,
        keywords=["TCP IP", "routing algorithms", "network topology", "software defined networking",
                  "network congestion", "quality of service", "mobile networks", "ad hoc networks",
                  "internet architecture", "network security", "bandwidth management", "5G networks", "IoT networks"],
        related_topics=["distributed_systems", "cybersecurity", "operating_systems"]
    ),

    # cs.OH - Other Computer Science (interdisciplinary: 2 years)
    ResearchDirection(
        topic="interdisciplinary_cs",
        focus_areas=["computational biology", "computational social science", "digital humanities"],
        years=2,
        keywords=["bioinformatics", "genomics", "protein folding", "network science", "text mining",
                  "social network analysis", "digital archives", "computational linguistics", "medical informatics",
                  "systems biology", "cultural analytics", "data driven humanities", "interdisciplinary computing"],
        related_topics=["computing_education", "natural_language_processing", "social_networks"]
    ),

    # cs.OS - Operating Systems (systems: 2 years)
    ResearchDirection(
        topic="operating_systems",
        focus_areas=["kernel design", "resource management", "virtualization"],
        years=2,
        keywords=["process scheduling", "memory management", "file systems", "device drivers",
                  "system calls", "containerization", "hypervisors", "virtual machines", "OS security",
                  "real time systems", "distributed operating systems", "microkernel", "system performance"],
        related_topics=["computer_architecture", "distributed_systems", "performance_analysis"]
    ),

    # cs.PF - Performance (systems: 2 years)
    ResearchDirection(
        topic="performance_analysis",
        focus_areas=["performance modeling", "benchmarking", "profiling"],
        years=2,
        keywords=["performance optimization", "bottleneck analysis", "latency reduction", "throughput improvement",
                  "performance metrics", "workload characterization", "system monitoring", "tracing tools",
                  "capacity planning", "performance tuning", "scalability analysis", "resource utilization"],
        related_topics=["parallel_computing", "operating_systems", "computer_architecture"]
    ),

    # cs.PL - Programming Languages (language design: 3 years)
    ResearchDirection(
        topic="programming_languages",
        focus_areas=["language design", "compilers", "type systems"],
        years=3,
        keywords=["syntax design", "semantics", "static analysis", "compiler optimization", "code generation",
                  "type inference", "functional programming", "object oriented programming",
                  "domain specific languages",
                  "runtime systems", "garbage collection", "language interoperability", "program transformation"],
        related_topics=["formal_methods", "software_engineering", "symbolic_computation"]
    ),

    # cs.RO - Robotics (hardware integration: 3 years)
    ResearchDirection(
        topic="robotics",
        focus_areas=["robot perception", "motion planning", "robot learning"],
        years=3,
        keywords=["sensor fusion", "computer vision for robotics", "path planning", "obstacle avoidance",
                  "manipulation", "grasping", "localization", "mapping", "autonomous navigation",
                  "human robot interaction", "robot control", "kinematics", "dynamics", "imitation learning"],
        related_topics=["reinforcement_learning", "3d_vision", "control_systems"]
    ),

    # cs.SC - Symbolic Computation (systems: 3 years)
    ResearchDirection(
        topic="symbolic_computation",
        focus_areas=["computer algebra", "automated reasoning", "symbolic integration"],
        years=3,
        keywords=["symbolic mathematics", "algebraic simplification", "equation solving", "symbolic differentiation",
                  "polynomial manipulation", "rational functions", "indefinite integration", "definite integration",
                  "special functions", "expression manipulation", "formula manipulation", "algebraic transformations"],
        related_topics=["mathematical_software", "programming_languages", "logic_in_cs"]
    ),

    # cs.SD - Sound (applied signal processing: 2 years)
    ResearchDirection(
        topic="audio_computing",
        focus_areas=["speech processing", "music information retrieval", "sound synthesis"],
        years=2,
        keywords=["speech recognition", "speaker identification", "audio features", "music classification",
                  "audio segmentation", "pitch detection", "rhythm analysis", "audio synthesis", "sound effects",
                  "audio enhancement", "noise reduction", "acoustic modeling", "audio signal processing"],
        related_topics=["multimedia_systems", "natural_language_processing", "deep_learning"]
    ),

    # cs.SE - Software Engineering (empirical systems: 2 years)
    ResearchDirection(
        topic="software_engineering",
        focus_areas=["software testing", "program analysis", "software maintenance"],
        years=2,
        keywords=["test automation", "unit testing", "integration testing", "code review", "static analysis",
                  "dynamic analysis", "bug detection", "code refactoring", "technical debt", "software quality",
                  "continuous integration", "version control", "software metrics", "development methodologies"],
        related_topics=["programming_languages", "formal_methods", "computing_education"]
    ),

    # cs.SI - Social and Information Networks (data science: 2 years)
    ResearchDirection(
        topic="social_networks",
        focus_areas=["network analysis", "community detection", "influence propagation"],
        years=2,
        keywords=["graph mining", "social network analysis", "link prediction", "centrality measures",
                  "clustering algorithms", "information diffusion", "viral marketing", "opinion dynamics",
                  "network topology", "temporal networks", "ego networks", "social influence", "network evolution"],
        related_topics=["interdisciplinary_cs", "multiagent_systems", "information_retrieval"]
    ),

    # cs.SY - Systems and Control (control theory: 2 years)
    ResearchDirection(
        topic="control_systems",
        focus_areas=["optimal control", "adaptive control", "cyber physical systems"],
        years=2,
        keywords=["feedback control", "state estimation", "Kalman filtering", "model predictive control",
                  "robust control", "nonlinear control", "stability analysis", "control design",
                  "embedded systems", "real time control", "sensor networks", "actuator control",
                  "system identification"],
        related_topics=["robotics", "parallel_computing", "computational_science"]
    ),
]

# Create a dictionary mapping from topic to ResearchDirection for easy lookup
DIRECTIONS_DICT = {direction.topic: direction for direction in AVAILABLE_DIRECTIONS}
