"""
Research Direction Assignment System for Multi-Agent Scientific Ecosystem

Each agent has inherent expertise (3 research directions sampled at initialization).
When selecting research directions, agents randomly sample from their expertise.
"""

from typing import List, Dict, Optional
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
    keywords: List[str] = None  # 10-15 keywords describing the research area


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


def assign_directions_batch(
    agents: List,
    directions: Optional[List[ResearchDirection]] = None
) -> Dict[str, ResearchDirection]:
    """Assign research directions to all agents simultaneously

    Each agent randomly selects from their inherent expertise (expertise_directions).
    All agents receive directions at the same time, simulating a synchronized
    research cycle (e.g., all labs starting projects after a funding cycle).

    Args:
        agents: List of agent objects to assign directions to
        directions: Available research directions (only used if agents don't have expertise initialized)

    Returns:
        Dictionary mapping agent_id to assigned ResearchDirection
    """
    if directions is None:
        directions = AVAILABLE_DIRECTIONS

    assignments = {}
    for agent in agents:
        # Randomly select from agent's expertise
        assignments[agent.id] = random.choice(agent.expertise)

    return assignments


def assign_directions_individual(
    agents: List,
    directions: Optional[List[ResearchDirection]] = None
) -> Dict[str, ResearchDirection]:
    """Assign research directions one agent at a time

    Each agent randomly selects from their inherent expertise (expertise_directions).
    Agents receive directions sequentially. This mode exists for compatibility but
    behaves the same as batch mode since each agent only samples from their own expertise.

    Args:
        agents: List of agent objects to assign directions to
        directions: Available research directions (only used if agents don't have expertise initialized)

    Returns:
        Dictionary mapping agent_id to assigned ResearchDirection
    """
    # In the new system, individual assignment is identical to batch assignment
    # since each agent only samples from their own expertise
    return assign_directions_batch(agents, directions)


AVAILABLE_DIRECTIONS = [
    # cs.AI - Artificial Intelligence (hot topic - multiple directions)
    # Fast-moving applied AI: 1-2 years
    ResearchDirection(
        topic="knowledge_representation",
        focus_areas=["ontologies", "semantic networks", "reasoning systems"],
        years=1,
        keywords=["knowledge graphs", "semantic web", "ontology engineering", "description logic",
                  "reasoning engines", "inference mechanisms", "knowledge bases", "linked data",
                  "conceptual modeling", "formal semantics", "knowledge extraction", "taxonomies"]
    ),
    ResearchDirection(
        topic="planning_and_scheduling",
        focus_areas=["automated planning", "constraint satisfaction", "search algorithms"],
        years=2,
        keywords=["task planning", "resource allocation", "heuristic search", "optimization algorithms",
                  "scheduling problems", "constraint programming", "temporal planning", "partial order planning",
                  "hierarchical planning", "planners", "goal reasoning", "action selection", "decision making"]
    ),
    ResearchDirection(
        topic="neural_symbolic_ai",
        focus_areas=["hybrid systems", "neuro symbolic integration", "logic based learning"],
        years=1,
        keywords=["symbolic reasoning", "neural networks", "differentiable logic", "inductive logic programming",
                  "rule extraction", "knowledge integration", "interpretable AI", "logical neural networks",
                  "symbolic grounding", "concept learning", "structured representations", "reasoning with neural nets"]
    ),

    # cs.AR - Hardware Architecture (requires fabrication time: 3 years)
    ResearchDirection(
        topic="computer_architecture",
        focus_areas=["processor design", "memory hierarchies", "hardware accelerators"],
        years=3,
        keywords=["chip design", "instruction set architecture", "cache systems", "pipeline optimization",
                  "parallel processing", "GPU architecture", "ASIC design", "system on chip",
                  "memory bandwidth", "performance modeling", "power efficiency", "hardware security", "FPGA"]
    ),

    # cs.CC - Computational Complexity (deep theory: 4 years)
    ResearchDirection(
        topic="complexity_theory",
        focus_areas=["complexity classes", "lower bounds", "circuit complexity"],
        years=4,
        keywords=["P versus NP", "hardness of approximation", "derandomization", "proof complexity",
                  "computational hardness", "time complexity", "space complexity", "parameterized complexity",
                  "fine grained complexity", "boolean circuits", "complexity hierarchies", "reduction techniques"]
    ),

    # cs.CE - Computational Engineering (standard engineering pace: 2 years)
    ResearchDirection(
        topic="computational_science",
        focus_areas=["scientific computing", "numerical methods", "simulation"],
        years=2,
        keywords=["high performance computing", "parallel algorithms", "multiscale modeling", "computational physics",
                  "finite element methods", "computational fluid dynamics", "molecular dynamics", "climate modeling",
                  "scientific visualization", "solver optimization", "computational chemistry", "engineering simulation"]
    ),

    # cs.CG - Computational Geometry (hot topic - multiple directions)
    # Algorithm-focused: 2-3 years
    ResearchDirection(
        topic="geometric_algorithms",
        focus_areas=["convex hulls", "voronoi diagrams", "mesh generation"],
        years=2,
        keywords=["computational geometry", "polygon triangulation", "spatial data structures", "geometric optimization",
                  "proximity problems", "intersection algorithms", "point location", "delaunay triangulation",
                  "geometric search", "arrangement algorithms", "visibility problems", "motion planning"]
    ),
    ResearchDirection(
        topic="computational_topology",
        focus_areas=["persistent homology", "shape analysis", "topological data analysis"],
        years=2,
        keywords=["topological features", "homology computation", "morse theory", "shape matching",
                  "topological signatures", "mapper algorithm", "filtration", "betti numbers",
                  "topological persistence", "shape descriptors", "surface reconstruction", "manifold learning"]
    ),

    # cs.CL - Computation and Language (fast-moving: 1 year)
    ResearchDirection(
        topic="natural_language_processing",
        focus_areas=["language modeling", "machine translation", "question answering"],
        years=1,
        keywords=["transformers", "large language models", "neural machine translation", "text generation",
                  "semantic parsing", "information extraction", "named entity recognition", "sentiment analysis",
                  "text classification", "language understanding", "contextual embeddings", "attention mechanisms", "tokenization"]
    ),
    ResearchDirection(
        topic="dialogue_systems",
        focus_areas=["conversational ai", "task oriented dialogue", "response generation"],
        years=1,
        keywords=["chatbots", "dialogue management", "intent recognition", "slot filling", "conversation flow",
                  "dialogue state tracking", "response selection", "natural language generation", "context modeling",
                  "user modeling", "chitchat systems", "multi turn dialogue", "dialogue evaluation"]
    ),

    # cs.CR - Cryptography and Security
    # Cryptography theory: 3 years, Applied security: 1 year
    ResearchDirection(
        topic="cryptography",
        focus_areas=["public key systems", "zero knowledge proofs", "post quantum crypto"],
        years=3,
        keywords=["encryption algorithms", "digital signatures", "key exchange", "lattice cryptography",
                  "homomorphic encryption", "secure multiparty computation", "cryptographic protocols",
                  "blockchain security", "quantum resistant cryptography", "elliptic curve cryptography", "hash functions"]
    ),
    ResearchDirection(
        topic="cybersecurity",
        focus_areas=["threat detection", "network security", "privacy protection"],
        years=2,
        keywords=["intrusion detection", "malware analysis", "vulnerability assessment", "penetration testing",
                  "security monitoring", "incident response", "access control", "authentication systems",
                  "data privacy", "secure communication", "firewall systems", "security policies", "threat intelligence"]
    ),

    # cs.CV - Computer Vision (hot topic - multiple directions)
    # Fast-moving applied vision: 1 year
    ResearchDirection(
        topic="image_recognition",
        focus_areas=["object detection", "semantic segmentation", "instance segmentation"],
        years=1,
        keywords=["convolutional neural networks", "image classification", "visual recognition", "feature extraction",
                  "bounding boxes", "region proposals", "mask generation", "panoptic segmentation",
                  "visual attention", "multi scale detection", "image understanding", "transfer learning"]
    ),
    ResearchDirection(
        topic="3d_vision",
        focus_areas=["depth estimation", "3d reconstruction", "point cloud processing"],
        years=1,
        keywords=["stereo vision", "structure from motion", "SLAM", "neural radiance fields",
                  "3D scene understanding", "volumetric reconstruction", "mesh generation", "depth sensors",
                  "point cloud segmentation", "3D object detection", "multiview geometry", "camera calibration"]
    ),
    ResearchDirection(
        topic="video_understanding",
        focus_areas=["action recognition", "video captioning", "temporal modeling"],
        years=1,
        keywords=["video analysis", "temporal convolution", "optical flow", "activity recognition",
                  "video generation", "motion detection", "spatiotemporal features", "video summarization",
                  "event detection", "video segmentation", "frame interpolation", "video prediction", "action localization"]
    ),

    # cs.CY - Computers and Society (empirical studies: 2 years)
    ResearchDirection(
        topic="ai_ethics",
        focus_areas=["fairness", "accountability", "transparency"],
        years=2,
        keywords=["algorithmic bias", "ethical AI", "responsible AI", "AI governance", "fairness metrics",
                  "explainable AI", "algorithmic accountability", "discrimination detection", "bias mitigation",
                  "ethical frameworks", "AI policy", "social impact", "trustworthy AI", "value alignment"]
    ),

    # cs.DB - Databases (systems work: 2 years)
    ResearchDirection(
        topic="database_systems",
        focus_areas=["query optimization", "distributed databases", "data management"],
        years=2,
        keywords=["SQL optimization", "indexing strategies", "transaction processing", "NoSQL databases",
                  "database architecture", "data warehousing", "OLAP", "query execution", "storage engines",
                  "database security", "data replication", "sharding", "consistency models"]
    ),

    # cs.DC - Distributed Computing (systems work: 2 years)
    ResearchDirection(
        topic="distributed_systems",
        focus_areas=["consensus algorithms", "fault tolerance", "distributed coordination"],
        years=2,
        keywords=["distributed protocols", "replicated state machines", "byzantine fault tolerance", "leader election",
                  "distributed transactions", "Paxos", "Raft", "eventual consistency", "distributed locks",
                  "clock synchronization", "failure detection", "distributed debugging", "microservices"]
    ),
    ResearchDirection(
        topic="parallel_computing",
        focus_areas=["parallel algorithms", "gpu computing", "high performance computing"],
        years=2,
        keywords=["CUDA programming", "parallel programming models", "MPI", "OpenMP", "task parallelism",
                  "data parallelism", "load balancing", "scalability", "performance tuning", "heterogeneous computing",
                  "parallel numerical methods", "supercomputing", "cluster computing"]
    ),

    # cs.DL - Digital Libraries (applied systems: 2 years)
    ResearchDirection(
        topic="digital_libraries",
        focus_areas=["information retrieval", "document management", "metadata systems"],
        years=2,
        keywords=["digital archives", "content management", "bibliographic systems", "document indexing",
                  "metadata standards", "digital preservation", "repository systems", "scholarly communication",
                  "open access", "citation analysis", "full text search", "Dublin Core", "institutional repositories"]
    ),

    # cs.DM - Discrete Mathematics (pure theory: 3 years)
    ResearchDirection(
        topic="discrete_mathematics",
        focus_areas=["graph theory", "combinatorics", "discrete optimization"],
        years=3,
        keywords=["graph algorithms", "network flows", "matching theory", "coloring problems", "combinatorial optimization",
                  "enumeration", "extremal combinatorics", "ramsey theory", "algebraic graph theory",
                  "matroid theory", "integer programming", "polyhedral combinatorics", "probabilistic methods"]
    ),

    # cs.DS - Data Structures and Algorithms (theory with proofs: 3 years)
    ResearchDirection(
        topic="algorithms",
        focus_areas=["approximation algorithms", "randomized algorithms", "online algorithms"],
        years=3,
        keywords=["algorithm design", "algorithm analysis", "NP hard problems", "greedy algorithms",
                  "dynamic programming", "streaming algorithms", "sublinear algorithms", "competitive analysis",
                  "primal dual methods", "linear programming", "algorithm complexity", "hardness results"]
    ),
    ResearchDirection(
        topic="data_structures",
        focus_areas=["advanced data structures", "succinct data structures", "cache efficient structures"],
        years=3,
        keywords=["tree structures", "hash tables", "priority queues", "spatial data structures", "persistent data structures",
                  "compressed data structures", "memory efficient structures", "I/O efficient algorithms",
                  "external memory algorithms", "dynamic data structures", "range queries", "dictionary structures"]
    ),

    # cs.ET - Emerging Technologies (quantum: very long-term: 4 years)
    ResearchDirection(
        topic="quantum_computing",
        focus_areas=["quantum algorithms", "quantum error correction", "quantum simulation"],
        years=4,
        keywords=["quantum gates", "quantum circuits", "superconducting qubits", "quantum supremacy",
                  "variational quantum algorithms", "quantum annealing", "topological quantum computing",
                  "quantum machine learning", "quantum cryptography", "entanglement", "quantum noise", "NISQ algorithms"]
    ),

    # cs.FL - Formal Languages and Automata (formal verification: 4 years)
    ResearchDirection(
        topic="formal_methods",
        focus_areas=["program verification", "model checking", "theorem proving"],
        years=4,
        keywords=["formal specification", "software verification", "temporal logic", "automated reasoning",
                  "proof assistants", "bounded model checking", "SAT solvers", "SMT solvers", "invariant generation",
                  "abstract interpretation", "symbolic execution", "program synthesis", "correctness proofs"]
    ),

    # cs.GL - General Literature (education research: 2 years)
    ResearchDirection(
        topic="computing_education",
        focus_areas=["cs pedagogy", "educational technology", "learning analytics"],
        years=2,
        keywords=["programming education", "computational thinking", "coding bootcamps", "online learning",
                  "student assessment", "curriculum design", "learning outcomes", "educational games",
                  "teaching methods", "CS1 courses", "active learning", "peer instruction", "MOOCs"]
    ),

    # cs.GR - Graphics (applied systems: 2 years)
    ResearchDirection(
        topic="computer_graphics",
        focus_areas=["rendering", "animation", "geometry processing"],
        years=2,
        keywords=["ray tracing", "rasterization", "shader programming", "physically based rendering",
                  "real time rendering", "character animation", "procedural generation", "texture mapping",
                  "lighting models", "mesh processing", "surface modeling", "visual effects", "GPU graphics"]
    ),

    # cs.GT - Computer Science and Game Theory (theory: 3 years)
    ResearchDirection(
        topic="algorithmic_game_theory",
        focus_areas=["mechanism design", "auction theory", "equilibrium computation"],
        years=3,
        keywords=["Nash equilibrium", "incentive compatibility", "price of anarchy", "voting theory",
                  "social choice", "market design", "combinatorial auctions", "fair division",
                  "strategic behavior", "payoff matrices", "cooperative games", "game tree search"]
    ),

    # cs.HC - Human-Computer Interaction (empirical: 2 years)
    ResearchDirection(
        topic="human_computer_interaction",
        focus_areas=["user interface design", "interaction techniques", "usability studies"],
        years=2,
        keywords=["user experience", "interface design", "usability testing", "accessibility", "gesture interaction",
                  "touchscreen interfaces", "virtual reality interfaces", "augmented reality", "user centered design",
                  "interaction design", "mobile interfaces", "information visualization", "human factors"]
    ),

    # cs.IR - Information Retrieval (applied: 1 year)
    ResearchDirection(
        topic="information_retrieval",
        focus_areas=["search engines", "ranking algorithms", "query understanding"],
        years=1,
        keywords=["web search", "relevance ranking", "document retrieval", "indexing methods", "search quality",
                  "query expansion", "learning to rank", "semantic search", "personalized search",
                  "recommendation systems", "text mining", "entity retrieval", "BM25"]
    ),

    # cs.IT - Information Theory (deep theory: 3 years)
    ResearchDirection(
        topic="information_theory",
        focus_areas=["coding theory", "compression", "channel capacity"],
        years=3,
        keywords=["Shannon entropy", "mutual information", "error correcting codes", "data compression",
                  "source coding", "channel coding", "rate distortion theory", "network information theory",
                  "LDPC codes", "turbo codes", "lossless compression", "lossy compression", "capacity theorems"]
    ),

    # cs.LG - Machine Learning (hot topic - multiple directions)
    # Fast-moving applied ML: 1 year
    ResearchDirection(
        topic="deep_learning",
        focus_areas=["neural architecture search", "efficient training", "transfer learning"],
        years=1,
        keywords=["neural networks", "backpropagation", "optimization", "batch normalization", "dropout",
                  "residual networks", "attention mechanisms", "model compression", "knowledge distillation",
                  "pre training", "fine tuning", "multi task learning", "domain adaptation"]
    ),
    ResearchDirection(
        topic="reinforcement_learning",
        focus_areas=["policy optimization", "multi agent rl", "model based rl"],
        years=1,
        keywords=["Q learning", "policy gradients", "actor critic", "deep RL", "reward shaping",
                  "exploration strategies", "temporal difference learning", "value functions", "Markov decision processes",
                  "model free RL", "sample efficiency", "off policy learning", "experience replay"]
    ),
    ResearchDirection(
        topic="generative_models",
        focus_areas=["diffusion models", "gans", "variational autoencoders"],
        years=1,
        keywords=["image generation", "text generation", "score based models", "denoising diffusion",
                  "adversarial training", "latent variable models", "energy based models", "flow models",
                  "conditional generation", "data augmentation", "synthetic data", "likelihood estimation"]
    ),
    ResearchDirection(
        topic="meta_learning",
        focus_areas=["few shot learning", "learning to learn", "task adaptation"],
        years=1,
        keywords=["rapid adaptation", "MAML", "prototypical networks", "metric learning", "transfer learning",
                  "multi task learning", "continual learning", "neural architecture search", "hyperparameter optimization",
                  "zero shot learning", "cross domain learning", "model agnostic methods"]
    ),
    ResearchDirection(
        topic="interpretable_ml",
        focus_areas=["explainable ai", "model interpretability", "feature attribution"],
        years=1,
        keywords=["model explanations", "feature importance", "attention visualization", "saliency maps",
                  "LIME", "SHAP", "counterfactual explanations", "model transparency", "decision rules",
                  "interpretable representations", "concept learning", "neural network visualization", "trust in AI"]
    ),

    # cs.LO - Logic in Computer Science (deep theory: 4 years)
    ResearchDirection(
        topic="logic_in_cs",
        focus_areas=["modal logic", "temporal logic", "constraint logic"],
        years=4,
        keywords=["propositional logic", "predicate logic", "logical inference", "satisfiability",
                  "proof theory", "model theory", "automated theorem proving", "logic programming",
                  "description logics", "epistemic logic", "deontic logic", "linear temporal logic"]
    ),

    # cs.MA - Multiagent Systems (systems work: 2 years)
    ResearchDirection(
        topic="multiagent_systems",
        focus_areas=["coordination", "cooperation", "agent communication"],
        years=2,
        keywords=["agent architectures", "multi agent planning", "distributed AI", "agent negotiation",
                  "coalition formation", "agent protocols", "emergent behavior", "swarm intelligence",
                  "agent based modeling", "team formation", "communication protocols", "agent cooperation"]
    ),

    # cs.MM - Multimedia (applied systems: 2 years)
    ResearchDirection(
        topic="multimedia_systems",
        focus_areas=["video processing", "audio processing", "multimodal learning"],
        years=2,
        keywords=["video encoding", "audio signal processing", "multimedia databases", "streaming systems",
                  "multimedia retrieval", "video compression", "audio video synchronization", "media formats",
                  "cross modal learning", "audio visual fusion", "multimedia indexing", "content delivery"]
    ),

    # cs.MS - Mathematical Software (systems: 2 years)
    ResearchDirection(
        topic="mathematical_software",
        focus_areas=["symbolic computation", "numerical software", "computational algebra"],
        years=2,
        keywords=["computer algebra systems", "mathematical libraries", "numerical solvers", "symbolic differentiation",
                  "polynomial computation", "algebraic algorithms", "Groebner bases", "mathematical optimization",
                  "linear algebra packages", "differential equation solvers", "interval arithmetic", "exact arithmetic"]
    ),

    # cs.NA - Numerical Analysis (applied math: 3 years)
    ResearchDirection(
        topic="numerical_analysis",
        focus_areas=["numerical linear algebra", "optimization methods", "differential equations"],
        years=3,
        keywords=["iterative methods", "matrix decomposition", "eigenvalue problems", "convex optimization",
                  "nonlinear optimization", "gradient methods", "finite difference methods", "spectral methods",
                  "convergence analysis", "numerical stability", "condition numbers", "preconditioning"]
    ),

    # cs.NE - Neural and Evolutionary Computing (applied: 2 years)
    ResearchDirection(
        topic="evolutionary_computation",
        focus_areas=["genetic algorithms", "evolutionary strategies", "neuroevolution"],
        years=2,
        keywords=["population based optimization", "fitness functions", "crossover operators", "mutation operators",
                  "selection mechanisms", "multi objective optimization", "coevolution", "genetic programming",
                  "differential evolution", "evolution of neural networks", "artificial life", "swarm optimization"]
    ),

    # cs.NI - Networking and Internet Architecture (systems: 2 years)
    ResearchDirection(
        topic="computer_networks",
        focus_areas=["network protocols", "wireless networks", "network optimization"],
        years=2,
        keywords=["TCP IP", "routing algorithms", "network topology", "software defined networking",
                  "network congestion", "quality of service", "mobile networks", "ad hoc networks",
                  "internet architecture", "network security", "bandwidth management", "5G networks", "IoT networks"]
    ),

    # cs.OH - Other Computer Science (interdisciplinary: 2 years)
    ResearchDirection(
        topic="interdisciplinary_cs",
        focus_areas=["computational biology", "computational social science", "digital humanities"],
        years=2,
        keywords=["bioinformatics", "genomics", "protein folding", "network science", "text mining",
                  "social network analysis", "digital archives", "computational linguistics", "medical informatics",
                  "systems biology", "cultural analytics", "data driven humanities", "interdisciplinary computing"]
    ),

    # cs.OS - Operating Systems (systems: 2 years)
    ResearchDirection(
        topic="operating_systems",
        focus_areas=["kernel design", "resource management", "virtualization"],
        years=2,
        keywords=["process scheduling", "memory management", "file systems", "device drivers",
                  "system calls", "containerization", "hypervisors", "virtual machines", "OS security",
                  "real time systems", "distributed operating systems", "microkernel", "system performance"]
    ),

    # cs.PF - Performance (systems: 2 years)
    ResearchDirection(
        topic="performance_analysis",
        focus_areas=["performance modeling", "benchmarking", "profiling"],
        years=2,
        keywords=["performance optimization", "bottleneck analysis", "latency reduction", "throughput improvement",
                  "performance metrics", "workload characterization", "system monitoring", "tracing tools",
                  "capacity planning", "performance tuning", "scalability analysis", "resource utilization"]
    ),

    # cs.PL - Programming Languages (language design: 3 years)
    ResearchDirection(
        topic="programming_languages",
        focus_areas=["language design", "compilers", "type systems"],
        years=3,
        keywords=["syntax design", "semantics", "static analysis", "compiler optimization", "code generation",
                  "type inference", "functional programming", "object oriented programming", "domain specific languages",
                  "runtime systems", "garbage collection", "language interoperability", "program transformation"]
    ),

    # cs.RO - Robotics (hardware integration: 3 years)
    ResearchDirection(
        topic="robotics",
        focus_areas=["robot perception", "motion planning", "robot learning"],
        years=3,
        keywords=["sensor fusion", "computer vision for robotics", "path planning", "obstacle avoidance",
                  "manipulation", "grasping", "localization", "mapping", "autonomous navigation",
                  "human robot interaction", "robot control", "kinematics", "dynamics", "imitation learning"]
    ),

    # cs.SC - Symbolic Computation (systems: 3 years)
    ResearchDirection(
        topic="symbolic_computation",
        focus_areas=["computer algebra", "automated reasoning", "symbolic integration"],
        years=3,
        keywords=["symbolic mathematics", "algebraic simplification", "equation solving", "symbolic differentiation",
                  "polynomial manipulation", "rational functions", "indefinite integration", "definite integration",
                  "special functions", "expression manipulation", "formula manipulation", "algebraic transformations"]
    ),

    # cs.SD - Sound (applied signal processing: 2 years)
    ResearchDirection(
        topic="audio_computing",
        focus_areas=["speech processing", "music information retrieval", "sound synthesis"],
        years=2,
        keywords=["speech recognition", "speaker identification", "audio features", "music classification",
                  "audio segmentation", "pitch detection", "rhythm analysis", "audio synthesis", "sound effects",
                  "audio enhancement", "noise reduction", "acoustic modeling", "audio signal processing"]
    ),

    # cs.SE - Software Engineering (empirical systems: 2 years)
    ResearchDirection(
        topic="software_engineering",
        focus_areas=["software testing", "program analysis", "software maintenance"],
        years=2,
        keywords=["test automation", "unit testing", "integration testing", "code review", "static analysis",
                  "dynamic analysis", "bug detection", "code refactoring", "technical debt", "software quality",
                  "continuous integration", "version control", "software metrics", "development methodologies"]
    ),

    # cs.SI - Social and Information Networks (data science: 2 years)
    ResearchDirection(
        topic="social_networks",
        focus_areas=["network analysis", "community detection", "influence propagation"],
        years=2,
        keywords=["graph mining", "social network analysis", "link prediction", "centrality measures",
                  "clustering algorithms", "information diffusion", "viral marketing", "opinion dynamics",
                  "network topology", "temporal networks", "ego networks", "social influence", "network evolution"]
    ),

    # cs.SY - Systems and Control (control theory: 2 years)
    ResearchDirection(
        topic="control_systems",
        focus_areas=["optimal control", "adaptive control", "cyber physical systems"],
        years=2,
        keywords=["feedback control", "state estimation", "Kalman filtering", "model predictive control",
                  "robust control", "nonlinear control", "stability analysis", "control design",
                  "embedded systems", "real time control", "sensor networks", "actuator control", "system identification"]
    ),
]
