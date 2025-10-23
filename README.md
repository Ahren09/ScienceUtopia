# AI-on-AI Scientific Ecosystem Simulation Framework

This framework simulates a fully AI-mediated scientific ecosystem where AI agents autonomously generate research papers, conduct peer reviews, and evolve scientific knowledge through closed-loop interactions.

## 🎯 Project Overview

This system creates a multi-agent simulation where different types of academic entities—universities, industry labs, freelancers, and funding agencies—interact in realistic scenarios. The goal is to study how academic ecosystems evolve under various conditions like funding shortages, toxic peer review environments, or institutional pressures.

**Key Features:**
- **Multi-Agent System**: Four distinct agent types with unique personalities and biases
- **Autonomous Paper Generation**: Extends AI-Scientist framework with agent-specific behaviors
- **Comprehensive Review System**: 3 reviewers + 1 meta-reviewer with realistic scoring mechanisms
- **Dynamic Citation Network**: Tracks paper relationships and influence over time
- **Intervention Framework**: Policy controls for improving diversity, novelty, and bias mitigation

## 🤖 Agent Personality Types

### 1. **Academic Rigorous** (`UniversityResearcher`)
- **Focus**: Theoretical contributions, fundamental knowledge advancement
- **Values**: Reproducibility, peer review rigor, long-term scientific impact
- **Biases**: Favors theoretical work, penalizes reproducibility concerns
- **Funding Level**: 70/100

### 2. **Commercial Practical** (`IndustryResearcher`)
- **Focus**: Practical applications, scalable solutions, market viability
- **Values**: Efficiency, real-world impact, competitive advantages  
- **Biases**: Rewards practical/scalable work, less penalty for theoretical gaps
- **Funding Level**: 90/100

### 3. **Innovative Disruptive** (`FreelancerAgent`)
- **Focus**: Unconventional approaches, paradigm-shifting discoveries
- **Values**: Radical innovation, interdisciplinary thinking, bold ideas
- **Biases**: Strong rewards for novelty, heavy penalties for incremental work
- **Funding Level**: 30/100

### 4. **Conservative Strategic** (`FundingAgent`)
- **Focus**: Strategic resource allocation, long-term impact assessment
- **Values**: Solid execution, funding alignment, proven approaches
- **Biases**: Slight bonus for strategic impact, penalty for high-risk work
- **Funding Level**: 100/100

## 🏗️ Implementation Status

### ✅ Phase 1: Agent Framework (COMPLETE)
- [x] Base `SimulationAgent` and `MultiAgentEcosystem` classes
- [x] Four specialized agent types with distinct personalities
- [x] Memory bank system for behavioral evolution
- [x] Resource management with cost deduction
- [x] Citation traceability with structured metadata
- [x] Internal review system (`ai_scientist/review/`)
- [x] Comprehensive testing suite (8/8 tests passing)

### 🔄 Phase 2: Simulation Engine (PENDING)
- [ ] Temporal simulation rounds with paper generation
- [ ] Multi-reviewer consensus system
- [ ] Paper archive and citation graph management
- [ ] Metrics calculation (novelty, bias, convergence)

## 🧪 Testing

### Run All Tests
```bash
# Phase 1: Agent Framework Tests
python tests/agents/test_runner.py

# Review System Tests  
python tests/agents/test_review_system.py
```

### VS Code Debug Configurations
- **[Test] Phase 1: Agent Framework** - Core agent functionality
- **[Test] Review System** - Internal review pipeline
- **[Test] Phase 1: Integration Tests** - Cross-component integration

## 🚀 Usage Examples

### Basic Agent Creation
```python
from ai_scientist.agents.specialized_agents import UniversityResearcher, IndustryResearcher

# Create different agent types
uni_agent = UniversityResearcher("MIT", funding_level=75)
industry_agent = IndustryResearcher("OpenAI", funding_level=95)

# Generate paper review
test_paper = {'content': 'Research paper content...', 'metadata': {...}}
review = uni_agent.review_paper(test_paper)
print(f"Score: {review['overall_score']}/10")
```

### Multi-Agent Ecosystem
```python
from ai_scientist.agents.base_agent import MultiAgentEcosystem
from ai_scientist.agents.specialized_agents import *

# Create ecosystem
ecosystem = MultiAgentEcosystem({}, [])

# Add diverse agent population
agents = [
    UniversityResearcher("Harvard", funding_level=80),
    IndustryResearcher("Google", funding_level=95),
    FreelancerAgent("IndependentResearcher", funding_level=25),
    FundingAgent("NSF", funding_level=100)
]

for agent in agents:
    ecosystem.add_agent(agent)

# Get available authors and reviewers
authors = ecosystem.get_available_authors()  
reviewers = ecosystem.get_available_reviewers() 
```

## 🔬 Research Applications

This framework enables investigation of:

1. **Academic Ecosystem Dynamics**: How do different institutional pressures affect research quality and diversity?

2. **AI Impact on Science**: Do AI-generated reviews show systematic biases toward AI-written papers?

3. **Funding Policy Effects**: How do funding shortages or strategic priorities influence research directions?

4. **Peer Review Toxicity**: What happens when review environments become hostile or biased?

5. **Innovation vs. Incrementalism**: How do different reward structures affect the balance between novel and incremental research?

## 📁 Project Structure

```
ai_scientist/
├── agents/                    # Multi-agent system
│   ├── base_agent.py         # Core SimulationAgent and MultiAgentEcosystem
│   └── specialized_agents.py # UniversityResearcher, IndustryResearcher, etc.
├── review/                   # Internal review system
│   ├── __init__.py
│   ├── prompts.py           # Personality-specific review prompts
│   └── review_pipeline.py   # AgentReviewPipeline implementation
tests/agents/                 # Comprehensive test suite
├── test_runner.py           # Phase 1 framework tests
└── test_review_system.py    # Review system tests
const.py                     # Cost constants (RESEARCH_COST, PAPER_SUBMISSION_COST)
```

## 🎯 Next Steps

1. **Phase 2 Implementation**: Simulation engine with temporal rounds
2. **Metrics Dashboard**: Real-time visualization of ecosystem health  
3. **Intervention System**: Policy controls for bias mitigation
4. **Human Evaluation**: Validate AI vs human preference patterns
5. **Publication Pipeline**: Generate results for academic publication

## 📚 Research Context

This project extends the **AI-Scientist-v2** framework to study emergent behaviors in AI-mediated scientific ecosystems. Unlike previous work that focuses on single-paper generation or review, we create a complete closed-loop system where agents both author and review papers, enabling study of complex interaction dynamics.

**Potential Publications:**
- "Emergent Behaviors in AI-Mediated Scientific Ecosystems" 
- "Do Large Language Models Prefer AI-Written Papers? A Multi-Agent Study"
- "The Impact of Funding Policies on AI-Driven Research Diversity"

---

This framework provides a comprehensive platform for understanding how AI might reshape scientific discovery and academic institutions in the coming decades.