# CLAUDE.md - Academic Society Ecosystem Simulation Framework

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI Guidance

* Ignore GEMINI.md and GEMINI-*.md files
* To save main context space, for code searches, inspections, troubleshooting or analysis, use code-searcher subagent where appropriate - giving the subagent full context background for the task(s) you assign it.
* After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.
* For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.
* Before you finish, please verify your solution
* Do what has been asked; nothing more, nothing less.
* NEVER create files unless they're absolutely necessary for achieving your goal.
* ALWAYS prefer editing an existing file to creating a new one.
* NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
* When you update or modify core context files, also update markdown documentation and memory bank
* When asked to commit changes, exclude CLAUDE.md and CLAUDE-*.md referenced memory bank system files from any commits. Never delete these files.
* Avoid adding many try-catch as it will obscure the problem. 
* Keep the code system succinct. DO NOT add many nested logics. 
* After editing, modify @.vscode/launch.json for all experiments included.
* Make sure your code can run according to all commands in @.vscode/launch.json

## Memory Bank System

This project uses a structured memory bank system with specialized context files. Always check these files for relevant information before starting work:

### Core Context Files

* **CLAUDE-activeContext.md** - Current session state, goals, and progress (if exists)
* **CLAUDE-patterns.md** - Established code patterns and conventions (if exists)
* **CLAUDE-decisions.md** - Architecture decisions and rationale (if exists)
* **CLAUDE-troubleshooting.md** - Common issues and proven solutions (if exists)
* **CLAUDE-config-variables.md** - Configuration variables reference (if exists)
* **CLAUDE-temp.md** - Temporary scratch pad (only read when referenced)

**Important:** Always reference the active context file first to understand what's currently being worked on and maintain session continuity.

### Memory Bank System Backups

When asked to backup Memory Bank System files, you will copy the core context files above and @.claude settings directory to directory @/path/to/backup-directory. If files already exist in the backup directory, you will overwrite them.

## Project Overview

This framework simulates a fully AI-mediated scientific ecosystem where AI agents (Universities, Industry Labs, Freelance Researchers, and Funding Agencies) autonomously generate research papers, conduct peer reviews, and evolve scientific knowledge through closed-loop interactions. Unlike previous works that focus only on paper generation or review in isolation, this system creates a completely "closed-loop" scientific ecosystem where knowledge creation, evaluation, and dissemination occur in dynamic feedback loops.

### Research Questions & Objectives

**Primary Goal:** Investigate how academic society will evolve under various environmental conditions such as funding shortages, toxic peer-review cultures, institutional pressures (e.g. some schools lean towards projects that get lots of citations), and resource constraints. This addresses fundamental questions about the resilience and adaptability of scientific ecosystems.

**Secondary Goal:** Investigate what happens when every role in the scientific lifecycle is mediated by AI agents, and understand the emergent behaviors, biases, and convergence patterns that arise in such AI-dominated ecosystems.

**Key Research Questions:**

*Primary Goal Questions:*
- How does funding scarcity affect research diversity, innovation, and collaboration patterns?
- What happens when toxic review cultures dominate the ecosystem? Do they self-perpetuate?
- How do different institutional pressures (tenure, profit, innovation) reshape scientific priorities?
- Can academic ecosystems recover from periods of resource constraint or cultural toxicity?
- What interventions can maintain ecosystem health under adverse conditions?

*Secondary Goal Questions:*
- Do AI reviewers exhibit preference biases toward AI-generated vs. human-written papers?
- How do research topics evolve in AI-mediated systems - toward innovation or intellectual convergence?
- How do citation patterns and academic inequalities evolve in AI-dominated ecosystems?
- What governance implications emerge for managing AI-dominated scientific ecosystems?

### Development Plan

Your development of code should follow a phased approach, leveraging existing AI-Scientist-v2 infrastructure AS MUCH AS POSSIBLE and extending it with multi-agent capabilities. NEVER introduce duplicate logics.

**CRITICAL: Non-Destructive Development Policy**
- **NEVER modify existing AI-Scientist-v2 files directly**
- **ALWAYS create child classes, extensions, or wrapper classes**
- **NEVER overwrite existing functions or logic**
- **Use inheritance and composition to extend functionality**
- **Keep all new multi-agent code in separate modules under ai_scientist/agents/ or ai_scientist/simulation/**

---

## **Implementation Summary**

### **Key Leverage Points from Existing Codebase:**

1. **Agent Management:** Extend `ai_scientist/treesearch/agent_manager.py` `AgentManager` class
2. **Paper Generation:** Use `ai_scientist/perform_ideation_temp_free.py` and `launch_scientist_bfts.py` pipeline  
3. **Review System:** Leverage `ai_scientist/perform_llm_review.py` and `ai_scientist/perform_vlm_review.py`
4. **Citation Management:** Use `ai_scientist/tools/semantic_scholar.py` `SemanticScholarSearchTool`
5. **Configuration:** Extend existing `bfts_config.yaml` system
6. **Experimental Tracking:** Use `ai_scientist/treesearch/journal.py` `Journal` and `Node` classes
7. **Model Integration:** Use existing `ai_scientist/llm.py` infrastructure

### **New Components to Implement:**

1. **Multi-agent specialization logic** (agent personality prompts and biases)
2. **Simulation orchestration engine** (yearly progression, agent interaction management)  
3. **Metrics calculation framework** (novelty, bias, convergence tracking)

### **Development Workflow:**

Each phase represents approximately 1-2 weeks of development work. The phases are designed to be incremental, with each phase building upon the previous while maintaining functional capabilities. The extensive use of existing AI-Scientist infrastructure ensures rapid development and consistency with established patterns.

**CRITICAL: Testing Requirements for Each Phase:**

Each phase MUST include comprehensive unit and integration tests to validate functionality before proceeding to the next phase. This ensures confidence in the system's stability and correctness at each development milestone.

**Phase-Specific Testing Requirements:**
