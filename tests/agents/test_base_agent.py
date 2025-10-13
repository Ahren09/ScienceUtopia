"""
Unit tests for base agent system
Tests Phase 1: Agent Framework implementation
"""

import pytest
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
sys.path.append('/nethome/yjin328/Workspace/PeerReview/AI-Scientist-v2')

from utopia.agents.base_agent import SimulationAgent, MultiAgentEcosystem
from utopia.agents.specialized_agents import UniversityAgent, IndustryAgent, FreelancerAgent, FundingAgent
from const import RESEARCH_COST, PAPER_SUBMISSION_COST


class TestSimulationAgent:
    """Test base SimulationAgent functionality"""
    
    def setup_method(self):
        """Set up test agent"""
        self.agent = SimulationAgent(
            personality="test_personality",
            reputation=5,
            funding_level=50,
            specializations=["test_field"],
            agent_id="test_agent_001"
        )
    
    def test_agent_initialization(self):
        """Test agent initialization with proper attributes"""
        assert self.agent.id == "test_agent_001"
        assert self.agent.personality == "test_personality"
        assert self.agent.reputation == 5
        assert self.agent.resources == 50
        assert self.agent.specializations == ["test_field"]
        assert isinstance(self.agent.memory_bank, list)
        assert isinstance(self.agent.review_experiences, list)
        
    def test_resource_management(self):
        """Test resource deduction for research activities"""
        initial_resources = self.agent.resources
        
        # Mock the ideation function to avoid actual API calls
        with patch('ai_scientist.agents.base_agent.generate_temp_free_idea') as mock_ideation:
            mock_ideation.return_value = [{"title": "Test Idea", "abstract": "Test abstract"}]
            
            ideas = self.agent.generate_research_idea([], "test focus")
            
            # Check resource deduction
            assert self.agent.resources == initial_resources - RESEARCH_COST
            assert ideas is not None
    
    def test_insufficient_resources_research(self):
        """Test behavior when agent has insufficient resources for research"""
        # Set resources below research cost
        self.agent.resources = RESEARCH_COST - 1
        
        with pytest.raises(ValueError, match="insufficient resources"):
            self.agent.generate_research_idea([], "test focus")
    
    def test_insufficient_resources_paper(self):
        """Test behavior when agent has insufficient resources for paper writing"""
        # Set resources below submission cost
        self.agent.resources = PAPER_SUBMISSION_COST - 1
        
        with pytest.raises(ValueError, match="insufficient resources"):
            self.agent.write_paper({"title": "Test"}, [])
    
    def test_memory_bank_functionality(self):
        """Test memory bank storage and retrieval"""
        # Add memory
        test_memory = {
            'type': 'test_memory',
            'thought': 'Test thought',
            'timestamp': time.time(),
            'trigger': 'test_trigger'
        }
        self.agent.memory_bank.append(test_memory)
        
        # Verify memory storage
        assert len(self.agent.memory_bank) == 1
        assert self.agent.memory_bank[0]['type'] == 'test_memory'
        
        # Test memory context extraction
        context = self.agent._extract_review_memory_context()
        assert isinstance(context, str)
    
    def test_review_experience_recording(self):
        """Test review experience recording functionality"""
        test_paper = {'id': 'test_paper_123', 'content': 'Test content'}
        test_review = {
            'overall_score': 7,
            'strengths': ['clear writing'],
            'weaknesses': ['limited scope']
        }
        
        # Record review experience
        self.agent._record_review_experience(test_paper, test_review, role="reviewer")
        
        # Verify experience recording
        assert len(self.agent.review_experiences) == 1
        assert self.agent.review_experiences[0]['role'] == 'reviewer'
        assert self.agent.review_experiences[0]['paper_id'] == 'test_paper_123'
    
    def test_citation_traceability(self):
        """Test citation metadata generation"""
        paper_output = {
            'writeup_md': 'Test paper content with citations',
            'plot_py': 'test plot code',
            'experiment_py': 'test experiment code'
        }
        references = []
        
        # Test citation addition
        enhanced_output = self.agent._add_traceable_citations(paper_output, references)
        
        # Verify citation structure
        assert 'citations' in enhanced_output
        assert 'citation_metadata' in enhanced_output
        assert isinstance(enhanced_output['citations'], list)
        assert 'total_citations' in enhanced_output['citation_metadata']
        assert 'citation_diversity_score' in enhanced_output['citation_metadata']
    
    def test_personality_prompt_generation(self):
        """Test personality prompt generation"""
        prompt = self.agent._get_personality_prompt()
        assert isinstance(prompt, str)
        assert "test_personality" in prompt or "test_field" in prompt
    
    def test_review_quality_assessment(self):
        """Test review quality assessment functionality"""
        constructive_review = {
            'overall_score': 8,
            'strengths': ['good methodology', 'clear results'],
            'weaknesses': ['minor issues']
        }
        
        toxic_review = {
            'overall_score': 2,
            'strengths': [],
            'weaknesses': ['terrible', 'awful', 'complete failure', 'waste of time']
        }
        
        assert self.agent._is_constructive_review(constructive_review) == True
        assert self.agent._is_toxic_review(toxic_review) == True
        assert self.agent._assess_review_quality(constructive_review) == "constructive"
        assert self.agent._assess_review_quality(toxic_review) == "toxic"


class TestMultiAgentEcosystem:
    """Test MultiAgentEcosystem functionality"""
    
    def setup_method(self):
        """Set up test ecosystem"""
        self.config = {"test_config": "value"}
        self.initial_corpus = []
        self.ecosystem = MultiAgentEcosystem(self.config, self.initial_corpus)
    
    def test_ecosystem_initialization(self):
        """Test ecosystem initialization"""
        assert self.ecosystem.paper_archive == []
        assert hasattr(self.ecosystem, 'citation_graph')
        assert hasattr(self.ecosystem, 'simulation_metrics')
        assert hasattr(self.ecosystem, 'agent_population')
        assert isinstance(self.ecosystem.agent_population, list)
    
    def test_agent_management(self):
        """Test adding and retrieving agents"""
        # Create test agents
        uni_agent = UniversityAgent("TestU", funding_level=70)
        ind_agent = IndustryAgent("TestCorp", funding_level=90)
        
        # Add agents to ecosystem
        self.ecosystem.add_agent(uni_agent)
        self.ecosystem.add_agent(ind_agent)
        
        # Verify agents added
        assert len(self.ecosystem.agent_population) == 2
        
        # Test getting available reviewers
        reviewers = self.ecosystem.get_available_reviewers()
        assert len(reviewers) == 2  # Both can review
        
        # Test getting available authors
        authors = self.ecosystem.get_available_authors()
        assert len(authors) == 2  # Both can author
    
    def test_metrics_tracker_initialization(self):
        """Test metrics tracker initialization"""
        metrics = self.ecosystem.simulation_metrics
        assert 'yearly_metrics' in metrics
        assert 'agent_interactions' in metrics
        assert 'citation_evolution' in metrics
        assert 'review_patterns' in metrics


class TestSpecializedAgents:
    """Test specialized agent types"""
    
    def test_university_agent_initialization(self):
        """Test UniversityAgent initialization and properties"""
        agent = UniversityAgent("Harvard", funding_level=80, agent_id="harvard_001")
        
        assert agent.university_name == "Harvard"
        assert agent.resources == 80
        assert agent.personality == "academic_rigorous"
        assert agent.can_author == True
        assert agent.can_review == True
        assert "theoretical_ml" in agent.specializations
        
        # Test personality prompt
        prompt = agent._get_personality_prompt()
        assert "university researcher" in prompt.lower()
        assert "theoretical contributions" in prompt.lower()
    
    def test_industry_agent_initialization(self):
        """Test IndustryAgent initialization and properties"""
        agent = IndustryAgent("Google", funding_level=95, agent_id="google_001")
        
        assert agent.company_name == "Google"
        assert agent.resources == 95
        assert agent.personality == "commercial_practical"
        assert agent.can_author == True
        assert agent.can_review == True
        assert "applied_ml" in agent.specializations
        
        # Test personality prompt
        prompt = agent._get_personality_prompt()
        assert "industry researcher" in prompt.lower()
        assert "practical applications" in prompt.lower()
    
    def test_freelancer_agent_initialization(self):
        """Test FreelancerAgent initialization and properties"""
        agent = FreelancerAgent("IndependentResearcher", funding_level=25, agent_id="indie_001")
        
        assert agent.researcher_name == "IndependentResearcher"
        assert agent.resources == 25
        assert agent.personality == "innovative_disruptive"
        assert agent.can_author == True
        assert agent.can_review == True
        assert "unconventional_methods" in agent.specializations
        
        # Test personality prompt
        prompt = agent._get_personality_prompt()
        assert "independent researcher" in prompt.lower()
        assert "unconventional approaches" in prompt.lower()
    
    def test_funding_agent_initialization(self):
        """Test FundingAgent initialization and properties"""
        agent = FundingAgent("NSF", funding_level=100, agent_id="nsf_001")
        
        assert agent.agency_name == "NSF"
        assert agent.resources == 100
        assert agent.personality == "conservative_strategic"
        assert agent.can_author == False  # Funding agents don't author
        assert agent.can_review == True
        assert "research_evaluation" in agent.specializations
        
        # Test personality prompt
        prompt = agent._get_personality_prompt()
        assert "funding agency" in prompt.lower()
        assert "strategic funding perspective" in prompt.lower()
    
    def test_agent_review_biases(self):
        """Test agent-specific review biases"""
        uni_agent = UniversityAgent("TestU")
        ind_agent = IndustryAgent("TestCorp")
        free_agent = FreelancerAgent("TestResearcher")
        fund_agent = FundingAgent("TestAgency")
        
        # Test review with theoretical strength
        theoretical_review = {
            'overall_score': 6,
            'strengths': ['strong theoretical_contribution', 'rigorous methodology'],
            'weaknesses': ['limited practical application']
        }
        
        # University agent should boost score for theoretical work
        uni_biased = uni_agent._apply_agent_review_biases(theoretical_review.copy())
        assert uni_biased['overall_score'] > theoretical_review['overall_score']
        
        # Test review with practical strength
        practical_review = {
            'overall_score': 6,
            'strengths': ['practical applications', 'scalable solution'],
            'weaknesses': ['limited theoretical depth']
        }
        
        # Industry agent should boost score for practical work
        ind_biased = ind_agent._apply_agent_review_biases(practical_review.copy())
        assert ind_biased['overall_score'] > practical_review['overall_score']
    
    def test_funding_decision_making(self):
        """Test funding agent decision making"""
        fund_agent = FundingAgent("TestAgency")
        
        # Mock proposal with high score
        high_score_proposal = {'id': 'prop_001', 'title': 'High Impact Research'}
        high_score_evaluation = {'overall_score': 8.5, 'summary': 'Excellent proposal'}
        
        decision = fund_agent._make_funding_decision(high_score_evaluation, high_score_proposal)
        
        assert decision['decision'] == 'fund'
        assert decision['funding_amount'] == 'high'
        assert decision['proposal_id'] == 'prop_001'
        assert decision['evaluation_score'] == 8.5
        
        # Mock proposal with low score
        low_score_proposal = {'id': 'prop_002', 'title': 'Low Impact Research'}
        low_score_evaluation = {'overall_score': 4.0, 'summary': 'Needs improvement'}
        
        decision = fund_agent._make_funding_decision(low_score_evaluation, low_score_proposal)
        
        assert decision['decision'] == 'reject'
        assert decision['funding_amount'] == 'none'


def test_phase1_integration():
    """
    Integration test for Phase 1 components
    Verifies all components work together
    """
    # Create ecosystem
    ecosystem = MultiAgentEcosystem({}, [])
    
    # Create diverse agent population
    agents = [
        UniversityAgent("MIT", funding_level=75),
        IndustryAgent("OpenAI", funding_level=95), 
        FreelancerAgent("IndieResearcher", funding_level=35),
        FundingAgent("NSF", funding_level=100)
    ]
    
    # Add all agents to ecosystem
    for agent in agents:
        ecosystem.add_agent(agent)
    
    # Verify ecosystem population
    assert len(ecosystem.agent_population) == 4
    
    # Test author/reviewer separation
    authors = ecosystem.get_available_authors()
    reviewers = ecosystem.get_available_reviewers()
    
    assert len(authors) == 3  # University, Industry, Freelancer can author
    assert len(reviewers) == 4  # All can review
    
    # Test that each agent has unique ID
    agent_ids = [agent.id for agent in ecosystem.agent_population]
    assert len(set(agent_ids)) == len(agent_ids)  # All unique
    
    print("✓ Phase 1 integration tests passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])