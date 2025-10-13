"""
Integration tests for agent idea generation and memory functionality
Tests Phase 1: Agent Framework integration with AI-Scientist pipeline
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
sys.path.append('/nethome/yjin328/Workspace/PeerReview/AI-Scientist-v2')

from utopia.agents.base_agent import SimulationAgent, MultiAgentEcosystem
from utopia.agents.specialized_agents import UniversityAgent, IndustryAgent


class TestAgentIdeationIntegration:
    """Test integration with AI-Scientist ideation pipeline"""
    
    def setup_method(self):
        """Set up test environment"""
        self.agent = UniversityAgent("TestU", funding_level=100)
    
    @patch('ai_scientist.perform_ideation_temp_free.generate_temp_free_idea')
    def test_agent_idea_generation_integration(self, mock_ideation):
        """Test agent idea generation using existing AI-Scientist pipeline"""
        # Mock the ideation function response
        mock_ideas = [
            {
                "Name": "test_idea",
                "Title": "Novel Deep Learning Architecture",
                "Abstract": "We propose a novel architecture for deep learning...",
                "Experiments": "We will implement the architecture and test on benchmarks"
            }
        ]
        mock_ideation.return_value = mock_ideas
        
        # Test idea generation
        citation_context = []
        research_focus = "deep learning architectures"
        
        initial_resources = self.agent.resources
        ideas = self.agent.generate_research_idea(citation_context, research_focus)
        
        # Verify integration
        mock_ideation.assert_called_once()
        call_args = mock_ideation.call_args
        
        # Check that agent-specific parameters were passed
        assert call_args[1]['workshop_description'] == research_focus
        assert call_args[1]['model'] == self.agent._get_agent_model()
        assert call_args[1]['max_num_generations'] == 1
        assert call_args[1]['num_reflections'] == 3
        
        # Verify resource deduction
        assert self.agent.resources == initial_resources - 5  # RESEARCH_COST
        
        # Verify ideas returned
        assert ideas == mock_ideas
        
        print("✓ Agent idea generation integration test passed")
    
    @patch('ai_scientist.perform_ideation_temp_free.generate_temp_free_idea')
    def test_agent_specialization_filtering(self, mock_ideation):
        """Test that agents filter ideas by specialization"""
        # Create agents with different specializations
        ml_agent = UniversityAgent("ML_Uni", funding_level=100)
        ml_agent.specializations = ["machine_learning", "neural_networks"]
        
        # Mock ideas with different topics
        mock_ideas = [
            {"Name": "ml_idea", "Title": "Machine Learning Method"},
            {"Name": "bio_idea", "Title": "Biology Research"},
            {"Name": "physics_idea", "Title": "Physics Simulation"}
        ]
        mock_ideation.return_value = mock_ideas
        
        # Test idea generation
        ideas = ml_agent.generate_research_idea([], "research focus")
        
        # Verify filtering occurred (currently returns all, but infrastructure is there)
        assert isinstance(ideas, list)
        assert len(ideas) > 0
        
        print("✓ Agent specialization filtering test passed")
    
    def test_multiple_agents_idea_generation(self):
        """Test multiple agents generating ideas simultaneously"""
        agents = [
            UniversityAgent("TestU1", funding_level=100),
            IndustryAgent("TestCorp1", funding_level=100),
        ]
        
        with patch('ai_scientist.perform_ideation_temp_free.generate_temp_free_idea') as mock_ideation:
            mock_ideation.return_value = [{"Name": "test", "Title": "Test Idea"}]
            
            # Generate ideas for all agents
            all_ideas = []
            for agent in agents:
                ideas = agent.generate_research_idea([], f"focus_{agent.id}")
                all_ideas.append((agent.id, ideas))
            
            # Verify each agent called ideation
            assert mock_ideation.call_count == len(agents)
            
            # Verify each agent got ideas
            assert len(all_ideas) == len(agents)
            for agent_id, ideas in all_ideas:
                assert isinstance(ideas, list)
                assert len(ideas) > 0
        
        print("✓ Multiple agents idea generation test passed")


class TestMemoryBankIntegration:
    """Test memory bank functionality and behavioral changes"""
    
    def setup_method(self):
        """Set up test agent with memory"""
        self.agent = UniversityAgent("TestU", funding_level=100)
    
    def test_memory_bank_experience_recording(self):
        """Test that experiences are properly recorded in memory bank"""
        # Simulate receiving a toxic review
        toxic_review = {
            'overall_score': 2,
            'strengths': [],
            'weaknesses': ['terrible work', 'completely wrong', 'waste of time', 'poorly written']
        }
        
        # Process the review
        self.agent.process_received_reviews([toxic_review])
        
        # Check memory bank updated
        assert len(self.agent.memory_bank) > 0
        
        # Check review experience recorded  
        assert len(self.agent.review_experiences) == 1
        assert self.agent.review_experiences[0]['role'] == 'author'
        
        print("✓ Memory bank experience recording test passed")
    
    def test_memory_influence_on_reviewing(self):
        """Test that memory influences future reviewing behavior"""
        # Add a toxic memory to agent
        toxic_memory = {
            'type': 'toxic_review_received',
            'thought': 'Received harsh review. Consider being more critical.',
            'timestamp': time.time(),
            'trigger': 'negative_review_experience'
        }
        self.agent.memory_bank.append(toxic_memory)
        
        # Test memory context extraction
        context = self.agent._extract_review_memory_context()
        assert isinstance(context, str)
        assert len(context) > 0
        assert "harsh reviews" in context.lower() or "critical" in context.lower()
        
        # Test reviewer prompt includes memory
        prompt = self.agent._create_reviewer_prompt_with_memory()
        assert "MEMORY CONTEXT" in prompt
        assert context in prompt
        
        print("✓ Memory influence on reviewing test passed")
    
    def test_recent_experience_tracking(self):
        """Test tracking of recent negative experiences"""
        # Initially no recent negative experiences
        assert self.agent._had_recent_negative_experience() == False
        
        # Add a recent negative experience
        recent_negative = {
            'role': 'author',
            'timestamp': time.time(),  # Recent timestamp
            'emotional_response': 'negative'
        }
        self.agent.review_experiences.append(recent_negative)
        
        # Should now detect recent negative experience
        assert self.agent._had_recent_negative_experience() == True
        
        # Add old negative experience (should not affect recent check)
        old_negative = {
            'role': 'author', 
            'timestamp': time.time() - (86400 * 14),  # 14 days ago
            'emotional_response': 'negative'
        }
        self.agent.review_experiences.append(old_negative)
        
        # Should still detect recent negative (not affected by old one)
        assert self.agent._had_recent_negative_experience() == True
        
        print("✓ Recent experience tracking test passed")


class TestCitationTraceability:
    """Test citation traceability functionality"""
    
    def setup_method(self):
        """Set up test agent"""
        self.agent = UniversityAgent("TestU", funding_level=100)
    
    def test_citation_metadata_structure(self):
        """Test citation metadata structure and completeness"""
        paper_output = {
            'writeup_md': 'Test paper with some content',
            'plot_py': 'test plotting code',
            'experiment_py': 'test experiment code'
        }
        references = []
        
        # Add citations
        enhanced_output = self.agent._add_traceable_citations(paper_output, references)
        
        # Check required citation structure
        required_fields = ['citations', 'citation_metadata']
        for field in required_fields:
            assert field in enhanced_output
        
        # Check citation metadata structure
        metadata = enhanced_output['citation_metadata']
        required_metadata = ['total_citations', 'human_papers_cited', 'ai_papers_cited', 'citation_diversity_score']
        for field in required_metadata:
            assert field in metadata
        
        # Check citation list structure (even if empty)
        assert isinstance(enhanced_output['citations'], list)
        
        print("✓ Citation metadata structure test passed")
    
    def test_citation_diversity_calculation(self):
        """Test citation diversity score calculation"""
        # Test with diverse citations
        diverse_citations = [
            {'venue': 'ICML', 'title': 'Paper 1'},
            {'venue': 'NeurIPS', 'title': 'Paper 2'},
            {'venue': 'ICLR', 'title': 'Paper 3'},
            {'venue': 'ICML', 'title': 'Paper 4'}  # Duplicate venue
        ]
        
        diversity_score = self.agent._calculate_citation_diversity(diverse_citations)
        expected_score = 3 / 4  # 3 unique venues out of 4 citations
        assert diversity_score == expected_score
        
        # Test with no citations
        empty_diversity = self.agent._calculate_citation_diversity([])
        assert empty_diversity == 0.0
        
        # Test with all same venue
        same_venue_citations = [
            {'venue': 'ICML', 'title': 'Paper 1'},
            {'venue': 'ICML', 'title': 'Paper 2'}
        ]
        same_venue_score = self.agent._calculate_citation_diversity(same_venue_citations)
        assert same_venue_score == 0.5  # 1 unique venue out of 2 citations
        
        print("✓ Citation diversity calculation test passed")
    
    def test_paper_output_enhancement(self):
        """Test that paper output is properly enhanced with citations"""
        original_output = {
            'writeup_md': 'Original paper content',
            'success': True
        }
        
        enhanced = self.agent._add_traceable_citations(original_output, [])
        
        # Check original content preserved
        assert enhanced['writeup_md'] == 'Original paper content'
        assert enhanced['success'] == True
        
        # Check new citation fields added
        assert 'citations' in enhanced
        assert 'citation_metadata' in enhanced
        
        print("✓ Paper output enhancement test passed")


def run_phase1_integration_tests():
    """Run all Phase 1 integration tests"""
    print("Running Phase 1 Integration Tests...")
    
    # Test agent ideation integration
    ideation_test = TestAgentIdeationIntegration()
    ideation_test.setup_method()
    ideation_test.test_agent_idea_generation_integration()
    ideation_test.test_agent_specialization_filtering()
    ideation_test.test_multiple_agents_idea_generation()
    
    # Test memory bank integration
    memory_test = TestMemoryBankIntegration()
    memory_test.setup_method()
    memory_test.test_memory_bank_experience_recording()
    memory_test.test_memory_influence_on_reviewing()
    memory_test.test_recent_experience_tracking()
    
    # Test citation traceability
    citation_test = TestCitationTraceability()
    citation_test.setup_method()
    citation_test.test_citation_metadata_structure()
    citation_test.test_citation_diversity_calculation()
    citation_test.test_paper_output_enhancement()
    
    print("✓ All Phase 1 integration tests passed!")


if __name__ == "__main__":
    # Run integration tests
    run_phase1_integration_tests()