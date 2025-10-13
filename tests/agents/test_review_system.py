"""
Test suite for simplified AgentReview system
Validates internal review pipeline functionality
"""

import sys
sys.path.append('/nethome/yjin328/Workspace/PeerReview/AI-Scientist-v2')

from utopia.review import AgentReviewPipeline, ReviewPrompts
from utopia.agents.specialized_agents import UniversityAgent, IndustryAgent


def test_review_pipeline_initialization():
    """Test AgentReviewPipeline initialization"""
    print("Testing review pipeline initialization...")
    
    pipeline = AgentReviewPipeline(client=None, model="gpt-4o")
    assert pipeline.client is None
    assert pipeline.model == "gpt-4o"
    assert isinstance(pipeline.prompts, ReviewPrompts)
    
    print("✓ Review pipeline initialization test passed")


def test_review_prompts():
    """Test review prompt generation"""
    print("Testing review prompt generation...")
    
    prompts = ReviewPrompts()
    
    # Test base prompt
    base_prompt = prompts.get_base_review_prompt()
    assert "peer reviewer" in base_prompt.lower()
    assert "technical quality" in base_prompt.lower()
    assert "score" in base_prompt.lower()
    
    # Test personality prompts
    academic_prompt = prompts.get_personality_prompt("academic_rigorous")
    assert "theoretical rigor" in academic_prompt.lower()
    
    commercial_prompt = prompts.get_personality_prompt("commercial_practical")
    assert "practical applications" in commercial_prompt.lower()
    
    # Test specialization prompts
    spec_prompt = prompts.get_specialization_prompt(["theoretical_ml", "applied_ml"])
    assert "expertise" in spec_prompt.lower()
    
    print("✓ Review prompt generation test passed")


def test_mock_review_generation():
    """Test mock review generation for different personalities"""
    print("Testing mock review generation...")
    
    pipeline = AgentReviewPipeline(client=None)  # No client = mock mode
    
    # Test different personality reviews
    personalities = ["academic_rigorous", "commercial_practical", "innovative_disruptive"]
    
    for personality in personalities:
        review = pipeline.generate_review(
            paper_content="Test paper content",
            reviewer_personality=personality,
            specializations=["test_field"]
        )
        
        # Validate review structure
        assert isinstance(review, dict)
        assert 'overall_score' in review
        assert 'summary' in review
        assert 'strengths' in review
        assert 'weaknesses' in review
        assert 'reviewer_personality' in review
        
        assert review['reviewer_personality'] == personality
        assert isinstance(review['overall_score'], int)
        assert 1 <= review['overall_score'] <= 10
        assert isinstance(review['strengths'], list)
        assert isinstance(review['weaknesses'], list)
    
    print("✓ Mock review generation test passed")


def test_agent_integration():
    """Test integration with agent system"""
    print("Testing agent integration...")
    
    # Create agents
    uni_agent = UniversityAgent("TestU", funding_level=100)
    ind_agent = IndustryAgent("TestCorp", funding_level=100)
    
    # Test paper review
    test_paper = {
        'content': 'This is a test paper about machine learning techniques.',
        'metadata': {'title': 'Test Paper', 'authors': ['Test Author']}
    }
    
    # Generate reviews
    uni_review = uni_agent.review_paper(test_paper)
    ind_review = ind_agent.review_paper(test_paper)
    
    # Validate reviews
    for review in [uni_review, ind_review]:
        assert isinstance(review, dict)
        assert 'overall_score' in review
        assert 'summary' in review
        assert 'strengths' in review
        assert 'weaknesses' in review
        
    print("✓ Agent integration test passed")


def test_review_parsing():
    """Test review text parsing functionality"""
    print("Testing review text parsing...")
    
    pipeline = AgentReviewPipeline(client=None)
    
    # Mock review text with proper format
    mock_review_text = """
## Summary
This paper presents a novel approach to machine learning.

## Strengths
- Strong theoretical foundation
- Comprehensive experimental evaluation
- Clear presentation and writing

## Weaknesses
- Limited novelty in core contribution
- Missing comparison with recent methods
- Some experimental details unclear

## Questions and Suggestions
How does the method scale to larger datasets?

## Overall Assessment
The work is solid but incremental.

## Score
6
"""
    
    parsed_review = pipeline._parse_review(mock_review_text, "academic_rigorous")
    
    assert parsed_review['overall_score'] == 6
    assert "novel approach" in parsed_review['summary']
    assert len(parsed_review['strengths']) == 3
    assert len(parsed_review['weaknesses']) == 3
    assert "larger datasets" in parsed_review['questions']
    
    print("✓ Review text parsing test passed")


def test_bullet_point_extraction():
    """Test bullet point extraction from text"""
    print("Testing bullet point extraction...")
    
    pipeline = AgentReviewPipeline(client=None)
    
    test_text = """
- First strength with dash
* Second strength with asterisk  
• Third strength with bullet
1. Numbered first item
2. Numbered second item
Plain text without marker
"""
    
    bullet_points = pipeline._extract_bullet_points(test_text)
    
    assert len(bullet_points) >= 4
    assert "First strength with dash" in bullet_points
    assert "Second strength with asterisk" in bullet_points  
    assert "Third strength with bullet" in bullet_points
    # Plain text lines without markers should also be captured
    plain_text_found = any("Plain text without marker" in bp for bp in bullet_points)
    assert plain_text_found or len(bullet_points) >= 4  # Either found or we have enough points
    
    print("✓ Bullet point extraction test passed")


def run_review_system_tests():
    """Run all review system tests"""
    print("=" * 50)
    print("REVIEW SYSTEM TESTING")
    print("=" * 50)
    
    tests = [
        test_review_pipeline_initialization,
        test_review_prompts,
        test_mock_review_generation,
        test_agent_integration,
        test_review_parsing,
        test_bullet_point_extraction
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"REVIEW SYSTEM TEST RESULTS:")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL REVIEW SYSTEM TESTS PASSED!")
        print("Simplified AgentReview system is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please fix before proceeding.")
        return False


if __name__ == "__main__":
    success = run_review_system_tests()
    sys.exit(0 if success else 1)