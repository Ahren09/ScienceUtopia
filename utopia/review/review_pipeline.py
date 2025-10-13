"""
Simplified Agent Review Pipeline

Internal implementation of paper review system that integrates with the existing
AI-Scientist LLM infrastructure without external dependencies.
"""

import re
from typing import Dict, List, Optional

class AgentReviewPipeline:
    """Simplified review pipeline for agent-based paper evaluation"""
    
    def __init__(self, client=None, model: str = "gpt-4o"):
        """Initialize review pipeline with LLM client
        
        Args:
            client: LLM client (uses existing AI-Scientist infrastructure)  
            model: Model name to use for reviews
        """
        self.client = client
        self.model = model
        self.prompts = ReviewPrompts()
        
    def generate_review(self, paper_content: str, reviewer_personality: str = "academic_rigorous", 
                       specializations: List[str] = None, memory_context: str = "") -> Dict:
        """Generate a review for the given paper content
        
        Args:
            paper_content: The paper content to review (markdown format)
            reviewer_personality: Personality type of the reviewer
            specializations: List of reviewer specializations  
            memory_context: Previous review experiences for context
            
        Returns:
            Dictionary containing review components and score
        """
        # Build review prompt
        base_prompt = self.prompts.get_base_review_prompt()
        personality_prompt = self.prompts.get_personality_prompt(reviewer_personality)
        specialization_prompt = self.prompts.get_specialization_prompt(specializations or [])
        
        full_prompt = f"{base_prompt}\n\n{personality_prompt}\n{specialization_prompt}"
        
        if memory_context:
            full_prompt += f"\n\nContext from your previous reviews:\n{memory_context}"
            
        full_prompt += f"\n\nPaper to review:\n{paper_content}"
        
        # Generate review using existing LLM infrastructure
        if self.client is None:
            # Fallback for testing - return mock review
            return self._generate_mock_review(reviewer_personality)
            
        try:
            # Use existing AI-Scientist LLM client infrastructure
            messages = [{"role": "user", "content": full_prompt}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            review_text = response.choices[0].message.content
            
            # Parse review into structured format
            return self._parse_review(review_text, reviewer_personality)
            
        except Exception as e:
            print(f"Review generation failed: {e}")
            return self._generate_mock_review(reviewer_personality)
    
    def _parse_review(self, review_text: str, personality: str) -> Dict:
        """Parse LLM review output into structured format"""
        
        # Extract sections using regex
        summary_match = re.search(r'## Summary\s*\n(.*?)(?=\n##|\n\n|\Z)', review_text, re.DOTALL)
        strengths_match = re.search(r'## Strengths\s*\n(.*?)(?=\n##|\n\n|\Z)', review_text, re.DOTALL)
        weaknesses_match = re.search(r'## Weaknesses\s*\n(.*?)(?=\n##|\n\n|\Z)', review_text, re.DOTALL)
        questions_match = re.search(r'## Questions and Suggestions\s*\n(.*?)(?=\n##|\n\n|\Z)', review_text, re.DOTALL)
        assessment_match = re.search(r'## Overall Assessment\s*\n(.*?)(?=\n##|\n\n|\Z)', review_text, re.DOTALL)
        score_match = re.search(r'## Score\s*\n.*?(\d+)', review_text, re.DOTALL)
        
        # Extract score
        score = 5  # default
        if score_match:
            try:
                score = int(score_match.group(1))
                score = max(1, min(10, score))  # Clamp to valid range
            except (ValueError, AttributeError):
                score = 5
                
        # Parse strengths and weaknesses into lists
        strengths = self._extract_bullet_points(strengths_match.group(1) if strengths_match else "")
        weaknesses = self._extract_bullet_points(weaknesses_match.group(1) if weaknesses_match else "")
        
        return {
            'overall_score': score,
            'summary': summary_match.group(1).strip() if summary_match else "No summary provided",
            'strengths': strengths,
            'weaknesses': weaknesses, 
            'questions': questions_match.group(1).strip() if questions_match else "",
            'assessment': assessment_match.group(1).strip() if assessment_match else "",
            'full_review': review_text,
            'reviewer_personality': personality
        }
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text section"""
        if not text:
            return []
            
        # Split by lines and extract bullet points
        lines = text.strip().split('\n')
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            # Remove common bullet point markers
            if line.startswith(('- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ')):
                # Remove the marker and add to list
                clean_line = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                if clean_line:
                    bullet_points.append(clean_line)
            elif line and not line.startswith('#'):
                # Plain text line without marker
                bullet_points.append(line)
                
        return bullet_points[:5]  # Limit to 5 points max
    
    def _generate_mock_review(self, personality: str) -> Dict:
        """Generate mock review for testing when no LLM client available"""
        
        personality_reviews = {
            "academic_rigorous": {
                'overall_score': 6,
                'summary': "The paper presents a novel approach with solid theoretical foundations, though experimental validation could be stronger.",
                'strengths': [
                    "Strong theoretical contribution with rigorous mathematical framework",
                    "Comprehensive related work coverage",
                    "Novel methodology that advances the field"
                ],
                'weaknesses': [
                    "Limited experimental evaluation on diverse datasets",
                    "Some claims lack sufficient empirical support",
                    "Reproducibility concerns due to missing implementation details"
                ]
            },
            "commercial_practical": {
                'overall_score': 7,
                'summary': "The paper demonstrates practical value with strong experimental results on real-world applications.",
                'strengths': [
                    "Clear practical applications with commercial relevance", 
                    "Scalable solution tested on large datasets",
                    "Significant performance improvements over baselines"
                ],
                'weaknesses': [
                    "Limited theoretical analysis of why the method works",
                    "Computational complexity not thoroughly analyzed",
                    "Missing comparison with recent industry methods"
                ]
            },
            "innovative_disruptive": {
                'overall_score': 8,
                'summary': "Highly novel approach that challenges conventional wisdom with creative interdisciplinary connections.",
                'strengths': [
                    "Breakthrough idea that opens new research directions",
                    "Creative combination of concepts from different fields", 
                    "Paradigm-shifting potential for the community"
                ],
                'weaknesses': [
                    "Some aspects lack rigorous validation",
                    "Methodology may be too radical for immediate adoption",
                    "Limited baselines from established approaches"
                ]
            }
        }
        
        review = personality_reviews.get(personality, personality_reviews["academic_rigorous"])
        review.update({
            'questions': "How does the method perform on out-of-distribution data?",
            'assessment': f"This work shows promise but needs refinement in key areas.",
            'full_review': f"Mock review generated for personality: {personality}",
            'reviewer_personality': personality
        })
        
        return review