"""
Specialized Agent Types for Multi-Agent Scientific Ecosystem

EXTENDS base agent classes WITHOUT modifying existing AI-Scientist code
"""

from .base_agent import SimulationAgent
from typing import List, Dict, Optional
import random
from utopia.models.models import LLM, GPT

class UniversityResearcher(SimulationAgent):
    """University researchers serve as both authors and reviewers in the ecosystem

    Multiple UniversityResearchers can be mapped to the same University.
    The university affiliation is tracked separately from the researcher identity.
    """

    def __init__(
        self,
        researcher_name: str,
        university_name: str,
        funding_level: int = 70,
        expertise: Optional[List] = None,
        llm = None
    ):
        """Initialize a university researcher

        Args:
            researcher_name: Name of the individual researcher
            university_name: Name of the university this researcher belongs to
            funding_level: Initial funding level (0-100)
            expertise_directions: List of 3 ResearchDirection objects representing inherent expertise.
                                 If None, will be randomly sampled during initialization.
        """
        super().__init__(
            reputation=5,
            funding_level=funding_level,
            agent_id=researcher_name,
            llm=llm
        )
        self.university_name = university_name
        self.can_author = True  # University researchers can write papers
        self.can_review = True  # University researchers can review papers

        # Store inherent expertise (3 research directions)
        self.expertise = expertise
        
    def _get_personality_prompt(self) -> str:
        return """You are a university researcher who both authors papers and reviews them for conferences. 
               You focus on theoretical contributions, novel methodologies, and advancing fundamental knowledge. 
               You value reproducibility, peer review rigor, and long-term scientific impact. Your reviewing 
               style is influenced by your own experiences as an author in the academic ecosystem."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply university-specific biases to reviews"""
        # Universities tend to value theoretical rigor and reproducibility
        strengths_text = ' '.join(review.get('strengths', []))
        if 'theoretical' in strengths_text.lower() or 'theoretical_contribution' in strengths_text:
            # Slight positive bias for theoretical work
            review['overall_score'] = min(10, review.get('overall_score', 5) + 0.5)
        
        if 'reproducibility' not in strengths_text.lower() and review.get('overall_score', 5) > 7:
            # Penalize high scores if reproducibility concerns exist
            review['overall_score'] = max(1, review.get('overall_score', 5) - 0.5)
            
        return review


class IndustryResearcher(SimulationAgent):
    """Industry researchers serve as both authors and reviewers in the ecosystem

    Multiple IndustryResearchers can be mapped to the same Company.
    The company affiliation is tracked separately from the researcher identity.
    """

    def __init__(
        self,
        researcher_name: str,
        company_name: str,
        funding_level: int = 90,
        expertise: Optional[List] = None,
        llm: LLM = None
    ):
        """Initialize an industry researcher

        Args:
            researcher_name: Name of the individual researcher
            company_name: Name of the company this researcher belongs to
            funding_level: Initial funding level (0-100)
            expertise_directions: List of 3 ResearchDirection objects representing inherent expertise.
                                 If None, will be randomly sampled during initialization.
        """

        super().__init__(
            reputation=6,
            funding_level=funding_level,
            agent_id=researcher_name,
            llm=llm
        )
        self.expertise = expertise
        self.company_name = company_name
        self.can_author = True  # Industry researchers can write papers
        self.can_review = True  # Industry researchers can review papers

        # Store inherent expertise (3 research directions)
        
    def _get_personality_prompt(self) -> str:
        return """You are an industry researcher who both authors papers and reviews them for conferences.
               You focus on practical applications, scalable solutions, and commercially viable innovations. 
               You prioritize efficiency, real-world impact, and competitive advantages. Your reviewing style 
               is influenced by your commercial perspective and your own publication experiences."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply industry-specific biases to reviews"""
        # Industry agents prefer practical applications and scalability
        strengths_text = ' '.join(review.get('strengths', []))
        if any(term in strengths_text.lower() for term in ['practical', 'scalable', 'real-world']):
            review['overall_score'] = min(10, review.get('overall_score', 5) + 0.7)
            
        weaknesses_text = ' '.join(review.get('weaknesses', []))
        if 'theoretical' in weaknesses_text.lower():
            # Less penalty for theoretical limitations
            review['overall_score'] = max(1, review.get('overall_score', 5) + 0.3)
            
        return review


class FreelancerAgent(SimulationAgent):
    """Freelancer agents primarily focus on authoring but may occasionally review"""

    def __init__(
        self,
        researcher_name: str,
        funding_level: int = 30,
        agent_id: str = None,
        expertise_directions: Optional[List] = None,
        llm: LLM = None
    ):

        assert agent_id is not None, "agent_id is required"
        super().__init__(
            reputation=4,
            funding_level=funding_level,
            agent_id=researcher_name,
            llm=llm
        )

        self.can_author = True   # Freelancers primarily write papers
        self.can_review = True   # Freelancers can review, but less frequently invited

        # Store inherent expertise (3 research directions)
        self.expertise_directions = expertise_directions or []
        
    def _get_personality_prompt(self) -> str:
        return """You are an independent researcher who challenges conventional wisdom 
               and pursues unconventional approaches. You value radical innovation, 
               interdisciplinary thinking, and paradigm-shifting discoveries. When reviewing,
               you appreciate bold ideas and are willing to champion unconventional work."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply freelancer-specific biases to reviews"""
        # Freelancers appreciate novelty and unconventional approaches
        if any(term in str(review.get('strengths', [])).lower() 
               for term in ['novel', 'innovative', 'unconventional', 'creative']):
            review['overall_score'] = min(10, review.get('overall_score', 5) + 1.0)
            
        if 'lacks novelty' in str(review.get('weaknesses', [])).lower():
            # Strong penalty for lack of novelty
            review['overall_score'] = max(1, review.get('overall_score', 5) - 1.5)
            
        return review


class FundingAgent(SimulationAgent):
    """Funding agents primarily review and allocate resources, occasionally author policy papers"""
    
    def __init__(self, agency_name: str, funding_level: int = 100, llm: LLM = None):
        
        
        super().__init__(
            reputation=8,
            funding_level=funding_level,
            agent_id=agency_name,
            llm=llm
        )
        self.llm = llm
        self.funding_priorities = []
        self.risk_aversion = "moderate"
        self.can_author = False  # Funding agents typically don't write research papers
        self.can_review = True   # Funding agents frequently serve as reviewers
        
    def _get_personality_prompt(self) -> str:
        return """You are a funding agency representative who primarily serves as a reviewer
               for academic conferences. You evaluate research from a strategic funding perspective,
               considering long-term impact, resource allocation efficiency, and alignment with
               funding priorities. Your reviewing style is conservative but fair."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply funding agency-specific biases to reviews"""
        # Funding agents are conservative and strategic
        if any(term in str(review.get('strengths', [])).lower() 
               for term in ['impact', 'strategic', 'significant']):
            review['overall_score'] = min(10, review.get('overall_score', 5) + 0.3)
            
        if any(term in str(review.get('weaknesses', [])).lower() 
               for term in ['high-risk', 'speculative', 'unproven']):
            # Conservative penalty for high-risk work
            review['overall_score'] = max(1, review.get('overall_score', 5) - 0.8)
            
        return review
        
    def allocate_funding(self, proposals: List[Dict]) -> List[Dict]:
        """Use existing review infrastructure for funding decisions"""
        funding_decisions = []
        for proposal in proposals:
            # Use review pipeline to evaluate funding proposals
            evaluation = self.review_paper(proposal)  # Reuse review infrastructure
            funding_decision = self._make_funding_decision(evaluation, proposal)
            funding_decisions.append(funding_decision)
        return funding_decisions
        
    def _make_funding_decision(self, evaluation: Dict, proposal: Dict) -> Dict:
        """Make funding decision based on evaluation"""
        score = evaluation.get('overall_score', 0)
        
        # Conservative funding thresholds
        if score >= 8.0:
            decision = "fund"
            amount = "high"
        elif score >= 6.5:
            decision = "fund"  
            amount = "medium"
        elif score >= 5.5:
            decision = "conditional_fund"
            amount = "low"
        else:
            decision = "reject"
            amount = "none"
            
        return {
            'proposal_id': proposal.get('id', 'unknown'),
            'decision': decision,
            'funding_amount': amount,
            'evaluation_score': score,
            'reviewer_id': self.id,
            'rationale': evaluation.get('summary', 'No rationale provided')
        }


# Import time for agent ID generation
import time


# Backward compatibility aliases (deprecated - will be removed in future versions)
UniversityAgent = UniversityResearcher
IndustryAgent = IndustryResearcher