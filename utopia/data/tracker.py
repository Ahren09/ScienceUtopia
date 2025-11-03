"""Metrics tracking for the simulation."""

from typing import Dict, List, Set, Union, Any
import logging
from utopia.utils.generail_utils import print_colored
from utopia.agents.researcher_agents import UniversityResearcher, IndustryResearcher

logger = logging.getLogger(__name__)


class FundingTracker:
    """Tracks global funding consumption across all agents to maintain constant total funding."""

    def __init__(self):
        self.funding_allocation_per_cycle: Dict[int, Dict[str, float]] = {}  # year -> total funding
        self.consumption_per_cycle: Dict[int, Dict[str, float]] = {}     # year -> total consumed
        
        self.growth_rate = 0.1 # 10% growth rate for the fundings

    def start_cycle(self, year: int, agents: Dict[str, Any]) -> float:
        """Record total funding at start of cycle.

        Args:
            year: Current year/cycle
            agents: Dictionary of agent_id -> agent with resources attribute

        Returns:
            Total funding across all agents
        """
        
        
        total_university = sum(
            agent.resources
            for agent in agents.values()
            if isinstance(agent, UniversityResearcher)
        )
        total_industry = sum(
            agent.resources
            for agent in agents.values()
            if isinstance(agent, IndustryResearcher)
        )
        
        
        total = total_university + total_industry
        self.funding_allocation_per_cycle[year] = {
            "university": 0.0,
            "industry": 0.0
        }
        self.consumption_per_cycle[year] = {
            "university": 0.0,
            "industry": 0.0
        }
        return total

    def record_funding_consumption(self, year: int, amount: float, agent_type: str):
        """Record funding consumed during this cycle.

        Args:
            year: Current year/cycle
            amount: Amount of funding consumed (positive value)
        """
        if year not in self.consumption_per_cycle:
            self.consumption_per_cycle[year] = {
                "university": 0.0,
                "industry": 0.0
            }
            
        original_amount = self.consumption_per_cycle[year][agent_type]
        self.consumption_per_cycle[year][agent_type] += amount
        logger.debug(f"[Funding Consumption] {agent_type} - {original_amount} -> {self.consumption_per_cycle[year][agent_type]}")
        
        
    def record_funding_allocation(self, year: int, amount: float, agent_type: str):
        """Record funding allocated during this cycle.

        Args:
            year: Current year/cycle
            amount: Amount of funding allocated (positive value)
        """
        assert year > 0, f"Year must be greater than 0. Got {year}"
        if year not in self.funding_allocation_per_cycle:
            self.funding_allocation_per_cycle[year] = {
                "university": 0.0,
                "industry": 0.0
            }
            
        target_total = self.consumption_per_cycle[year][agent_type] * (1 + self.growth_rate)
        for yr in range(year - 1, 0, -1):
                
            gap = self.consumption_per_cycle[yr][agent_type] * (1 + self.growth_rate) - self.funding_allocation_per_cycle[yr][agent_type]
            target_total += gap
            
        target_total = int(target_total)
        
            
        original_amount = self.funding_allocation_per_cycle[year][agent_type]
        self.funding_allocation_per_cycle[year][agent_type] += amount
        
        logger.debug(f"[Funding Allocation] {agent_type} - {original_amount} -> {self.funding_allocation_per_cycle[year][agent_type]} (Target: {target_total})")
        
        assert self.funding_allocation_per_cycle[year][agent_type] <= target_total, f"Funding allocation for {agent_type} in year {year} is greater than the target total {target_total}. Current: {self.funding_allocation_per_cycle[year][agent_type]}, Target: {target_total}."
        
        
    def get_funding_consumption(self, year: int):
        """Get funding consumption for a given year and agent type."""
        return self.consumption_per_cycle[year]
        
    def get_funding_allocation_amount(self, year: int, agent_type: str):
        """Record funding allocated during this cycle.

        Args:
            year: Current year/cycle
            amount: Amount of funding allocated (positive value)
        """
        
        amount = 0
        for yr in range(year, 0, -1):
            if yr not in self.funding_allocation_per_cycle:
                self.funding_allocation_per_cycle[yr] = {
                    "university": 0.0,
                    "industry": 0.0
                }
                
            gap = self.consumption_per_cycle[yr][agent_type] * (1 + self.growth_rate) - self.funding_allocation_per_cycle[yr][agent_type]
            
            logger.debug(f"[Funding Allocation] {year} - Consumed: {self.consumption_per_cycle[year][agent_type]} - Allocated: {self.funding_allocation_per_cycle[year][agent_type]}, Diff: {gap}")
            amount += gap
        
        logger.debug(f"[Funding Allocation] Total amount: {amount:.2f}")
        return int(amount)
        # if year not in self.funding_allocation_per_cycle:
        #     self.funding_allocation_per_cycle[year] = {
        #         "university": 0.0,
        #         "industry": 0.0
        #     }
        

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "funding_allocation_per_cycle": self.funding_allocation_per_cycle,
            "consumption_per_cycle": self.consumption_per_cycle
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FundingTracker":
        """Deserialize from dict."""
        tracker = cls()
        tracker.funding_allocation_per_cycle = data.get("funding_allocation_per_cycle", {})
        tracker.funding_allocation_per_cycle = {int(year): allocation for year, allocation in tracker.funding_allocation_per_cycle.items()}
        
        tracker.consumption_per_cycle = data.get("consumption_per_cycle", {})
        tracker.consumption_per_cycle = {int(year): consumption for year, consumption in tracker.consumption_per_cycle.items()}
        return tracker

class AgentTracker:
    """Tracks metrics for the simulation."""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        
        
    def record_agent_resources(self, agent_id: str, resources: Union[float, int], year: int):
        """Record the resources of an agent for a given year."""
        if agent_id not in self.resources:
            self.resources[agent_id] = []

        # Check for existing records
        for record in self.resources[agent_id]:
            assert record['year'] <= year, f"Previous record for agent {agent_id} in year {record['year']} has a greater year than the current year {year}. Record: {record}"

            # If record already exists for this year, verify consistency and skip adding
            if record['year'] == year:
                assert record['resources'] == resources, f"Resources for agent {agent_id} in year {year} are not the same as the current resources {resources}. Record: {record}"
                return  # Don't add duplicate
            
        if len(self.resources[agent_id]) == 0:
            logger.debug(f"[Resource] {agent_id} - {resources} (First record)")
            
        else:
            logger.debug(f"[Resource] {agent_id} - {self.resources[agent_id][-1]['resources']} -> {resources}")

        # Add new record only if it doesn't exist for this year
        self.resources[agent_id].append({
            'year': year,
            'resources': resources
        })
        
        

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {"resources": self.resources}

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentTracker":
        """Deserialize from dict."""
        tracker = cls()
        tracker.resources = data.get("resources", {})
        if not tracker.resources:
            print_colored("AgentTracker: No resources found in checkpoint data", "red")
        return tracker


class CitationTracker:
    """Tracks which papers cite which papers."""

    def __init__(self):
        self.citations: Dict[str, Set[str]] = {}  # cited_paper_id -> [citing_paper_ids]

    def add_citations(self, citing_paper_id: str, cited_paper_ids: Union[str,List[str], Set[str]], year: int):
        """Add citations from a paper to its references."""
        for cited_id in cited_paper_ids:
            if cited_id not in self.citations:
                self.citations[cited_id] = set()
            if citing_paper_id not in self.citations[cited_id]:
                self.citations[cited_id].add(citing_paper_id)

    def get_citations(self, paper_id: str) -> List[str]:
        """Get papers that cite this paper."""
        return self.citations.get(paper_id, [])

    def get_citation_count(self, paper_id: str) -> int:
        """Get number of citations for a paper."""
        return len(self.citations.get(paper_id, []))

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {"citations": {k: list(v) for k, v in self.citations.items()}}

    @classmethod
    def from_dict(cls, data: Dict) -> "CitationTracker":
        """Deserialize from dict."""
        tracker = cls()
        tracker.citations = {k: set(v) for k, v in data["citations"].items()}
        return tracker
