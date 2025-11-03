from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Citation:
    """Citation entry - frozen to allow use in sets and as dict keys"""
    current_paper_id: str
    cited_paper_id: str

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'current_paper_id': self.current_paper_id,
            'cited_paper_id': self.cited_paper_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Citation':
        """Deserialize from dictionary"""
        return cls(
            current_paper_id=data['current_paper_id'],
            cited_paper_id=data['cited_paper_id']
        )
