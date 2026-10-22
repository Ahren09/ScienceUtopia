"""Portable P/F state is kept separate from the retired sequential protocol."""
from dataclasses import asdict
from types import SimpleNamespace
import pytest
from utopia.funding.feedback import FundingFeedbackMixin, Mechanisms
from utopia.runtime.checkpointing import PortableCheckpointError

class Base:
    def __init__(self, **kwargs):
        self.args = kwargs.get("args", SimpleNamespace(portable_checkpoint=True))
    def extra_checkpoint_state(self):
        return {"base_marker": 7}
    def restore_extra_checkpoint_state(self, state):
        assert state == {"base_marker": 7}
    def load_checkpoint(self, year):
        return ("portable", year)

class World(FundingFeedbackMixin, Base):
    pass

@pytest.mark.parametrize("cell", ["P1F1", "P0F1", "P1F0", "P0F0"])
def test_funding_state_round_trip_and_subclass_chain(cell):
    original = World(mechanisms=Mechanisms.from_cell(cell))
    original.funding_feedback_years = {1: {"metrics": {"accepted_papers_per_founder": 0.25}}}
    original.initial_world = {"agents": ["a", "b"]}
    original.initial_world_hash = "world"
    original.founder_ids = ("a", "b")
    original.corpus_hash = "corpus"
    restored = World(mechanisms=Mechanisms.from_cell(cell))
    restored.restore_extra_checkpoint_state(original.extra_checkpoint_state())
    assert restored.funding_feedback_years == original.funding_feedback_years
    assert restored.initial_world == original.initial_world
    assert restored.initial_world_hash == "world"
    assert restored.founder_ids == ("a", "b")
    assert restored.corpus_hash == "corpus"
    assert restored.load_checkpoint(1) == ("portable", 1)

def test_cannot_resume_a_different_funding_treatment():
    saved = World(mechanisms=Mechanisms.from_cell("P1F1")).extra_checkpoint_state()
    with pytest.raises(PortableCheckpointError):
        World(mechanisms=Mechanisms.from_cell("P1F0")).restore_extra_checkpoint_state(saved)

def test_legacy_unlabelled_snapshot_still_rejected():
    with pytest.raises(RuntimeError, match="fresh worlds"):
        World(args=SimpleNamespace()).load_checkpoint(1)
