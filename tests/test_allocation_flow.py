import pytest

from mechanics.task_flow import build_allocation_prompt, sanitize_allocation
from mechanics.models import PoolDef

class DummyResources:
    def __init__(self):
        self.pools = [
            PoolDef(id="hp", name="Health Points", kind="health", current_path="hp.current", max_value=20, max_path=None, recovery=[]),
            PoolDef(id="mp", name="Mana Points", kind="health", current_path="mp.current", max_path="mp.max", recovery=[])
        ]

class DummyPack:
    def __init__(self):
        self.resources = DummyResources()

@pytest.fixture
def dummy_pack():
    return DummyPack()

@pytest.fixture
def dummy_sheet():
    return {
        "hp": {"current": 10},
        "mp": {"current": 5, "max": 15}
    }

def test_build_allocation_prompt(dummy_pack, dummy_sheet):
    pending = {"points": 5, "among": ["hp", "mp", "unknown"]}
    prompt = build_allocation_prompt(dummy_pack, dummy_sheet, pending, "je donne tout aux pv")

    # Check that defensive formatting worked
    assert "Points to allocate: 5" in prompt
    assert "- hp (Health Points) : 10/20 (manque 10)" in prompt
    assert "- mp (Mana Points) : 5/15 (manque 10)" in prompt
    assert "- unknown (?) : ?/?" in prompt
    assert "je donne tout aux pv" in prompt

def test_build_allocation_prompt_defensive(dummy_pack):
    # Empty pending, empty sheet
    prompt = build_allocation_prompt(dummy_pack, {}, {}, "hello")
    assert "Points to allocate: ?" in prompt
    assert "hello" in prompt

def test_sanitize_allocation_decline():
    pending = {"points": 5, "among": ["hp", "mp"]}

    # Valid decline
    raw = {"understood": True, "decline": True, "allocation": {}}
    res = sanitize_allocation(raw, pending)
    assert res == {"action": "decline", "allocation": {}}

def test_sanitize_allocation_valid():
    pending = {"points": 5, "among": ["hp", "mp"]}

    # Valid allocation (mix of str and int is allowed if strictly convertible to int)
    raw = {"understood": True, "decline": False, "allocation": {"hp": 2, "mp": "3"}}
    res = sanitize_allocation(raw, pending)
    assert res == {"action": "allocate", "allocation": {"hp": 2, "mp": 3}}

    # Float convertible to int
    raw = {"understood": True, "decline": False, "allocation": {"hp": 2.0, "mp": 3}}
    res = sanitize_allocation(raw, pending)
    assert res == {"action": "allocate", "allocation": {"hp": 2, "mp": 3}}

    # Key not in among -> ignored
    raw = {"understood": True, "decline": False, "allocation": {"hp": 5, "sp": 2}}
    res = sanitize_allocation(raw, pending)
    assert res == {"action": "allocate", "allocation": {"hp": 5}}

def test_sanitize_allocation_invalid():
    pending = {"points": 5, "among": ["hp", "mp"]}

    # Not understood
    assert sanitize_allocation({"understood": False}, pending) is None
    assert sanitize_allocation({"understood": "true"}, pending) is None

    # Not a dict
    assert sanitize_allocation("not a dict", pending) is None

    # Total sum > points
    assert sanitize_allocation({"understood": True, "allocation": {"hp": 4, "mp": 2}}, pending) is None

    # Total sum == 0 without decline
    assert sanitize_allocation({"understood": True, "allocation": {"hp": 0, "mp": 0}}, pending) is None

    # Negative allocation
    assert sanitize_allocation({"understood": True, "allocation": {"hp": -1, "mp": 6}}, pending) is None

    # Invalid value format (not convertible to int)
    assert sanitize_allocation({"understood": True, "allocation": {"hp": "abc"}}, pending) is None
    assert sanitize_allocation({"understood": True, "allocation": {"hp": 2.5}}, pending) is None

    # Boolean value
    assert sanitize_allocation({"understood": True, "allocation": {"hp": True}}, pending) is None

def test_sanitize_allocation_invalid_pending():
    # Pending without points or invalid points
    assert sanitize_allocation({"understood": True}, {"among": ["hp"]}) is None
    assert sanitize_allocation({"understood": True}, {"points": -1, "among": ["hp"]}) is None

    # Pending without among or invalid among
    assert sanitize_allocation({"understood": True}, {"points": 5}) is None
    assert sanitize_allocation({"understood": True}, {"points": 5, "among": []}) is None
    assert sanitize_allocation({"understood": True}, {"points": 5, "among": [1, 2]}) is None
