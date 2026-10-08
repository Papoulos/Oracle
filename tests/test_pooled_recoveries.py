import pytest
import os
import json
import random
from game_state_engine import GameStateEngine
from mechanics.models import ResourcesConfig
from systems.validate import validate_pack
from systems.loader import load_pack

def get_engine(character_file, pack_id):
    pack = load_pack(pack_id, base_dir='tests/fixtures')
    return GameStateEngine(character_file, pack=pack)


@pytest.fixture
def mock_systems_dir(monkeypatch):
    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")
    return "tests/fixtures"

@pytest.fixture
def character_file(tmp_path):
    char_path = tmp_path / "character.json"
    data = {
        "tier": 1,
        "resources": {
            "might": {"current": 5, "max": 10},
            "speed": {"current": 2, "max": 10},
            "intellect": {"current": 8, "max": 10}
        }
    }
    with open(char_path, "w") as f:
        json.dump(data, f)
    return str(char_path)

def test_rest_player_choice_pending_allocation(character_file, monkeypatch):
    monkeypatch.setenv("SYSTEM_PACK", "cypher_recovery_pack")
    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")


    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")



    engine = get_engine(character_file, os.environ['SYSTEM_PACK'])
    # Fix rng to return 4 + 1 = 5 points
    random.seed(42)
    engine.rng = random.Random(42)
    # test random.randint equivalent to simulate d(1,6)

    # We should override the rng in expr evaluator but the pack uses rng context
    res = engine.rest("one_action_rest")

    assert res.success
    assert "pending_allocation" in res.state_changes
    pending = res.state_changes["pending_allocation"]
    assert pending["rule_id"] == "recovery_roll"
    assert pending["among"] == ["might", "speed", "intellect"]

    # Check save
    with open(character_file, "r") as f:
        data = json.load(f)
    assert "pending_allocation" in data
    assert data["pending_allocation"]["points"] > 0
    points = data["pending_allocation"]["points"]

def test_allocate_recovery_valid(character_file, monkeypatch):
    monkeypatch.setenv("SYSTEM_PACK", "cypher_recovery_pack")
    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")



    # Setup pending allocation
    with open(character_file, "w") as f:
        json.dump({
            "tier": 1,
            "resources": {
                "might": {"current": 5, "max": 10},
                "speed": {"current": 2, "max": 10},
                "intellect": {"current": 8, "max": 10}
            },
            "pending_allocation": {
                "rule_id": "recovery_roll",
                "trigger": "one_action_rest",
                "points": 5,
                "among": ["might", "speed", "intellect"]
            }
        }, f)

    engine = get_engine(character_file, os.environ['SYSTEM_PACK'])
    res = engine.allocate_recovery({"might": 2, "speed": 3})

    assert res.success
    assert "pending_allocation" not in engine.state
    assert engine.state["resources"]["might"]["current"] == 7
    assert engine.state["resources"]["speed"]["current"] == 5

    changes = res.state_changes
    assert "allocated" in changes
    assert changes["allocated"]["might"]["apres"] == 7
    assert changes["wasted"]["total"] == 0

def test_allocate_recovery_invalid(character_file, monkeypatch):
    monkeypatch.setenv("SYSTEM_PACK", "cypher_recovery_pack")
    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")



    with open(character_file, "w") as f:
        json.dump({
            "tier": 1,
            "resources": {
                "might": {"current": 5, "max": 10}
            },
            "pending_allocation": {
                "rule_id": "recovery_roll",
                "trigger": "one_action_rest",
                "points": 3,
                "among": ["might"]
            }
        }, f)

    engine = get_engine(character_file, os.environ['SYSTEM_PACK'])

    # Invalid key
    res = engine.allocate_recovery({"speed": 1})
    assert not res.success

    # Invalid value (negative)
    res = engine.allocate_recovery({"might": -1})
    assert not res.success

    # Exceeds total
    res = engine.allocate_recovery({"might": 5})
    assert not res.success

    # Wasted via cap (valeur > manque plafonnée avec wasted)
    # might missing is 5, but points is 3, let's change missing to 2
    engine.state["resources"]["might"]["current"] = 8
    engine.save()

    res = engine.allocate_recovery({"might": 3})
    assert res.success
    assert res.state_changes["wasted"]["capped"]["might"] == 1
    assert res.state_changes["wasted"]["total"] == 1

def test_rest_blocked_by_pending(character_file, monkeypatch):
    monkeypatch.setenv("SYSTEM_PACK", "cypher_recovery_pack")
    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")



    with open(character_file, "w") as f:
        json.dump({
            "pending_allocation": {"rule_id": "recovery_roll", "points": 5, "among": ["might"]}
        }, f)

    engine = get_engine(character_file, os.environ['SYSTEM_PACK'])
    res = engine.rest("one_action_rest")
    assert not res.success

def test_auto_in_order(character_file, monkeypatch):
    monkeypatch.setenv("SYSTEM_PACK", "cypher_recovery_auto_pack")
    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")


    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")



    engine = get_engine(character_file, os.environ['SYSTEM_PACK'])
    random.seed(42)
    engine.rng = random.Random(42)
    # Say points_expr yields 6 points
    # Might missing = 5. Speed missing = 8. Intellect missing = 2
    # 5 goes to Might (filling it). 1 goes to Speed.

    res = engine.rest("one_action_rest")
    assert res.success

    assert engine.state["resources"]["might"]["current"] == 10
    assert engine.state["resources"]["speed"]["current"] == 4
    assert engine.state["resources"]["intellect"]["current"] == 8

def test_next_rest_sequence(character_file, monkeypatch):
    monkeypatch.setenv("SYSTEM_PACK", "cypher_recovery_pack")
    monkeypatch.setenv("SYSTEMS_DIR", "tests/fixtures")



    engine = get_engine(character_file, os.environ['SYSTEM_PACK'])
    random.seed(42)
    engine.rng = random.Random(42)

    # Call 1 -> one_action_rest
    res = engine.next_rest("cypher_rests")
    assert res.success
    assert res.state_changes["sequence"]["next_index"] == 1
    assert engine.state.get("recovery_step") == 1

    # Clear pending so we can rest again
    del engine.state["pending_allocation"]
    engine.save()

    # Call 2 -> ten_minute_rest
    res = engine.next_rest("cypher_rests")
    assert res.success
    assert res.state_changes["sequence"]["next_index"] == 2
    assert engine.state.get("recovery_step") == 2

    # Clear again
    del engine.state["pending_allocation"]
    engine.save()

    # Call 3 -> one_hour_rest
    engine.next_rest("cypher_rests")
    del engine.state["pending_allocation"]

    # Call 4 -> ten_hour_rest
    res = engine.next_rest("cypher_rests")
    assert res.state_changes["sequence"]["next_index"] == 0 # wraps around

def test_validate_pack_errors():
    # test syntax and validation
    issues = validate_pack("tests/fixtures/cypher_recovery_pack")
    # should have no errors (maybe warnings)
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 0
