import pytest
import json
import random
from game_state_engine import GameStateEngine
from systems.loader import load_pack

@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMS_DIR", "systems")
    pack = load_pack("cypher_system")

    char_file = tmp_path / "character.json"
    char_data = {
        "effort": 2,
        "edge": {
            "might": 1,
            "speed": 0,
            "intellect": 0
        },
        "resources": {
            "might": {"current": 10, "max": 12},
            "speed": {"current": 8, "max": 10},
            "intellect": {"current": 2, "max": 10}
        }
    }
    char_file.write_text(json.dumps(char_data))

    # Graîne fixe 42
    rng = random.Random(42)
    return GameStateEngine(character_path=str(char_file), pack=pack, rng=rng)

def test_execute_task_success_with_effort(engine):
    # With effort_levels=2, difficulty=4, edge=1.
    # Initial diff = 4, target 12. Adjustments = -2 effort. Final diff = 2, target = 6.
    # Random 42 -> expected roll is 4 for Python 3.12, but we can just check it exactly.
    rng2 = random.Random(42)
    expected_roll = rng2.randint(1, 20)

    res = engine.execute_task("task", "might", 4, effort_levels=2)
    assert res.success is True

    expected_outcome = "SUCCESS" if expected_roll >= 6 else "FAILURE"
    assert expected_outcome in res.state_changes["task"]["outcome"]
    assert res.state_changes["task"]["difficulty"]["initial"] == 4
    assert res.state_changes["task"]["difficulty"]["final"] == 2
    assert res.state_changes["task"]["difficulty"]["name"] == "Standard"
    assert res.state_changes["task"]["target"] == 6
    assert res.state_changes["task"]["rolled"] == [expected_roll]

    # Cost check: effort levels = 2, edge = 1 -> 3 + 2(2-1) + 0 - 1 = 4
    assert res.state_changes["effort"]["cost"] == 4
    assert res.state_changes["might"]["avant"] == 10
    assert res.state_changes["might"]["apres"] == 6

    # Make sure JSON serializable
    json.dumps(res.state_changes)

    # File written
    with open(engine.character_path) as f:
        data = json.load(f)
        assert data["resources"]["might"]["current"] == 6

def test_execute_task_success_no_effort(engine):
    res = engine.execute_task("task", "might", 4)
    assert res.success is True
    assert "effort" not in res.state_changes
    assert "might" not in res.state_changes

    with open(engine.character_path) as f:
        data = json.load(f)
        assert data["resources"]["might"]["current"] == 10

def test_execute_task_invalid_effort(engine):
    with open(engine.character_path) as f:
        before = f.read()

    res = engine.execute_task("task", "might", 4, effort_levels=3)
    assert res.success is False
    assert res.blocked_reason == "invalid_effort"

    with open(engine.character_path) as f:
        after = f.read()
    assert before == after

def test_execute_task_insufficient_resources(engine):
    # Intellect has 2, cost for 1 level edge 0 is 3
    rng_state = engine.rng.getstate()
    res = engine.execute_task("task", "intellect", 4, effort_levels=1)
    assert res.success is False
    assert res.blocked_reason == "insufficient_resources"
    assert engine.rng.getstate() == rng_state

def test_execute_task_impaired(engine):
    res = engine.execute_task("task", "speed", 4, effort_levels=1, impaired=True)
    assert res.success is True
    # 1 level, 0 edge, impaired -> 3 + 0 + 1 - 0 = 4 cost
    assert res.state_changes["effort"]["cost"] == 4

def test_execute_task_step_adjustments(engine):
    res = engine.execute_task("task", "might", 4, effort_levels=1, step_adjustments={"skill": -1, "asset": -1})
    assert res.success is True
    assert res.state_changes["task"]["difficulty"]["initial"] == 4
    # effort 1 gives -1, skill -1, asset -1 = total -3. Diff 4 - 3 = 1
    assert res.state_changes["task"]["difficulty"]["final"] == 1
    assert res.state_changes["task"]["target"] == 3

def test_execute_task_effort_conflict(engine):
    res = engine.execute_task("task", "might", 4, effort_levels=1, step_adjustments={"effort": -1})
    assert res.success is False
    assert res.blocked_reason == "effort_conflict"

def test_execute_task_auto_success_zero(engine):
    res = engine.execute_task("task", "might", 0)
    assert res.success is True
    assert res.state_changes["task"]["difficulty"]["final"] == 0
    assert res.state_changes["task"]["rolled"] == []

def test_execute_task_invalid_difficulty(engine):
    res = engine.execute_task("task", "might", 11)
    assert res.success is False
    assert res.blocked_reason == "invalid_difficulty"

    res2 = engine.execute_task("task", "might", True)
    assert res2.success is False
    assert res2.blocked_reason == "invalid_difficulty"

def test_execute_task_invalid_pool_action_unknown(engine):
    res = engine.execute_task("task", "unknown_pool", 4)
    assert res.success is False
    assert res.blocked_reason == "invalid_pool"

    res2 = engine.execute_task("unknown_action", "might", 4)
    assert res2.success is False
    assert res2.blocked_reason == "unknown_action"
    assert "task" in res2.message

    res3 = engine.execute_task("task", None, 4)
    assert res3.success is False
    assert res3.blocked_reason == "invalid_pool"

def test_execute_task_no_actions(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMS_DIR", "systems")
    pack = load_pack("dnd5e_srd")
    e = GameStateEngine(character_path=str(tmp_path / "c.json"), pack=pack)
    res = e.execute_task("task", "hp", 4)
    assert res.success is False
    assert res.blocked_reason == "no_actions"

def test_execute_task_formula_error(engine):
    engine.pack.actions.actions[0].effort.cost_expr = "character.absent"
    res = engine.execute_task("task", "might", 4, effort_levels=1)
    assert res.success is False
    assert res.blocked_reason == "formula_error"

def test_describe_difficulty_scale(engine):
    desc = engine.describe_difficulty_scale()
    assert "Niveau 4 (Difficult, cible 12)" in desc
    assert "50/50 pour des gens formés" in desc

def test_describe_difficulty_scale_dnd5e(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMS_DIR", "systems")
    pack = load_pack("dnd5e_srd")
    e = GameStateEngine(character_path=str(tmp_path / "c.json"), pack=pack)
    desc = e.describe_difficulty_scale()
    assert desc == ""
