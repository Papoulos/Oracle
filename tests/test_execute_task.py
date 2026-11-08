import pytest
import json
import random
from pathlib import Path
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
    pack = load_pack("pbta_minimal", base_dir=Path(__file__).parent / "fixtures")
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
    assert "DD 5 (Très facile)" in desc
    assert "DD 15 (Moyen)" in desc

def test_execute_task_impossible_waives_effort(engine):
    # Diff 10, target 30 initially, reduced to target 27. Impossible.
    # Speed pool starts at 8. Effort cost would be something.
    rng_state = engine.rng.getstate()
    with open(engine.character_path) as f:
        before = f.read()

    res = engine.execute_task("task", "speed", 10, effort_levels=1)

    assert res.success is True
    assert "Effort not spent: the task is impossible." in res.message
    assert res.state_changes["task"]["impossible"] is True
    assert res.state_changes["effort"]["levels"] == 1
    assert res.state_changes["effort"]["cost"] == 0
    assert res.state_changes["effort"]["waived"] is True
    assert "speed" not in res.state_changes

    assert engine.rng.getstate() == rng_state
    with open(engine.character_path) as f:
        after = f.read()
    assert before == after

def test_execute_task_possible_spends_effort(engine):
    # Diff 7 (Impossible), target 21, but 1 level of effort makes it diff 6, target 18 (possible).
    rng_state = engine.rng.getstate()
    res = engine.execute_task("task", "speed", 7, effort_levels=1)

    assert res.success is True
    assert "Effort not spent" not in res.message
    assert res.state_changes["task"]["impossible"] is False
    assert res.state_changes["effort"]["levels"] == 1
    assert res.state_changes["effort"]["cost"] > 0
    assert res.state_changes["effort"]["waived"] is False
    assert res.state_changes["speed"]["avant"] == 8
    assert res.state_changes["speed"]["apres"] == 8 - res.state_changes["effort"]["cost"]

    assert engine.rng.getstate() != rng_state

def test_execute_task_still_impossible_waives_effort(engine):
    # Diff 8, target 24. 1 level of effort -> diff 7, target 21. Still impossible.
    rng_state = engine.rng.getstate()
    with open(engine.character_path) as f:
        before = f.read()

    res = engine.execute_task("task", "speed", 8, effort_levels=1)

    assert res.success is True
    assert "Effort not spent: the task is impossible." in res.message
    assert res.state_changes["task"]["impossible"] is True
    assert res.state_changes["effort"]["levels"] == 1
    assert res.state_changes["effort"]["cost"] == 0
    assert res.state_changes["effort"]["waived"] is True
    assert "speed" not in res.state_changes

    assert engine.rng.getstate() == rng_state
    with open(engine.character_path) as f:
        after = f.read()
    assert before == after

def test_execute_task_no_actions_step_target_fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMS_DIR", "systems")
    # Load step target pack which shouldn't have actions.json or might not have one?
    pack = load_pack("step_target_pack", base_dir=Path(__file__).parent / "fixtures")
    e = GameStateEngine(character_path=str(tmp_path / "c.json"), pack=pack)
    res = e.execute_task("task", "hp", 4)
    assert res.success is False
    assert res.blocked_reason == "no_actions"


class _FixedRollRng:
    def __init__(self, roll_val: int):
        self.roll_val = roll_val

    def randint(self, a: int, b: int) -> int:
        return self.roll_val


def test_attack_damage_effort_and_npc_defeat(engine):
    # Roll 18 on d20 (+2 bonus damage), base_damage=4 (medium weapon), 1 level of damage effort (+3 dmg)
    engine.rng = _FixedRollRng(18)
    engine.state["effort"] = 2

    res = engine.execute_task(
        "attack",
        "might",
        difficulty=3,
        effort_levels=0,
        damage_effort_levels=1,
        base_damage=4,
        target_npc="GARDE_01",
        npc_health_max=9,
        npc_armor=1,
    )
    assert res.success is True
    assert res.state_changes["task"]["outcome"] == "SUCCESS"
    # Effort cost: 1 level with edge 1 = 3 - 1 = 2 points from might (10 -> 8)
    assert res.state_changes["effort"]["cost"] == 2
    assert res.state_changes["effort"]["damage_levels"] == 1
    assert res.state_changes["might"] == {"avant": 10, "apres": 8}

    # Attack damage: base 4 + effort 3 + roll_bonus 2 (on 18) = 9
    atk = res.state_changes["attack_damage"]
    assert atk == {"base": 4, "effort_bonus": 3, "roll_bonus": 2, "total": 9}

    # NPC takes 9 - 1 (armor) = 8 damage -> 9 -> 1 HP
    npc = res.state_changes["npc_damage"]
    assert npc["avant"] == 9
    assert npc["apres"] == 1
    assert npc["defeated"] is False

    # Second hit defeats the NPC
    res2 = engine.execute_task(
        "attack",
        "might",
        difficulty=3,
        base_damage=4,
        target_npc="GARDE_01",
    )
    assert res2.success is True
    assert res2.state_changes["npc_damage"]["apres"] == 0
    assert res2.state_changes["npc_damage"]["defeated"] is True
    assert engine.state["npc_health"]["GARDE_01"]["current"] == 0


def test_natural_20_refunds_effort_and_major_effect(engine):
    engine.rng = _FixedRollRng(20)
    res = engine.execute_task(
        "attack",
        "might",
        difficulty=4,
        effort_levels=1,
        damage_effort_levels=1,
        base_damage=4,
    )
    assert res.success is True
    assert res.state_changes["task"]["special_roll"] == "major_effect"
    assert res.state_changes["effort"]["refunded"] is True
    assert res.state_changes["effort"]["cost"] == 0
    # Pool was not deducted!
    assert engine.state["resources"]["might"]["current"] == 10
    # Damage includes +4 from natural 20 and +3 from 1 damage effort level
    assert res.state_changes["attack_damage"]["total"] == 4 + 3 + 4


def test_natural_1_gm_intrusion(engine):
    engine.rng = _FixedRollRng(1)
    res = engine.execute_task("attack", "might", difficulty=3)
    assert res.success is True
    assert res.state_changes["task"]["outcome"] == "FAILURE"
    assert res.state_changes["task"]["special_roll"] == "gm_intrusion"


def test_defend_failure_applies_armor_pool_cascade_and_damage_track(engine):
    engine.rng = _FixedRollRng(2)  # fails difficulty 4 (target 12)
    engine.state["armor"] = 2
    engine.state["damage_track"] = "hale"
    # might=10, speed=8, intellect=2

    # Incoming damage 14 - 2 armor = 12 effective -> drains might (10->0) and speed (8->6)
    res = engine.execute_task(
        "defend",
        "speed",
        difficulty=4,
        incoming_damage=14,
        damage_pool="might",
    )
    assert res.success is True
    assert res.state_changes["task"]["outcome"] == "FAILURE"
    dmg = res.state_changes["damage_taken"]
    assert dmg["raw"] == 14
    assert dmg["armor"] == 2
    assert dmg["effective"] == 12
    assert engine.state["resources"]["might"]["current"] == 0
    assert engine.state["resources"]["speed"]["current"] == 6
    assert engine.state["damage_track"] == "impaired"

    # Now character is impaired: effort costs +1 per level automatically, roll bonus on 19 is capped at +1, and minor_effect is suppressed
    engine.rng = _FixedRollRng(19)
    res_imp = engine.execute_task("attack", "speed", difficulty=3, effort_levels=1, base_damage=4)
    assert res_imp.success is True
    # 1 level on speed (edge 0) while impaired = 3 + 1 = 4
    assert res_imp.state_changes["effort"]["cost"] == 4
    assert res_imp.state_changes["attack_damage"]["roll_bonus"] == 1
    assert "special_roll" not in res_imp.state_changes["task"]

    # Take 2 more damage ignoring armor -> drains remaining speed (2->0) while intellect stays at 2 -> debilitated
    dmg_res = engine.apply_damage(2, pool_id="speed", ignore_armor=True)
    assert dmg_res.success is True
    assert engine.state["resources"]["speed"]["current"] == 0
    assert engine.state["resources"]["intellect"]["current"] == 2
    assert engine.state["damage_track"] == "debilitated"

    # Debilitated blocks task execution
    blocked = engine.execute_task("task", "intellect", difficulty=2)
    assert blocked.success is False
    assert blocked.blocked_reason == "debilitated"

    # Recovering points via allocate_recovery restores damage_track from debilitated -> hale
    engine.state["pending_allocation"] = {
        "rule_id": "recovery_roll",
        "trigger": "one_action",
        "points": 5,
        "among": ["might", "speed", "intellect"],
    }
    alloc_res = engine.allocate_recovery({"might": 3, "speed": 2})
    assert alloc_res.success is True
    assert engine.state["damage_track"] == "hale"
    assert alloc_res.state_changes["damage_track"] == {"avant": "debilitated", "apres": "hale"}

    # Full cascade to dead when all 3 pools hit 0
    dead_res = engine.apply_damage(20, ignore_armor=True)
    assert dead_res.success is True
    assert engine.state["damage_track"] == "dead"
    blocked_dead = engine.execute_task("task", "might", difficulty=1)
    assert blocked_dead.success is False
    assert blocked_dead.blocked_reason == "dead"


def test_execute_task_dnd5e_d20vstarget(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMS_DIR", "systems")
    pack = load_pack("dnd5e_srd")
    char_file = tmp_path / "dnd_char.json"
    char_file.write_text(json.dumps({
        "name": "Gimli",
        "statistics": {"Force": 16, "Dextérité": 12},
        "resources": {"hit_points": {"current": 20, "max": 20}},
    }))
    e = GameStateEngine(character_path=str(char_file), pack=pack, rng=_FixedRollRng(14))

    # Ability check with stat lookup (Force 16 -> +3 modifier) vs DC 15: 14 + 3 = 17 -> SUCCESS
    res = e.execute_task("ability_check", None, difficulty=15, stat="Force")
    assert res.success is True
    assert res.state_changes["task"]["outcome"] == "SUCCESS"
    assert res.state_changes["task"]["modifier"] == 3
    assert res.state_changes["task"]["total"] == 17

    # Attack with base_damage and target_npc
    res_atk = e.execute_task("attack", None, difficulty=13, modifier=5, base_damage=8, target_npc="GOBELIN_01", npc_health_max=7)
    assert res_atk.success is True
    assert res_atk.state_changes["attack_damage"]["total"] == 8
    assert res_atk.state_changes["npc_damage"]["defeated"] is True

    # Saving throw failure with incoming_damage
    e.rng = _FixedRollRng(3)
    res_save = e.execute_task("saving_throw", None, difficulty=15, stat="Dextérité", incoming_damage=6)
    assert res_save.success is True
    assert res_save.state_changes["task"]["outcome"] == "FAILURE"
    assert e.state["resources"]["hit_points"]["current"] == 14


def test_execute_task_pbta_and_dice_pool(tmp_path):
    from mechanics.actions import ActionsConfig, TaskAction

    # PbtA 2d6
    pbta_pack = load_pack("pbta_minimal", base_dir=Path(__file__).parent / "fixtures")
    pbta_pack.actions = ActionsConfig(actions=[TaskAction(id="act_under_pressure", name="Agir sous pression", kind="task", pools=[])])
    e_pbta = GameStateEngine(character_path=str(tmp_path / "pbta.json"), pack=pbta_pack, rng=_FixedRollRng(4))
    res_pbta = e_pbta.execute_task("act_under_pressure", None, difficulty=0, modifier=1, stat="Cool")
    assert res_pbta.success is True
    assert res_pbta.state_changes["task"]["total"] == 9
    assert res_pbta.state_changes["task"]["outcome"] == "PARTIAL_SUCCESS"
    assert res_pbta.state_changes["task"]["tier_id"] == "weak"

    # DicePoolSuccess (success_threshold is 8 in dice_pool_pack)
    pool_pack = load_pack("dice_pool_pack", base_dir=Path(__file__).parent / "fixtures")
    pool_pack.actions = ActionsConfig(actions=[TaskAction(id="skill_test", name="Test de compétence", kind="task", pools=[])])
    e_pool = GameStateEngine(character_path=str(tmp_path / "pool.json"), pack=pool_pack, rng=_FixedRollRng(8))
    res_pool = e_pool.execute_task("skill_test", None, difficulty=2, pool_size=4)
    assert res_pool.success is True
    assert res_pool.state_changes["task"]["outcome"] == "SUCCESS"
    assert res_pool.state_changes["task"]["total"] == 4


