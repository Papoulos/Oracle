import sys
import copy
from typing import Any
import pytest

from mechanics.task_flow import (
    build_task_analysis_prompt,
    sanitize_task_decision,
    format_task_roll_line,
    enforce_engine_owned_fields
)
from systems.loader import load_pack

@pytest.fixture
def cypher_pack():
    return load_pack("cypher_system")

def test_no_langchain_dependency():
    """Ensure task_flow can be imported without langchain."""
    if "mechanics.task_flow" in sys.modules:
        del sys.modules["mechanics.task_flow"]
    sys.modules["langchain_core"] = None
    import mechanics.task_flow
    assert "mechanics.task_flow" in sys.modules

def test_build_task_analysis_prompt(cypher_pack):
    sheet = {
        "resources": {
            "might": {"current": 10, "max": 12},
            "speed": {"current": "abc", "max": 15}, # Invalid current
            "intellect": {"current": 10} # Missing max
        }
    }

    prompt = build_task_analysis_prompt(
        pack=cypher_pack,
        sheet=sheet,
        user_input="Je frappe le gobelin",
        chronicle_text="Aventure...",
        npcs_json="[]",
        state_summary="Might: 10/12",
        mechanical_context="Nothing",
        scale_text="0 (Routine) - 10 (Impossible)"
    )

    assert "Je frappe le gobelin" in prompt
    assert "Aventure..." in prompt
    assert "0 (Routine) - 10 (Impossible)" in prompt

    # Check pool formatting
    # Note: Cypher System pack uses pool names without translated suffix in the tests here,
    # or the test needs to match exactly what is in the pack
    assert "- might (Might) : 10/12" in prompt or "- might (Puissance (Might)) : 10/12" in prompt
    assert "- speed (Speed) : ?/15" in prompt or "- speed (Vitesse (Speed)) : ?/15" in prompt
    assert "- intellect (Intellect) : 10/?" in prompt or "- intellect (Intellect (Intellect)) : 10/?" in prompt

    # Check guidance from pack
    assert "Procédure du MJ pour toute tâche" in prompt

def test_sanitize_task_decision_types(cypher_pack):
    # Test valid dict
    valid_raw = {
        "need_roll": True,
        "action_id": "task",
        "pool": "might",
        "difficulty": 4,
        "ease": {"skill": 1, "asset": "2", "invalid": 3},
        "hinder": "1",
        "effort_levels": 2.0
    }
    decision = sanitize_task_decision(valid_raw, cypher_pack)
    assert decision is not None
    assert decision["difficulty"] == 4
    assert decision["ease"]["skill"] == 1
    assert decision["ease"]["asset"] == 2
    assert "invalid" not in decision["ease"]
    assert decision["hinder"] == 1
    assert decision["effort_levels"] == 2

    # Test invalid need_roll
    assert sanitize_task_decision({"need_roll": "true"}, cypher_pack) is None
    assert sanitize_task_decision({"need_roll": False}, cypher_pack) is None

    # Test invalid difficulty
    assert sanitize_task_decision({"need_roll": True, "difficulty": "4"}, cypher_pack) is None
    assert sanitize_task_decision({"need_roll": True, "difficulty": 11}, cypher_pack) is None
    assert sanitize_task_decision({"need_roll": True, "difficulty": True}, cypher_pack) is None

    # Test invalid pool
    assert sanitize_task_decision({"need_roll": True, "difficulty": 4, "pool": "unknown"}, cypher_pack) is None

    # Test weird types map to 0
    weird_raw = {
        "need_roll": True,
        "action_id": "task",
        "pool": "might",
        "difficulty": 4,
        "ease": {"skill": 2.5, "asset": -1, "tool": True},
        "hinder": -5,
        "effort_levels": "abc"
    }
    decision2 = sanitize_task_decision(weird_raw, cypher_pack)
    assert decision2 is not None
    assert decision2["ease"] == {}
    assert decision2["hinder"] == 0
    assert decision2["effort_levels"] == 0

def test_sanitize_task_decision_empty_pack():
    class DummyPack:
        actions = None
    assert sanitize_task_decision({"need_roll": True}, DummyPack()) is None

def test_sanitize_task_decision_missing_pool(cypher_pack):
    # Action requires a pool
    valid_raw_no_pool = {
        "need_roll": True,
        "action_id": "task",
        "difficulty": 4
    }
    # Test absent
    assert sanitize_task_decision(valid_raw_no_pool, cypher_pack) is None
    # Test None
    valid_raw_no_pool["pool"] = None
    assert sanitize_task_decision(valid_raw_no_pool, cypher_pack) is None
    # Test empty string
    valid_raw_no_pool["pool"] = ""
    assert sanitize_task_decision(valid_raw_no_pool, cypher_pack) is None

    # For an action that does NOT require pools, pool_id should become None
    from mechanics.actions import TaskAction
    cypher_pack.actions.actions.append(TaskAction(
        id="no_pool_task",
        name="No Pool Task",
        kind="task",
        pools=[]
    ))
    decision = sanitize_task_decision({"need_roll": True, "action_id": "no_pool_task", "difficulty": 4}, cypher_pack)
    assert decision is not None
    assert decision["pool"] is None

def test_format_task_roll_line(cypher_pack, tmp_path, monkeypatch):
    import random
    import copy
    from game_state_engine import GameStateEngine

    # We use a real engine and true outputs
    def build_engine():
        engine = GameStateEngine(pack=cypher_pack)
        engine.state = {
            "effort": 2,
            "edge": {"might": 1, "speed": 0, "intellect": 0},
            "resources": {
                "might": {"current": 10, "max": 12},
                "speed": {"current": 8, "max": 10},
                "intellect": {"current": 2, "max": 10}
            }
        }
        # Disable saving to avoid creating Memory/
        monkeypatch.setattr(engine, "save", lambda: None)
        return engine

    # 1. Success with effort
    engine = build_engine()
    engine.rng = random.Random(5) # seed 5 : dé = 20 -> SUCCESS
    res1 = engine.execute_task(action_id="task", pool_id="might", difficulty=4, effort_levels=1)
    assert res1.state_changes["task"]["outcome"] == "SUCCESS"
    line1 = format_task_roll_line(res1.state_changes, cypher_pack)
    assert line1 == "Tâche — Puissance (Might) : difficulté 4 (Difficult) → 3 (Demanding), cible 9, d20 = 20 → Succès (marge +11). Effort : 1 niveau(x), coût 2, Puissance (Might) 10→8."
    assert "None" not in line1

    # 2. Failure without effort
    engine = build_engine()
    engine.rng = random.Random(42) # seed 42 : dé = 4 (target 12 -> fail)
    res2 = engine.execute_task(action_id="task", pool_id="speed", difficulty=4, effort_levels=0)
    assert res2.state_changes["task"]["outcome"] == "FAILURE"
    line2 = format_task_roll_line(res2.state_changes, cypher_pack)
    assert line2 == "Tâche — Vitesse (Speed) : difficulté 4 (Difficult) → 4 (Difficult), cible 12, d20 = 4 → Échec (marge -8)."
    assert "None" not in line2

    # 3. Auto success (difficulty 0)
    engine = build_engine()
    engine.rng = random.Random(42)
    res3 = engine.execute_task(action_id="task", pool_id="intellect", difficulty=0, effort_levels=0)
    line3 = format_task_roll_line(res3.state_changes, cypher_pack)
    assert line3 == "Tâche — Intellect : difficulté 0 (Routine) → 0 (Routine) : réussite automatique, aucun dé."
    assert "None" not in line3

    # 4. Impossible with effort waived (difficulty 10, 1 level)
    engine = build_engine()
    engine.rng = random.Random(42)
    res4 = engine.execute_task(action_id="task", pool_id="might", difficulty=10, effort_levels=1)
    line4 = format_task_roll_line(res4.state_changes, cypher_pack)
    assert line4 == "Tâche — Puissance (Might) : difficulté 10 (Impossible) → 9 (Immortal), cible 27 inatteignable : aucun dé, échec. Effort non dépensé."
    assert "None" not in line4

    # 5. Impossible without requested effort (no 'Effort non dépensé')
    engine = build_engine()
    engine.rng = random.Random(42)
    res5 = engine.execute_task(action_id="task", pool_id="might", difficulty=10, effort_levels=0)
    line5 = format_task_roll_line(res5.state_changes, cypher_pack)
    assert line5 == "Tâche — Puissance (Might) : difficulté 10 (Impossible) → 10 (Impossible), cible 30 inatteignable : aucun dé, échec."
    assert "Effort non dépensé" not in line5
    assert "None" not in line5

    # 6. Roll with warning (skill -3 capped)
    engine = build_engine()
    engine.rng = random.Random(42) # seed 42: dé = 4
    res6 = engine.execute_task(action_id="task", pool_id="speed", difficulty=4, step_adjustments={"skill": -3}, effort_levels=0)
    line6 = format_task_roll_line(res6.state_changes, cypher_pack)
    assert line6 == "Tâche — Vitesse (Speed) : difficulté 4 (Difficult) → 2 (Standard), cible 6, d20 = 4 → Échec (marge -2). Attention : skill capped at 2."
    assert "None" not in line6

    # 7. Missing level name
    import copy
    pack_copy = copy.deepcopy(cypher_pack)
    pack_copy.actions.difficulty_scale = [d for d in pack_copy.actions.difficulty_scale if d.level not in (4, 2)]

    engine_mod = GameStateEngine(pack=pack_copy)
    engine_mod.state = {
        "effort": 2,
        "edge": {"might": 1, "speed": 0, "intellect": 0},
        "resources": {
            "might": {"current": 10, "max": 12},
            "speed": {"current": 8, "max": 10},
            "intellect": {"current": 2, "max": 10}
        }
    }
    monkeypatch.setattr(engine_mod, "save", lambda: None)
    engine_mod.rng = random.Random(42) # seed 42: dé = 4

    res7 = engine_mod.execute_task(action_id="task", pool_id="speed", difficulty=4, step_adjustments={"skill": -3}, effort_levels=0)

    line7 = format_task_roll_line(res7.state_changes, pack_copy)
    assert line7 == "Tâche — Vitesse (Speed) : difficulté 4 → 2, cible 6, d20 = 4 → Échec (marge -2). Attention : skill capped at 2."
    assert "  " not in line7
    assert "()" not in line7
    assert "None" not in line7

def test_enforce_engine_owned_fields(cypher_pack):
    # Dummy pool group so we can test it since cypher pack has no pool_groups
    from mechanics.models import PoolGroupDef
    cypher_pack.resources.pool_groups.append(
        PoolGroupDef(
            id="recoveries",
            name="Recoveries",
            path="resources.pool_groups.recoveries",
            items=[],
            recovery=[]
        )
    )

    snapshot = {
        "resources": {
            "might": {"current": 10, "max": 12},
            "speed": {"current": 10, "max": 12},
            "pool_groups": {
                "recoveries": {"1": True, "2": False}
            }
        },
        "recovery_step": 1,
        "equipment": "Sword"
    }

    sheet = {
        "resources": {
            "might": {"current": 15, "max": 12}, # Modifié illégalement
            # speed supprimé
            "luck": {"current": 5}, # Nouveau champ hors pack (doit rester)
            "pool_groups": {
                "recoveries": {"1": True, "2": True, "3": False} # Modifié illégalement (ajout)
            }
        },
        "equipment": "Axe", # Modifié légalement (narratif)
        "pending_allocation": {"points": 2} # Ajouté illégalement
    }

    import copy
    snap_copy = copy.deepcopy(snapshot)
    sheet_copy = copy.deepcopy(sheet)

    corrected, restored = enforce_engine_owned_fields(sheet, snapshot, cypher_pack)

    # Verify non-mutation
    assert snapshot == snap_copy
    assert sheet == sheet_copy

    # Verify restorations
    assert corrected["resources"]["might"]["current"] == 10
    assert corrected["resources"]["speed"]["current"] == 10
    # Wait, the snapshot recoveries had {"1": True, "2": False}. The restored dict must be exactly that.
    assert corrected["resources"]["pool_groups"]["recoveries"] == {"1": True, "2": False}
    assert corrected["recovery_step"] == 1
    assert "pending_allocation" not in corrected

    # Verify non-engine modifications kept
    assert corrected["equipment"] == "Axe"
    assert corrected["resources"]["luck"]["current"] == 5

    # Restored paths list
    assert "resources.might.current" in restored
    assert "resources.speed.current" in restored
    assert "resources.speed.max" in restored
    assert "resources.pool_groups.recoveries" in restored
    assert "recovery_step" in restored
    assert "pending_allocation" in restored

def test_enforce_engine_owned_fields_empty_and_string_parent(cypher_pack):
    snapshot = {
        "resources": {}
    }

    sheet_empty = {
        "resources": {
            "might": {
                "current": 5,
                "max": 12
            }
        }
    }

    # engine path includes resources.might.current. But it's absent in snapshot.
    # Therefore it should be deleted from sheet.
    # Without _clean_empty_dicts, the parent might and resources should stay.
    corrected_empty, _ = enforce_engine_owned_fields(sheet_empty, snapshot, cypher_pack)
    assert corrected_empty == {"resources": {"might": {}}}

    # Test parent as string
    sheet_str = {
        "resources": "n/a"
    }
    snapshot_str = {
        "resources": {
            "might": {
                "current": 10,
                "max": 12
            }
        }
    }
    # It should not crash, and should recreate the dicts.
    corrected_str, restored_str = enforce_engine_owned_fields(sheet_str, snapshot_str, cypher_pack)
    assert corrected_str["resources"]["might"]["current"] == 10
    assert "resources.might.current" in restored_str
