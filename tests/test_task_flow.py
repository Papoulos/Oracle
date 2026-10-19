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
    assert "- might (Might): 10/12" in prompt or "- might (Puissance (Might)): 10/12" in prompt
    assert "- speed (Speed): ?/15" in prompt or "- speed (Vitesse (Speed)): ?/15" in prompt
    assert "- intellect (Intellect): 10/?" in prompt or "- intellect (Intellect (Intellect)): 10/?" in prompt

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
        "difficulty": 4,
        "ease": {"skill": 2.5, "asset": -1, "tool": True},
        "hinder": -5,
        "effort_levels": "abc"
    }
    decision2 = sanitize_task_decision(weird_raw, cypher_pack)
    assert decision2["ease"] == {}
    assert decision2["hinder"] == 0
    assert decision2["effort_levels"] == 0

def test_sanitize_task_decision_empty_pack():
    class DummyPack:
        actions = None
    assert sanitize_task_decision({"need_roll": True}, DummyPack()) is None

def test_format_task_roll_line(cypher_pack):
    # Normal success with effort
    state_changes = {
        "task": {
            "pool": "might",
            "difficulty": {"initial": 4, "final": 2, "name": "Standard"},
            "target": 6,
            "roll": 10,
            "outcome": "SUCCESS",
            "margin": 4
        },
        "effort": {
            "cost": 3,
            "levels": 1,
            "before": 10,
            "after": 7
        },
        "warnings": ["Low pool"]
    }

    line = format_task_roll_line(state_changes, cypher_pack)
    assert "Tâche — Puissance (Might) : difficulté 4 (Difficult) → 2 (Standard), cible 6, d20 = 10 → Succès (marge +4)." in line
    assert "Effort : 1 niveau(x), coût 3, Puissance (Might) 10→7." in line
    assert "Attention : Low pool." in line

    # Auto success
    state_changes_auto = {
        "task": {
            "pool": "speed",
            "difficulty": {"initial": 2, "final": 0, "name": "Routine"},
            "auto_success": True
        }
    }
    line2 = format_task_roll_line(state_changes_auto, cypher_pack)
    assert "Tâche — Vitesse (Speed) : difficulté 2 (Standard) → 0 (Routine) : réussite automatique, aucun dé." in line2

    # Impossible
    state_changes_imp = {
        "task": {
            "pool": "intellect",
            "difficulty": {"initial": 8, "final": 8, "name": "Heroic"},
            "impossible": True,
            "target": 24,
            "outcome": "FAILURE",
            "margin": -24
        },
        "effort": {
            "cost": 5,
            "levels": 2,
            "before": 10,
            "after": 10,
            "waived": True
    }
    }
    line3 = format_task_roll_line(state_changes_imp, cypher_pack)
    assert "cible 24 inatteignable : aucun dé, échec. Effort non dépensé." in line3

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
