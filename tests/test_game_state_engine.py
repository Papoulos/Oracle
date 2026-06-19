import os
import json
import pytest
from game_state_engine import GameStateEngine, ActionResult

@pytest.fixture
def temp_character_file(tmp_path):
    char_file = tmp_path / "character.json"
    data = {
        "nom": "Test Hero",
        "niveau": 1,
        "xp": 0,
        "xp_prochain_niveau": 1000,
        "pv": 10,
        "ressources": {
            "points_de_vie": {"actuels": 10, "max": 10},
            "sorts_par_jour": {
                "niveau_1": {"restants": 2, "max": 2}
            },
            "points_de_rage": {"restants": 2, "max": 2}
        }
    }
    char_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(char_file)

def test_gse_load_save(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    assert gse.state["nom"] == "Test Hero"

    gse.state["nom"] = "Updated Hero"
    gse.save()

    with open(temp_character_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["nom"] == "Updated Hero"

def test_gse_get_hp(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    hp_cur, hp_max = gse.get_hp()
    assert hp_cur == 10
    assert hp_max == 10

def test_gse_apply_damage(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    res = gse.apply_damage(3)
    assert res.success is True
    assert gse.get_hp()[0] == 7
    assert gse.state["pv"] == 7 # legacy sync

def test_gse_apply_healing(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    gse.apply_damage(5)
    res = gse.apply_healing(3)
    assert res.success is True
    assert gse.get_hp()[0] == 8

def test_gse_consume_spell_slot(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    res = gse.consume_spell_slot(1)
    assert res.success is True
    assert gse.get_resource("sorts_par_jour", level=1)[0] == 1

    gse.consume_spell_slot(1)
    res = gse.consume_spell_slot(1)
    assert res.success is False
    assert res.blocked_reason == "no_spell_slot_remaining"

def test_gse_rest_long(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    gse.apply_damage(5)
    gse.consume_spell_slot(1)
    gse.consume_resource("points_de_rage")

    res = gse.rest("long")
    assert res.success is True
    assert gse.get_hp()[0] == 10
    assert gse.get_resource("sorts_par_jour", level=1)[0] == 2
    assert gse.get_resource("points_de_rage")[0] == 2

def test_gse_detect_action_type():
    gse = GameStateEngine()
    assert gse.detect_action_type("Je lance un sort") == "spell"
    assert gse.detect_action_type("Je me repose") == "rest"
    assert gse.detect_action_type("Je rentre en rage") == "rage"
    assert gse.detect_action_type("Je marche") is None

def test_gse_apply_orchestrator_decision(temp_character_file):
    gse = GameStateEngine(temp_character_file)

    # Damage
    gse.apply_orchestrator_decision({"action": "damage", "amount": 2})
    assert gse.get_hp()[0] == 8

    # Heal
    gse.apply_orchestrator_decision({"action": "heal", "amount": 1})
    assert gse.get_hp()[0] == 9

    # XP
    res = gse.apply_orchestrator_decision({"action": "xp", "amount": 100})
    assert gse.state["xp"] == 100
    assert res.success is True
