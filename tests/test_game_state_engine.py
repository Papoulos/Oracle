import os
import json
import pytest
from game_state_engine import GameStateEngine, ActionResult

@pytest.fixture
def temp_character_file(tmp_path):
    char_file = tmp_path / "character.json"
    data = {
        "name": "Test Hero",
        "level": 1,
        "xp": 0,
        "next_level_xp": 1000,
        "pv": 10,
        "resources": {
            "hit_points": {"current": 10, "max": 10},
            "spells_per_day": {
                "level_1": {"current": 2, "max": 2}
            },
            "points_de_rage": {"current": 2, "max": 2}
        }
    }
    char_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(char_file)

@pytest.fixture
def temp_character_file_pack(tmp_path):
    char_file = tmp_path / "character.json"
    data = {
        "name": "Test Hero",
        "level": 1,
        "xp": 0,
        "next_level_xp": 1000,
        "resources": {
            "hit_points": {
                "current": 10,
                "max": 10
            },
            "spells_per_day": {
                "level_1": {"current": 2, "max": 2}
            }
        }
    }
    char_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(char_file)

def test_gse_load_save(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    assert gse.state["name"] == "Test Hero"

    gse.state["name"] = "Updated Hero"
    gse.save()

    with open(temp_character_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["name"] == "Updated Hero"

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
    assert gse.get_resource("spells_per_day", level=1)[0] == 1

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
    assert gse.get_resource("spells_per_day", level=1)[0] == 2
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

from systems.loader import load_pack

def test_gse_pack_dnd5e_srd(temp_character_file_pack, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        gse = GameStateEngine(temp_character_file_pack)
        assert "mode expérimental, aucun system pack chargé" in caplog.text

    pack = load_pack("dnd5e_srd")

    with caplog.at_level(logging.INFO):
        gse_pack = GameStateEngine(temp_character_file_pack, pack=pack)
        assert "System pack loaded: dnd5e_srd" in caplog.text

    summary = gse_pack.get_state_summary()
    assert "Points de vie: 10/10" in summary
    assert "Emplacements de sorts level_1: 2/2" in summary

    # Assert specific exact string for D&D 5e SRD
    expected_summary = "Level: 1 | XP: 0/1000 | Points de vie: 10/10 | Emplacements de sorts level_1: 2/2"
    assert summary == expected_summary

    # Detect action using triggers.json
    assert gse_pack.detect_action_type("Je lance une boule de feu.") == "cast_spell"
    assert gse_pack.detect_action_type("Je fais un repos long.") == "rest:long_rest"

    # Faux positif test
    assert gse_pack.detect_action_type("Ceci est une relance.") is None

    # Test Rest (mode pack)
    gse_pack.state["resources"]["hit_points"]["current"] = 5
    gse_pack.state["resources"]["spells_per_day"]["level_1"]["current"] = 1
    gse_pack.state["pv"] = 5

    res = gse_pack.rest("long_rest")
    assert res.success is True
    assert gse_pack.state["resources"]["hit_points"]["current"] == 10
    assert gse_pack.state["resources"]["spells_per_day"]["level_1"]["current"] == 2
    assert gse_pack.state["pv"] == 10

    # Test short_rest (ne modifie rien pour l'instant dans dnd5e_srd)
    res_short = gse_pack.rest("short_rest")
    assert res_short.success is True
    assert gse_pack.state["resources"]["hit_points"]["current"] == 10

    # Test inconnu
    res_unknown = gse_pack.rest("inconnu")
    assert res_unknown.success is False
    assert "inconnu" in res_unknown.message
    assert "long_rest" in res_unknown.message
    assert "short_rest" in res_unknown.message

def test_gse_pack_pbta(tmp_path):
    char_file = tmp_path / "character_pbta.json"
    data = {
        "blessures": 2,
        "stress": 1
    }
    char_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    pack = load_pack("pbta_minimal", base_dir="tests/fixtures")
    gse = GameStateEngine(str(char_file), pack=pack)

    summary = gse.get_state_summary()
    assert "Jauge de Blessures: 2/5" in summary
    assert "Jauge de Stress: 1/3" in summary

    assert gse.detect_action_type("Je prends le temps de dormir.") is None # dormir n'y est pas, seulement dors
    assert gse.detect_action_type("Je dors.") == "rest:repos"

    res = gse.rest("soins")
    assert res.success is True
    assert gse.state["blessures"] == 3

    # Test consume_pool
    res_consume = gse.consume_pool("stress", amount=1)
    assert res_consume.success is True
    assert gse.state["stress"] == 0

    res_consume_fail = gse.consume_pool("stress", amount=1)
    assert res_consume_fail.success is False
    assert "Not enough" in res_consume_fail.message

    res_repos = gse.rest("repos")
    assert res_repos.success is True
    assert gse.state["blessures"] == 5
    assert gse.state["stress"] == 3

    # Invalid trigger
    res_invalid = gse.rest("long_rest")
    assert res_invalid.success is False

def test_import_without_langchain(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "langchain_ollama", None)
    monkeypatch.setitem(sys.modules, "langchain", None)
    # Reload game_state_engine to test if it imports fine
    import importlib
    import game_state_engine
    importlib.reload(game_state_engine)
    # Should not raise exception

def test_trigger_false_positives():
    from systems.loader import load_pack
    from game_state_engine import GameStateEngine
    pack = load_pack("dnd5e_srd")
    gse = GameStateEngine(pack=pack)

    assert gse.detect_action_type("Il sort de la pièce") is None
    assert gse.detect_action_type("la nuit tombe") is None
    assert gse.detect_action_type("le camp ennemi") is None

def test_detect_action_with_key_template(tmp_path, monkeypatch):
    import json
    from mechanics.models import TriggersConfig
    from systems.loader import load_pack
    from game_state_engine import GameStateEngine

    char_file = tmp_path / "char_template.json"
    data = {
        "resources": {
            "spells_per_day": {
                "level_2": {
                    "current": 2,
                    "max": 3
                }
            }
        }
    }
    char_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # Load normal pack and inject our custom rule
    pack = load_pack("dnd5e_srd")

    # Let's add a key_template to cast_spell manually
    for rule in pack.triggers.rules:
        if rule.id == "cast_spell":
            rule.key_template = "level_{key}"

    gse = GameStateEngine(str(char_file), pack=pack)

    # E2E test
    # This should detect cast_spell, extract "2", format it into "level_2"
    action = gse.detect_action("je lance un sort de niveau 2")
    assert action is not None
    assert action.key == "level_2"

    # consume_pool should now work successfully
    res = gse.consume_pool(action.target, key=action.key, amount=action.amount)
    assert res.success is True

    # Check that current went from 2 to 1
    assert gse.state["resources"]["spells_per_day"]["level_2"]["current"] == 1
