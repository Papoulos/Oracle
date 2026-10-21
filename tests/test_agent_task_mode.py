import pytest
import os
import json
import random
import logging
from unittest import mock
from agent import RPGAgent
from game_state_engine import GameStateEngine
from systems.loader import load_pack

@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    os.makedirs("Memory", exist_ok=True)
    yield tmp_path

def create_agent(tmp_path, pack_id, game_state="ADVENTURE"):
    pack = None
    if pack_id:
        pack = load_pack(pack_id)

    char_data = {
        "name": "Test",
        "status": "complet",
        "tier": 1,
        "resources": {
            "might": {"current": 10, "max": 10, "edge": 0},
            "speed": {"current": 10, "max": 10, "edge": 0},
            "intellect": {"current": 2, "max": 10, "edge": 0}
        }
    }

    with open("Memory/character.json", "w") as f:
        json.dump(char_data, f)

    gse = GameStateEngine(character_path="Memory/character.json", pack=pack, rng=random.Random(42))

    agent = RPGAgent()
    agent.game_state = game_state
    agent.gse = gse
    agent.character_data = agent.gse.state
    agent.scenario_structure = {}
    agent.progression = {} # Mock progression dictionary properly

    mock_llm = mock.Mock()
    agent.llm = mock_llm

    agent.narrator.generate_response = mock.Mock(return_value="Narrator response")
    agent.sheet_manager.update_sheet = mock.Mock(return_value=agent.character_data)
    agent.get_scenario_context = mock.Mock(return_value="Scenario context")
    agent.get_npcs_context = mock.Mock(return_value="NPCs context")
    agent.update_chronicle = mock.Mock()

    return agent

def test_task_mode_property(tmp_path):
    agent = create_agent(tmp_path, "cypher_system")
    assert agent._task_mode() is True

    agent2 = create_agent(tmp_path, "dnd5e_srd")
    assert agent2._task_mode() is False

    agent3 = create_agent(tmp_path, None)
    assert agent3._task_mode() is False

def test_full_turn_success(tmp_path):
    agent = create_agent(tmp_path, "cypher_system")
    agent.current_scene_id = None

    mock_llm_response = mock.Mock()
    mock_llm_response.content = json.dumps({
        "need_roll": True,
        "pool": "might",
        "difficulty": 4,
        "ease": {"skill": 1, "asset": 0},
        "hinder": 0,
        "effort_levels": 2
    })
    agent.llm.invoke.return_value = mock_llm_response

    with mock.patch.object(agent.gse, "execute_task", wraps=agent.gse.execute_task) as spy_execute:
        res = agent.chat("Je pousse le rocher")

        spy_execute.assert_called_once_with(
            "task", # Based on cypher system actions.json the action id is 'task'
            "might",
            4,
            step_adjustments={"skill": -1},
            effort_levels=2
        )

        assert agent.llm.invoke.call_count == 1

        assert agent.character_data["resources"]["might"]["current"] == 5

        call_args = agent.narrator.generate_response.call_args[0]
        instruction = call_args[2]

        assert "The roll below was resolved by the game engine: narrate its outcome faithfully, never recompute, reroll or alter it." in instruction
        assert "TECHNICAL RESULT: Tâche — " in instruction
        assert "Effort : 2 niveau(x)" in instruction

        assert "🎲" in res
        # Technically assert the "Succès" / "Échec" appears only ONCE by checking the dice format outcome
        assert res.count("Échec") + res.count("Succès") == 1

def test_full_turn_no_roll(tmp_path):
    agent = create_agent(tmp_path, "cypher_system")
    agent.current_scene_id = None

    mock_llm_response = mock.Mock()
    mock_llm_response.content = json.dumps({"need_roll": False})
    agent.llm.invoke.return_value = mock_llm_response

    res = agent.chat("Je regarde autour")

    call_args = agent.narrator.generate_response.call_args[0]
    instruction = call_args[2]

    assert "The roll below was resolved by the game engine" not in instruction
    assert "TECHNICAL RESULT: Aucun jet requis" in instruction
    assert "🎲" not in res
    assert agent.character_data["resources"]["might"]["current"] == 10

def test_full_turn_blocked_action(tmp_path):
    agent = create_agent(tmp_path, "cypher_system")
    agent.current_scene_id = None

    mock_llm_response = mock.Mock()
    mock_llm_response.content = json.dumps({
        "need_roll": True,
        "pool": "intellect",
        "difficulty": 3,
        "ease": {},
        "hinder": 0,
        "effort_levels": 1
    })
    agent.llm.invoke.return_value = mock_llm_response

    res = agent.chat("J'étudie la rune")

    call_args = agent.narrator.generate_response.call_args[0]
    instruction = call_args[2]

    assert "ACTION BLOQUÉE : insufficient_resources" in instruction
    assert "The roll below was resolved by the game engine" not in instruction
    assert "TECHNICAL RESULT: Aucun jet requis" in instruction
    assert "🎲" not in res
    assert agent.character_data["resources"]["intellect"]["current"] == 2

def test_safeguard(tmp_path, caplog):
    agent = create_agent(tmp_path, "cypher_system")
    agent.current_scene_id = None

    mock_llm_response = mock.Mock()
    mock_llm_response.content = json.dumps({"need_roll": False})
    agent.llm.invoke.return_value = mock_llm_response

    altered_sheet = {
        "name": "Test",
        "status": "complet",
        "tier": 1,
        "equipment": ["Sword"],
        "resources": {
            "might": {"current": 999, "max": 10, "edge": 0},
            "speed": {"current": 10, "max": 10, "edge": 0},
            "intellect": {"current": 2, "max": 10, "edge": 0}
        }
    }

    agent.sheet_manager.update_sheet.return_value = altered_sheet

    with caplog.at_level(logging.WARNING, logger="agent"):
        agent.chat("Je trouve une épée")

        assert agent.character_data["resources"]["might"]["current"] == 10
        assert "equipment" in agent.character_data
        assert agent.character_data["equipment"] == ["Sword"]

        assert any("Engine-owned field restored after sheet update: resources.might.current" in record.message for record in caplog.records)

        with open("Memory/character.json", "r") as f:
            disk_data = json.load(f)
        assert disk_data["resources"]["might"]["current"] == 10
        assert "equipment" in disk_data

def test_historical_fallback(tmp_path):
    agent = create_agent(tmp_path, "dnd5e_srd")
    agent.current_scene_id = None

    mock_llm_response = mock.Mock()
    mock_llm_response.content = json.dumps({
        "covered_by_catalog": False,
        "need_roll": True,
        "stat": "Strength",
        "bonus": 2,
        "dc": 12,
        "calculation_breakdown": "+2 STR",
        "mechanical_decision": {}
    })
    agent.llm.invoke.return_value = mock_llm_response

    agent.roll_dice = mock.Mock(return_value=15)
    agent._check_resources = mock.Mock(return_value={"ok": True})

    res = agent.chat("Je tape")

    assert agent._check_resources.called
    assert agent.roll_dice.called

    call_args = agent.narrator.generate_response.call_args[0]
    instruction = call_args[2]

    assert "The roll below was resolved by the game engine" not in instruction
    assert "TECHNICAL RESULT: Jet de Strength (DC 12) : 15 + 2 (+2 STR) = 17 → Succès" in instruction
    assert "🎲 Jet de Strength (DC 12) : 15 + 2 (+2 STR) = 17 (Succès)" in res
