import pytest
import os
import json
import random
from unittest import mock
from agent import RPGAgent
from game_state_engine import GameStateEngine
from systems.loader import load_pack
import systems.promote

def create_agent(tmp_path, pack_id, character_data=None, game_state="ADVENTURE"):
    pack = None
    if pack_id:
        pack = load_pack(pack_id)

    char_data = character_data or {
        "name": "Test",
        "status": "complet",
        "tier": 1,
        "resources": {
            "might": {"current": 5, "max": 10, "edge": 0},
            "speed": {"current": 5, "max": 10, "edge": 0},
            "intellect": {"current": 5, "max": 10, "edge": 0}
        }
    }

    os.makedirs("Memory", exist_ok=True)
    with open("Memory/character.json", "w") as f:
        json.dump(char_data, f)
    with open("Memory/progression.json", "w") as f:
        json.dump({"global_clocks": {}}, f)

    gse = GameStateEngine(character_path="Memory/character.json", pack=pack, rng=random.Random(42))

    with mock.patch("agent.chromadb.PersistentClient"), mock.patch("agent.Chroma"):
        agent = RPGAgent(verbose=False)
    agent.game_state = game_state
    agent.gse = gse
    agent.character_data = agent.gse.state
    agent.scenario_structure = {}
    agent.progression = {"clocks": {}}
    agent.current_scene_id = None

    mock_llm = mock.Mock()
    agent.llm = mock_llm

    agent.narrator.generate_response = mock.Mock(return_value="Narrator response")
    agent.sheet_manager.update_sheet = mock.Mock(return_value=agent.character_data)
    agent.get_scenario_context = mock.Mock(return_value="Scenario context")
    agent.get_npcs_context = mock.Mock(return_value="NPCs context")
    agent.update_chronicle = mock.Mock()

    return agent

@pytest.fixture
def mock_systems_dir(monkeypatch):
    monkeypatch.setattr(systems.promote, "SYSTEMS_DIR", "systems")

def test_cypher_system_rests_and_allocation(monkeypatch, tmp_path, mock_systems_dir):
    agent = create_agent(tmp_path, pack_id="cypher_system")

    agent.llm.invoke.return_value = mock.Mock(content=json.dumps({"need_roll": False}))

    # Turn 1: "Je me repose"
    agent.chat("Je me repose")

    pending = agent.gse.state.get("pending_allocation")
    assert pending is not None
    assert pending["points"] >= 1

    instruction = agent.narrator.generate_response.call_args[0][2]
    assert "RÉPARTITION EN ATTENTE" in instruction

    assert agent.llm.invoke.call_count == 1

    # Turn 2: Valid allocation
    points_to_allocate = pending["points"]
    expected_gain = min(points_to_allocate, 5) # Current is 5, max is 10.

    agent.llm.invoke.reset_mock()
    agent.llm.invoke.side_effect = [
        mock.Mock(content=json.dumps({"understood": True, "allocation": {"might": points_to_allocate}})),
        mock.Mock(content=json.dumps({"need_roll": False}))
    ]

    agent.chat(f"Je mets tout dans Might")

    assert agent.gse.state.get("pending_allocation") is None
    assert agent.gse.state["resources"]["might"]["current"] == 5 + expected_gain

    instruction2 = agent.narrator.generate_response.call_args[0][2]
    assert "MECANIQUE" in instruction2
    assert agent.llm.invoke.call_count == 2

    # Turn 3: "Je me repose encore"
    agent.llm.invoke.reset_mock()
    agent.llm.invoke.side_effect = [
        mock.Mock(content=json.dumps({"need_roll": False}))
    ]

    agent.chat("Je me repose encore")
    assert agent.gse.state["recovery_step"] == 2

def test_allocation_not_understood(monkeypatch, tmp_path, mock_systems_dir):
    agent = create_agent(tmp_path, pack_id="cypher_system")

    agent.llm.invoke.return_value = mock.Mock(content=json.dumps({"need_roll": False}))
    agent.chat("Je me repose")
    pending = agent.gse.state.get("pending_allocation")
    assert pending is not None

    agent.llm.invoke.reset_mock()
    agent.llm.invoke.side_effect = [
        mock.Mock(content=json.dumps({"understood": False})),
        mock.Mock(content=json.dumps({"need_roll": False}))
    ]

    agent.chat("Quel temps fait-il ?")

    assert agent.gse.state.get("pending_allocation") is not None
    assert agent.gse.state["resources"]["might"]["current"] == 5
    instruction = agent.narrator.generate_response.call_args[0][2]
    assert "RÉPARTITION EN ATTENTE" in instruction

def test_allocation_decline(monkeypatch, tmp_path, mock_systems_dir):
    agent = create_agent(tmp_path, pack_id="cypher_system")

    agent.llm.invoke.return_value = mock.Mock(content=json.dumps({"need_roll": False}))
    agent.chat("Je me repose")

    agent.llm.invoke.reset_mock()
    agent.llm.invoke.side_effect = [
        mock.Mock(content=json.dumps({"understood": True, "decline": True})),
        mock.Mock(content=json.dumps({"need_roll": False}))
    ]

    agent.chat("Je refuse")

    assert agent.gse.state.get("pending_allocation") is None
    assert agent.gse.state["resources"]["might"]["current"] == 5

def test_allocation_invalid_at_engine(monkeypatch, tmp_path, mock_systems_dir):
    agent = create_agent(tmp_path, pack_id="cypher_system")

    agent.llm.invoke.return_value = mock.Mock(content=json.dumps({"need_roll": False}))
    agent.chat("Je me repose")
    pending = agent.gse.state.get("pending_allocation")

    agent.llm.invoke.reset_mock()
    with mock.patch("agent.sanitize_allocation") as mock_sanitize:
        mock_sanitize.return_value = {"action": "allocate", "allocation": {"might": 999}}

        agent.llm.invoke.side_effect = [
            mock.Mock(content=json.dumps({})),
            mock.Mock(content=json.dumps({"need_roll": False}))
        ]

        agent.chat("Je triche")

    instruction = agent.narrator.generate_response.call_args[0][2]
    assert "ACTION BLOQUÉE" in instruction
    assert agent.gse.state.get("pending_allocation") is not None
    assert agent.gse.state["resources"]["might"]["current"] == 5


def test_dnd5e_consume_and_long_rest(monkeypatch, tmp_path, mock_systems_dir):
    char = {
        "pv": 3,
        "resources": {
            "hit_points": {"current": 3, "max": 30},
            "spells_per_day": {
                "level_1": {"current": 2, "max": 4},
                "level_2": {"current": 2, "max": 2}
            }
        }
    }
    agent = create_agent(tmp_path, pack_id="dnd5e_srd", character_data=char)
    agent.llm.invoke.return_value = mock.Mock(content=json.dumps({"need_roll": False}))

    agent.chat("Je lance un sort de niveau 2")
    assert agent.gse.state["resources"]["spells_per_day"]["level_2"]["current"] == 1

    agent.chat("Je lance un sort")
    instruction = agent.narrator.generate_response.call_args[0][2]
    assert "ACTION BLOQUÉE" in instruction
    assert agent.gse.state["resources"]["spells_per_day"]["level_2"]["current"] == 1

    agent.chat("On prend un repos long")
    assert agent.gse.state["resources"]["hit_points"]["current"] == 30
    assert agent.gse.state["resources"]["spells_per_day"]["level_2"]["current"] == 2


def test_experimental_mode(monkeypatch, tmp_path):
    agent = create_agent(tmp_path, pack_id=None, character_data={"pv": {"current": 5}})
    agent.llm.invoke.return_value = mock.Mock(content=json.dumps({"need_roll": False}))

    with mock.patch.object(agent.gse, "consume_spell_slot") as mock_consume:
        mock_consume.return_value = mock.Mock(success=True, message="Sort lancé")
        agent.chat("Je lance un sort")
        mock_consume.assert_called_once_with(1)

    with mock.patch.object(agent.gse, "rest") as mock_rest:
        mock_rest.return_value = mock.Mock(success=True, message="Repos pris")
        agent.chat("repos long")
        mock_rest.assert_called_once_with("long")

def test_engine_owned_fields_survive_sheet_manager(monkeypatch, tmp_path, mock_systems_dir):
    agent = create_agent(tmp_path, pack_id="cypher_system")

    agent.llm.invoke.return_value = mock.Mock(content=json.dumps({"need_roll": False}))

    def malicious_update(sheet, *args, **kwargs):
        new_sheet = sheet.copy()
        if "pending_allocation" in new_sheet:
            del new_sheet["pending_allocation"]
        if "recovery_step" in new_sheet:
            del new_sheet["recovery_step"]
        return new_sheet

    agent.sheet_manager.update_sheet.side_effect = malicious_update

    agent.chat("Je me repose")

    assert agent.gse.state.get("pending_allocation") is not None
    assert agent.gse.state.get("recovery_step") == 1
