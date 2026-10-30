import os
import json
import pytest
import logging
from unittest import mock
from game_state_engine import GameStateEngine
from agent import RPGAgent
from scenario_agents import SceneGraphAgent

@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("Memory", exist_ok=True)
    yield tmp_path

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
            "hit_points": {"current": 4, "max": 10},
            "spells_per_day": {
                "level_1": {"current": 0, "max": 2}
            },
            "points_de_rage": {"current": 1, "max": 2}
        }
    }
    char_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(char_file)


def test_gse_rest_palier_inexistant(temp_character_file):
    gse = GameStateEngine(temp_character_file)
    res = gse.rest("palier_inexistant")
    assert res.success is False
    assert "Unknown" in res.message or "inconnu" in res.message


def test_agent_chat_non_task_mode_roll(monkeypatch):
    monkeypatch.delenv("SYSTEM_PACK", raising=False)
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.chronicle_data = {"summary": "L'aventure commence."}
    agent.npcs_data = []
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
        "metadata": {"title": "La Quête", "global_pitch": "...", "starting_scene": "SCENE_01"},
        "entities": {"npcs": [], "locations": []},
        "acts": [
            {"act_id": "ACTE_1", "title": "Acte 1", "included_scenes": ["SCENE_01"]}
        ],
        "scene_nodes": [
            {
                "scene_id": "SCENE_01",
                "act_id": "ACTE_1",
                "title": "Scène 1",
                "gm_objective": "...",
                "resolution_condition": "Le PJ avance",
                "logical_exits": []
            }
        ],
        "global_clocks": []
    }
    agent._build_lookups()
    agent.progression = {
        "current_act": "ACTE_1",
        "current_scene": "SCENE_01",
        "resolved_scenes": [],
        "bypassed_scenes": [],
        "clocks": {},
        "notable_deviations": []
    }
    agent.character_data = {
        "name": "Hero",
        "level": 1,
        "xp": 0,
        "resources": {"hit_points": {"current": 10, "max": 10}}
    }

    def invoke_mock(prompt, **kwargs):
        p_str = str(prompt)
        if "RPG referee. Analyze the player's action" in p_str:
            return mock.Mock(content=json.dumps({"consumes": False}))
        elif "Analyze the player's action relative to the current scene" in p_str:
            return mock.Mock(content=json.dumps({"category": "improvisation", "next_scene": None}))
        elif "According to the character" in p_str:
            return mock.Mock(content=json.dumps({
                "need_roll": True,
                "stat": "athlétisme",
                "bonus": 1,
                "calculation_breakdown": "+1 Force",
                "dc": 10,
                "reason": "Escalade de la falaise",
                "mechanical_decision": {"action": None, "amount": None}
            }))
        else:
            return mock.Mock(content="{}")

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=invoke_mock)
    agent.narrator.generate_response = mock.Mock(return_value="Vous escaladez la falaise !")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    response = agent.chat("Je grimpe sur la falaise")

    assert agent.llm.invoke.call_count == 3
    assert "Jet de athlétisme" in response


def test_indexer_log_option_not_passed():
    """
    --log not passed -> no indexer_debug.log file is created.
    """
    log_file = "indexer_debug.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    from indexer import main
    with mock.patch("sys.argv", ["indexer.py", "--clear"]):
        with mock.patch("indexer.index_directory"):
            with mock.patch("chromadb.PersistentClient"):
                main()

    assert not os.path.exists(log_file)


def test_indexer_log_option_passed():
    """
    --log passed -> indexer_debug.log contains at least one PROMPT and RAW RESPONSE entry.
    """
    log_file = "indexer_debug.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    mock_store = mock.Mock()
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Scenario RAG content")
    ]

    agent = SceneGraphAgent(mock_store, verbose=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        encoding="utf-8"
    )
    logging.root.setLevel(logging.DEBUG)

    with mock.patch.object(agent.llm.__class__, "invoke", return_value=mock.Mock(content='{"scene_initiale": "1.1", "scenes": []}')):
        agent.generate({})

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert "PROMPT" in log_content
    assert "RAW RESPONSE" in log_content
