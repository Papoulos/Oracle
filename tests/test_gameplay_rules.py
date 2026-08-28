import os
import json
import pytest
import logging
from unittest import mock
from game_state_engine import GameStateEngine
from agent import RPGAgent
from scenario_agents import GameplayRulesAgent, EXTRACTION_RECOVERY_PROMPT

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

@pytest.fixture
def mock_recovery_rules_file():
    path = "Memory/recovery_rules.json"
    backup = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                backup = f.read()
        except Exception:
            pass

    rules = {
        "recovery_tiers": [
            {
                "id": "repos_long",
                "name": "Repos Long",
                "text_triggers": ["dormir dans un lit", "bivouac sécurisé"],
                "effects": [
                    {"resource": "resources.hit_points", "action": "restore_full", "value": None},
                    {"resource": "resources.spells_per_day", "action": "restore_full", "value": None}
                ]
            },
            {
                "id": "repos_court",
                "name": "Repos Court",
                "text_triggers": ["récupération de 10 minutes", "petite pause"],
                "effects": [
                    {"resource": "resources.hit_points", "action": "restore_percentage", "value": 25}
                ]
            }
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=4, ensure_ascii=False)

    yield path

    if backup is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(backup)
    elif os.path.exists(path):
        os.remove(path)


@pytest.fixture
def mock_action_catalog_file():
    path = "Memory/action_catalog.json"
    backup = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                backup = f.read()
        except Exception:
            pass

    catalog = {
        "common_actions": [
            {
                "name": "Attaquer",
                "triggers": ["attaque", "frapper"],
                "resolution": "Jet de combat",
                "on_success": "Ennemi blessé",
                "on_failure": "Rien"
            }
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4, ensure_ascii=False)

    yield path

    if backup is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(backup)
    elif os.path.exists(path):
        os.remove(path)


def test_gse_rest_repos_long_dynamic(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    hp, hp_max = gse.get_hp()
    assert hp == 4
    assert gse.get_resource("spells_per_day", level=1)[0] == 0

    res = gse.rest("repos_long")
    assert res.success is True
    assert gse.get_hp()[0] == 10
    assert gse.get_resource("spells_per_day", level=1)[0] == 2
    assert gse.state["pv"] == 10


def test_gse_rest_repos_court_dynamic(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    res = gse.rest("repos_court")
    assert res.success is True
    assert gse.get_hp()[0] == 6
    assert gse.state["pv"] == 6


def test_gse_rest_palier_inexistant(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    res = gse.rest("palier_inexistant")
    assert res.success is False
    assert "Unknown" in res.message or "inconnu" in res.message


def test_gse_detect_action_type_dynamic(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    assert gse.detect_action_type("Je prends une petite pause") == "rest:repos_court"
    assert gse.detect_action_type("Je décide de dormir dans un lit") == "rest:repos_long"


def test_agent_chat_covered_by_catalog(mock_action_catalog_file):
    with mock.patch("agent.RPGAgent.get_core_context") as mock_get_core:
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

        # Mock prompt-aware LLM invoke
        def invoke_mock(prompt, **kwargs):
            p_str = str(prompt)
            if "RPG referee. Analyze the player's action" in p_str:
                return mock.Mock(content=json.dumps({"consumes": False}))
            elif "Analyze the player's action relative to the current scene" in p_str:
                return mock.Mock(content=json.dumps({"category": "improvisation", "next_scene": None}))
            elif "COMMON ACTIONS CATALOG FOR THIS SYSTEM" in p_str:
                return mock.Mock(content=json.dumps({
                    "covered_by_catalog": True,
                    "need_roll": True,
                    "stat": "combat",
                    "bonus": 2,
                    "calculation_breakdown": "+2 dague",
                    "dc": 12,
                    "reason": "Attaque à la dague",
                    "mechanical_decision": {"action": None, "amount": None}
                }))
            else:
                return mock.Mock(content="{}")

        agent.llm = mock.Mock()
        agent.llm.invoke = mock.Mock(side_effect=invoke_mock)
        agent.narrator.generate_response = mock.Mock(return_value="Vous attaquez avec succès !")
        agent.sheet_manager.update_sheet = mock.Mock(return_value={})
        agent.update_chronicle = mock.Mock()

        response = agent.chat("J'attaque le gobelin")

        mock_get_core.assert_not_called()
        assert "Jet de combat" in response
        assert "12" in response


def test_agent_chat_not_covered_by_catalog(mock_action_catalog_file):
    with mock.patch("agent.RPGAgent.get_core_context", return_value="Règles de crochetage...") as mock_get_core:
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

        # Mock prompt-aware LLM invoke
        def invoke_mock(prompt, **kwargs):
            p_str = str(prompt)
            if "RPG referee. Analyze the player's action" in p_str:
                return mock.Mock(content=json.dumps({"consumes": False}))
            elif "Analyze the player's action relative to the current scene" in p_str:
                return mock.Mock(content=json.dumps({"category": "improvisation", "next_scene": None}))
            elif "COMMON ACTIONS CATALOG FOR THIS SYSTEM" in p_str:
                return mock.Mock(content=json.dumps({"covered_by_catalog": False}))
            elif "Based on the following CODEX RULES" in p_str:
                return mock.Mock(content=json.dumps({
                    "need_roll": True,
                    "stat": "crochetage",
                    "bonus": 3,
                    "calculation_breakdown": "+3 Dextérité",
                    "dc": 15,
                    "reason": "Crochetage de serrure complexe",
                    "mechanical_decision": {"action": None, "amount": None}
                }))
            else:
                return mock.Mock(content="{}")

        agent.llm = mock.Mock()
        agent.llm.invoke = mock.Mock(side_effect=invoke_mock)
        agent.narrator.generate_response = mock.Mock(return_value="Vous crochetez la serrure !")
        agent.sheet_manager.update_sheet = mock.Mock(return_value={})
        agent.update_chronicle = mock.Mock()

        response = agent.chat("Je crochette la porte")

        mock_get_core.assert_called_once_with("Je crochette la porte", k=mock.ANY)
        assert "Jet de crochetage" in response
        assert "15" in response


def test_agent_chat_empty_catalog():
    if os.path.exists("Memory/action_catalog.json"):
        os.remove("Memory/action_catalog.json")

    with mock.patch("agent.RPGAgent.get_core_context", return_value="Règles d'escalade...") as mock_get_core:
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

        # Mock prompt-aware LLM invoke
        def invoke_mock(prompt, **kwargs):
            p_str = str(prompt)
            if "RPG referee. Analyze the player's action" in p_str:
                return mock.Mock(content=json.dumps({"consumes": False}))
            elif "Analyze the player's action relative to the current scene" in p_str:
                return mock.Mock(content=json.dumps({"category": "improvisation", "next_scene": None}))
            elif "Based on the following CODEX RULES" in p_str:
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

        mock_get_core.assert_called_once_with("Je grimpe sur la falaise", k=mock.ANY)
        assert agent.llm.invoke.call_count == 3
        assert "Jet de athlétisme" in response


# --- NEW DIAGNOSTIC & MIGRATION TESTS ---

def test_recovery_rules_empty_schema_saving():
    """
    Test that EXTRACTION_RECOVERY_PROMPT/generate_recovery_rules with an empty character_schema.json
    still saves the discovered tiers with consistent resource names (no systematic rejection).
    """
    mock_store = mock.Mock()
    mock_store.get.return_value = {
        "documents": ["Healing rules info..."],
        "metadatas": [{"page": 1}]
    }
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Règles de repos.")
    ]

    agent = GameplayRulesAgent(mock_store)
    agent._load_character_schema = mock.Mock(return_value={"required_fields": []})

    # The first LLM invoke is for discovery -> returns ["Natural Healing"]
    # The second is for extraction -> returns structured recovery rules JSON
    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"tiers": ["Natural Healing"]}'),
        mock.Mock(content=json.dumps({
            "recovery_tiers": [
                {
                    "id": "natural_healing",
                    "name": "Natural Healing",
                    "text_triggers": ["night of sleep"],
                    "effects": [{"resource": "resources.hit_points", "action": "restore_full", "value": None}]
                }
            ]
        }))
    ]):
        rules = agent.generate_recovery_rules()

    assert len(rules.get("recovery_tiers", [])) == 1
    assert rules["recovery_tiers"][0]["id"] == "natural_healing"
    assert rules["recovery_tiers"][0]["effects"][0]["resource"] == "resources.hit_points"


def test_indexer_log_option_not_passed():
    """
    --log not passed -> no indexer_debug.log file is created.
    """
    log_file = "indexer_debug.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Mock indexer run without --log
    from indexer import main
    with mock.patch("sys.argv", ["indexer.py", "--clear"]):
        with mock.patch("indexer.index_directory"):
            with mock.patch("indexer.ManualGeneratorAgent") as mock_gen:
                with mock.patch("indexer.GameplayRulesAgent") as mock_gameplay:
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
    mock_store.get.return_value = {
        "documents": ["Core rules content"],
        "metadatas": [{"page": 1}]
    }
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Core rules RAG content")
    ]

    agent = GameplayRulesAgent(mock_store, verbose=True)

    # Clear existing handlers to allow basicConfig to write during tests
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up basic config for the file log
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        encoding="utf-8"
    )
    logging.root.setLevel(logging.DEBUG)

    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"tiers": ["Natural Healing"]}'),
        mock.Mock(content='{"recovery_tiers": []}')
    ]):
        agent.generate_recovery_rules()

    # Reset root logger so it closes the file and we can read it
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert "PROMPT" in log_content
    assert "RAW RESPONSE" in log_content


def test_extract_json_failed_warning():
    """
    A failed extract_json (LLM returns non-JSON) triggers warning logging even without --log.
    """
    mock_store = mock.Mock()
    mock_store.get.return_value = {
        "documents": ["Core rules content"],
        "metadatas": [{"page": 1}]
    }
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Core rules RAG")
    ]

    agent = GameplayRulesAgent(mock_store, verbose=False)

    log_messages = []
    def custom_log(msg):
        log_messages.append(msg)

    # First invoke returns valid tiers list, second invoke returns invalid non-JSON to trigger extraction json warning
    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"tiers": ["Natural Healing"]}'),
        mock.Mock(content="Definitely NOT JSON!")
    ]):
        agent.generate_recovery_rules(log_callback=custom_log)

    assert any("JSON parsing failed" in msg for msg in log_messages)
    assert any("Definitely NOT JSON!" in msg for msg in log_messages)
