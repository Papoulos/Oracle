import os
import json
import pytest
from unittest import mock
from validation import validate_scenario_structure
from agent import RPGAgent
import config

@pytest.fixture
def clean_scenario_data():
    return {
        "metadata": {
            "title": "La Larme de l'Oubli",
            "global_pitch": "Un scénario d'exploration fantastique.",
            "starting_scene": "SCENE_01_AUBERGE"
        },
        "acts": [
            {
                "act_id": "ACTE_1",
                "title": "Le Départ",
                "entry_condition": "Création terminée",
                "completion_condition": "Le joueur quitte l'auberge",
                "included_scenes": ["SCENE_01_AUBERGE", "SCENE_02_ROUTE"]
            }
        ],
        "global_clocks": [
            {
                "name": "L'Orage",
                "trigger": "Le temps passe",
                "consequence": "La foudre tombe",
                "threshold": 6
            }
        ],
        "entities": {
            "npcs": [
                {
                    "id": "MAITRE_ELROND",
                    "full_name": "Maître Elrond",
                    "usual_location": "LIEU_FONDCOMBE",
                    "agenda_and_motivation": "Aider",
                    "fears_and_weaknesses": "Aucune",
                    "initial_attitude": "Neutre",
                    "stats_and_abilities": "Fort"
                }
            ],
            "locations": [
                {
                    "id": "LIEU_FONDCOMBE",
                    "full_name": "Fondcombe",
                    "sensory_atmosphere": "Calme",
                    "interactive_elements": "Livre"
                }
            ]
        },
        "scene_nodes": [
            {
                "scene_id": "SCENE_01_AUBERGE",
                "act_id": "ACTE_1",
                "location_id": "LIEU_FONDCOMBE",
                "title": "Rencontre à l'auberge",
                "present_npcs": ["MAITRE_ELROND"],
                "gm_objective": "Présenter Elrond",
                "resolution_condition": "Le PJ parle à Elrond",
                "local_rules_and_limits": "Pas de combat",
                "challenges_and_encounters": [],
                "logical_exits": [
                    {
                        "action_or_direction": "Prendre la route",
                        "destination_scene_id": "SCENE_02_ROUTE"
                    }
                ]
            },
            {
                "scene_id": "SCENE_02_ROUTE",
                "act_id": "ACTE_1",
                "location_id": "LIEU_FONDCOMBE",
                "title": "Sur la route",
                "present_npcs": [],
                "gm_objective": "Avancer",
                "resolution_condition": "La route se termine",
                "local_rules_and_limits": "",
                "challenges_and_encounters": [],
                "logical_exits": []
            }
        ]
    }

def test_validation_propre(clean_scenario_data):
    # Case 1: Extractions propre -> 0 warnings, 0 errors
    data, warnings, errors = validate_scenario_structure(clean_scenario_data)
    assert len(warnings) == 0
    assert len(errors) == 0

def test_validation_idempotente(clean_scenario_data):
    # Verify idempotency
    data, warnings, errors = validate_scenario_structure(clean_scenario_data)
    data2, warnings2, errors2 = validate_scenario_structure(data)
    assert len(warnings2) == 0
    assert len(errors2) == 0

def test_validation_casse_et_reparable(clean_scenario_data):
    # Case 2: Sabotaged but repairable version
    saboted = clean_scenario_data
    # 1. Unknown location on scene
    saboted["scene_nodes"][0]["location_id"] = "LIEU_INCONNU"
    # 2. Orphan NPC
    saboted["scene_nodes"][0]["present_npcs"].append("NPC_ORPHELIN")
    # 3. Logical exit to unknown scene
    saboted["scene_nodes"][0]["logical_exits"].append({
        "action_or_direction": "Inconnu",
        "destination_scene_id": "SCENE_INCONNUE"
    })
    # 4. Bidirectional incoherence: act does not list the scene
    saboted["acts"][0]["included_scenes"].remove("SCENE_02_ROUTE")
    # 5. Orphan included_scene in the act
    saboted["acts"][0]["included_scenes"].append("SCENE_ORPHELINE")
    # 6. Missing clock threshold
    del saboted["global_clocks"][0]["threshold"]

    data, warnings, errors = validate_scenario_structure(saboted)

    # All of these should be fixed and yield warnings, not errors
    assert len(errors) == 0
    assert len(warnings) > 0

    # Assertions on corrections
    assert data["scene_nodes"][0]["location_id"] is None
    assert "NPC_ORPHELIN" not in data["scene_nodes"][0]["present_npcs"]
    assert len(data["scene_nodes"][0]["logical_exits"]) == 1 # unknown output removed
    assert "SCENE_02_ROUTE" in data["acts"][0]["included_scenes"] # bidirectionally repaired
    assert "SCENE_ORPHELINE" not in data["acts"][0]["included_scenes"] # orphan removed
    assert data["global_clocks"][0]["threshold"] == 6 # fallback to default

def test_validation_erreurs_bloquantes(clean_scenario_data):
    # Case 3: Blocking errors that cannot be fixed
    # 1. Missing resolution_condition
    del clean_scenario_data["scene_nodes"][0]["resolution_condition"]
    # 2. Invalid act_id
    clean_scenario_data["scene_nodes"][0]["act_id"] = "ACTE_INEXISTANT"

    data, warnings, errors = validate_scenario_structure(clean_scenario_data)
    assert len(errors) == 2
    assert any("resolution_condition missing" in e for e in errors)
    assert any("act_id" in e for e in errors)

def test_validation_doublons_et_localisation_pnj(clean_scenario_data):
    # Test new rules added: duplicates detection and npc usual_location validation
    data = clean_scenario_data
    # Duplicate NPC ID
    data["entities"]["npcs"].append({
        "id": "MAITRE_ELROND",
        "full_name": "Autre Elrond",
        "usual_location": "LIEU_INCONNU",
        "agenda_and_motivation": "Rien",
        "fears_and_weaknesses": "Aucune",
        "initial_attitude": "Amical",
        "stats_and_abilities": "Inconnu"
    })

    corrected, warnings, errors = validate_scenario_structure(data)
    assert len(errors) == 0
    assert any("duplicate id 'MAITRE_ELROND'" in w for w in warnings)
    assert any("usual_location 'LIEU_INCONNU' unknown -> set to null." in w for w in warnings)

    assert corrected["entities"]["npcs"][0]["usual_location"] == "LIEU_FONDCOMBE"
    assert corrected["entities"]["npcs"][1]["usual_location"] is None

def test_setup_world_json_scenarios(tmp_path, monkeypatch):
    # Setup test configuration environment paths
    monkeypatch.setattr(config, "SCENARIO_DATA_PATH", str(tmp_path))

    agent = RPGAgent()
    agent.scenario_extractor_agent = mock.Mock()
    agent._check_collections = mock.Mock(return_value=True)

    # 1. Scenario with zero JSON files -> calls ScenarioExtractorAgent
    # Prepare some mock generation return value
    mock_sc_structure = {
        "metadata": {"title": "Aventure", "global_pitch": "Un pitch", "starting_scene": "SCENE_01"},
        "acts": [{"act_id": "ACTE_1", "title": "Acte", "included_scenes": ["SCENE_01"]}],
        "global_clocks": [],
        "entities": {"npcs": [], "locations": []},
        "scene_nodes": [{"scene_id": "SCENE_01", "act_id": "ACTE_1", "resolution_condition": "PJ sort"}]
    }
    agent.scenario_extractor_agent.generate = mock.Mock(return_value=mock_sc_structure)

    # Clean Memory/ directory if present
    for f in ["character.json", "scenario_structure.json", "progression.json", "scenario.json"]:
        p = os.path.join("Memory", f)
        if os.path.exists(p):
            os.remove(p)

    agent.setup_world()
    assert agent.scenario_extractor_agent.generate.called
    assert agent.scenario_structure["metadata"]["title"] == "Aventure"

    # 2. Scenario with exactly one JSON file -> loads directly, never calls ScenarioExtractorAgent
    agent.scenario_extractor_agent.generate.reset_mock()
    one_json = tmp_path / "scenario_structure.json"
    with open(one_json, "w", encoding="utf-8") as f:
        json.dump(mock_sc_structure, f)

    agent.setup_world()
    assert not agent.scenario_extractor_agent.generate.called
    assert agent.scenario_structure["metadata"]["title"] == "Aventure"

    # 3. Scenario with multiple JSON files -> raises ValueError
    another_json = tmp_path / "another_scenario.json"
    with open(another_json, "w", encoding="utf-8") as f:
        json.dump(mock_sc_structure, f)

    with pytest.raises(ValueError, match="Multiple scenario JSON files found"):
        agent.setup_world()

def test_full_text_below_threshold_avoid_similarity_search():
    # Setup scenario store mock
    store = mock.Mock()
    # Mocking get returns few documents
    store.get = mock.Mock(return_value={
        "documents": ["Mon petit scénario court sur une page."],
        "metadatas": [{"page": 0}]
    })
    store.similarity_search = mock.Mock(return_value=[])

    from scenario_agents import ScenarioExtractorAgent
    extractor = ScenarioExtractorAgent(store)

    # Set threshold to a value that is higher than length of the documents (length = 37)
    import config
    original_threshold = config.SCENARIO_FULLTEXT_THRESHOLD_CHARS
    config.SCENARIO_FULLTEXT_THRESHOLD_CHARS = 100

    # Mock LLM and extract entities
    mock_res = mock.Mock()
    mock_res.content = "{}"
    extractor.llm = mock.Mock()
    extractor.llm.invoke = mock.Mock(return_value=mock_res)

    extractor._extract_entities(log=lambda x: None)

    # similarity_search should NOT be called because length of text is 37 <= 100
    assert not store.similarity_search.called

    # Set threshold to a very low value so similarity_search is called (length = 37 > 10)
    config.SCENARIO_FULLTEXT_THRESHOLD_CHARS = 10
    extractor._extract_entities(log=lambda x: None)
    assert store.similarity_search.called

    # Restore threshold
    config.SCENARIO_FULLTEXT_THRESHOLD_CHARS = original_threshold

def test_truncated_json_repair():
    from base_utils import extract_json
    truncated_input = """```json
{
    "steps": [
        {
            "step": 1,
            "name": "Étape 1",
            "description": "Procédure."
        },
        {
            "step": 2,
            "name": "Étape 2",
            "description": "Seconde"
"""
    result = extract_json(truncated_input)
    assert result is not None
    assert "steps" in result
    assert len(result["steps"]) == 1
    assert result["steps"][0]["name"] == "Étape 1"
