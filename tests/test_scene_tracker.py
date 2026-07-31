import json
import pytest
import os
from unittest import mock
from scenario_agents import ScenarioExtractorAgent, SceneGraphAgent
from agent import RPGAgent

class MockDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MockScenarioStore:
    def similarity_search(self, query, k=15, filter=None):
        return [
            MockDocument("FONDCOMBE est la demeure d'ELROND.", {"scene_id": "1.1"}),
            MockDocument("SCENE_01_DEPART : Le joueur quitte Fondcombe.", {"scene_id": "1.1"})
        ]

def test_scenario_extractor_validation():
    # Test that invalid references (orphan NPC, location, or scene) are filtered or logged
    store = MockScenarioStore()
    extractor = ScenarioExtractorAgent(store)

    # Mock the LLM calls for each pass
    # Pass 1: Entities
    mock_entities = mock.Mock()
    mock_entities.content = """
    {
      "npcs": [
        {"id": "ELROND", "full_name": "Maître Elrond", "usual_location": "FONDCOMBE", "agenda_and_motivation": "Aider les PJ", "fears_and_weaknesses": "Inconnu", "initial_attitude": "Amical", "stats_and_abilities": "Inconnu"}
      ],
      "locations": [
        {"id": "FONDCOMBE", "full_name": "Fondcombe", "sensory_atmosphere": "Calme", "interactive_elements": "Inconnu"}
      ]
    }
    """
    # Pass 2: Scene nodes
    mock_scenes = mock.Mock()
    mock_scenes.content = """
    {
      "scene_nodes": [
        {
          "scene_id": "SCENE_01_DEPART",
          "act_id": "ACTE_1",
          "location_id": "FONDCOMBE",
          "title": "Départ",
          "present_npcs": ["ELROND"],
          "gm_objective": "Lancer l'aventure",
          "resolution_condition": "Le PJ part",
          "local_rules_and_limits": "Aucune",
          "challenges_and_encounters": [],
          "logical_exits": [{"action_or_direction": "Suivre la route", "destination_scene_id": "SCENE_02_ROUTE"}]
        },
        {
          "scene_id": "SCENE_02_ROUTE",
          "act_id": "ACTE_1",
          "location_id": "LIEU_INVALIDE",
          "title": "La route",
          "present_npcs": ["PNJ_INVALIDE"],
          "gm_objective": "Surmonter les dangers",
          "resolution_condition": "La route est libre",
          "local_rules_and_limits": "Aucune",
          "challenges_and_encounters": [],
          "logical_exits": []
        }
      ]
    }
    """
    # Pass 3: Acts
    mock_macro = mock.Mock()
    mock_macro.content = """
    {
      "acts": [
        {
          "act_id": "ACTE_1",
          "title": "Le départ",
          "entry_condition": "Création terminée",
          "completion_condition": "Arrivée à Fondcombe",
          "included_scenes": ["SCENE_01_DEPART", "SCENE_02_ROUTE", "SCENE_INVALIDE"]
        }
      ]
    }
    """
    # Pass 4: Global clocks
    mock_horloges = mock.Mock()
    mock_horloges.content = """
    {
      "global_clocks": [
        {"name": "Le réveil de Sauron", "trigger": "Le temps passe", "consequence": "Le ciel s'assombrit", "threshold": "6"}
      ]
    }
    """
    # Pass 5: Metadata
    mock_meta = mock.Mock()
    mock_meta.content = """
    {
      "metadata": {
        "title": "Le Hobbit",
        "global_pitch": "Un voyage inattendu.",
        "starting_scene": "SCENE_01_DEPART"
      }
    }
    """

    extractor.llm = mock.Mock()
    extractor.llm.invoke = mock.Mock(side_effect=[mock_entities, mock_scenes, mock_macro, mock_horloges, mock_meta])

    structure = extractor.generate()

    assert structure is not None
    # Verify that invalid references were validated and cleaned/removed
    scenes = {s["scene_id"]: s for s in structure["scene_nodes"]}
    assert "SCENE_02_ROUTE" in scenes
    assert scenes["SCENE_02_ROUTE"]["location_id"] is None
    assert scenes["SCENE_02_ROUTE"]["present_npcs"] == []

    # Verify macro structure cleaned included_scenes
    actes = {a["act_id"]: a for a in structure["acts"]}
    assert "ACTE_1" in actes
    assert "SCENE_INVALIDE" not in actes["ACTE_1"]["included_scenes"]
    assert "SCENE_01_DEPART" in actes["ACTE_1"]["included_scenes"]

def test_get_current_context_lookup():
    agent = RPGAgent()
    agent.scenario_structure = {
      "metadata": {"title": "La Quête", "global_pitch": "Pitch...", "starting_scene": "SCENE_01"},
      "entities": {
        "npcs": [{"id": "ELROND", "full_name": "Maître Elrond", "usual_location": "FONDCOMBE", "agenda_and_motivation": "Aider", "initial_attitude": "Amical", "stats_and_abilities": "Inconnu"}],
        "locations": [{"id": "FONDCOMBE", "full_name": "Fondcombe", "sensory_atmosphere": "Merveilleux", "interactive_elements": "Livre de Lore"}]
      },
      "acts": [{"act_id": "ACTE_1", "title": "Le début", "completion_condition": "Parler à Elrond", "included_scenes": ["SCENE_01"]}],
      "scene_nodes": [
        {
          "scene_id": "SCENE_01",
          "act_id": "ACTE_1",
          "location_id": "FONDCOMBE",
          "title": "Rencontre",
          "present_npcs": ["ELROND"]
        }
      ]
    }
    agent._build_lookups()
    agent.progression = {
        "current_scene": "SCENE_01",
        "resolved_scenes": [],
        "bypassed_scenes": [],
        "clocks": {},
        "notable_deviations": []
    }

    context = agent.get_current_context()
    assert "CURRENT SCENE : Rencontre" in context
    assert "Fondcombe" in context
    assert "Merveilleux" in context
    assert "Maître Elrond" in context
    assert "Le début" in context

def test_transition_deterministic_logic():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"title": "La Quête", "global_pitch": "...", "starting_scene": "SCENE_01"},
      "entities": {"npcs": [], "locations": []},
      "acts": [
          {"act_id": "ACTE_1", "title": "Acte 1", "included_scenes": ["SCENE_01"]},
          {"act_id": "ACTE_2", "title": "Acte 2", "included_scenes": ["SCENE_02"]}
      ],
      "scene_nodes": [
        {
          "scene_id": "SCENE_01",
          "act_id": "ACTE_1",
          "title": "Scène 1",
          "gm_objective": "...",
          "resolution_condition": "Le PJ avance",
          "logical_exits": [{"action_or_direction": "Avancer", "destination_scene_id": "SCENE_02"}]
        },
        {
          "scene_id": "SCENE_02",
          "act_id": "ACTE_2",
          "title": "Scène 2",
          "gm_objective": "...",
          "resolution_condition": "Le PJ finit",
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

    # Mock RAG calls
    agent.get_current_scene = mock.Mock(return_value="Scène de l'auberge")
    agent.get_scenario_context = mock.Mock(return_value="Contexte RAG")
    agent.get_core_context = mock.Mock(return_value="Règles")

    # Mock LLM for mechanical analysis
    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    # Mock LLM for classification (transition to SCENE_02)
    mock_classif = mock.Mock()
    mock_classif.content = '{"category": "transition", "next_scene": "SCENE_02", "notable_deviation": "Le PJ a résolu l\'énigme."}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="Le narrateur décrit la transition.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    agent.chat("Je marche vers la sortie")

    assert agent.current_scene_id == "SCENE_02"
    assert agent.progression["current_scene"] == "SCENE_02"
    assert agent.progression["current_act"] == "ACTE_2"
    assert "SCENE_01" in agent.progression["resolved_scenes"]
    agent.update_chronicle.assert_called_once_with("Je marche vers la sortie", "Le narrateur décrit la transition.", "Le PJ a résolu l'énigme.")

def test_transition_invalid_fallback_to_improvisation():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"title": "La Quête", "global_pitch": "...", "starting_scene": "SCENE_01"},
      "entities": {"npcs": [], "locations": []},
      "acts": [{"act_id": "ACTE_1", "title": "Acte 1", "included_scenes": ["SCENE_01"]}],
      "scene_nodes": [
        {
          "scene_id": "SCENE_01",
          "act_id": "ACTE_1",
          "title": "Scène 1",
          "gm_objective": "...",
          "resolution_condition": "Le PJ avance"
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

    agent.get_current_scene = mock.Mock(return_value="Scène de l'auberge")
    agent.get_scenario_context = mock.Mock(return_value="Contexte RAG")
    agent.get_core_context = mock.Mock(return_value="Règles")

    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    # transition to an invalid scene ID
    mock_classif = mock.Mock()
    mock_classif.content = '{"category": "transition", "next_scene": "SCENE_INVALIDE", "notable_deviation": null}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="Narration.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    agent.chat("Je marche vers la sortie")

    # Should fallback to improvisation: no scene change, no exception raised
    assert agent.current_scene_id == "SCENE_01"
    assert "SCENE_01" not in agent.progression["resolved_scenes"]

def test_contournement_non_blocking():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"title": "La Quête", "global_pitch": "...", "starting_scene": "SCENE_01"},
      "entities": {"npcs": [], "locations": []},
      "acts": [{"act_id": "ACTE_1", "title": "Acte 1", "included_scenes": ["SCENE_01"]}],
      "scene_nodes": [
        {
          "scene_id": "SCENE_01",
          "act_id": "ACTE_1",
          "title": "Scène 1"
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

    agent.get_current_scene = mock.Mock(return_value="Scène de l'auberge")
    agent.get_scenario_context = mock.Mock(return_value="Contexte RAG")
    agent.get_core_context = mock.Mock(return_value="Règles")

    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    mock_classif = mock.Mock()
    mock_classif.content = '{"category": "bypassed", "next_scene": null, "notable_deviation": "Le PJ a contourné."}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="Narration contournement.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    response = agent.chat("Je m'enfuis")

    assert response == "Narration contournement."
    assert agent.current_scene_id == "SCENE_01"
    assert "SCENE_01" in agent.progression["bypassed_scenes"]

def test_clock_trigger_consequence_injection():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"title": "La Quête", "global_pitch": "...", "starting_scene": "SCENE_01"},
      "entities": {"npcs": [], "locations": []},
      "acts": [{"act_id": "ACTE_1", "title": "Acte 1", "included_scenes": ["SCENE_01"]}],
      "scene_nodes": [{"scene_id": "SCENE_01", "act_id": "ACTE_1", "title": "Scène 1"}],
      "global_clocks": [
        {"name": "Le Volcan", "trigger": "Chaleur", "consequence": "Le volcan entre en éruption !", "threshold": 3}
      ]
    }
    agent._build_lookups()
    # Clock segments at 2 (so adding 1 triggers it)
    agent.progression = {
        "current_act": "ACTE_1",
        "current_scene": "SCENE_01",
        "resolved_scenes": [],
        "bypassed_scenes": [],
        "clocks": {
            "Le Volcan": {"segments": 2, "declenchee": False}
        },
        "notable_deviations": []
    }

    agent.get_current_scene = mock.Mock(return_value="Scène")
    agent.get_scenario_context = mock.Mock(return_value="Contexte RAG")
    agent.get_core_context = mock.Mock(return_value="Règles")

    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    # Classification output adds 1 segment to "Le Volcan"
    mock_classif = mock.Mock()
    mock_classif.content = '{"category": "improvisation", "next_scene": null, "impacted_clocks": [{"name": "Le Volcan", "segments_added": 1}], "notable_deviation": null}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])

    agent.narrator.generate_response = mock.Mock(return_value="Le volcan explose.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    agent.chat("Je fais une action")

    # Verify clock segment was incremented and declenchee set to True
    assert agent.progression["clocks"]["Le Volcan"]["segments"] == 3
    assert agent.progression["clocks"]["Le Volcan"]["declenchee"] is True

    # Verify that consequence was injected in instructions to Narrator
    call_args = agent.narrator.generate_response.call_args[0]
    instructions = call_args[2]
    assert "ÉVÉNEMENT DÉCLENCHÉ (horloge 'Le Volcan') : Le volcan entre en éruption !" in instructions

    # Verify that the consequence was passed as notable_deviation to Chronicle update
    agent.update_chronicle.assert_called_once_with("Je fais une action", "Le volcan explose.", "Le volcan entre en éruption !")

def test_load_game_restores_scene_progression():
    agent = RPGAgent()

    os.makedirs("Memory", exist_ok=True)
    with open("Memory/scenario_structure.json", "w", encoding="utf-8") as f:
        json.dump({
          "metadata": {"title": "La Quête", "global_pitch": "Un anneau...", "starting_scene": "SCENE_01"},
          "entities": {"npcs": [], "locations": []},
          "acts": [{"act_id": "ACTE_1", "title": "Acte 1", "included_scenes": ["SCENE_01"]}],
          "scene_nodes": [{"scene_id": "SCENE_01", "act_id": "ACTE_1", "title": "Scène 1"}],
          "global_clocks": []
        }, f)

    with open("Memory/progression.json", "w", encoding="utf-8") as f:
        json.dump({
            "current_act": "ACTE_1",
            "current_scene": "SCENE_01",
            "resolved_scenes": [],
            "bypassed_scenes": [],
            "clocks": {},
            "notable_deviations": ["Quelque chose de remarquable"]
        }, f)

    with open("Memory/character.json", "w", encoding="utf-8") as f:
        json.dump({"name": "Frodon", "class": "Rogue"}, f)

    loaded = agent.load_game()
    assert loaded is True
    assert agent.current_scene_id == "SCENE_01"
    assert agent.progression["current_act"] == "ACTE_1"
    assert agent.progression["notable_deviations"] == ["Quelque chose de remarquable"]

    # Cleanup
    for file in ["character.json", "scenario_structure.json", "progression.json"]:
        p = os.path.join("Memory", file)
        if os.path.exists(p):
            os.remove(p)
