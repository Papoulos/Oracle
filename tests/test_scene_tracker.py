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
    # Test that invalid references (orphan PNJ, lieux, or scenes) are filtered or logged
    store = MockScenarioStore()
    extractor = ScenarioExtractorAgent(store)

    # Mock the LLM calls for each pass
    # Pass 1: Entités
    mock_entities = mock.Mock()
    mock_entities.content = """
    {
      "pnj": [
        {"id": "ELROND", "nom_complet": "Maître Elrond", "localisation_habituelle": "FONDCOMBE", "agenda_et_motivation": "Aider les PJ", "peurs_et_faiblesses": "Inconnu", "attitude_initiale": "Amical", "stats_et_capacites": "Inconnu"}
      ],
      "lieux": [
        {"id": "FONDCOMBE", "nom_complet": "Fondcombe", "ambiance_sensorielle": "Calme", "elements_interactifs": "Inconnu"}
      ]
    }
    """
    # Pass 2: Nœuds scéniques (scène 1 has valid lieu ELROND is valid, scene 2 has invalid lieu_rattache_id and invalid pnj_presents)
    mock_scenes = mock.Mock()
    mock_scenes.content = """
    {
      "noeuds_sceniques": [
        {
          "id_scene": "SCENE_01_DEPART",
          "acte_rattache_id": "ACTE_1",
          "lieu_rattache_id": "FONDCOMBE",
          "titre": "Départ",
          "pnj_presents": ["ELROND"],
          "objectif_mj": "Lancer l'aventure",
          "condition_resolution": "Le PJ part",
          "limites_et_regles_locales": "Aucune",
          "defis_et_rencontres": [],
          "sorties_logiques": [{"action_ou_direction": "Suivre la route", "destination_scene_id": "SCENE_02_ROUTE"}]
        },
        {
          "id_scene": "SCENE_02_ROUTE",
          "acte_rattache_id": "ACTE_1",
          "lieu_rattache_id": "LIEU_INVALIDE",
          "titre": "La route",
          "pnj_presents": ["PNJ_INVALIDE"],
          "objectif_mj": "Surmonter les dangers",
          "condition_resolution": "La route est libre",
          "limites_et_regles_locales": "Aucune",
          "defis_et_rencontres": [],
          "sorties_logiques": []
        }
      ]
    }
    """
    # Pass 3: Macro-structure (ACTE_1 has valid scenes but references an invalid SCENE_INVALIDE)
    mock_macro = mock.Mock()
    mock_macro.content = """
    {
      "macro_structure": [
        {
          "id_acte": "ACTE_1",
          "titre": "Le départ",
          "condition_entree": "Création terminée",
          "condition_validation": "Arrivée à Fondcombe",
          "scenes_incluses": ["SCENE_01_DEPART", "SCENE_02_ROUTE", "SCENE_INVALIDE"]
        }
      ]
    }
    """
    # Pass 4: Horloges globales
    mock_horloges = mock.Mock()
    mock_horloges.content = """
    {
      "horloges_globales": [
        {"nom": "Le réveil de Sauron", "declencheur": "Le temps passe", "consequence": "Le ciel s'assombrit", "seuil": "6"}
      ]
    }
    """
    # Pass 5: Métadonnées
    mock_meta = mock.Mock()
    mock_meta.content = """
    {
      "metadata": {
        "titre": "Le Hobbit",
        "pitch_global": "Un voyage inattendu.",
        "scene_initiale": "SCENE_01_DEPART"
      }
    }
    """

    extractor.llm = mock.Mock()
    extractor.llm.invoke = mock.Mock(side_effect=[mock_entities, mock_scenes, mock_macro, mock_horloges, mock_meta])

    structure = extractor.generate()

    assert structure is not None
    # Verify that invalid references were validated and cleaned/removed
    scenes = {s["id_scene"]: s for s in structure["noeuds_sceniques"]}
    assert "SCENE_02_ROUTE" in scenes
    assert scenes["SCENE_02_ROUTE"]["lieu_rattache_id"] is None
    assert scenes["SCENE_02_ROUTE"]["pnj_presents"] == []

    # Verify macro structure cleaned scenes_incluses
    actes = {a["id_acte"]: a for a in structure["macro_structure"]}
    assert "ACTE_1" in actes
    assert "SCENE_INVALIDE" not in actes["ACTE_1"]["scenes_incluses"]
    assert "SCENE_01_DEPART" in actes["ACTE_1"]["scenes_incluses"]

def test_get_current_context_lookup():
    agent = RPGAgent()
    agent.scenario_structure = {
      "metadata": {"titre": "La Quête", "pitch_global": "Pitch...", "scene_initiale": "SCENE_01"},
      "entites": {
        "pnj": [{"id": "ELROND", "nom_complet": "Maître Elrond", "localisation_habituelle": "FONDCOMBE", "agenda_et_motivation": "Aider", "attitude_initiale": "Amical", "stats_et_capacites": "Inconnu"}],
        "lieux": [{"id": "FONDCOMBE", "nom_complet": "Fondcombe", "ambiance_sensorielle": "Merveilleux", "elements_interactifs": "Livre de Lore"}]
      },
      "macro_structure": [{"id_acte": "ACTE_1", "titre": "Le début", "condition_validation": "Parler à Elrond", "scenes_incluses": ["SCENE_01"]}],
      "noeuds_sceniques": [
        {
          "id_scene": "SCENE_01",
          "acte_rattache_id": "ACTE_1",
          "lieu_rattache_id": "FONDCOMBE",
          "titre": "Rencontre",
          "pnj_presents": ["ELROND"]
        }
      ]
    }
    agent._build_lookups()
    agent.progression = {
        "scene_courante": "SCENE_01",
        "scenes_resolues": [],
        "scenes_contournees": [],
        "horloges": {},
        "ecarts_notables": []
    }

    context = agent.get_current_context()
    assert "SCÈNE COURANTE : Rencontre" in context
    assert "Fondcombe" in context
    assert "Merveilleux" in context
    assert "Maître Elrond" in context
    assert "Le début" in context

def test_transition_deterministic_logic():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"titre": "La Quête", "pitch_global": "...", "scene_initiale": "SCENE_01"},
      "entites": {"pnj": [], "lieux": []},
      "macro_structure": [
          {"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]},
          {"id_acte": "ACTE_2", "titre": "Acte 2", "scenes_incluses": ["SCENE_02"]}
      ],
      "noeuds_sceniques": [
        {
          "id_scene": "SCENE_01",
          "acte_rattache_id": "ACTE_1",
          "titre": "Scène 1",
          "objectif_mj": "...",
          "condition_resolution": "Le PJ avance",
          "sorties_logiques": [{"action_ou_direction": "Avancer", "destination_scene_id": "SCENE_02"}]
        },
        {
          "id_scene": "SCENE_02",
          "acte_rattache_id": "ACTE_2",
          "titre": "Scène 2",
          "objectif_mj": "...",
          "condition_resolution": "Le PJ finit",
          "sorties_logiques": []
        }
      ],
      "horloges_globales": []
    }
    agent._build_lookups()
    agent.progression = {
        "acte_courant": "ACTE_1",
        "scene_courante": "SCENE_01",
        "scenes_resolues": [],
        "scenes_contournees": [],
        "horloges": {},
        "ecarts_notables": []
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
    mock_classif.content = '{"categorie": "transition", "scene_suivante": "SCENE_02", "ecart_notable": "Le PJ a résolu l\'énigme."}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="Le narrateur décrit la transition.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    agent.chat("Je marche vers la sortie")

    assert agent.current_scene_id == "SCENE_02"
    assert agent.progression["scene_courante"] == "SCENE_02"
    assert agent.progression["acte_courant"] == "ACTE_2"
    assert "SCENE_01" in agent.progression["scenes_resolues"]
    agent.update_chronicle.assert_called_once_with("Je marche vers la sortie", "Le narrateur décrit la transition.", "Le PJ a résolu l'énigme.")

def test_transition_invalid_fallback_to_improvisation():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"titre": "La Quête", "pitch_global": "...", "scene_initiale": "SCENE_01"},
      "entites": {"pnj": [], "lieux": []},
      "macro_structure": [{"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]}],
      "noeuds_sceniques": [
        {
          "id_scene": "SCENE_01",
          "acte_rattache_id": "ACTE_1",
          "titre": "Scène 1",
          "objectif_mj": "...",
          "condition_resolution": "Le PJ avance"
        }
      ],
      "horloges_globales": []
    }
    agent._build_lookups()
    agent.progression = {
        "acte_courant": "ACTE_1",
        "scene_courante": "SCENE_01",
        "scenes_resolues": [],
        "scenes_contournees": [],
        "horloges": {},
        "ecarts_notables": []
    }

    agent.get_current_scene = mock.Mock(return_value="Scène de l'auberge")
    agent.get_scenario_context = mock.Mock(return_value="Contexte RAG")
    agent.get_core_context = mock.Mock(return_value="Règles")

    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    # transition to an invalid scene ID
    mock_classif = mock.Mock()
    mock_classif.content = '{"categorie": "transition", "scene_suivante": "SCENE_INVALIDE", "ecart_notable": null}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="Narration.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    agent.chat("Je marche vers la sortie")

    # Should fallback to improvisation: no scene change, no exception raised
    assert agent.current_scene_id == "SCENE_01"
    assert "SCENE_01" not in agent.progression["scenes_resolues"]

def test_contournement_non_blocking():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"titre": "La Quête", "pitch_global": "...", "scene_initiale": "SCENE_01"},
      "entites": {"pnj": [], "lieux": []},
      "macro_structure": [{"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]}],
      "noeuds_sceniques": [
        {
          "id_scene": "SCENE_01",
          "acte_rattache_id": "ACTE_1",
          "titre": "Scène 1"
        }
      ],
      "horloges_globales": []
    }
    agent._build_lookups()
    agent.progression = {
        "acte_courant": "ACTE_1",
        "scene_courante": "SCENE_01",
        "scenes_resolues": [],
        "scenes_contournees": [],
        "horloges": {},
        "ecarts_notables": []
    }

    agent.get_current_scene = mock.Mock(return_value="Scène de l'auberge")
    agent.get_scenario_context = mock.Mock(return_value="Contexte RAG")
    agent.get_core_context = mock.Mock(return_value="Règles")

    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    mock_classif = mock.Mock()
    mock_classif.content = '{"categorie": "contournement", "scene_suivante": null, "ecart_notable": "Le PJ a contourné."}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="Narration contournement.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    response = agent.chat("Je m'enfuis")

    assert response == "Narration contournement."
    assert agent.current_scene_id == "SCENE_01"
    assert "SCENE_01" in agent.progression["scenes_contournees"]

def test_clock_trigger_consequence_injection():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "SCENE_01"
    agent.scenario_structure = {
      "metadata": {"titre": "La Quête", "pitch_global": "...", "scene_initiale": "SCENE_01"},
      "entites": {"pnj": [], "lieux": []},
      "macro_structure": [{"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]}],
      "noeuds_sceniques": [{"id_scene": "SCENE_01", "acte_rattache_id": "ACTE_1", "titre": "Scène 1"}],
      "horloges_globales": [
        {"nom": "Le Volcan", "declencheur": "Chaleur", "consequence": "Le volcan entre en éruption !", "seuil": 3}
      ]
    }
    agent._build_lookups()
    # Clock segments at 2 (so adding 1 triggers it)
    agent.progression = {
        "acte_courant": "ACTE_1",
        "scene_courante": "SCENE_01",
        "scenes_resolues": [],
        "scenes_contournees": [],
        "horloges": {
            "Le Volcan": {"segments": 2, "declenchee": False}
        },
        "ecarts_notables": []
    }

    agent.get_current_scene = mock.Mock(return_value="Scène")
    agent.get_scenario_context = mock.Mock(return_value="Contexte RAG")
    agent.get_core_context = mock.Mock(return_value="Règles")

    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    # Classification output adds 1 segment to "Le Volcan"
    mock_classif = mock.Mock()
    mock_classif.content = '{"categorie": "improvisation", "scene_suivante": null, "horloges_impactees": [{"nom": "Le Volcan", "segments_ajoutes": 1}], "ecart_notable": null}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])

    # We want to capture the instructions passed to narrator
    agent.narrator.generate_response = mock.Mock(return_value="Le volcan explose.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    agent.chat("Je fais une action")

    # Verify clock segment was incremented and declenchee set to True
    assert agent.progression["horloges"]["Le Volcan"]["segments"] == 3
    assert agent.progression["horloges"]["Le Volcan"]["declenchee"] is True

    # Verify that consequence was injected in instructions to Narrator
    call_args = agent.narrator.generate_response.call_args[0]
    instructions = call_args[2]
    assert "ÉVÉNEMENT DÉCLENCHÉ (horloge 'Le Volcan') : Le volcan entre en éruption !" in instructions

    # Verify that the consequence was passed as ecart_notable to Chronicle update
    agent.update_chronicle.assert_called_once_with("Je fais une action", "Le volcan explose.", "Le volcan entre en éruption !")

def test_load_game_restores_scene_progression():
    agent = RPGAgent()

    os.makedirs("Memory", exist_ok=True)
    with open("Memory/scenario_structure.json", "w", encoding="utf-8") as f:
        json.dump({
          "metadata": {"titre": "La Quête", "pitch_global": "Un anneau...", "scene_initiale": "SCENE_01"},
          "entites": {"pnj": [], "lieux": []},
          "macro_structure": [{"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]}],
          "noeuds_sceniques": [{"id_scene": "SCENE_01", "acte_rattache_id": "ACTE_1", "titre": "Scène 1"}],
          "horloges_globales": []
        }, f)

    with open("Memory/progression.json", "w", encoding="utf-8") as f:
        json.dump({
            "acte_courant": "ACTE_1",
            "scene_courante": "SCENE_01",
            "scenes_resolues": [],
            "scenes_contournees": [],
            "horloges": {},
            "ecarts_notables": ["Quelque chose de remarquable"]
        }, f)

    with open("Memory/character.json", "w", encoding="utf-8") as f:
        json.dump({"nom": "Frodon", "classe": "Rogue"}, f)

    loaded = agent.load_game()
    assert loaded is True
    assert agent.current_scene_id == "SCENE_01"
    assert agent.progression["acte_courant"] == "ACTE_1"
    assert agent.progression["ecarts_notables"] == ["Quelque chose de remarquable"]

    # Cleanup
    for file in ["character.json", "scenario_structure.json", "progression.json"]:
        p = os.path.join("Memory", file)
        if os.path.exists(p):
            os.remove(p)
