import json
import pytest
import os
from unittest import mock
from scenario_agents import SceneGraphAgent
from agent import RPGAgent
from indexer import index_scenes

class MockDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MockScenarioStore:
    def similarity_search(self, query, k=15, filter=None):
        # On renvoie des faux documents
        return [
            MockDocument("Scène 1 : l'auberge du Poney Fringant à Bree. Les PJ rencontrent l'aubergiste.", {"scene_id": "1.1"}),
            MockDocument("Scène 2 : la route vers Fondcombe. Traversée des marais et attaque de loups.", {"scene_id": "1.2"})
        ]

    def delete(self, filter=None):
        pass

    def add_documents(self, documents):
        pass

def test_scenegraph_agent_generate():
    store = MockScenarioStore()
    agent = SceneGraphAgent(store)

    # Mock de l'invocation du LLM de l'agent
    mock_response = mock.Mock()
    mock_response.content = """
    ```json
    {
      "scene_initiale": "1.1",
      "scenes": [
        {
          "id": "1.1",
          "titre": "L'auberge",
          "lieu": "Bree",
          "pnjs": ["aubergiste"],
          "esprit_de_la_scene": "Le PJ doit en apprendre plus sur l'anneau.",
          "elements_a_preserver": ["L'aubergiste est amical mais méfiant"],
          "reactions_anticipees": [
            {"action_probable": "Discuter", "consequence": "Il donne une lettre"}
          ],
          "objectif_atteint_si": "le joueur obtient des informations",
          "statut": "a_venir"
        },
        {
          "id": "1.2",
          "titre": "La route",
          "lieu": "Les marais",
          "pnjs": [],
          "esprit_de_la_scene": "Surmonter les dangers de la route.",
          "elements_a_preserver": [],
          "reactions_anticipees": [],
          "objectif_atteint_si": "le joueur survit aux marais",
          "statut": "a_venir"
        }
      ]
    }
    ```
    """

    # Mocking self.llm with a mock object since Pydantic prevents direct attribute setting on ChatOllama
    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(return_value=mock_response)

    scenes_data = agent.generate(scenario_summary={"pitch": "Une quête légendaire."})
    assert scenes_data is not None
    assert scenes_data["scene_initiale"] == "1.1"
    assert len(scenes_data["scenes"]) == 2
    assert scenes_data["scenes"][0]["statut"] == "en_cours"
    assert scenes_data["scenes"][1]["statut"] == "a_venir"
    assert os.path.exists("Memory/scenes.json")

def test_transition_deterministic_logic():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "1.1"
    agent.scenes_data = {
      "scene_initiale": "1.1",
      "scenes": [
        {
          "id": "1.1",
          "titre": "L'auberge",
          "statut": "en_cours",
          "esprit_de_la_scene": "...",
          "objectif_atteint_si": "..."
        },
        {
          "id": "1.2",
          "titre": "La route",
          "statut": "a_venir",
          "esprit_de_la_scene": "...",
          "objectif_atteint_si": "..."
        }
      ]
    }

    # Mock get_current_scene et get_scenario_context
    agent.get_current_scene = mock.Mock(return_value="Scène de l'auberge")
    agent.get_scenario_context = mock.Mock(return_value="Contexte de Bree")
    agent.get_core_context = mock.Mock(return_value="Règles du jeu")

    # Mock LLM pour l'analyse mécanique
    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    # Mock LLM pour la classification de scène (transition)
    mock_classif = mock.Mock()
    mock_classif.content = '{"categorie": "transition", "scene_suivante": "1.2", "note_chronique": "Le PJ a résolu l\'énigme de l\'aubergiste."}'

    # Mock LLM pour la réponse finale du narrateur
    mock_narrator = mock.Mock()
    mock_narrator.content = "Le narrateur vous répond avec éloquence."

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="Le narrateur décrit la transition.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    agent.chat("Je parle à l'aubergiste")

    # Vérification de la transition de scène déterministe
    assert agent.current_scene_id == "1.2"
    assert agent.scenes_data["scenes"][0]["statut"] == "resolue"
    assert agent.scenes_data["scenes"][1]["statut"] == "en_cours"
    agent.update_chronicle.assert_called_once_with("Je parle à l'aubergiste", "Le narrateur décrit la transition.", "Le PJ a résolu l'énigme de l'aubergiste.")

def test_contournement_non_blocking():
    agent = RPGAgent()
    agent.game_state = "ADVENTURE"
    agent.current_scene_id = "1.1"
    agent.scenes_data = {
      "scene_initiale": "1.1",
      "scenes": [
        {
          "id": "1.1",
          "titre": "L'auberge",
          "statut": "en_cours",
          "esprit_de_la_scene": "...",
          "objectif_atteint_si": "..."
        }
      ]
    }

    agent.get_current_scene = mock.Mock(return_value="Scène de l'auberge")
    agent.get_scenario_context = mock.Mock(return_value="Contexte de Bree")
    agent.get_core_context = mock.Mock(return_value="Règles du jeu")

    mock_mech = mock.Mock()
    mock_mech.content = '{"need_roll": false, "mechanical_decision": null}'

    mock_classif = mock.Mock()
    mock_classif.content = '{"categorie": "contournement", "scene_suivante": null, "note_chronique": "Le PJ a brûlé l\'auberge !"}'

    agent.llm = mock.Mock()
    agent.llm.invoke = mock.Mock(side_effect=[mock_mech, mock_classif])
    agent.narrator.generate_response = mock.Mock(return_value="L'auberge s'embrase.")
    agent.sheet_manager.update_sheet = mock.Mock(return_value={})
    agent.update_chronicle = mock.Mock()

    # L'appel ne doit pas planter ni lever d'exception
    response = agent.chat("Je jette une torche sur le toit")

    assert response == "L'auberge s'embrase."
    assert agent.current_scene_id == "1.1"
    assert agent.scenes_data["scenes"][0]["statut"] == "contournee"
    agent.update_chronicle.assert_called_once_with("Je jette une torche sur le toit", "L'auberge s'embrase.", "Le PJ a brûlé l'auberge !")

def test_load_game_restores_scene():
    agent = RPGAgent()

    # Écriture d'une scène sauvegardée
    os.makedirs("Memory", exist_ok=True)
    with open("Memory/scenes.json", "w", encoding="utf-8") as f:
        json.dump({
            "scene_initiale": "1.1",
            "scenes": [
                {"id": "1.1", "statut": "resolue"},
                {"id": "1.2", "statut": "en_cours"}
            ]
        }, f)

    with open("Memory/character.json", "w", encoding="utf-8") as f:
        json.dump({"nom": "Frodon", "classe": "Rogue"}, f)

    with open("Memory/scenario.json", "w", encoding="utf-8") as f:
        json.dump({"titre": "La quête", "pitch": "Un anneau..."}, f)

    loaded = agent.load_game()
    assert loaded is True
    assert agent.current_scene_id == "1.2"
    assert agent.scenes_data["scenes"][0]["statut"] == "resolue"
    assert agent.scenes_data["scenes"][1]["statut"] == "en_cours"

    # Nettoyage des fichiers temporaires du test
    for file in ["character.json", "scenario.json", "scenes.json"]:
        p = os.path.join("Memory", file)
        if os.path.exists(p):
            os.remove(p)
