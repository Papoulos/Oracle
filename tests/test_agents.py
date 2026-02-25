import pytest
from unittest.mock import MagicMock, patch
from agents.agent_garde import AgentGarde
from agents.agent_regles import AgentRegles
from agents.agent_memoire import AgentMemoire

def test_garde_anti_hallucination():
    mock_codex = MagicMock()
    mock_intrigue = MagicMock()
    # Mock search to return empty
    mock_intrigue.similarity_search.return_value = []

    agent = AgentGarde(mock_codex, mock_intrigue)

    # Mock LLM response for impossible action
    with patch("agents.agent_garde.safe_chain_invoke") as mock_invoke:
        mock_invoke.return_value = {
            "possible": False,
            "raison": "NON, parce que le lieu mentionné n'existe pas."
        }

        res = agent.valider_action("Je vais à la taverne fantôme", '{"monde": {"lieu_actuel": "Forêt"}}')
        assert res["possible"] is False
        assert res["raison"].startswith("NON")

def test_regles_simple_action():
    mock_codex = MagicMock()
    agent = AgentRegles(mock_codex)

    with patch("agents.agent_regles.safe_chain_invoke") as mock_invoke:
        mock_invoke.return_value = {
            "besoin_jet": False,
            "jet_format": None,
            "explication_regle": "Action simple",
            "seuil": None
        }

        analyse, _ = agent.evaluer_besoin_jet("Je marche", "{}", "OUI")
        assert analyse["besoin_jet"] is False

def test_memoire_no_fake_facts():
    agent = AgentMemoire()

    with patch("agents.agent_memoire.safe_chain_invoke") as mock_invoke:
        # Mocking a response where the LLM might hallucinate,
        # but our goal is to check if we handle the validated output correctly
        mock_invoke.return_value = {
            "personnage_updates": {},
            "monde_updates": {},
            "resume_action": "Rien de spécial"
        }

        updates = agent.extract_updates("Quête", "...", "...", "Le joueur ne fait rien.")
        assert updates["resume_action"] == "Rien de spécial"
        assert updates["personnage_updates"] == {}
