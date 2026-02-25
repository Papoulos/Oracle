import pytest
from unittest.mock import MagicMock, patch
from agents.agent_garde import AgentGarde
from agents.agent_regles import AgentRegles
from agents.agent_memoire import AgentMemoire
from agents.agent_personnage import AgentPersonnage

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


def test_personnage_creation_completion_is_computed_from_sheet():
    agent = AgentPersonnage()
    memory = {
        "personnage": {
            "nom": "Lyra",
            "race": "Elfe",
            "classe": "Rôdeuse",
            "stats": {"agilite": 14},
            "inventaire": ["Arc court", "Cape"]
        }
    }

    with patch("agents.agent_personnage.safe_chain_invoke") as mock_invoke:
        # 1st call = analyst, 2nd call = DM
        mock_invoke.side_effect = [
            {
                "updates": {},
                "player_agreed_to_roll": False,
                "internal_thought": "No new updates"
            },
            {
                "message": "Parfait, tout est prêt.",
                "reflexion": "Done",
                "creation_terminee": False
            }
        ]

        res = agent.interagir_creation("On valide", memory, journal=[])
        assert res["creation_terminee"] is True


def test_personnage_sheet_normalization_uses_inventory_for_equipment():
    agent = AgentPersonnage()
    sheet = {
        "nom": "Ari",
        "race": "Humain",
        "classe": "Guerrier",
        "stats": {"force": 12},
        "inventaire": ["Épée", "Bouclier"]
    }

    normalized = agent._normalize_character_sheet(sheet)
    assert normalized["equipement"] == "Épée, Bouclier"
    assert agent._compute_missing_fields(normalized) == []
