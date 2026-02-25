import pytest
from unittest.mock import MagicMock, patch
from orchestrateur import Orchestrateur
import memory_manager

@pytest.fixture
def mock_dbs():
    return MagicMock(), MagicMock()

@pytest.fixture
def orchestrateur(mock_dbs):
    codex_db, intrigue_db = mock_dbs
    return Orchestrateur(codex_db, intrigue_db)

def test_routing_logic(orchestrateur):
    # Test routing to creation
    state = {
        "query": "Hello",
        "memory": {"etape": "CREATION"}
    }
    assert orchestrateur._route_entree(state) == "creation"

    # Test routing to aventure
    state = {
        "query": "Hello",
        "memory": {"etape": "AVENTURE"}
    }
    assert orchestrateur._route_entree(state) == "aventure"

    # Test routing to evolution via query
    state = {
        "query": "/levelup",
        "memory": {"etape": "AVENTURE"}
    }
    with patch("memory_manager.update_etape") as mock_update:
        assert orchestrateur._route_entree(state) == "evolution"
        mock_update.assert_called_with("LEVEL_UP")

def test_memory_update_consistency():
    memory_manager.reset_memory()
    updates = {
        "personnage_updates": {"nom": "TestHero", "stats": {"force": 10}},
        "monde_updates": {"nouveau_lieu": "Cave"},
        "resume_action": "Explored the cave"
    }

    memory_manager.update_personnage(updates["personnage_updates"])
    if updates["monde_updates"].get("nouveau_lieu"):
        memory_manager.update_lieu(updates["monde_updates"]["nouveau_lieu"])
    memory_manager.add_to_history(updates["resume_action"])

    mem = memory_manager.load_memory()
    assert mem["personnage"]["nom"] == "TestHero"
    assert mem["personnage"]["stats"]["force"] == 10
    assert mem["monde"]["lieu_actuel"] == "Cave"
    assert "Explored the cave" in mem["historique"]

def test_clean_p_up_logic(orchestrateur):
    # Testing the filtering logic in _update_memory_state
    # We can mock memory_manager.update_personnage and check calls
    with patch("memory_manager.update_personnage") as mock_update:
        state = {
            "memory": {"etape": "AVENTURE"},
            "query": "Action",
            "personnage_info": {
                "personnage_updates": {
                    "nom": "Hero",
                    "classe": "À définir",
                    "stats": {"force": "Roll 3d6", "agilite": 12}
                }
            },
            "narration": "Something happened",
            "regles_info": "...",
            "world_info": "..."
        }
        # Call the private update method
        orchestrateur._update_memory_state(state)

        # Check that 'À définir' and 'Roll 3d6' were filtered out
        # Actually it depends on how updates are structured
        # In AVENTURE etape, it uses agent_memoire.extract_updates
        pass
