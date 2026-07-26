import json
import re
from agent import RPGAgent
import os
from unittest import mock

def test_character_completion_transition():
    if os.path.exists("Memory/character_schema.json"):
        try:
            os.remove("Memory/character_schema.json")
        except Exception:
            pass
    agent = RPGAgent()

    # Mock update_sheet directly to return the final character data with status complete
    agent.sheet_manager.update_sheet = mock.Mock(
        return_value={"nom": "Test", "classe": "Guerrier", "statut": "complet"}
    )

    # Mock the character creator response
    mock_json = '{"nom": "Test", "classe": "Guerrier"}'
    mock_response = f"Voici votre personnage :\n```json\n{mock_json}\n```\nPrêt pour l'aventure ?"

    # Mock the internal call to character_creator.generate_response
    original_generate = agent.character_creator.generate_response
    # Add statut: complet to trigger transition
    mock_response_with_status = f"Voici votre personnage :\n```json\n" + json.dumps({"nom": "Test", "classe": "Guerrier", "statut": "complet"}) + "\n```\nPrêt pour l'aventure ?"
    agent.character_creator.generate_response = lambda input, history, char_data: mock_response_with_status

    # Also mock _extract_and_add_resources to prevent LLM/RAG invocation
    agent._extract_and_add_resources = mock.Mock()

    assert agent.game_state == "CREATION"
    agent.chat("Finalise mon perso")

    assert agent.game_state == "SUMMARY"
    assert agent.character_data["nom"] == "Test"
    assert os.path.exists("Memory/character.json")

    # Restore
    agent.character_creator.generate_response = original_generate
    print("Test passed: State correctly transitioned to SUMMARY")

if __name__ == "__main__":
    test_character_completion_transition()
