import json
import re
from agent import RPGAgent
import os
from unittest import mock

def test_character_completion_transition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import os; os.makedirs("Memory", exist_ok=True)
    agent = RPGAgent()
    agent._load_character_schema = mock.Mock(
        return_value={"required_fields": [{"path": "name", "type": "string"}]}
    )

    # Mock update_sheet directly to return the final character data with status complete
    agent.sheet_manager.update_sheet = mock.Mock(
        return_value={"name": "Test", "class": "Guerrier", "status": "complet"}
    )

    # Mock the character creator response
    mock_json = '{"name": "Test", "class": "Guerrier"}'
    mock_response = f"Voici votre personnage :\n```json\n{mock_json}\n```\nPrêt pour l'aventure ?"

    # Mock the internal call to character_creator.generate_response
    original_generate = agent.character_creator.generate_response
    mock_response_with_status = f"Voici votre personnage :\n```json\n" + json.dumps({"name": "Test", "class": "Guerrier", "status": "complet"}) + "\n```\nPrêt pour l'aventure ?"
    agent.character_creator.generate_response = lambda input, history, char_data: mock_response_with_status

    # Also mock _extract_and_add_resources to prevent LLM/RAG invocation
    agent._extract_and_add_resources = mock.Mock()

    assert agent.game_state == "CREATION"
    agent.chat("Finalise mon perso")

    assert agent.game_state == "SUMMARY"
    assert agent.character_data["name"] == "Test"
    assert os.path.exists("Memory/character.json")

    # Restore
    agent.character_creator.generate_response = original_generate
    print("Test passed: State correctly transitioned to SUMMARY")

if __name__ == "__main__":
    test_character_completion_transition()
