import json
import os
import shutil
from agent import RPGAgent
from unittest import mock

def setup_module():
    if os.path.exists("Memory"):
        shutil.rmtree("Memory")
    os.makedirs("Memory", exist_ok=True)

def test_incremental_creation():
    # Write a test schema before initializing the agent
    schema = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "race", "type": "string"},
            {"path": "class", "type": "string"}
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open("Memory/character_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f)

    agent = RPGAgent()

    # Mock sheet manager update_sheet behavior to avoid calling actual LLM/connection
    def mock_update_sheet(char_sheet, user_input, response, mode="ADVENTURE"):
        if "Aragorn" in user_input:
            return {"name": "Aragorn", "status": "en_cours"}
        elif "Humain" in user_input:
            return {"name": "Aragorn", "race": "Humain", "status": "en_cours"}
        elif "Rôdeur" in user_input:
            return {"name": "Aragorn", "race": "Humain", "class": "Rôdeur", "status": "complet"}
        return char_sheet

    agent.sheet_manager.update_sheet = mock_update_sheet
    agent._extract_and_add_resources = mock.Mock()

    # Step 1: Initial creation (Name)
    mock_json_1 = '{"name": "Aragorn", "status": "en_cours"}'
    mock_response_1 = f"Bonjour Aragorn ! Quelle est votre race ?\n```json\n{mock_json_1}\n```"

    original_generate = agent.character_creator.generate_response
    agent.character_creator.generate_response = lambda input, history, char_data: mock_response_1

    agent.chat("Je m'appelle Aragorn")

    assert agent.game_state == "CREATION"
    assert agent.character_data["name"] == "Aragorn"
    assert agent.character_data["status"] == "en_cours"
    assert os.path.exists("Memory/character.json")

    # Step 2: Second step (Race) - GM should receive existing data
    mock_json_2 = '{"name": "Aragorn", "race": "Humain", "status": "en_cours"}'

    def mock_gen_2(input, history, char_data):
        assert char_data["name"] == "Aragorn"
        return f"Un Humain, très bien !\n```json\n{mock_json_2}\n```"

    agent.character_creator.generate_response = mock_gen_2
    agent.chat("Je suis un Humain")

    assert agent.game_state == "CREATION"
    assert agent.character_data["race"] == "Humain"

    # Step 3: Final step
    mock_json_3 = '{"name": "Aragorn", "race": "Humain", "class": "Rôdeur", "status": "complet"}'
    agent.character_creator.generate_response = lambda input, history, char_data: f"Terminé !\n```json\n{mock_json_3}\n```"

    agent.chat("Je suis un Rôdeur")

    assert agent.game_state == "SUMMARY"
    assert agent.character_data["class"] == "Rôdeur"
    assert agent.character_data["status"] == "complet"

    # Restore
    agent.character_creator.generate_response = original_generate
    print("Test incremental creation passed!")

if __name__ == "__main__":
    setup_module()
    test_incremental_creation()
