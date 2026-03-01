import json
import re
from agent import RPGAgent
import os

def test_character_completion_transition():
    agent = RPGAgent()
    # Mock the character creator response
    mock_json = '{"nom": "Test", "classe": "Guerrier"}'
    mock_response = f"Voici votre personnage :\n```json\n{mock_json}\n```\nPrêt pour l'aventure ?"

    # We need to mock the internal call to character_creator.generate_response
    # For a simple test, we can just call the logic in chat but replace the creator's output

    # Save original method
    original_generate = agent.character_creator.generate_response
    agent.character_creator.generate_response = lambda input, history: mock_response

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
