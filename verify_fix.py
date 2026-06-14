import json
import os
import sys

# Ajout du chemin actuel pour importer les modules du projet
sys.path.append(os.getcwd())

from base_utils import extract_json

def test_extract_json_nested():
    print("Test: extract_json avec JSON imbriqué")
    text = """Voici le JSON:
```json
{
  "titre": "Aventure",
  "actes": [
    {"num": 1, "nom": "Debut"},
    {"num": 2, "nom": "Fin"}
  ]
}
```
Texte après."""
    result = extract_json(text)
    assert result is not None
    assert result["titre"] == "Aventure"
    assert len(result["actes"]) == 2
    print("✅ Réussi")

def test_extract_json_no_markdown():
    print("Test: extract_json sans balises markdown")
    text = """Texte avant { "cle": "valeur", "liste": [1, 2] } texte après"""
    result = extract_json(text)
    assert result is not None
    assert result["cle"] == "valeur"
    print("✅ Réussi")

def test_unwrap_character_data():
    print("Test: _unwrap_character_data")
    from agent import RPGAgent
    # On mocke les stores pour éviter l'init de Chroma
    class MockStore:
        def __init__(self): pass

    agent = RPGAgent.__new__(RPGAgent) # On ne veut pas appeler __init__

    nested_data = {
        "personnage": {
            "nom": "Elias",
            "classe": "Rodeur"
        }
    }
    unwrapped = agent._unwrap_character_data(nested_data)
    assert unwrapped["nom"] == "Elias"

    flat_data = {"nom": "Test", "classe": "Guerrier"}
    unwrapped2 = agent._unwrap_character_data(flat_data)
    assert unwrapped2["nom"] == "Test"
    print("✅ Réussi")

if __name__ == "__main__":
    try:
        test_extract_json_nested()
        test_extract_json_no_markdown()
        test_unwrap_character_data()
        print("\nTous les tests de vérification sont passés !")
    except Exception as e:
        print(f"\n❌ Échec d'un test : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
