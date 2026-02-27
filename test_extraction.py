import json
import re
import os

def save_character_json(text):
    """Extrait et sauvegarde le JSON du personnage si présent."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            char_data = json.loads(json_match.group(1))
            os.makedirs("Memory", exist_ok=True)
            with open("Memory/character.json", "w", encoding="utf-8") as f:
                json.dump(char_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du JSON : {e}")
    return False

# Test case
test_text = """
Bravo ! Voici votre personnage :

```json
{
    "nom": "Grog l'Invincible",
    "race": "Orque",
    "classe": "Guerrier",
    "statistiques": {
        "force": 18,
        "agilité": 12,
        "intelligence": 8
    }
}
```

Voulez-vous commencer l'aventure ?
"""

if __name__ == "__main__":
    if os.path.exists("Memory/character.json"):
        os.remove("Memory/character.json")

    result = save_character_json(test_text)
    print(f"Extraction result: {result}")

    if os.path.exists("Memory/character.json"):
        with open("Memory/character.json", "r") as f:
            content = json.load(f)
            print("File content:")
            print(json.dumps(content, indent=4))
    else:
        print("File not created!")
