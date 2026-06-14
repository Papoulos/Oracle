from base_utils import extract_json

test_json_with_nesting = """
Voici le scénario :
```json
{
  "titre": "Test",
  "lieux": [
    {"nom": "Lieu 1", "desc": "Desc 1"},
    {"nom": "Lieu 2", "desc": "Desc 2"}
  ],
  "autre": {
    "cle": "valeur"
  }
}
```
Fin.
"""

def test():
    result = extract_json(test_json_with_nesting)
    if result and result.get("titre") == "Test" and len(result.get("lieux", [])) == 2:
        print("✅ Extraction avec imbrication réussie")
    else:
        print("❌ Échec de l'extraction avec imbrication")
        print(f"Résultat : {result}")

if __name__ == "__main__":
    test()
