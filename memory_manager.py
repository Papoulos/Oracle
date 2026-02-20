import json
import os

MEMORY_FILE = "memory.json"

DEFAULT_MEMORY = {
    "etape": "CREATION",
    "personnage": {
        "nom": "À définir",
        "stats": {},
        "inventaire": [],
        "xp": 0,
        "niveau": 1
    },
    "monde": {
        "lieu_actuel": "",
        "factions": {},
        "evenements_marquants": [],
        "secrets_decouverts": []
    },
    "historique": [],
    "chronique": [],
    "compteur_actions": 0
}

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return DEFAULT_MEMORY
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return DEFAULT_MEMORY
            return json.loads(content)
    except (json.JSONDecodeError, IOError):
        return DEFAULT_MEMORY

def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_etape(nouvelle_etape):
    memory = load_memory()
    memory["etape"] = nouvelle_etape
    save_memory(memory)

def get_etape():
    memory = load_memory()
    return memory.get("etape", "CREATION")

def update_stats(stats_updates):
    memory = load_memory()
    if "stats" not in memory["personnage"]:
        memory["personnage"]["stats"] = {}
    memory["personnage"]["stats"].update(stats_updates)
    save_memory(memory)

def update_personnage(updates):
    memory = load_memory()
    memory["personnage"].update(updates)
    save_memory(memory)

def add_to_inventory(item):
    memory = load_memory()
    if item not in memory["personnage"]["inventaire"]:
        memory["personnage"]["inventaire"].append(item)
    save_memory(memory)

def update_lieu(nouveau_lieu):
    memory = load_memory()
    memory["monde"]["lieu_actuel"] = nouveau_lieu
    save_memory(memory)

def add_evenement(evenement):
    memory = load_memory()
    memory["monde"]["evenements_marquants"].append(evenement)
    save_memory(memory)

def add_to_history(event_summary):
    memory = load_memory()
    if "historique" not in memory:
        memory["historique"] = []
    memory["historique"].append(event_summary)
    save_memory(memory)

def increment_action_count():
    memory = load_memory()
    memory["compteur_actions"] = memory.get("compteur_actions", 0) + 1
    save_memory(memory)
    return memory["compteur_actions"]

def add_chronique_chapter(chapter_text):
    memory = load_memory()
    if "chronique" not in memory:
        memory["chronique"] = []
    memory["chronique"].append(chapter_text)
    save_memory(memory)

def reset_memory():
    save_memory(DEFAULT_MEMORY)
