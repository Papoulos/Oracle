import json
import os
from filelock import FileLock

MEMORY_FILE = "memory.json"
LOCK_FILE = "memory.json.lock"

DEFAULT_MEMORY = {
    "etape": "CREATION",
    "personnage": {
        "nom": "À définir",
        "race": "À définir",
        "classe": "À définir",
        "stats": {},
        "inventaire": [],
        "points_de_passage": {},
        "journal_creation": [],
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

lock = FileLock(LOCK_FILE)

def load_memory():
    with lock:
        if not os.path.exists(MEMORY_FILE):
            return DEFAULT_MEMORY.copy()
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return DEFAULT_MEMORY.copy()
                return json.loads(content)
        except (json.JSONDecodeError, IOError):
            return DEFAULT_MEMORY.copy()

def save_memory(data):
    with lock:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def update_etape(nouvelle_etape):
    with lock:
        memory = load_memory_internal()
        memory["etape"] = nouvelle_etape
        save_memory_internal(memory)

def get_etape():
    memory = load_memory()
    return memory.get("etape", "CREATION")

def update_stats(stats_updates):
    with lock:
        memory = load_memory_internal()
        if "stats" not in memory["personnage"]:
            memory["personnage"]["stats"] = {}
        memory["personnage"]["stats"].update(stats_updates)
        save_memory_internal(memory)

def update_personnage(updates):
    with lock:
        memory = load_memory_internal()
        for key, value in updates.items():
            if isinstance(value, dict) and key in memory["personnage"] and isinstance(memory["personnage"][key], dict):
                memory["personnage"][key].update(value)
            else:
                memory["personnage"][key] = value
        save_memory_internal(memory)

def add_to_inventory(item):
    with lock:
        memory = load_memory_internal()
        if item not in memory["personnage"]["inventaire"]:
            memory["personnage"]["inventaire"].append(item)
        save_memory_internal(memory)

def update_lieu(nouveau_lieu):
    with lock:
        memory = load_memory_internal()
        memory["monde"]["lieu_actuel"] = nouveau_lieu
        save_memory_internal(memory)

def add_evenement(evenement):
    with lock:
        memory = load_memory_internal()
        memory["monde"]["evenements_marquants"].append(evenement)
        save_memory_internal(memory)

def add_to_history(event_summary):
    with lock:
        memory = load_memory_internal()
        if "historique" not in memory:
            memory["historique"] = []
        memory["historique"].append(event_summary)
        save_memory_internal(memory)

def increment_action_count():
    with lock:
        memory = load_memory_internal()
        memory["compteur_actions"] = memory.get("compteur_actions", 0) + 1
        save_memory_internal(memory)
        return memory["compteur_actions"]

def add_chronique_chapter(chapter_text):
    with lock:
        memory = load_memory_internal()
        if "chronique" not in memory:
            memory["chronique"] = []
        memory["chronique"].append(chapter_text)
        save_memory_internal(memory)

def add_to_journal_creation(message):
    with lock:
        memory = load_memory_internal()
        if "journal_creation" not in memory["personnage"]:
            memory["personnage"]["journal_creation"] = []
        memory["personnage"]["journal_creation"].append(message)
        save_memory_internal(memory)

def reset_memory():
    save_memory(DEFAULT_MEMORY.copy())

# Internal helpers that don't acquire lock (to be used inside with lock)
def load_memory_internal():
    if not os.path.exists(MEMORY_FILE):
        return DEFAULT_MEMORY.copy()
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return DEFAULT_MEMORY.copy()
            return json.loads(content)
    except (json.JSONDecodeError, IOError):
        return DEFAULT_MEMORY.copy()

def save_memory_internal(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
