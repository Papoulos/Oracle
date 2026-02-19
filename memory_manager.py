import json
import os

MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_stats(stats_updates):
    memory = load_memory()
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
