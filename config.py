import os
import sys
from dotenv import load_dotenv

load_dotenv()

# --- Configuration LLM (Texte) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

# --- Configuration Embeddings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# --- Configuration par Agent ---
CHARACTER_MODEL = os.getenv("CHARACTER_MODEL")
CHARACTER_TEMP = float(os.getenv("CHARACTER_TEMP")) if os.getenv("CHARACTER_TEMP") else None

NARRATOR_MODEL = os.getenv("NARRATOR_MODEL")
NARRATOR_TEMP = float(os.getenv("NARRATOR_TEMP")) if os.getenv("NARRATOR_TEMP") else None

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL")
ORCHESTRATOR_TEMP = float(os.getenv("ORCHESTRATOR_TEMP")) if os.getenv("ORCHESTRATOR_TEMP") else None

CHRONICLE_MODEL = os.getenv("CHRONICLE_MODEL")
CHRONICLE_TEMP = float(os.getenv("CHRONICLE_TEMP")) if os.getenv("CHRONICLE_TEMP") else None

# --- Configuration Serveur ---
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS")
SERVER_PORT = int(os.getenv("SERVER_PORT")) if os.getenv("SERVER_PORT") else None

# --- Autres paramètres ---
CHROMA_PATH = os.getenv("CHROMA_PATH")
CORE_DATA_PATH = os.getenv("CORE_DATA_PATH")
SCENARIO_DATA_PATH = os.getenv("SCENARIO_DATA_PATH")

# Noms des collections VectorDB
CORE_COLLECTION_NAME = os.getenv("CORE_COLLECTION_NAME")
SCENARIO_COLLECTION_NAME = os.getenv("SCENARIO_COLLECTION_NAME")

# --- RAG ---
RAG_SEARCH_K = int(os.getenv("RAG_SEARCH_K")) if os.getenv("RAG_SEARCH_K") else None

# --- Compatibilité Ollama (Anciennes variables) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")

def check_config():
    """
    Vérifie que toutes les variables de configuration nécessaires sont présentes.
    S'arrête si une variable obligatoire est manquante.
    """
    required_vars = [
        "LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL",
        "EMBEDDING_PROVIDER", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL",
        "CHARACTER_MODEL", "CHARACTER_TEMP",
        "NARRATOR_MODEL", "NARRATOR_TEMP",
        "ORCHESTRATOR_MODEL", "ORCHESTRATOR_TEMP",
        "CHRONICLE_MODEL", "CHRONICLE_TEMP",
        "SERVER_ADDRESS", "SERVER_PORT",
        "CHROMA_PATH", "CORE_DATA_PATH", "SCENARIO_DATA_PATH",
        "CORE_COLLECTION_NAME", "SCENARIO_COLLECTION_NAME",
        "RAG_SEARCH_K"
    ]

    missing_vars = []
    for var in required_vars:
        val = os.getenv(var)
        if val is None or val.strip() == "":
            missing_vars.append(var)

    if missing_vars:
        print("\n❌ ERREUR DE CONFIGURATION")
        print("Les variables d'environnement suivantes sont manquantes dans votre fichier .env :")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nVeuillez copier .env.example vers .env et configurer les variables manquantes.")
        sys.exit(1)
