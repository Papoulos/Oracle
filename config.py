import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration LLM (Texte) ---
# Provider: "ollama" ou "openai" (pour llama-cpp ou autre API compatible)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3")

# --- Configuration Embeddings ---
# Provider: "ollama" ou "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# --- Configuration par Agent ---
CHARACTER_MODEL = os.getenv("CHARACTER_MODEL", LLM_MODEL)
CHARACTER_TEMP = float(os.getenv("CHARACTER_TEMP", 0.7))

NARRATOR_MODEL = os.getenv("NARRATOR_MODEL", LLM_MODEL)
NARRATOR_TEMP = float(os.getenv("NARRATOR_TEMP", 0.7))

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", LLM_MODEL)
ORCHESTRATOR_TEMP = float(os.getenv("ORCHESTRATOR_TEMP", 0.7))

# --- Autres paramètres ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CORE_DATA_PATH = os.getenv("CORE_DATA_PATH", "./data/core")
SCENARIO_DATA_PATH = os.getenv("SCENARIO_DATA_PATH", "./data/scenario")

# Noms des collections VectorDB
CORE_COLLECTION_NAME = "core_collection"
SCENARIO_COLLECTION_NAME = "scenario_collection"

# --- Compatibilité Ollama (Anciennes variables) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", LLM_BASE_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", LLM_MODEL)
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", EMBEDDING_MODEL)
