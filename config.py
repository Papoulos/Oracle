import os
from dotenv import load_dotenv

load_dotenv()

# Configuration Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Chemins
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CORE_DATA_PATH = os.getenv("CORE_DATA_PATH", "./data/core")
SCENARIO_DATA_PATH = os.getenv("SCENARIO_DATA_PATH", "./data/scenario")

# Noms des collections VectorDB
CORE_COLLECTION_NAME = "core_collection"
SCENARIO_COLLECTION_NAME = "scenario_collection"
