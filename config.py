import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DATA_PATH = os.getenv("DATA_PATH", "./data")

# Noms des collections
COLLECTION_CODEX = "codex"
COLLECTION_INTRIGUE = "intrigue"

def check_ollama_connectivity():
    import requests
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        return response.status_code == 200
    except:
        return False
