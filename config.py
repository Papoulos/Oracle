import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DATA_PATH = os.getenv("DATA_PATH", "./data")

# Noms des collections
COLLECTION_CODEX = "codex"
COLLECTION_INTRIGUE = "intrigue"
