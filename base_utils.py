from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config
import re
import json

def extract_json(text: str, expected_type: type = dict):
    """
    Extrait un bloc JSON (objet ou tableau) d'un texte.
    expected_type = dict ou list
    """
    if not text:
        return None

    open_char  = "{" if expected_type == dict else "["
    close_char = "}" if expected_type == dict else "]"

    # 1. Balises ```json
    m = re.search(rf"```(?:json)?\s*({re.escape(open_char)}[\s\S]*?{re.escape(close_char)})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2. Premier bloc complet
    m = re.search(rf"({re.escape(open_char)}[\s\S]*{re.escape(close_char)})", text)
    if m:
        try:
            # On prend le plus long match possible pour éviter les faux positifs si plusieurs blocs existent
            # Mais re.search avec [\s\S]* est déjà gourmand par défaut
            return json.loads(m.group(1))
        except Exception:
            pass

    return None

def get_llm(model_name, temperature):
    if config.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=config.LLM_BASE_URL,
            temperature=temperature
        )
    else: # openai / llama-cpp
        return ChatOpenAI(
            model=model_name,
            base_url=config.LLM_BASE_URL,
            temperature=temperature,
            api_key="sk-no-key-required"
        )

def get_embeddings():
    if config.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.EMBEDDING_BASE_URL
        )
    else: # openai / llama-cpp
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.EMBEDDING_BASE_URL,
            api_key="sk-no-key-required"
        )

class BaseAgent:
    def __init__(self, model=None, temperature=0.7):
        model_name = model if model else config.LLM_MODEL
        self.llm = get_llm(model_name, temperature)
