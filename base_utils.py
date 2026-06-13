from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config
import re
import json

def extract_json(text):
    """
    Extrait le premier bloc JSON (objet ou tableau) d'un texte.
    Gère les balises ```json ... ``` et tente de trouver le contenu entre { } ou [ ].
    """
    # Nettoyage préliminaire : enlever d'éventuelles balises markdown
    # On cherche d'abord entre balises ```json ... ```
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        content = match.group(1).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Si le contenu dans les balises échoue, on continue avec la recherche globale sur ce contenu
            text = content

    # Recherche du premier '{' ou '[' et du dernier '}' ou ']'
    start_brace = text.find('{')
    start_bracket = text.find('[')

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        # On a probablement un objet
        end_brace = text.rfind('}')
        if end_brace != -1:
            content = text[start_brace:end_brace+1]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
    elif start_bracket != -1:
        # On a probablement un tableau
        end_bracket = text.rfind(']')
        if end_bracket != -1:
            content = text[start_bracket:end_bracket+1]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
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
