from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config
import re
import json

def extract_json(text: str, expected_type: type = dict):
    """
    Extrait un bloc JSON (objet ou tableau) d'un texte de manière robuste.
    Tente de trouver tous les blocs JSON et retourne le dernier valide.
    """
    if not text:
        return None

    open_char  = "{" if expected_type == dict else "["
    close_char = "}" if expected_type == dict else "]"

    # 1. Extraction de tous les blocs markdown ```json ... ```
    # On utilise une recherche non-gourmande pour les blocs eux-mêmes,
    # mais on veut quand même supporter l'imbrication à l'intérieur d'un bloc.
    # Note : Le regex [\s\S]*? s'arrête au premier ``` suivant.
    # Prétraitement : supprimer certains préfixes courants que les LLM ajoutent parfois
    text = re.sub(r"^(?:JSON|Résultat|Voici le JSON|Output)\s*:\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)

    markdown_blocks = re.findall(rf"```(?:json)?\s*({re.escape(open_char)}[\s\S]*?{re.escape(close_char)})\s*```", text)

    # Inverser pour tester du plus récent (bas du message) au plus ancien
    for block in reversed(markdown_blocks):
        try:
            # Nettoyage minimal des espaces/caractères invisibles
            block_clean = block.strip()
            # Nettoyage des virgules traînantes avant les fermetures
            block_clean = re.sub(r",\s*([\]}])", r"\1", block_clean)
            return json.loads(block_clean)
        except json.JSONDecodeError as e:
            print(f"[extract_json] DEBUG: Échec décodage bloc markdown : {e}")
            continue

    # 2. Si aucun bloc markdown n'est valide, on cherche des structures JSON brutes
    # On cherche tous les blocs commençant par open_char et finissant par close_char
    # de manière gourmande pour capturer le maximum.
    # On utilise une recherche qui s'assure que le premier open_char n'est pas précédé d'un autre open_char sans close_char
    raw_matches = re.findall(rf"({re.escape(open_char)}[\s\S]*{re.escape(close_char)})", text)
    for match in reversed(raw_matches):
        # On tente de trouver le JSON valide à l'intérieur (au cas où il y aurait du texte autour)
        # On essaie de réduire le match de la fin vers le début pour trouver le bon crochet fermant
        temp_match = match.strip()
        while temp_match:
            try:
                # Nettoyage des virgules traînantes avant de tenter le chargement
                clean_json = re.sub(r",\s*([\]}])", r"\1", temp_match)
                return json.loads(clean_json)
            except json.JSONDecodeError:
                # Retirer le dernier caractère et chercher le précédent close_char
                last_close = temp_match.rfind(close_char, 0, -1)
                if last_close == -1:
                    break
                temp_match = temp_match[:last_close + 1]

    return None

def get_llm(model_name, temperature):
    if config.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=config.LLM_BASE_URL,
            temperature=temperature,
            num_ctx=8192,  # Fenêtre de contexte
            num_predict=2048  # Limite de tokens en sortie pour éviter les troncatures
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
