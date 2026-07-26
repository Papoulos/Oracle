from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config
import re
import json

def extract_json(text: str, expected_type: type = dict):
    """
    Extrait un bloc JSON (objet ou tableau) d'un texte de manière robuste.
    Tente de trouver tous les blocs JSON et retourne le dernier valide (avec réparation si tronqué).
    """
    if not text:
        return None

    open_char  = "{" if expected_type == dict else "["
    close_char = "}" if expected_type == dict else "]"

    # Prétraitement : supprimer certains préfixes courants que les LLM ajoutent parfois
    text = re.sub(r"^(?:JSON|Résultat|Voici le JSON|Output)\s*:\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Nettoyer une chaîne candidate et tenter de la charger directement
    def clean_and_load(s: str):
        s_clean = s.strip()
        s_clean = re.sub(r",\s*([\]}])", r"\1", s_clean)
        return json.loads(s_clean)

    # Tenter de réparer une chaîne tronquée en cherchant les accolades/crochets de droite à gauche
    def try_repair(s: str):
        indices = [i for i, char in enumerate(s) if char == close_char]
        for idx in reversed(indices):
            candidate = s[:idx+1]
            for suffix in ["", "}", "]}", "}}]}", "]", "]}", "]]}"]:
                try:
                    return clean_and_load(candidate + suffix)
                except json.JSONDecodeError:
                    continue
        return None

    # 1. Extraction via les blocs markdown ```json ... ``` (fermés ou non)
    markdown_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", text)
    for block in reversed(markdown_blocks):
        block_str = block.strip()
        if not block_str.startswith(open_char):
            first_open = block_str.find(open_char)
            if first_open != -1:
                block_str = block_str[first_open:]
            else:
                continue
        try:
            return clean_and_load(block_str)
        except json.JSONDecodeError:
            repaired = try_repair(block_str)
            if repaired is not None:
                return repaired

    # 2. Si aucun bloc markdown n'est valide, on cherche dans le texte brut
    first_open_idx = text.find(open_char)
    if first_open_idx != -1:
        raw_text = text[first_open_idx:]
        try:
            return clean_and_load(raw_text)
        except json.JSONDecodeError:
            repaired = try_repair(raw_text)
            if repaired is not None:
                return repaired

    return None

def get_llm(model_name, temperature):
    if config.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=config.LLM_BASE_URL,
            temperature=temperature,
            num_ctx=16384, # Fenêtre de contexte doublée pour supporter RAG + historique + réponses longues
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

def get_full_store_text(store, log) -> str:
    """Récupère et concatène tout le texte indexé dans `store`, trié par page."""
    try:
        result = store.get(include=["documents", "metadatas"])
        paired = sorted(
            zip(result.get("documents", []), result.get("metadatas", []) or [{}] * len(result.get("documents", []))),
            key=lambda x: x[1].get("page", 0) if x[1] else 0
        )
        return "\n\n".join(doc for doc, _ in paired)
    except Exception as e:
        log(f"⚠ Erreur lors de la récupération du texte complet : {e}")
        return ""


def get_relevant_context(store, queries, log, threshold_chars, k=15) -> str:
    """Texte source complet si sa taille le permet, sinon RAG par requêtes dédupliqué."""
    full_text = get_full_store_text(store, log)
    if full_text and len(full_text) <= threshold_chars:
        log(f"Utilisation du texte source complet (taille: {len(full_text)} <= {threshold_chars} car.)")
        return full_text
    log(f"Utilisation de requêtes RAG par similarité (taille de full_text: {len(full_text)})")
    all_docs = []
    for q in queries:
        all_docs.extend(store.similarity_search(q, k=k))
    unique_contents = {d.page_content: d for d in all_docs}
    return "\n\n---\n\n".join(unique_contents.keys())

class BaseAgent:
    def __init__(self, model=None, temperature=0.7):
        model_name = model if model else config.LLM_MODEL
        self.llm = get_llm(model_name, temperature)
