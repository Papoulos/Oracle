from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import config
import re
import json
import logging



def extract_json(text: str, expected_type: type = dict):
    """
    Robustly extracts a JSON block (object or array) from a text string.
    Attempts to find all JSON blocks and returns the last valid one (with repair if truncated).
    """
    if not text:
        return None

    open_char  = "{" if expected_type == dict else "["
    close_char = "}" if expected_type == dict else "]"

    # Preprocessing: remove some common prefixes that LLMs sometimes add
    text = re.sub(r"^(?:JSON|Result|Here is the JSON|Output)\s*:\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Clean a candidate string and try to load it directly
    def clean_and_load(s: str):
        s_clean = s.strip()
        s_clean = re.sub(r",\s*([\]}])", r"\1", s_clean)
        return json.loads(s_clean)

    # Try to repair a truncated string by looking for braces/brackets from right to left
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

    # 1. Extraction via markdown blocks ```json ... ``` (closed or not)
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

    # 2. If no markdown block is valid, search the raw text
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

def get_llm(model_name, temperature, **options):
    # Ensure model_name is a string even if config values are None
    model_name = str(model_name) if model_name is not None else "default-model"
    if config.LLM_PROVIDER == "ollama":
        ollama_kwargs = {
            "model": model_name,
            "base_url": config.LLM_BASE_URL,
            "temperature": temperature,
            "num_ctx": options.get("num_ctx", 16384),
            "num_predict": options.get("num_predict", 2048),
        }
        reasoning = options.get("reasoning", None)
        if reasoning is not None:
            ollama_kwargs["reasoning"] = reasoning
        keep_alive = options.get("keep_alive", None)
        if keep_alive is not None:
            ollama_kwargs["keep_alive"] = keep_alive

        return ChatOllama(**ollama_kwargs)
    elif config.LLM_PROVIDER == "anthropic":
        return ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            anthropic_api_key=config.ANTHROPIC_API_KEY
        )
    elif config.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=config.GEMINI_API_KEY
        )
    else: # openai / llama-cpp
        return ChatOpenAI(
            model=model_name,
            base_url=config.LLM_BASE_URL,
            temperature=temperature,
            api_key=config.OPENAI_API_KEY if config.OPENAI_API_KEY else "sk-no-key-required"
        )

def get_embeddings():
    model_name = str(config.EMBEDDING_MODEL) if config.EMBEDDING_MODEL is not None else "default-embedding-model"
    if config.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(
            model=model_name,
            base_url=config.EMBEDDING_BASE_URL
        )
    elif config.EMBEDDING_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=config.GEMINI_API_KEY
        )
    else: # openai / llama-cpp
        return OpenAIEmbeddings(
            model=model_name,
            base_url=config.EMBEDDING_BASE_URL,
            api_key=config.OPENAI_API_KEY if config.OPENAI_API_KEY else "sk-no-key-required"
        )

def get_full_store_text(store, log) -> str:
    """Retrieves and concatenates all indexed text in `store`, sorted by page."""
    try:
        result = store.get(include=["documents", "metadatas"])
        paired = sorted(
            zip(result.get("documents", []), result.get("metadatas", []) or [{}] * len(result.get("documents", []))),
            key=lambda x: x[1].get("page", 0) if x[1] else 0
        )
        return "\n\n".join(doc for doc, _ in paired)
    except Exception as e:
        log(f"⚠ Error while retrieving full text: {e}")
        return ""


def get_relevant_context(store, queries, log, threshold_chars, k=15) -> str:
    """Full source text if its size permits, otherwise deduplicated similarity-based RAG."""
    full_text = get_full_store_text(store, log)
    if full_text and len(full_text) <= threshold_chars:
        log(f"Using full source text (size: {len(full_text)} <= {threshold_chars} chars)")
        return full_text
    log(f"Using similarity-based RAG queries (full_text size: {len(full_text)})")
    all_docs = []
    for q in queries:
        all_docs.extend(store.similarity_search(q, k=k))
    unique_contents = {d.page_content: d for d in all_docs}
    return "\n\n---\n\n".join(unique_contents.keys())

class BaseAgent:
    def __init__(self, model=None, temperature=0.7, verbose=False, llm_options: dict | None = None):
        model_name = model if model else config.LLM_MODEL
        options = llm_options if llm_options is not None else {}
        if "reasoning" not in options and config.LLM_REASONING is not None:
            options["reasoning"] = config.LLM_REASONING
        if "keep_alive" not in options and config.LLM_KEEP_ALIVE is not None:
            options["keep_alive"] = config.LLM_KEEP_ALIVE

        self.llm = get_llm(model_name, temperature, **options)
        self.verbose = verbose

    def _invoke_logged(self, prompt_template, inputs, label=""):
        messages = prompt_template.format_messages(**inputs)
        if self.verbose:
            rendered = "\n".join(f"[{m.type}] {m.content}" for m in messages)
            logging.debug(f"=== PROMPT [{self.__class__.__name__}{' - ' + label if label else ''}] ===\n{rendered}")
        response = self.llm.invoke(messages)
        if self.verbose:
            logging.debug(f"=== RAW RESPONSE [{self.__class__.__name__}{' - ' + label if label else ''}] ===\n{response.content}")
        return response
