import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.exceptions import LangChainException
import config

logger = logging.getLogger(__name__)

# Error message when LLM fails
ERROR_MESSAGE = "L'Oracle est momentanément indisponible. Veuillez réessayer dans quelques instants."

def safe_chain_invoke(chain, input_data, timeout=20):
    """
    Safely invoke a LangChain chain with timeout, retry, and error handling.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception, LangChainException)),
        reraise=True
    )
    def _invoke():
        # Using timeout in config if supported by the provider
        return chain.invoke(input_data, config={
            "recursion_limit": 50,
            "run_name": "AgentInvoke",
            "timeout": timeout
        })

    try:
        # Note: ChatOllama request_timeout is usually set at initialization.
        # We will ensure it's set in the Agent constructors as well.
        return _invoke()
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        # Return a structure that won't break the application
        # If the output parser is StrOutputParser, return the string error.
        # If it's JsonOutputParser, we might need a dummy JSON or raise to be handled by the caller.
        raise e

def handle_llm_error(e):
    """
    Returns a technical short error message for the user.
    """
    logger.error(f"Technical error: {e}")
    return ERROR_MESSAGE
