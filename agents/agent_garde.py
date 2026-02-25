from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.models import GardeResponse
from llm_utils import safe_chain_invoke, handle_llm_error
import config
import json
import logging

logger = logging.getLogger(__name__)

class AgentGarde:
    def __init__(self, codex_db, intrigue_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            timeout=20
        )
        self.codex_db = codex_db
        self.intrigue_db = intrigue_db

    def valider_action(self, query, memory):
        # CONTEXTE : Solo Game
        codex_docs = self.codex_db.similarity_search(query, k=3) if self.codex_db else []
        codex_context = "\n\n".join([doc.page_content for doc in codex_docs])

        intrigue_docs = self.intrigue_db.similarity_search(query, k=3) if self.intrigue_db else []
        intrigue_context = "\n\n".join([doc.page_content for doc in intrigue_docs])

        prompt = ChatPromptTemplate.from_template("""
        You are the Game Guard (Agent Garde).
        Your role is to verify if the player's action is possible, realistic, and consistent with the universe and the plot.
        **IMPORTANT: This is a SOLO game. There is only one player character.**

        CURRENT MEMORY:
        {memory}

        CODEX CONTEXT (Rules, Universe):
        {codex_context}

        INTRIGUE CONTEXT (Scenario, Location):
        {intrigue_context}

        PLAYER ACTION:
        {query}

        INSTRUCTIONS:
        1. Evaluate if the action is physically possible in this universe (Codex).
        2. Evaluate if the action is consistent with the current situation and the plot (Intrigue).
        3. **ANTI-HALLUCINATION**: Never assume a default location (like an inn). If the action mentions a place not present in Intrigue or Memory, the action is IMPOSSIBLE.
        4. If the action is possible, "possible" must be true and "raison" must be EXACTLY "OUI".
        5. If the action is impossible, "possible" must be false and "raison" must mandatory start with "NON, parce que..." (in French).
        6. PROVIDE NO ADVICE, no suggestions, and no alternatives. Be a binary, cold, and strict judge.
        7. Output MUST be valid JSON.

        Output JSON structure:
        {{
            "possible": boolean,
            "raison": "OUI" or "NON, parce que [short explanation in French]"
        }}
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            response_dict = safe_chain_invoke(chain, {
                "memory": memory,
                "codex_context": codex_context,
                "intrigue_context": intrigue_context,
                "query": query
            })
            # Validation with Pydantic
            validated = GardeResponse(**response_dict)
            return validated.model_dump()
        except Exception as e:
            logger.error(f"AgentGarde error: {e}")
            return {
                "possible": False,
                "raison": f"NON, parce que l'Oracle rencontre une erreur technique : {handle_llm_error(e)}"
            }
