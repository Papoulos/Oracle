import random
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from agents.models import RulesAnalyse
from llm_utils import safe_chain_invoke, handle_llm_error
import config

def simulate_dice_roll(dice_expr):
    # Simple regex for NdM (+/-) K
    match = re.match(r"(\d+)d(\d+)([+-]\d+)?", dice_expr.replace(" ", ""))
    if match:
        n = int(match.group(1))
        m = int(match.group(2))
        mod = int(match.group(3)) if match.group(3) else 0
        rolls = [random.randint(1, m) for _ in range(n)]
        total = sum(rolls) + mod
        return {
            "expression": dice_expr,
            "details": f"({'+'.join(map(str, rolls))}){mod:+}",
            "total": total,
            "texte": f"Jet: {n}d{m}{mod:+} = ({'+'.join(map(str, rolls))}){mod:+} = {total}"
        }
    return None

class AgentRegles:
    def __init__(self, codex_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            timeout=20
        )
        self.llm_json = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            timeout=20
        )
        self.codex_db = codex_db

    def evaluer_besoin_jet(self, query, char_sheet, world_info):
        # CONTEXTE : Solo Game
        context_docs = self.codex_db.similarity_search(query, k=3) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        You are the Rules Expert (Agent Règles).
        Your role is to determine if the player's action requires a dice roll according to the CODEX.
        **IMPORTANT: This is a SOLO game. There is only one player character.**

        CHARACTER SHEET:
        {char_sheet}

        WORLD INFO (Agent Monde or Garde context):
        {world_info}

        CODEX CONTEXT:
        {context}

        PLAYER ACTION:
        {query}

        INSTRUCTIONS:
        1. Analyze if the action requires a skill test or resolution by dice.
        2. **IMPORTANT: Absolute Neutrality**: NEVER assume hidden intention, method, or style (e.g., stealth, speed, violence, caution) if it's not EXPLICITLY written by the player.
           - Example: "I approach the house" is a simple action. DO NOT impose a stealth roll.
           - Example: "I approach stealthily" requires a stealth roll.
        3. If the action is simple, routine, and without direct opposition, no roll is necessary.
        4. If the Game Guard (garde_context) has indicated an impossibility, no roll is necessary.
        5. Respond in JSON format with the following fields:
           - "besoin_jet": boolean
           - "jet_format": "NdM+K" (e.g., "1d20+2") or null if no roll
           - "explication_regle": short explanation of the applied rule in French
           - "seuil": the score to reach or difficulty (if applicable)

        Respond ONLY in JSON.
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            response_dict = safe_chain_invoke(chain, {
                "char_sheet": char_sheet,
                "world_info": world_info,
                "context": context_text,
                "query": query
            })
            validated = RulesAnalyse(**response_dict)
            return validated.model_dump(), context_text
        except Exception as e:
            return {
                "besoin_jet": False,
                "jet_format": None,
                "explication_regle": f"Erreur technique de l'Oracle : {handle_llm_error(e)}",
                "seuil": None
            }, context_text

    def interpreter_reussite(self, query, roll_result, explication_regle, context_text):
        prompt = ChatPromptTemplate.from_template("""
        You are the Rules Expert (Agent Règles).
        You must determine if the action is a success based on the dice result and the rules.

        ACTION: {query}
        APPLIED RULE: {explication_regle}
        CODEX CONTEXT: {context}
        ROLL RESULT: {roll_result}

        INSTRUCTIONS:
        1. Compare the roll result to the CODEX rules.
        2. Declare if it's a SUCCESS (RÉUSSITE) or FAILURE (ÉCHEC) in French.
        3. Briefly explain the technical consequences in French.
        4. Be concise and technical. No narration.

        RESPONSE (in French):
        """)

        chain = prompt | self.llm | StrOutputParser()
        try:
            return safe_chain_invoke(chain, {
                "query": query,
                "explication_regle": explication_regle,
                "context": context_text,
                "roll_result": roll_result
            })
        except Exception as e:
            return f"L'Oracle n'a pas pu interpréter le résultat : {handle_llm_error(e)}"
