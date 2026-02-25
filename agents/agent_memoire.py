from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.models import MemoryUpdates
from llm_utils import safe_chain_invoke, handle_llm_error
import config
import json

class AgentMemoire:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            timeout=20
        )

    def extract_updates(self, query, rules_info, world_info, narration):
        prompt = ChatPromptTemplate.from_template("""
        You are the Memory Agent. Your role is to analyze the turn that just happened to extract ONLY what actually took place.

        INFORMATION SOURCES:
        - GM Narration (MAIN SOURCE): This is what the player saw and experienced. If a fact is not here, it did NOT happen.
        - Rules & World: Used only to confirm technical stats (HP, XP) or validated geographical details.

        LAST TURN:
        Player: {query}
        Rules: {rules_info}
        World: {world_info}
        GM Narration: {narration}

        CRITICAL INSTRUCTIONS:
        - DO NOT invent actions or encounters. If the player didn't talk to a character in the Narration, they did NOT contact them.
        - DO NOT extract secrets or info from the World Agent that were not explicitly revealed in the Narration.
        - The field "nouveau_lieu" must be a LOCATION (e.g., "The marketplace"), not a person's name.
        - The summary "resume_action" must be strictly factual in French based on the Narration.

        Respond ONLY with a JSON object:
        {{
            "personnage_updates": {{
                "stats": {{...}},
                "inventaire_ajouts": [...]
            }},
            "monde_updates": {{
                "nouveau_lieu": "string or null",
                "nouvel_evenement": "string or null"
            }},
            "resume_action": "Concise summary in French (Who did what / Result)"
        }}

        JSON:
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            updates_dict = safe_chain_invoke(chain, {
                "query": query,
                "rules_info": rules_info,
                "world_info": world_info,
                "narration": narration
            })
            validated = MemoryUpdates(**updates_dict)
            return validated.model_dump()
        except Exception as e:
            print(f"Error memory extraction: {e}")
            return {}
