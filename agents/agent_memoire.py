from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import config
import json

class AgentMemoire:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )

    def extract_updates(self, query, rules_info, world_info, narration):
        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Agent Mémoire. Ton rôle est d'analyser le tour qui vient de se dérouler pour en extraire les informations essentielles à conserver.
        Tu dois résumer ce qu'il s'est passé en définissant précisément "qui a fait quoi" et quel a été le résultat.

        DERNIER TOUR:
        Joueur: {query}
        Règles: {rules_info}
        Monde: {world_info}
        Narration MJ: {narration}

        INSTRUCTIONS:
        1. Extrais les changements structurels (stats, inventaire, lieu).
        2. Rédige un résumé factuel et concis de l'action (ex: "Le joueur a tenté de crocheter la porte de la cave, réussite, il est entré discrètement").
        3. Réponds UNIQUEMENT avec un objet JSON structuré comme suit:
        {{
            "personnage_updates": {{
                "stats": {{...}},
                "inventaire_ajouts": [...],
                "xp_gain": 0
            }},
            "monde_updates": {{
                "nouveau_lieu": "...",
                "nouvel_evenement": "..."
            }},
            "resume_action": "Résumé concis de qui a fait quoi et du résultat"
        }}

        JSON:
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            updates = chain.invoke({
                "query": query,
                "rules_info": rules_info,
                "world_info": world_info,
                "narration": narration
            })
            return updates
        except Exception as e:
            print(f"Erreur extraction mémoire: {e}")
            return {}
