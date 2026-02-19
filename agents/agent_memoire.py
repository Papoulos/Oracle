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
        Tu es l'Agent Mémoire. Ton rôle est d'extraire les changements structurels du jeu pour mettre à jour le fichier JSON de mémoire.

        DERNIER TOUR:
        Joueur: {query}
        Règles: {rules_info}
        Monde: {world_info}
        Narration MJ: {narration}

        INSTRUCTIONS:
        Extrais uniquement les changements concernant:
        - Le personnage (PV, inventaire, XP)
        - Le monde (nouveau lieu, nouveaux faits marquants)

        Réponds UNIQUEMENT avec un objet JSON structuré comme suit:
        {{
            "personnage_updates": {{ "stats": {{...}}, "inventaire_ajouts": [...], "xp_gain": 0 }},
            "monde_updates": {{ "nouveau_lieu": "...", "nouvel_evenement": "..." }}
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
