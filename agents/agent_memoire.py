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
        Tu es l'Agent Mémoire. Ton rôle est d'analyser le tour qui vient de se dérouler pour en extraire UNIQUEMENT ce qui s'est réellement passé.

        SOURCES D'INFORMATION:
        - Narration MJ (SOURCE PRINCIPALE): C'est ce que le joueur a vu et vécu. Si un fait n'est pas ici, il ne s'est PAS passé.
        - Règles & Monde: Utilisés uniquement pour confirmer des stats techniques (PV, XP) ou des détails géographiques validés.

        DERNIER TOUR:
        Joueur: {query}
        Règles: {rules_info}
        Monde: {world_info}
        Narration MJ: {narration}

        CONSIGNES CRITIQUES:
        - NE PAS inventer d'actions ou de rencontres. Si le joueur n'a pas parlé à un personnage dans la Narration, il ne l'a PAS contacté.
        - NE PAS extraire de secrets ou d'infos de l'Agent Monde qui n'ont pas été explicitement révélés dans la Narration.
        - Le champ "nouveau_lieu" doit être un LIEU (ex: "La place du marché"), pas un nom de personne.
        - Le résumé "resume_action" doit être strictement factuel basé sur la Narration.

        Réponds UNIQUEMENT avec un objet JSON:
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
            "resume_action": "Résumé concis (Qui a fait quoi / Résultat)"
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
