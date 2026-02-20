import random
import re
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import config

class AgentPersonnage:
    def __init__(self, codex_db=None, intrigue_db=None):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7
        )
        self.llm_json = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )
        self.codex_db = codex_db
        self.intrigue_db = intrigue_db

    def interagir_creation(self, query, memory, historique=[]):
        context_docs = self.codex_db.similarity_search("règles création personnage caractéristiques classes", k=5) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Agent Personnage, un Maître de Jeu expert en création de personnage.
        Ton rôle est d'accompagner le joueur pour définir son Nom, sa Classe, ses Caractéristiques et son Équipement.

        HISTORIQUE DES ÉCHANGES (Contexte):
        {historique}

        ÉTAT ACTUEL DE LA FICHE:
        {char_sheet}

        RÉPONSE ACTUELLE DU JOUEUR:
        {query}

        RÈGLES DU CODEX (Source de vérité):
        {context}

        INSTRUCTIONS CRITIQUES :
        1. ANALYSE : Lis la réponse du joueur. S'il a répondu à ta question précédente (même de façon brève comme "Arthur" ou "Guerrier"), tu DOIS extraire cette info. La réponse du joueur est PRIORITAIRE sur l'état de la fiche.
        2. PERSISTANCE : Toute information extraite doit être placée dans l'objet "personnage_updates".
        3. EXPLICIT_CONFIRMATION : Dans ton message, commence par confirmer ce que tu as enregistré (ex: "Très bien, ton nom est donc [Nom].").
        4. PROGRESSION : Pose ensuite la question SUIVANTE. Ne boucle pas sur une question déjà répondue.
           - Si Nom == "À définir" -> Demande le nom.
           - Si Classe absente -> LISTE EXPLICITEMENT les classes trouvées dans le CODEX et demande un choix.
           - Si Stats vides -> Explique le calcul (ex: 3d6) et propose de tirer les dés.
        5. UNE SEULE QUESTION : Ne demande jamais deux choses en même temps.
        6. JETS DE DÉS : Si le joueur te demande de tirer les dés, simule-le et donne les scores obtenus.
        7. FIN : Quand Nom, Classe, Stats et Équipement sont OK, mets "creation_terminee" à true.

        Réponds UNIQUEMENT en JSON avec cette structure:
        {{
            "reflexion": "Analyse de la situation : qu'est-ce qui est acquis, que manque-t-il, quelle est la prochaine étape ?",
            "message": "Ta réponse immersive (Confirmation des acquis + Prochaine question)",
            "personnage_updates": {{
                "nom": "valeur extraite ou inchangée",
                "classe": "valeur extraite ou inchangée",
                "stats": {{...}},
                "inventaire": [...]
            }},
            "creation_terminee": boolean
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        res = chain.invoke({
            "context": context_text,
            "char_sheet": json.dumps(memory.get("personnage", {})),
            "historique": json.dumps(historique),
            "query": query
        })
        return res

    def calculer_xp(self, action_query, narration, regles_info, world_info):
        context_query = f"récompense XP action {action_query}"
        codex_docs = self.codex_db.similarity_search(context_query, k=2) if self.codex_db else []
        intrigue_docs = self.intrigue_db.similarity_search(context_query, k=2) if self.intrigue_db else []

        context_text = "CODEX:\n" + "\n".join([d.page_content for d in codex_docs])
        context_text += "\n\nINTRIGUE:\n" + "\n".join([d.page_content for d in intrigue_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Agent Personnage. Détermine le gain d'XP.

        CONTEXTE:
        {context}

        ACTION: {query}
        RÈGLES: {regles}
        NARRATION: {narration}

        Réponds en JSON:
        {{
            "xp_gagne": int,
            "raison": "..."
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({
            "context": context_text,
            "query": action_query,
            "regles": regles_info,
            "narration": narration
        })

    def verifier_niveau(self, personnage_memory):
        xp = personnage_memory.get("xp", 0)
        niveau = personnage_memory.get("niveau", 1)

        context_docs = self.codex_db.similarity_search("table progression niveau XP", k=3) if self.codex_db else []
        context_text = "\n".join([d.page_content for d in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Vérifie le passage de niveau.

        RÈGLES:
        {context}

        ACTUEL: Niveau {niveau}, XP {xp}

        Réponds en JSON:
        {{
            "passage_niveau": boolean,
            "nouveau_niveau": int
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({
            "context": context_text,
            "niveau": niveau,
            "xp": xp
        })

    def gerer_evolution(self, query, memory, historique=[]):
        context_docs = self.codex_db.similarity_search("bonus montée de niveau caractéristiques compétences", k=5) if self.codex_db else []
        context_text = "\n".join([d.page_content for d in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Agent Personnage, chargé d'accompagner le joueur dans sa montée de niveau.

        HISTORIQUE (Contexte):
        {historique}

        FICHE ACTUELLE:
        {char_sheet}

        RÉPONSE DU JOUEUR:
        {query}

        RÈGLES DE MONTÉE DE NIVEAU (CODEX):
        {context}

        INSTRUCTIONS:
        1. Analyse les choix du joueur par rapport aux règles du CODEX.
        2. Confirme explicitement les changements que tu enregistres.
        3. Si des choix sont encore à faire, liste les options possibles clairement.
        4. Mets à jour "personnage_updates" avec les nouveaux bonus (stats, nouvelles compétences, etc.).
        5. Quand tous les bonus du niveau sont choisis, mets "evolution_terminee" à true.

        Réponds en JSON:
        {{
            "message": "Ta réponse (Confirmation + Questions éventuelles)",
            "personnage_updates": {{...}},
            "evolution_terminee": boolean
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({
            "context": context_text,
            "char_sheet": json.dumps(memory.get("personnage", {})),
            "historique": json.dumps(historique),
            "query": query
        })
