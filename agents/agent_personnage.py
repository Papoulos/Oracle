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

    def generer_guide_creation(self):
        # Recherche exhaustive pour comprendre le système de création
        queries = [
            "étapes création personnage",
            "races disponibles et bonus",
            "classes métiers professions",
            "calcul caractéristiques attributs",
            "équipement de départ"
        ]
        context_text = ""
        if self.codex_db:
            for q in queries:
                docs = self.codex_db.similarity_search(q, k=3)
                context_text += f"\n\n--- EXTRAIT SUR '{q}' ---\n"
                context_text += "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es le Gardien du Savoir (Agent Personnage). Ton but est de prouver ta compréhension parfaite des règles de création de personnage du CODEX.

        DOCUMENTS DU CODEX :
        {context}

        MISSION :
        Génère un guide complet de création de personnage. Ce guide doit être structuré, clair et exhaustif selon le CODEX.

        CONTENU ATTENDU :
        1. LES ÉTAPES : Liste les étapes dans l'ordre (ex: 1. Nom, 2. Race...).
        2. LES OPTIONS : Pour chaque étape de choix (Race, Classe, etc.), liste TOUTES les options mentionnées dans le CODEX avec leurs spécificités.
        3. LES MÉCANIQUES : Explique comment sont définies les statistiques (tirage de dés, répartition de points...).
        4. L'ÉQUIPEMENT : Détaille ce qu'un personnage reçoit en commençant.

        Ton message doit montrer au joueur que tu as bien "lu" le CODEX et que tu es prêt à le guider.

        RÉPONDS UNIQUEMENT AVEC CE FORMAT JSON :
        {{
            "reflexion": "Résumé technique de ce que j'ai trouvé dans le CODEX.",
            "message": "Le guide complet, formaté pour être lu par le joueur (utilise le Markdown)."
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            return chain.invoke({"context": context_text})
        except Exception as e:
            return {
                "reflexion": f"Erreur lors de la génération : {e}",
                "message": "Je n'ai pas pu compiler le guide de création. Vérifiez que le CODEX est bien indexé."
            }

    def interagir_creation(self, query, memory, journal=[]):
        # Pour l'instant, on se contente de renvoyer le guide si on est en création
        guide = self.generer_guide_creation()
        return {
            "personnage_info": {},
            "message": guide["message"],
            "creation_terminee": False
        }

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
