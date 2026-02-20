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

    def interagir_creation(self, query, memory):
        context_docs = self.codex_db.similarity_search("règles création personnage caractéristiques classes", k=5) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Agent Personnage. Ton but est d'accompagner le joueur dans la création de son personnage.
        Tu dois poser des questions une par une pour définir : Nom, Classe/Métier, Caractéristiques, et Équipement de départ.
        Utilise le CODEX pour respecter les règles du jeu.

        RÈGLES DU CODEX:
        {context}

        ÉTAT ACTUEL DU PERSONNAGE:
        {char_sheet}

        DERNIER ÉCHANGE / RÉPONSE DU JOUEUR:
        {query}

        INSTRUCTIONS:
        1. Analyse la fiche actuelle pour identifier ce qui manque (Nom, Classe, Stats, Équipement).
        2. Si le nom est "Nouveau Personnage" ou vide, demande au joueur quel nom il souhaite.
        3. Si le nom est défini mais que la classe manque, demande sa classe.
        4. Si la classe est définie mais que les statistiques sont vides, procède à leur définition (ou tirage de dés).
        5. Si le joueur répond à ta dernière question, valide sa réponse selon le CODEX et mets à jour la fiche dans "personnage_updates".
        6. Pose TOUJOURS une seule question à la fois.
        7. Sois immersif et encourageant.
        8. Quand TOUS les éléments (Nom, Classe, Stats, Inventaire de départ) sont définis et validés, mets "creation_terminee" à true.

        Réponds UNIQUEMENT avec ce format JSON:
        {{
            "message": "Ton message au joueur",
            "personnage_updates": {{
                "nom": "...",
                "stats": {{...}},
                "inventaire": [...],
                "classe": "..."
            }},
            "creation_terminee": false
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        res = chain.invoke({
            "context": context_text,
            "char_sheet": json.dumps(memory.get("personnage", {})),
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

    def gerer_evolution(self, query, memory):
        context_docs = self.codex_db.similarity_search("bonus montée de niveau caractéristiques compétences", k=5) if self.codex_db else []
        context_text = "\n".join([d.page_content for d in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Accompagne le joueur pour sa montée de niveau.

        RÈGLES:
        {context}

        FICHE:
        {char_sheet}

        RÉPONSE JOUEUR:
        {query}

        Réponds en JSON:
        {{
            "message": "...",
            "personnage_updates": {{...}},
            "evolution_terminee": boolean
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({
            "context": context_text,
            "char_sheet": json.dumps(memory.get("personnage", {})),
            "query": query
        })
