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

    def interagir_creation(self, query, memory, journal=[]):
        context_docs = self.codex_db.similarity_search("règles création personnage caractéristiques classes", k=5) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Agent Personnage, un Maître de Jeu expert en création de personnage.
        Ton but est de guider le joueur à travers 4 étapes : 1. Nom, 2. Classe, 3. Caractéristiques, 4. Équipement.

        JOURNAL COMPLET DE LA CRÉATION :
        {journal}

        ÉTAT ACTUEL DE LA FICHE :
        {char_sheet}

        RÉPONSE DU JOUEUR :
        {query}

        RÈGLES DU CODEX :
        {context}

        INSTRUCTIONS IMPÉRATIVES :
        1. ANALYSE DU JOURNAL : Regarde ce qui a été demandé et répondu. Ne redemande JAMAIS une information déjà donnée.
        2. EXTRACTION PRIORITAIRE : Si le joueur donne une info (même en dehors de ta question), tu DOIS l'extraire, mettre à jour la fiche et passer le point de passage à True.
        3. MISE À JOUR : Dans "personnage_updates", n'inclus QUE les champs qui changent ce tour. Ne remets pas "À définir".
        4. PROGRESSION : Suis strictement l'ordre : Nom -> Classe -> Stats -> Équipement.
           - Si points_de_passage["nom"] est False : demande le nom.
           - Si points_de_passage["classe"] est False : liste les classes du CODEX et demande un choix.
           - Si points_de_passage["stats"] est False : liste les stats du CODEX, explique le jet (ex: 3d6) et propose de le faire.
           - Si points_de_passage["equipement"] est False : propose un pack selon la classe.
        5. MESSAGE AU JOUEUR :
           - Commence par confirmer ce qui a été validé (ex: "Très bien, tu t'appelles Arthur.").
           - Affiche la checklist de progression (ex: "[X] Nom, [ ] Classe...").
           - Pose la question suivante de manière immersive.
        6. FIN : Quand tout est à True, mets "creation_terminee" à true.

        Réponds UNIQUEMENT en JSON :
        {{
            "reflexion": "Résumé des acquis, analyse de la réponse, identification du prochain point de passage.",
            "message": "Confirmation + Checklist + Prochaine Question",
            "personnage_updates": {{
                "nom": "valeur réelle",
                "classe": "valeur réelle",
                "stats": {{...}},
                "inventaire": [...],
                "points_de_passage": {{ "nom": bool, "classe": bool, "stats": bool, "equipement": bool }}
            }},
            "creation_terminee": bool
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        res = chain.invoke({
            "context": context_text,
            "char_sheet": json.dumps(memory.get("personnage", {})),
            "journal": json.dumps(journal),
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
