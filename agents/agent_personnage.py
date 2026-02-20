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
        1. PREMIER MESSAGE : Si l'historique est vide, ton premier message DOIT lister les 4 étapes : 1. Nom, 2. Classe, 3. Caractéristiques, 4. Équipement.
        2. CHECKLIST : Utilise "points_de_passage". Ne passe à l'étape N+1 que si l'étape N est à True.
        3. EXTRACTION : Si le joueur donne une info (ex: "Je m'appelle Arthur"), tu DOIS l'extraire et passer le point de passage à True.
        4. PERSISTANCE : Dans "personnage_updates", n'inclus QUE les champs qui changent réellement ce tour. NE METS JAMAIS de valeurs fictives comme "..." ou "À définir". Si un champ ne change pas, ne l'inclus pas dans "personnage_updates".
        5. ÉTAPE 1 (Nom) : Le nom par défaut est "À définir". Si tu extrais un nom, mets-le dans "nom" et passe "points_de_passage": {"nom": true}.
        6. ÉTAPE 2 (Classe) : Liste les classes du CODEX. Si le joueur choisit, mets à jour "classe" et passe "points_de_passage": {"classe": true}.
        7. ÉTAPE 3 (Stats) : Propose de tirer les dés pour TOUTES les stats du CODEX. Une fois fait, enregistre dans "stats" et passe "points_de_passage": {"stats": true}.
        8. ÉTAPE 4 (Équipement) : Propose un pack selon la classe. Une fois validé, ajoute à "inventaire" et passe "points_de_passage": {"equipement": true}.
        9. UNE SEULE QUESTION : Ne demande qu'une seule chose à la fois. Confirme toujours l'info précédente avant de demander la suite.

        Réponds UNIQUEMENT en JSON avec cette structure:
        {{
            "reflexion": "Analyse de l'historique et de la réponse. État de la checklist. Décision pour le prochain message.",
            "message": "Ta réponse au joueur (Confirmation + Question unique)",
            "personnage_updates": {{
                "nom": "valeur réelle extraite",
                "classe": "valeur réelle extraite",
                "stats": {{...}},
                "inventaire": [...],
                "points_de_passage": {{ ... }}
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
