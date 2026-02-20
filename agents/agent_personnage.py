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

    def definir_etapes_creation(self):
        context_docs = self.codex_db.similarity_search("étapes création personnage règles obligatoires", k=5) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert Technique du système de jeu.
        En te basant UNIQUEMENT sur le CODEX, liste les étapes obligatoires pour créer un personnage.

        RÈGLES DU CODEX :
        {context}

        INSTRUCTIONS :
        1. Identifie les étapes (ex: Nom, Race, Classe, Caractéristiques, Compétences, Sorts, Équipement...).
        2. Retourne un dictionnaire JSON où chaque étape est une clé avec la valeur false.
        3. Ajoute TOUJOURS "nom" en première étape si non mentionné.

        RÉPONDS UNIQUEMENT EN JSON :
        {{
            "nom": false,
            "etape_2": false,
            ...
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({"context": context_text})

    def interagir_creation(self, query, memory, journal=[]):
        # On cherche des infos spécifiques sur l'étape en cours
        pdp = memory.get("personnage", {}).get("points_de_passage", {})
        prochaine_etape = next((k for k, v in pdp.items() if not v), "règles générales")

        context_docs = self.codex_db.similarity_search(f"règles création personnage {prochaine_etape} options disponibles", k=8) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Agent Personnage, Maître de Jeu expert.
        Ton but est de guider le joueur à travers ces étapes : {pdp_keys}.

        JOURNAL DE CRÉATION :
        {journal}

        ÉTAT DE LA FICHE :
        {char_sheet}

        RÉPONSE DU JOUEUR :
        {query}

        EXTRAITS DU CODEX SUR L'ÉTAPE ACTUELLE :
        {context}

        INSTRUCTIONS IMPÉRATIVES :
        1. ANALYSE DU JOURNAL : Ne redemande JAMAIS ce qui est déjà acquis.
        2. EXTRACTION : Si le joueur donne une info (ex: une classe parmi celles du CODEX), valide-la et passe son point de passage à True.
        3. ÉTAPE PAR ÉTAPE : Ne traite qu'UNE SEULE étape à la fois, dans l'ordre de la checklist.
        4. OPTIONS DU CODEX : Pour des étapes comme la Classe, la Race ou l'Équipement, tu DOIS lister CLAIREMENT les options trouvées dans le CODEX. Ne les invente pas.
        5. MESSAGE AU JOUEUR :
           - Confirme l'acquis précédent.
           - Affiche la checklist avec l'état actuel (ex: "[X] Nom, [ ] Classe...").
           - Pose la question pour la prochaine étape NON VALIDÉE.
        6. FIN : Quand tout est à True, mets "creation_terminee" à true.

        Réponds UNIQUEMENT en JSON :
        {{
            "reflexion": "Étape actuelle, options trouvées dans le CODEX, analyse de la réponse du joueur.",
            "message": "Ta réponse (Confirmation + Checklist + Prochaine Question avec options)",
            "personnage_updates": {{
                "nom": "valeur",
                "classe": "valeur",
                "stats": {{...}},
                "inventaire": [...],
                "points_de_passage": {{ ... }}
            }},
            "creation_terminee": bool
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        res = chain.invoke({
            "pdp_keys": ", ".join(pdp.keys()),
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
