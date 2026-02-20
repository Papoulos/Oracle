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
        # On fait une recherche large pour ne rien rater des règles de création
        queries = ["étapes création personnage", "caractéristiques obligatoires", "choix classe race", "équipement de départ"]
        context_text = ""
        if self.codex_db:
            for q in queries:
                docs = self.codex_db.similarity_search(q, k=3)
                context_text += "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert Technique du système de jeu.
        Ton rôle est d'analyser le CODEX pour extraire la liste ordonnée des étapes de création d'un personnage joueur.

        EXTRAITS DU CODEX :
        {context}

        INSTRUCTIONS :
        1. Liste les étapes indispensables (ex: "nom", "race", "classe", "attributs", "competences", "equipement").
        2. Les clés du JSON doivent être en minuscules, sans accents, et simples (un seul mot).
        3. Retourne un dictionnaire JSON où chaque clé est une étape avec la valeur false.
        4. "nom" doit TOUJOURS être la première étape.

        RÉPONDS UNIQUEMENT AVEC LE JSON (pas de texte avant ou après) :
        {{
            "nom": false,
            "etape_suivante": false
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            return chain.invoke({"context": context_text})
        except:
            return {"nom": False, "classe": False, "caracteristiques": False, "equipement": False}

    def interagir_creation(self, query, memory, journal=[]):
        # On cherche des infos spécifiques sur l'étape en cours
        pdp = memory.get("personnage", {}).get("points_de_passage", {})
        prochaine_etape = next((k for k, v in pdp.items() if not v), "règles générales")

        # On élargit la recherche pour avoir toutes les options d'un coup pour cette étape
        context_docs = self.codex_db.similarity_search(f"options création {prochaine_etape} liste choix", k=10) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es le Maître de Jeu (MJ) chargé de la création du personnage. Ton ton est immersif, solennel et bienveillant.
        Tu ne parles JAMAIS comme un robot technique ("aucune info extraite", "joueur a dit"). Tu ES le MJ.

        PLAN DE CRÉATION : {pdp_keys}

        HISTORIQUE DES ÉCHANGES :
        {journal}

        FICHE ACTUELLE :
        {char_sheet}

        RÈGLES ET OPTIONS DU CODEX POUR CETTE ÉTAPE ({prochaine_etape}) :
        {context}

        ACTION DU JOUEUR :
        {query}

        MISSION :
        1. ANALYSE : Si le joueur répond à la question précédente, valide son choix par rapport au CODEX.
        2. EXTRACTION : Toute info validée (nom, classe, etc.) doit être mise dans "personnage_updates" et le point de passage correspondant doit passer à true.
        3. PREMIER CONTACT : Si le journal est vide, commence par une introduction chaleureuse, présente le plan ({pdp_keys}) et pose la première question (le nom).
        4. OPTIONS EXPLICITES : Si tu demandes de choisir une Classe ou une Race, tu DOIS lister TOUTES les options trouvées dans le CODEX ci-dessus.
        5. MESSAGE : Ton message au joueur doit être 100% narratif.
           - Confirme l'étape validée.
           - Affiche la progression : [X] Étape validée / [ ] Étape à venir.
           - Pose la question suivante.
        6. UNE SEULE ÉTAPE : Ne demande pas deux choses à la fois.

        RÉPONDS UNIQUEMENT EN JSON :
        {{
            "reflexion": "Quelle étape est en cours ? Quelle info a été donnée ? Quelle est la suite ?",
            "message": "Ton message immersif au joueur (Narratif + Checklist + Question)",
            "personnage_updates": {{
                "nom": "valeur réelle",
                "classe": "valeur réelle",
                "stats": {{...}},
                "points_de_passage": {{ ... }}
            }},
            "creation_terminee": bool
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        res = chain.invoke({
            "pdp_keys": ", ".join(pdp.keys()),
            "prochaine_etape": prochaine_etape,
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
