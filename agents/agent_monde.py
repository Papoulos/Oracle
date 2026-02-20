from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config
import json

class AgentMonde:
    def __init__(self, intrigue_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )
        self.intrigue_db = intrigue_db

    def consult(self, query, memory_dict):
        # On essaie de chercher par rapport à l'action ET au lieu actuel pour rester dans le thème
        lieu_actuel = memory_dict.get("monde", {}).get("lieu_actuel", "")
        search_query = f"{lieu_actuel} {query}" if lieu_actuel else query

        context_docs = self.intrigue_db.similarity_search(search_query, k=5) if self.intrigue_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert du Monde et du Scénario (Agent Monde).
        Ton rôle est d'extraire de l'INTRIGUE les éléments nécessaires pour que le MJ Narrateur puisse décrire la scène.

        MÉMOIRE ACTUELLE:
        {memory}

        CONTEXTE DE L'INTRIGUE (Scénario, PNJ, Lieux):
        {context}

        ACTION DU JOUEUR:
        {query}

        TES MISSIONS :
        1. Identifie le lieu, les PNJ présents et l'ambiance décrits dans l'INTRIGUE correspondant à l'action ou au lieu actuel.
        2. Si l'action du joueur déclenche un événement du scénario, décris-le.
        3. Donne des détails concrets (descriptions, dialogues possibles, secrets révélables) pour aider le Narrateur.
        4. Si la mémoire est vide, cherche les éléments de l'introduction ou du début de l'aventure.

        RÈGLES CRITIQUES:
        - Ne dis JAMAIS "RAS". Donne toujours du contexte narratif issu de l'INTRIGUE.
        - NE RÉVÈLE PAS de secrets futurs, seulement ce qui est pertinent ICI et MAINTENANT.
        - Réponds avec des faits bruts et des descriptions techniques du scénario, pas de narration (le Narrateur s'en chargera).
        - **IMPORTANT** : Si aucune information pertinente n'est trouvée dans l'INTRIGUE (contexte vide), réponds explicitement : "ERREUR : Aucune information trouvée dans le scénario pour ce contexte. Veuillez vérifier l'indexation de l'INTRIGUE." N'invente JAMAIS de lieux ou de PNJ qui ne sont pas dans le contexte.

        RÉPONSE (Facts & Scenario details):
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "memory": json.dumps(memory_dict),
            "context": context_text,
            "query": query
        })

        return response

    def chercher_introduction(self):
        # On cherche les éléments de début avec plusieurs angles et un k plus grand
        queries = [
            "introduction début aventure scène initiale point de départ",
            "chapitre 1 prologue",
            "personnage joueur commence situation initiale",
            " " # Recherche large
        ]

        all_docs = []
        if self.intrigue_db:
            for q in queries:
                all_docs.extend(self.intrigue_db.similarity_search(q, k=5))

        # Déduplication simple par contenu
        seen = set()
        context_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                context_docs.append(doc)
                seen.add(doc.page_content)

        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        if not context_text:
            return "ERREUR CRITIQUE : Aucune donnée disponible dans la base INTRIGUE. Assurez-vous d'avoir indexé vos fichiers PDF."

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert du Monde et du Scénario (Agent Monde).
        Ton rôle est d'extraire de l'INTRIGUE les éléments pour COMMENCER l'aventure.

        CONTEXTE DE L'INTRIGUE :
        {context}

        TES MISSIONS :
        1. Décris la situation initiale en te basant sur les informations fournies.
        2. Identifie le lieu, l'ambiance et les personnages présents.
        3. Donne tous les détails factuels nécessaires pour que le MJ Narrateur puisse lancer la partie.

        RÈGLES CRITIQUES:
        - Utilise le contexte fourni comme la source absolue du début de l'aventure.
        - Ne dis jamais "RAS" ou "ERREUR". Fais le meilleur usage possible des informations extraites.
        - N'INVENTE RIEN qui ne soit pas suggéré par le contexte (pas d'Auberge du Dragon Vert par défaut).
        - Reste purement factuel.

        RÉPONSE (Facts & Intro details):
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context_text
        })

        return response
