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
