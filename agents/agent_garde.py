from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import config

class AgentGarde:
    def __init__(self, codex_db, intrigue_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )
        self.codex_db = codex_db
        self.intrigue_db = intrigue_db

    def valider_action(self, query, memory):
        # Recherche dans le Codex (règles/physique)
        codex_docs = self.codex_db.similarity_search(query, k=3) if self.codex_db else []
        codex_context = "\n\n".join([doc.page_content for doc in codex_docs])

        # Recherche dans l'Intrigue (scénario/cohérence narrative)
        intrigue_docs = self.intrigue_db.similarity_search(query, k=3) if self.intrigue_db else []
        intrigue_context = "\n\n".join([doc.page_content for doc in intrigue_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es le Garde du Jeu (Agent Garde).
        Ton rôle est de vérifier si l'action du joueur est possible, réaliste et cohérente avec l'univers et l'intrigue.

        MÉMOIRE ACTUELLE:
        {memory}

        CONTEXTE DU CODEX (Règles, Univers):
        {codex_context}

        CONTEXTE DE L'INTRIGUE (Scénario, Lieu):
        {intrigue_context}

        ACTION DU JOUEUR:
        {query}

        INSTRUCTIONS:
        1. Évalue si l'action est physiquement possible dans cet univers (Codex).
        2. Évalue si l'action est cohérente avec la situation actuelle et l'intrigue (Intrigue).
        3. **ANTI-HALLUCINATION** : Ne présume jamais d'un lieu par défaut (comme une auberge). Si l'action mentionne un lieu non présent dans l'Intrigue ou la Mémoire, l'action est IMPOSSIBLE.
        4. Si l'action est possible, "possible" doit être true et "raison" doit être EXACTEMENT "OUI".
        5. Si l'action est impossible, "possible" doit être false et "raison" doit obligatoirement commencer par "NON, parce que...".
        6. NE DONNE AUCUN CONSEIL, aucune suggestion et aucune alternative. Sois un juge binaire, froid et strict.

        Réponds UNIQUEMENT avec un objet JSON:
        {{
            "possible": boolean,
            "raison": "OUI" ou "NON, parce que [explication courte]"
        }}

        JSON:
        """)

        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({
            "memory": memory,
            "codex_context": codex_context,
            "intrigue_context": intrigue_context,
            "query": query
        })

        return response
