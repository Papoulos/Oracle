from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

class AgentMonde:
    def __init__(self, intrigue_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )
        self.intrigue_db = intrigue_db

    def consult(self, query, memory):
        context_docs = self.intrigue_db.similarity_search(query, k=5) if self.intrigue_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert du Monde et du Scénario (Agent Monde).
        Ton rôle est de vérifier la cohérence de l'action du joueur avec l'intrigue, le lieu et l'état actuel du monde.
        Tu es le premier à intervenir pour valider si l'action est réaliste et possible dans le contexte narratif.

        MÉMOIRE ACTUELLE (Lieu, Histoire):
        {memory}

        CONTEXTE DE L'INTRIGUE (Secrets, Scénario):
        {context}

        ACTION DU JOUEUR:
        {query}

        RÈGLES CRITIQUES:
        - Si l'action est cohérente et ne nécessite aucune intervention particulière du scénario, réponds UNIQUEMENT "RAS".
        - Si l'action pose un problème (incohérence, impossibilité), explique-le brièvement.
        - NE RÉVÈLE PAS de secrets, de noms de PNJ cachés ou d'événements futurs si le joueur ne les a pas encore découverts.
        - Ne donne des informations que si elles sont publiques ou si l'action du joueur permet de les découvrir.
        - Réponds avec des faits bruts, pas de narration.
        - Indique si l'action entraîne une conséquence scénaristique majeure.
        - NE FAIS JAMAIS référence à des actions passées du joueur qui ne sont pas explicitement présentes dans la MÉMOIRE ACTUELLE. Si la mémoire est vide, considère que c'est le tout début.

        RÉPONSE:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "memory": memory,
            "context": context_text,
            "query": query
        })

        return response
