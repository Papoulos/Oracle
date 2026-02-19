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
        Ton rôle est de vérifier la cohérence de l'action avec l'intrigue et l'état du monde.

        MÉMOIRE ACTUELLE (Lieu, Histoire):
        {memory}

        CONTEXTE DE L'INTRIGUE (Secrets, Scénario):
        {context}

        ACTION DU JOUEUR:
        {query}

        RÈGLES CRITIQUES:
        - Protège les informations cachées (ne les révèle pas si le joueur ne peut pas les savoir).
        - Ne révèle que ce que le personnage peut raisonnablement savoir ou découvrir.
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
