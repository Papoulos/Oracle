from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

class AgentNarrateur:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.8
        )

    def narrate(self, query, rules_info, world_info, garde_info, memory):
        prompt = ChatPromptTemplate.from_template("""
        Tu es le MJ Narrateur. Ton rôle est de décrire la résolution de l'action du joueur et de jouer les PNJ.
        Tu transformes les informations techniques des autres agents en une narration immersive.

        ACTION DU JOUEUR:
        {query}

        VALIDATION DU GARDE (Agent Garde):
        {garde_info}

        INFOS DES RÈGLES (Agent Règles):
        {rules_info}

        INFOS DU MONDE/SCÉNARIO (Agent Monde):
        {world_info}

        MÉMOIRE DU JEU:
        {memory}

        CONSIGNES:
        - Concentre-toi sur la résolution de l'action actuelle.
        - Si l'Agent Garde indique que l'action est IMPOSSIBLE, explique-le de manière narrative et immersive (sans dire "L'Agent Garde a dit que"). Raconte pourquoi l'action échoue ou est bloquée.
        - Si l'action est possible, utilise les INFOS DES RÈGLES et du MONDE pour construire ton récit.
        - Ne redécris pas inutilement le lieu si le joueur s'y trouve déjà, sauf changement.
        - Incorpore les résultats des tests de règles (succès/échec) de manière fluide.
        - Fais parler les PNJ si nécessaire.
        - Pose un choix ou demande une action au joueur à la fin.
        - Reste fidèle à l'ambiance et à l'intrigue.

        NARRE LA RÉPONSE AU JOUEUR:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "rules_info": rules_info,
            "world_info": world_info,
            "garde_info": garde_info,
            "memory": memory
        })

        return response
