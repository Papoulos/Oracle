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

    def narrate(self, query, rules_info, world_info, memory):
        prompt = ChatPromptTemplate.from_template("""
        Tu es le MJ Narrateur. Ton rôle est de décrire la résolution de l'action du joueur et de jouer les PNJ.
        Tu transformes les informations techniques des autres agents en une narration immersive.

        ACTION DU JOUEUR:
        {query}

        INFOS DES RÈGLES (Agent Règles):
        {rules_info}

        INFOS DU MONDE (Agent Monde):
        {world_info}

        MÉMOIRE DU JEU:
        {memory}

        CONSIGNES:
        - Concentre-toi sur la résolution de l'action actuelle.
        - **IMPORTANT** : Ne redécris pas inutilement le lieu (ambiance, décor) si le joueur s'y trouve déjà depuis plusieurs tours. Ne le fais que s'il y a un changement notable ou un nouveau détail important.
        - Incorpore les résultats des tests de règles (succès/échec) de manière fluide dans le récit.
        - Fais parler les PNJ si nécessaire.
        - Pose un choix ou demande une action au joueur à la fin.
        - Ne décide PAS des règles, utilise uniquement ce que l'Agent Règles t'a transmis.
        - Reste fidèle à l'ambiance et à l'intrigue.
        - Si l'Agent Monde signale une incohérence majeure, intègre-la comme un obstacle ou une impossibilité narrative.

        NARRE LA RÉPONSE AU JOUEUR:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "rules_info": rules_info,
            "world_info": world_info,
            "memory": memory
        })

        return response
