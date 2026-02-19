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
        Tu es le MJ Narrateur. Ton rôle est de décrire la scène et de jouer les PNJ.
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
        - Décris la scène de manière immersive.
        - Incorpore les résultats des jets de dés (succès/échec) s'il y en a.
        - Pose un choix ou demande une action au joueur à la fin.
        - Ne décide PAS des règles.
        - Ne connais pas le scénario complet au-delà de ce que l'Agent Monde t'a transmis.
        - Reste fidèle à l'ambiance du jeu.
        - NE FAIS JAMAIS référence à des discussions ou actions précédentes du joueur qui ne figurent pas dans la MÉMOIRE DU JEU. Si la mémoire est vide, c'est que la partie vient de commencer. Ne dis pas "Comme vous l'avez mentionné" si ce n'est pas écrit dans la mémoire.

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
