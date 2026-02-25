from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_utils import safe_chain_invoke, handle_llm_error
import config

class AgentChronique:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7,
            timeout=20
        )

    def generer_chapitre(self, derniers_evenements):
        """
        Generates a new chapter of the chronicle (adventure log) in the first person
        based on the last 10 history events.
        """
        prompt = ChatPromptTemplate.from_template("""
        You are the hero of an epic adventure. You keep an adventure log.
        Your role is to write a new chapter of your adventures based on the latest events that happened to you.

        INSTRUCTIONS:
        - Write in the FIRST PERSON in French ("Je", "Moi", "Mon").
        - The tone should be that of an adventure log (personal reflections, emotions, narrative tone).
        - Fluidly summarize the provided events, do not just make a list.
        - Be concise but immersive.
        - DO NOT invent major events not in the list, but you can elaborate on your feelings.

        LAST EVENTS:
        {evenements}

        NEW LOG CHAPTER IN FRENCH:
        """)

        chain = prompt | self.llm | StrOutputParser()

        # Preparation of events as a text list
        evenements_texte = "\n".join([f"- {ev}" for ev in derniers_evenements])

        try:
            chapitre = safe_chain_invoke(chain, {"evenements": evenements_texte})
            return chapitre.strip()
        except Exception as e:
            print(f"Error chronicle generation: {e}")
            return "Une page blanche dans mon journal... (Erreur de génération de l'Oracle)"
