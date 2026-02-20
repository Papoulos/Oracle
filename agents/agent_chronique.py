from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

class AgentChronique:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7  # Un peu plus de créativité pour la narration
        )

    def generer_chapitre(self, derniers_evenements):
        """
        Génère un nouveau chapitre de la chronique (journal de bord) à la première personne
        basé sur les 10 derniers événements de l'historique.
        """
        prompt = ChatPromptTemplate.from_template("""
        Tu es le héros d'une aventure épique. Tu tiens ton journal de bord.
        Ton rôle est de rédiger un nouveau chapitre de tes aventures en te basant sur les derniers événements qui te sont arrivés.

        CONSIGNES :
        - Rédige à la PREMIÈRE PERSONNE ("Je", "Moi", "Mon").
        - Le ton doit être celui d'un journal de bord (réflexions personnelles, émotions, ton narratif).
        - Résume de manière fluide les événements fournis, ne te contente pas de faire une liste.
        - Sois concis mais immersif.
        - N'invente pas d'événements majeurs qui ne sont pas dans la liste, mais tu peux broder sur tes sentiments.

        DERNIERS ÉVÉNEMENTS :
        {evenements}

        NOUVEAU CHAPITRE DU JOURNAL :
        """)

        chain = prompt | self.llm | StrOutputParser()

        # Préparation des événements sous forme de liste textuelle
        evenements_texte = "\n".join([f"- {ev}" for ev in derniers_evenements])

        try:
            chapitre = chain.invoke({"evenements": evenements_texte})
            return chapitre.strip()
        except Exception as e:
            print(f"Erreur génération chronique : {e}")
            return "Une page blanche dans mon journal... (Erreur de génération)"
