from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import chromadb
import config

class RPGAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7
        )

        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )

        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.vector_store = Chroma(
            client=self.client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings
        )

        self.history = ChatMessageHistory()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un Maître du Jeu (MJ) expert en jeux de rôle.
            Ton but actuel est de guider le joueur pas à pas dans la création de son personnage en te basant sur les règles et les informations contenues dans le CODEX ci-dessous.

            CONSIGNES :
            1. Sois proactif : pose une seule question à la fois pour guider le joueur.
            2. Utilise le CODEX pour proposer des options valides (races, classes, statistiques, compétences, etc.).
            3. Garde un ton immersif, médiéval-fantastique et encourageant.
            4. Ne sors jamais de ton rôle de MJ.
            5. Dès que tu considères que le personnage est complet (tous les éléments nécessaires selon les règles ont été définis), tu DOIS conclure la création et générer un bloc JSON final récapitulant toutes les caractéristiques du personnage.

            IMPORTANT : Le bloc JSON doit être unique, complet et entouré des balises ```json et ```.

            CODEX :
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucun contexte trouvé."

    def chat(self, user_input):
        context = self.get_context(user_input)

        # Préparation des entrées pour la chaîne
        inputs = {
            "context": context,
            "history": self.history.messages,
            "input": user_input
        }

        response = self.chain.invoke(inputs)

        # Mise à jour de l'historique
        self.history.add_user_message(user_input)
        self.history.add_ai_message(response.content)

        return response.content

    def clear_history(self):
        self.history.clear()
