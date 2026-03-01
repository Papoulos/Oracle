import json
import re
import random
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import chromadb
import config

class BaseAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7
        )

class CharacterCreator(BaseAgent):
    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un Maître du Jeu (MJ) expert en jeux de rôle.
            Ton but actuel est de guider le joueur pas à pas dans la création de son personnage en te basant sur les règles et les informations contenues dans le CODEX ci-dessous.

            CONSIGNES :
            1. Sois proactif : pose une seule question à la fois pour guider le joueur.
            2. Utilise le CODEX pour proposer des options valides (races, classes, statistiques, compétences, etc.).
            3. Garde un ton immersif, médiéval-fantastique et encourageant.
            4. Ne sors jamais de ton rôle de MJ.
            5. Dès que tu considères que le personnage est complet, tu DOIS conclure la création et générer un bloc JSON final récapitulant toutes les caractéristiques du personnage.
            6. Une fois le JSON généré, ne commence PAS l'aventure. Contente-toi de dire au joueur que son personnage est prêt et que l'aventure va pouvoir commencer.

            IMPORTANT : Le bloc JSON doit être unique, complet et entouré des balises ```json et ```. C'est ce bloc qui signale techniquement la fin de cette phase.

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

    def generate_response(self, user_input, history):
        context = self.get_context(user_input)
        inputs = {
            "context": context,
            "history": history,
            "input": user_input
        }
        response = self.chain.invoke(inputs)
        return response.content

class Narrator(BaseAgent):
    def __init__(self):
        super().__init__()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Narrateur d'une aventure de jeu de rôle.
            Ton rôle est de décrire les scènes, de jouer les PNJs et de présenter les choix au joueur.
            Tu reçois des instructions de l'Orchestrateur (MJ) et tu dois les transformer en un récit immersif en français.

            CONSIGNES :
            - Ne décide JAMAIS des règles ou des résultats des actions (c'est l'Orchestrateur qui le fait).
            - Ne modifie JAMAIS l'état du jeu.
            - Utilise un ton narratif riche et immersif.
            - Réagis en fonction de l'historique de la conversation pour rester cohérent.
            - Termine toujours par une question ou une incitation à l'action pour le joueur.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("system", "CONSIGNES DE L'ORCHESTRATEUR : {instructions}"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm

    def generate_response(self, user_input, history, instructions):
        inputs = {
            "history": history,
            "instructions": instructions,
            "input": user_input
        }
        response = self.chain.invoke(inputs)
        return response.content

class RPGAgent(BaseAgent):
    def __init__(self):
        super().__init__()
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

        self.character_creator = CharacterCreator(self.vector_store)
        self.narrator = Narrator()

        self.history = ChatMessageHistory()
        self.game_state = "CREATION" # CREATION, SUMMARY, ADVENTURE
        self.character_data = None
        self.scenario_data = None

    def get_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucun contexte trouvé."

    def roll_dice(self, sides=20):
        return random.randint(1, sides)

    def generate_scenario(self):
        # Prompt to generate a scenario based on character
        scenario_prompt = f"""Génère un scénario de départ pour un jeu de rôle basé sur ce personnage :
        {json.dumps(self.character_data, indent=2, ensure_ascii=False)}

        Le scénario doit être au format JSON avec les clés suivantes :
        - titre: Le titre de l'aventure
        - intrigue: Un résumé de l'intrigue
        - situation_initiale: La scène de départ où se trouve le joueur
        - objectifs: Liste des objectifs immédiats

        Réponds uniquement avec le bloc JSON entouré de ```json et ```.
        """
        response = self.llm.invoke(scenario_prompt)
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response.content, re.DOTALL)
        if json_match:
            try:
                self.scenario_data = json.loads(json_match.group(1))
                os.makedirs("Memory", exist_ok=True)
                with open("Memory/scenario.json", "w", encoding="utf-8") as f:
                    json.dump(self.scenario_data, f, indent=4, ensure_ascii=False)
                return True
            except Exception:
                return False
        return False

    def chat(self, user_input):
        if self.game_state == "CREATION":
            # Add character generation history if it's the first message
            if not self.history.messages:
                # This logic might be handled in app.py but good to have safety
                pass

            response = self.character_creator.generate_response(user_input, self.history.messages)

            # Check for character completion (handle potential text before/after JSON)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                try:
                    # Nettoyage supplémentaire si nécessaire pour json.loads
                    json_str = json_match.group(1).strip()
                    self.character_data = json.loads(json_str)

                    # Sauvegarde
                    os.makedirs("Memory", exist_ok=True)
                    with open("Memory/character.json", "w", encoding="utf-8") as f:
                        json.dump(self.character_data, f, indent=4, ensure_ascii=False)

                    # Transition d'état
                    self.game_state = "SUMMARY"
                except Exception as e:
                    # Log error but don't break the flow
                    print(f"Error parsing character JSON: {e}")

            self.history.add_user_message(user_input)
            self.history.add_ai_message(response)
            return response

        elif self.game_state == "ADVENTURE":
            context = self.get_context(user_input)

            # 1. L'Orchestrateur décide si un jet est nécessaire
            analysis_prompt = f"""Analyse l'action du joueur : "{user_input}"
            Selon le personnage et le scénario, un jet de dé est-il nécessaire ?
            Si oui, précise la CARACTÉRISTIQUE à utiliser et la DIFFICULTÉ (DC).
            Réponds au format JSON :
            {{
                "need_roll": boolean,
                "stat": "nom_stat_ou_null",
                "dc": integer_ou_null,
                "reason": "explication courte"
            }}
            """
            analysis_response = self.llm.invoke(analysis_prompt).content
            roll_info = ""
            roll_result = None

            try:
                # Extraction du JSON d'analyse
                json_match = re.search(r"(\{.*?\})", analysis_response, re.DOTALL)
                analysis_data = json.loads(json_match.group(1)) if json_match else {"need_roll": False}

                if analysis_data.get("need_roll"):
                    die_roll = self.roll_dice(20)
                    stat_name = analysis_data.get("stat", "Inconnu")
                    dc = analysis_data.get("dc", 10)
                    # On pourrait ici chercher le bonus dans character_data, mais on va rester simple
                    roll_info = f"Jet de {stat_name} (DC {dc}) : {die_roll}"
                    roll_result = "Succès" if die_roll >= dc else "Échec"
            except Exception:
                analysis_data = {"need_roll": False}

            # 2. L'Orchestrateur donne ses instructions finales au Narrateur
            decision_instruction = f"""Action Joueur: {user_input}
            Contexte Codex: {context}
            Résultat technique : {"Pas de jet nécessaire" if not roll_info else f"{roll_info} -> {roll_result}"}
            Instructions: Décris les conséquences de cette action en restant fidèle au résultat technique.
            """

            final_response = self.narrator.generate_response(user_input, self.history.messages, decision_instruction)

            if roll_info:
                final_response += f"\n\n---\n*🎲 {roll_info} ({roll_result})*"

            self.history.add_user_message(user_input)
            self.history.add_ai_message(final_response)
            return final_response

    def start_adventure(self):
        if self.generate_scenario():
            self.game_state = "ADVENTURE"
            # Introduction par le narrateur
            intro_instruction = f"L'aventure commence. Voici le scénario : {self.scenario_data['intrigue']}. Présente la situation initiale : {self.scenario_data['situation_initiale']}"
            intro_response = self.narrator.generate_response("L'aventure commence !", self.history.messages, intro_instruction)
            self.history.add_ai_message(intro_response)
            return intro_response
        return "Erreur lors de la génération du scénario."

    def clear_history(self):
        self.history.clear()
        self.game_state = "CREATION"
        self.character_data = None
        self.scenario_data = None
