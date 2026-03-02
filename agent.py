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
    def __init__(self, model=None, temperature=0.7):
        self.llm = ChatOllama(
            model=model if model else config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature
        )

class CharacterCreator(BaseAgent):
    def __init__(self, vector_store):
        super().__init__(model=config.CHARACTER_MODEL, temperature=config.CHARACTER_TEMP)
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un Maître du Jeu (MJ) expert en jeux de rôle.
            Ton but actuel est de guider le joueur pas à pas dans la création de son personnage en te basant sur les règles et les informations contenues dans le CODEX ci-dessous.

            CONSIGNES :
            1. Sois proactif : pose une seule question à la fois pour guider le joueur.
            2. Utilise le CODEX pour proposer des options valides (races, classes, statistiques, compétences, équipement, etc.).
            3. Lors de la détermination des caractéristiques (Force, Dextérité, etc.), propose CLAIREMENT au joueur de lancer les dés pour lui ou de le laisser faire/utiliser une autre méthode.
            4. N'oublie JAMAIS l'étape de l'équipement de départ en suivant scrupuleusement les règles du CODEX pour la classe choisie.
            5. Garde un ton immersif, médiéval-fantastique et encourageant.
            6. Ne sors jamais de ton rôle de MJ.
            7. Dès que tu considères que le personnage est complet, tu DOIS conclure la création et générer un bloc JSON final récapitulant toutes les caractéristiques du personnage.
            8. Une fois le JSON généré, ne commence PAS l'aventure. Contente-toi de dire au joueur que son personnage est prêt et que l'aventure va pouvoir commencer.

            IMPORTANT : Le bloc JSON doit être unique, complet et entouré des balises ```json et ```. C'est ce bloc qui signale techniquement la fin de cette phase.

            CODEX (Règles et Monde) :
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
        super().__init__(model=config.NARRATOR_MODEL, temperature=config.NARRATOR_TEMP)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Narrateur d'une aventure de jeu de rôle.
            Ton rôle est de décrire les scènes, de jouer les PNJs et de présenter les choix au joueur.
            Tu reçois des instructions de l'Orchestrateur (MJ) et tu dois les transformer en un récit immersif en français.

            CONSIGNES :
            - Ne décide JAMAIS des règles ou des résultats des actions (c'est l'Orchestrateur qui le fait).
            - Ne modifie JAMAIS l'état du jeu.
            - Utilise un ton narratif riche et immersif.
            - Réagis en fonction de l'historique de la conversation pour rester cohérent.
            - Termine la narration par une question ou une incitation à l'action pour le joueur.
            - APRÈS la question, ajoute une ligne de séparation "---" suivie d'un bloc intitulé "📌 Résumé des informations" contenant les points clés de l'action, les indices trouvés ou les informations importantes récoltées.
            - Ne liste JAMAIS les "PNJs présents" ou "Lieux présents" sous forme de liste technique à la fin.
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
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=config.ORCHESTRATOR_TEMP)
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)

        # Collection pour les règles (Core)
        self.core_store = Chroma(
            client=self.client,
            collection_name=config.CORE_COLLECTION_NAME,
            embedding_function=self.embeddings
        )

        # Collection pour le scénario
        self.scenario_store = Chroma(
            client=self.client,
            collection_name=config.SCENARIO_COLLECTION_NAME,
            embedding_function=self.embeddings
        )

        self.character_creator = CharacterCreator(self.core_store)
        self.narrator = Narrator()

        self.history = ChatMessageHistory()
        self.game_state = "CREATION" # CREATION, SUMMARY, ADVENTURE
        self.character_data = None
        self.scenario_data = None

    def get_core_context(self, query):
        try:
            docs = self.core_store.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucune règle trouvée."

    def get_scenario_context(self, query):
        try:
            docs = self.scenario_store.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucun élément de scénario trouvé."

    def roll_dice(self, sides=20):
        return random.randint(1, sides)

    def generate_scenario(self):
        # On tente d'extraire des infos du vector store scenario
        try:
            # On cherche de manière large pour l'extraction globale
            scenario_docs = self.scenario_store.similarity_search("intrigue personnages lieux objectifs", k=10)
            raw_context = "\n\n".join([doc.page_content for doc in scenario_docs])
        except Exception:
            raw_context = "Aucun document de scénario trouvé."

        # Prompt pour extraire le scénario
        extraction_prompt = f"""Tu es un assistant MJ. À partir des extraits de documents suivants, extrais les informations clés pour lancer une partie de JDR.
        Si les documents sont vides ou insuffisants, invente une suite cohérente basée sur le personnage du joueur.

        PERSONNAGE :
        {json.dumps(self.character_data, indent=2, ensure_ascii=False)}

        EXTRAITS DU SCÉNARIO :
        {raw_context}

        Génère un scénario structuré au format JSON avec les clés suivantes :
        - titre: Le titre de l'aventure
        - intrigue: Un résumé global de l'intrigue
        - personnages_cles: Liste des PNJs importants mentionnés
        - lieux_cles: Liste des lieux importants
        - situation_initiale: La scène exacte où commence le joueur
        - objectifs: Liste des objectifs immédiats

        Réponds uniquement avec le bloc JSON entouré de ```json et ```.
        """
        response = self.llm.invoke(extraction_prompt)
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
            response = self.character_creator.generate_response(user_input, self.history.messages)

            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1).strip()
                    self.character_data = json.loads(json_str)
                    os.makedirs("Memory", exist_ok=True)
                    with open("Memory/character.json", "w", encoding="utf-8") as f:
                        json.dump(self.character_data, f, indent=4, ensure_ascii=False)
                    self.game_state = "SUMMARY"
                except Exception as e:
                    print(f"Error parsing character JSON: {e}")

            self.history.add_user_message(user_input)
            self.history.add_ai_message(response)
            return response

        elif self.game_state == "ADVENTURE":
            core_context = self.get_core_context(user_input)
            scenario_context = self.get_scenario_context(user_input)

            # 1. L'Orchestrateur analyse l'action avec le Codex (règles)
            analysis_prompt = f"""Analyse l'action du joueur : "{user_input}"
            Basé sur les RÈGLES suivantes :
            {core_context}

            Selon le personnage ({json.dumps(self.character_data, ensure_ascii=False)}), un jet de dé est-il nécessaire ?
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
                json_match = re.search(r"(\{.*?\})", analysis_response, re.DOTALL)
                analysis_data = json.loads(json_match.group(1)) if json_match else {"need_roll": False}

                if analysis_data.get("need_roll"):
                    die_roll = self.roll_dice(20)
                    stat_name = analysis_data.get("stat", "Inconnu")
                    dc = analysis_data.get("dc", 10)
                    roll_info = f"Jet de {stat_name} (DC {dc}) : {die_roll}"
                    roll_result = "Succès" if die_roll >= dc else "Échec"
            except Exception:
                analysis_data = {"need_roll": False}

            # 2. L'Orchestrateur donne ses instructions basées sur le SCÉNARIO
            decision_instruction = f"""Action Joueur: {user_input}
            Contexte Scénario (Faits): {scenario_context}
            Résumé Scénario Global: {self.scenario_data['intrigue']}
            Résultat technique : {"Pas de jet nécessaire" if not roll_info else f"{roll_info} -> {roll_result}"}
            Instructions: Décris les conséquences en utilisant les éléments du SCÉNARIO et le résultat technique. Inclus les points clés/indices dans le résumé final.
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
            intro_instruction = f"L'aventure commence. Voici le scénario : {self.scenario_data['intrigue']}. Présente la situation initiale : {self.scenario_data['situation_initiale']}. N'oublie pas le résumé des points clés à la fin."
            intro_response = self.narrator.generate_response("L'aventure commence !", self.history.messages, intro_instruction)
            self.history.add_ai_message(intro_response)
            return intro_response
        return "Erreur lors de la génération du scénario."

    def clear_history(self):
        self.history.clear()
        self.game_state = "CREATION"
        self.character_data = None
        self.scenario_data = None
