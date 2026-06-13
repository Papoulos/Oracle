import json
import re
import random
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import chromadb
import config
from base_utils import BaseAgent, get_llm, get_embeddings, extract_json
from scenario_agents import NPCSetupAgent, ScenarioSetupAgent

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

class ChronicleAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=config.CHRONICLE_MODEL, temperature=config.CHRONICLE_TEMP)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Chroniqueur d'une aventure de jeu de rôle.
            Ton rôle est de tenir à jour un résumé factuel et concis de l'histoire jusqu'à présent.
            Tu reçois l'ancien résumé, l'action du joueur et la réponse du narrateur.
            Tu dois produire un NOUVEAU résumé mis à jour qui intègre ces nouveaux événements.

            CONSIGNES :
            - Sois concis et factuel.
            - Garde les éléments importants (lieux, rencontres, objets obtenus, blessures).
            - Utilise le français.
            - Réponds uniquement avec le nouveau résumé, sans fioritures.
            """),
            ("human", """ANCIEN RÉSUMÉ : {old_chronicle}
            ACTION JOUEUR : {user_input}
            RÉPONSE NARRATEUR : {narrator_response}

            Nouveau résumé mis à jour :"""),
        ])
        self.chain = self.prompt | self.llm

    def update(self, old_chronicle, user_input, narrator_response):
        inputs = {
            "old_chronicle": old_chronicle if old_chronicle else "L'aventure commence à peine.",
            "user_input": user_input,
            "narrator_response": narrator_response
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
        self.embeddings = get_embeddings()
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
        self.chronicle_agent = ChronicleAgent()

        # Agents de setup (one-shot, début de partie)
        self.npc_setup_agent = NPCSetupAgent(self.scenario_store)
        self.scenario_setup_agent = ScenarioSetupAgent(self.scenario_store, self.core_store)

        # Données PNJ en mémoire vive
        self.npcs_data = None

        self.history = ChatMessageHistory()
        self.game_state = "CREATION" # CREATION, SUMMARY, ADVENTURE
        self.character_data = None
        self.scenario_data = None
        self.chronicle_data = None

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

    def setup_world(self) -> bool:
        """
        Pipeline de setup complet (one-shot, exécuté après la création du PJ).

        Étape 1 → NPCSetupAgent  : génère Memory/npcs.json
        Étape 2 → ScenarioSetupAgent : génère Memory/scenario.json (enrichi)

        Retourne True si le scénario a été généré avec succès.
        """
        print("[RPGAgent] ── Setup du monde ──")

        # Étape 1 : PNJ
        print("[RPGAgent] Génération des fiches PNJ...")
        npcs = self.npc_setup_agent.generate_npcs(self.character_data)
        self.npcs_data = npcs  # peut être [] si le scénario est vide

        # Étape 2 : Trame scénario
        print("[RPGAgent] Génération de la trame scénario...")
        scenario = self.scenario_setup_agent.generate_scenario(
            self.character_data, self.npcs_data or []
        )

        if not scenario:
            return False

        self.scenario_data = scenario
        return True

    def chat(self, user_input):
        if self.game_state == "CREATION":
            response = self.character_creator.generate_response(user_input, self.history.messages)

            character_data = extract_json(response)
            if isinstance(character_data, dict):
                self.character_data = character_data
                os.makedirs("Memory", exist_ok=True)
                with open("Memory/character.json", "w", encoding="utf-8") as f:
                    json.dump(self.character_data, f, indent=4, ensure_ascii=False)
                self.game_state = "SUMMARY"

            self.history.add_user_message(user_input)
            self.history.add_ai_message(response)
            return response

        elif self.game_state == "ADVENTURE":
            core_context = self.get_core_context(user_input)
            scenario_context = self.get_scenario_context(user_input)
            npcs_summary = self.get_npcs_context()

            # 1. L'Orchestrateur analyse l'action avec le Codex (règles)
            analysis_prompt = f"""Analyse l'action du joueur : "{user_input}"
            Basé sur les RÈGLES suivantes :
            {core_context}

            PNJs présents et leurs secrets (si pertinent) :
            {json.dumps(self.npcs_data, ensure_ascii=False, indent=2)}

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
                analysis_data = extract_json(analysis_response)
                if not analysis_data:
                    print(f"[RPGAgent] ✗ Échec de l'extraction JSON de l'analyse : {analysis_response[:200]}...")
                    analysis_data = {"need_roll": False}

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
            Résumé Scénario Global: {self.scenario_data['intrigue_complete']}
            PNJs disponibles (sans secrets) :
            {npcs_summary}
            Résultat technique : {"Pas de jet nécessaire" if not roll_info else f"{roll_info} -> {roll_result}"}
            Instructions: Décris les conséquences en utilisant les éléments du SCÉNARIO, les PNJs si nécessaire, et le résultat technique. Inclus les points clés/indices dans le résumé final.
            """

            final_response = self.narrator.generate_response(user_input, self.history.messages, decision_instruction)

            if roll_info:
                final_response += f"\n\n---\n*🎲 {roll_info} ({roll_result})*"

            # Mise à jour de la chronique
            self.update_chronicle(user_input, final_response)

            self.history.add_user_message(user_input)
            self.history.add_ai_message(final_response)
            return final_response

    def start_adventure(self):
        """Lance le pipeline de setup puis démarre la narration."""
        if not self.setup_world():
            return "Erreur lors de la génération du monde."

        self.game_state = "ADVENTURE"

        acte1 = self.scenario_data.get("actes", [{}])[0]
        intro_instruction = (
            f"Scénario : {self.scenario_data['intrigue_complete']}\n"
            f"Situation initiale : {self.scenario_data['situation_initiale']}\n"
            f"Premier acte — '{acte1.get('titre', '')}' : {acte1.get('objectif_principal', '')}\n"
            "Présente la scène d'ouverture de manière immersive. "
            "Termine par le bloc 📌 Résumé des informations."
        )

        intro_response = self.narrator.generate_response(
            "L'aventure commence !", self.history.messages, intro_instruction
        )

        pitch = self.scenario_data.get('pitch', '')
        full_response = f"**{self.scenario_data.get('titre', 'Aventure')}**\n\n*{pitch}*\n\n{intro_response}"

        self.update_chronicle("L'aventure commence !", full_response)
        self.history.add_ai_message(full_response)
        return full_response

    def update_chronicle(self, user_input, response):
        old_summary = ""
        if self.chronicle_data and isinstance(self.chronicle_data, dict):
            old_summary = self.chronicle_data.get("summary", "")

        new_summary = self.chronicle_agent.update(old_summary, user_input, response)
        self.chronicle_data = {"summary": new_summary}

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/Chronicle.json", "w", encoding="utf-8") as f:
            json.dump(self.chronicle_data, f, indent=4, ensure_ascii=False)

    def load_character(self):
        """Charge uniquement le personnage et passe en mode SUMMARY."""
        try:
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    self.character_data = json.load(f)
                self.game_state = "SUMMARY"
                print(f"[RPGAgent] Personnage chargé : {self.character_data.get('nom')}")
                return True
        except Exception as e:
            print(f"[RPGAgent] Erreur chargement personnage : {e}")
        return False

    def load_game(self):
        """Charge la sauvegarde complète (PJ + PNJ + Scénario + Chronique)."""
        try:
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    self.character_data = json.load(f)

            if os.path.exists("Memory/npcs.json"):
                with open("Memory/npcs.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.npcs_data = data.get("npcs", [])

            if os.path.exists("Memory/scenario.json"):
                with open("Memory/scenario.json", "r", encoding="utf-8") as f:
                    self.scenario_data = json.load(f)

            if os.path.exists("Memory/Chronicle.json"):
                with open("Memory/Chronicle.json", "r", encoding="utf-8") as f:
                    self.chronicle_data = json.load(f)

            if self.character_data and self.scenario_data:
                self.game_state = "ADVENTURE"
                nb_npcs = len(self.npcs_data) if self.npcs_data else 0
                print(f"[RPGAgent] Partie chargée — {nb_npcs} PNJ disponibles.")
                return True
            elif self.character_data:
                self.game_state = "SUMMARY"
                print(f"[RPGAgent] Personnage chargé (partie incomplète).")
                return True

        except Exception as e:
            print(f"[RPGAgent] Erreur chargement : {e}")

        return False

    def clear_history(self):
        """Réinitialise complètement la partie."""
        self.history.clear()
        self.game_state = "CREATION"
        self.character_data = None
        self.scenario_data = None
        self.chronicle_data = None
        self.npcs_data = None

        for file in ["character.json", "npcs.json", "scenario.json", "Chronicle.json"]:
            path = os.path.join("Memory", file)
            if os.path.exists(path):
                os.remove(path)
                print(f"[RPGAgent] Supprimé : {path}")

    def get_npc(self, npc_id: str) -> dict | None:
        """Retourne la fiche d'un PNJ par son id, ou None."""
        if not self.npcs_data:
            return None
        return next((n for n in self.npcs_data if n.get("id") == npc_id), None)

    def get_npcs_context(self) -> str:
        """
        Retourne un résumé des PNJ (SANS leurs secrets) pour le Narrateur.
        Les secrets restent côté Orchestrateur uniquement.
        """
        if not self.npcs_data:
            return "Aucun PNJ disponible."
        lines = []
        for n in self.npcs_data:
            lines.append(
                f"- {n['nom']} ({n['classe']}, niv.{n['niveau']}) "
                f"| Relation PJ: {n['relation_pj']} "
                f"| Lieu: {n.get('localisation_actuelle', '?')}"
            )
        return "\n".join(lines)
