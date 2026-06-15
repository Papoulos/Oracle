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
from scenario_agents import NPCExtractorAgent, ScenarioSummaryAgent

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
            5. Détermine et calcule TOUTES les statistiques dérivées à partir des règles du CODEX : Points de Vie (PV), Classe d'Armure (CA), Jets de Protection (Saves), et toute autre caractéristique pertinente selon la classe et la race choisies. Guide le joueur si un choix ou un jet de dé est nécessaire pour ces valeurs.
            6. Garde un ton immersif, médiéval-fantastique et encourageant.
            7. Ne sors jamais de ton rôle de MJ.
            8. Dès que tu considères que le personnage est complet, tu DOIS conclure la création et générer un bloc JSON final récapitulant TOUTES les caractéristiques du personnage, y compris les statistiques dérivées (PV, CA, Jets de Protection, etc.).
            9. Une fois le JSON généré, ne commence PAS l'aventure. Contente-toi de dire au joueur que son personnage est prêt et que l'aventure va pouvoir commencer.

            IMPORTANT : Le bloc JSON doit être unique, complet (incluant nom, classe, race, niveau, statistiques, PV, CA, jets de protection, équipement, compétences) et entouré des balises ```json et ```. C'est ce bloc qui signale techniquement la fin de cette phase.

            CODEX (Règles et Monde) :
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=config.RAG_SEARCH_K)
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

class SheetManagerAgent(BaseAgent):
    def __init__(self, vector_store):
        super().__init__(model=config.SHEET_MANAGER_MODEL, temperature=config.SHEET_MANAGER_TEMP)
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Gestionnaire de Fiche de Personnage.
            Ton rôle est de mettre à jour la fiche JSON du personnage en fonction des événements qui viennent de se produire, en respectant les règles du CODEX.
            Tu reçois la fiche actuelle, l'action du joueur, la réponse du narrateur et les règles pertinentes.
            Tu dois retourner la NOUVELLE fiche JSON complète et à jour.

            CONSIGNES :
            - Mets à jour les Points de Vie (PV) si le personnage a été blessé ou soigné.
            - Ajoute ou retire des objets de l'inventaire si nécessaire.
            - Mets à jour l'expérience (XP) ou le niveau si mentionné, en suivant les tables de progression du CODEX.
            - Ne modifie pas les statistiques de base (Force, etc.) sauf si un événement permanent l'exige.
            - Assure-toi que le JSON est valide et complet.
            - Réponds UNIQUEMENT avec le bloc JSON entouré de ```json et ```.

            RÈGLES DU CODEX (Contexte) :
            {context}
            """),
            ("human", """FICHE ACTUELLE :
            {character_sheet}

            DERNIERS ÉVÉNEMENTS :
            Action joueur : {user_input}
            Réponse narrateur : {narrator_response}

            Nouvelle fiche JSON mise à jour :"""),
        ])
        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=config.RAG_SEARCH_K)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucune règle trouvée pour la mise à jour."

    def update_sheet(self, character_sheet, user_input, narrator_response):
        context = self.get_context(f"Règles pour : {user_input} {narrator_response}")
        inputs = {
            "context": context,
            "character_sheet": json.dumps(character_sheet, ensure_ascii=False, indent=2),
            "user_input": user_input,
            "narrator_response": narrator_response
        }
        response = self.chain.invoke(inputs)
        new_sheet = extract_json(response.content)
        return new_sheet

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
        self.sheet_manager = SheetManagerAgent(self.core_store)

        # Agents de setup (one-shot, début de partie)
        self.npc_extractor_agent = NPCExtractorAgent(self.scenario_store)
        self.scenario_summary_agent = ScenarioSummaryAgent(self.scenario_store)

        # Données PNJ en mémoire vive
        self.npcs_data = None

        self.history = ChatMessageHistory()
        self.game_state = "CREATION" # CREATION, SUMMARY, ADVENTURE
        self.character_data = None
        self.scenario_data = None
        self.chronicle_data = None

    def _check_collections(self):
        checks = [
            (config.CORE_COLLECTION_NAME,     "Règles"),
            (config.SCENARIO_COLLECTION_NAME, "Scénario"),
        ]
        all_ok = True
        for coll_name, label in checks:
            try:
                collection = self.client.get_collection(coll_name)
                count = collection.count()
                if count == 0:
                    print(f"⚠ [{label}] Collection '{coll_name}' est vide — lance 'python indexer.py'")
                    all_ok = False
                else:
                    print(f"✓ [{label}] {count} chunks disponibles.")
            except Exception:
                print(f"✗ [{label}] Collection '{coll_name}' introuvable.")
                all_ok = False
        return all_ok

    def get_core_context(self, query):
        try:
            docs = self.core_store.similarity_search(query, k=config.RAG_SEARCH_K)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucune règle trouvée."

    def get_scenario_context(self, query):
        try:
            docs = self.scenario_store.similarity_search(query, k=config.RAG_SEARCH_K)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucun élément de scénario trouvé."

    def roll_dice(self, sides=20):
        return random.randint(1, sides)

    def setup_world(self) -> bool:
        """
        Pipeline de setup complet (one-shot, exécuté après la création du PJ).

        Étape 1 → ScenarioSummaryAgent : génère Memory/scenario.json
        Étape 2 → NPCExtractorAgent  : génère Memory/npcs.json

        Retourne True si le scénario a été généré avec succès.
        """
        if not self._check_collections():
            return False

        print("[RPGAgent] ── Setup du monde ──")

        # Étape 1 : Trame scénario
        print("[RPGAgent] Extraction du scénario...")
        scenario = self.scenario_summary_agent.generate()
        if not scenario:
            return False
        self.scenario_data = scenario

        # Étape 2 : PNJ
        print("[RPGAgent] Extraction des PNJ...")
        npcs = self.npc_extractor_agent.extract(self.scenario_data)
        self.npcs_data = npcs  # peut être [] sans bloquer

        return True

    def _unwrap_character_data(self, data):
        """Désadresse les données du personnage si elles sont imbriquées dans une clé racine."""
        if not isinstance(data, dict):
            return data

        # Liste des clés racines courantes
        root_keys = ["personnage", "character", "pj", "sheet", "fiche"]

        # Si on n'a qu'une seule clé et qu'elle est dans notre liste
        if len(data) == 1:
            key = list(data.keys())[0]
            if key.lower() in root_keys and isinstance(data[key], dict):
                return data[key]

        # Sinon on cherche si une de ces clés existe au premier niveau et contient un dictionnaire significatif
        for key in root_keys:
            if key in data and isinstance(data[key], dict) and len(data[key]) > 2:
                return data[key]

        return data

    def chat(self, user_input):
        if self.game_state == "CREATION":
            response = self.character_creator.generate_response(user_input, self.history.messages)

            character_data = extract_json(response)
            if isinstance(character_data, dict):
                self.character_data = self._unwrap_character_data(character_data)
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
            chronicle_text = self.chronicle_data.get("summary", "L'aventure commence.") if self.chronicle_data else "L'aventure commence."

            # 1. L'Orchestrateur analyse l'action avec le Codex (règles)
            analysis_prompt = f"""Analyse l'action du joueur : "{user_input}"
            Basé sur les RÈGLES du CODEX suivantes :
            {core_context}

            Historique de l'aventure (Chronique) :
            {chronicle_text}

            PNJs présents et leurs secrets (si pertinent) :
            {json.dumps(self.npcs_data, ensure_ascii=False, indent=2)}

            Selon le personnage ({json.dumps(self.character_data, ensure_ascii=False)}), un jet de dé est-il nécessaire ?
            Si oui, identifie le bonus approprié en appliquant RIGOUREUSEMENT les règles du CODEX ci-dessus à la fiche de personnage du joueur.

            Réponds au format JSON :
            {{
                "need_roll": boolean,
                "stat": "nom_stat_ou_null",
                "bonus": integer_ou_null,
                "calculation_breakdown": "explication du bonus (ex: +3 Force, +2 Athlétisme)",
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
                    bonus = analysis_data.get("bonus")
                    if bonus is None:
                        bonus = 0
                    total = die_roll + bonus
                    stat_name = analysis_data.get("stat", "Inconnu")
                    dc = analysis_data.get("dc", 10)
                    breakdown = analysis_data.get("calculation_breakdown", f"bonus +{bonus}")

                    roll_info = f"Jet de {stat_name} (DC {dc}) : {die_roll} + {bonus} ({breakdown}) = {total}"
                    roll_result = "Succès" if total >= dc else "Échec"
            except Exception:
                analysis_data = {"need_roll": False}

            # 2. L'Orchestrateur donne ses instructions basées sur le SCÉNARIO et la CHRONIQUE
            decision_instruction = f"""Action Joueur: {user_input}
            Contexte Scénario (Faits du RAG): {scenario_context}
            Historique de l'aventure (Chronique - progression réelle): {chronicle_text}
            PNJs disponibles (sans secrets) :
            {npcs_summary}
            Résultat technique : {"Pas de jet nécessaire" if not roll_info else f"{roll_info} -> {roll_result}"}
            Instructions: Décris les conséquences en utilisant les éléments du SCÉNARIO (RAG) et de la CHRONIQUE (pour la cohérence de la progression). Utilise les PNJs si nécessaire, et le résultat technique. Inclus les points clés/indices dans le résumé final.
            """

            final_response = self.narrator.generate_response(user_input, self.history.messages, decision_instruction)

            if roll_info:
                final_response += f"\n\n---\n*🎲 {roll_info} ({roll_result})*"

            # Mise à jour de la fiche de personnage
            try:
                new_sheet = self.sheet_manager.update_sheet(self.character_data, user_input, final_response)
                if new_sheet and isinstance(new_sheet, dict):
                    self.character_data = self._unwrap_character_data(new_sheet)
                    os.makedirs("Memory", exist_ok=True)
                    with open("Memory/character.json", "w", encoding="utf-8") as f:
                        json.dump(self.character_data, f, indent=4, ensure_ascii=False)
                    print("[RPGAgent] Fiche de personnage mise à jour.")
            except Exception as e:
                print(f"[RPGAgent] ⚠ Erreur lors de la mise à jour de la fiche : {e}")

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

        pitch = self.scenario_data.get('pitch', 'Une nouvelle aventure commence.')
        situation = self.scenario_data.get('situation_initiale', 'Le héros se tient prêt.')

        intro_instruction = (
            f"Pitch de l'aventure : {pitch}\n"
            f"Situation initiale : {situation}\n"
            "En te basant sur ces éléments et sur tes connaissances du monde (RAG), présente la scène d'ouverture de manière immersive. "
            "Le joueur doit être immédiatement plongé dans l'action ou l'ambiance. "
            "Termine par le bloc 📌 Résumé des informations."
        )

        intro_response = self.narrator.generate_response(
            "L'aventure commence !", self.history.messages, intro_instruction
        )

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
                    data = json.load(f)
                    self.character_data = self._unwrap_character_data(data)
                self.game_state = "SUMMARY"
                nom = self.character_data.get('nom') if self.character_data else "Inconnu"
                print(f"[RPGAgent] Personnage chargé : {nom}")
                return True
        except Exception as e:
            print(f"[RPGAgent] Erreur chargement personnage : {e}")
        return False

    def load_game(self):
        """Charge la sauvegarde complète (PJ + PNJ + Scénario + Chronique)."""
        try:
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.character_data = self._unwrap_character_data(data)

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
