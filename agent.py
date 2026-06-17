import json
import re
import random
import os
import time
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
            Ton but actuel est de guider le joueur dans la création de son personnage.
            Utilise le MANUEL DE CRÉATION comme une liste de contrôle (checklist) pour ne rien oublier, mais reste interactif et flexible.

            MANUEL DE CRÉATION (Structure globale) :
            {manual}

            CONSIGNES DE DIALOGUE ET MÉCANIQUES :
            1. Réponds TOUJOURS en français.
            2. Pose UNE SEULE QUESTION à la fois.
            3. Pour les statistiques/caractéristiques : Demande TOUJOURS au joueur s'il souhaite que tu lances les dés pour lui (selon la méthode du Codex) ou s'il préfère le faire lui-même.
            4. Pour les choix de Race et de Classe : Ne te contente pas de ce qui est dans le manuel. Interroge le CODEX (RAG) pour obtenir la liste complète et exacte des options disponibles et présente-les au joueur.
            5. Calcule les statistiques dérivées (PV, CA, modificateurs) en suivant scrupuleusement les formules du CODEX.

            CONSIGNES TECHNIQUES (JSON) :
            1. À CHAQUE RÉPONSE, tu DOIS inclure un bloc JSON valide à la toute fin.
            2. Le bloc JSON doit être entouré des balises ```json et ```.
            3. NE METS RIEN APRÈS LE BLOC JSON.
            4. Si TOUTES les étapes du manuel sont terminées, mets `"statut": "complet"`. Sinon, `"statut": "en_cours"`.

            STRUCTURE DU JSON ATTENDUE :
            {{
                "nom": "...",
                "race": "...",
                "classe": "...",
                "statistiques": {{ "Force": 10, ... }},
                "équipement": [...],
                "pv": 10,
                "ca": 10,
                "statut": "en_cours" | "complet"
            }}

            ÉTAT ACTUEL DU PERSONNAGE :
            {current_character}

            CODEX (Détails des règles à interroger pour chaque choix) :
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=config.RAG_K_CREATION)
            print(f"[CharacterCreator] DEBUG: Recherche contextuelle pour '{query}' -> {len(docs)} docs trouvés.")
            if docs:
                print(f"[CharacterCreator] DEBUG: Premier extrait : {docs[0].page_content[:200]}...")
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[CharacterCreator] DEBUG: Erreur RAG : {e}")
            return "Aucun contexte trouvé."

    def _load_manual(self):
        path = "Memory/creation_manual.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.dumps(json.load(f), ensure_ascii=False, indent=2)
            except Exception:
                return "Manuel non disponible (utiliser le Codex)."
        return "Manuel non disponible (utiliser le Codex)."

    def generate_response(self, user_input, history, character_data=None):
        # On enrichit la requête RAG avec les derniers messages pour avoir du contexte sur l'étape de création
        rag_query = user_input
        if len(history) >= 1:
            last_msg = history[-1].content
            # On extrait une partie du dernier message du MJ pour aider le RAG
            rag_query = f"{last_msg[:100]} {user_input}"

        context = self.get_context(rag_query)
        manual = self._load_manual()

        inputs = {
            "manual": manual,
            "context": context,
            "history": history,
            "input": user_input,
            "current_character": json.dumps(character_data, ensure_ascii=False, indent=2) if character_data else "Aucune donnée pour le moment."
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
            docs = self.vector_store.similarity_search(query, k=config.RAG_K_ADVENTURE)
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
            ("system", """Tu es le Narrateur d'une aventure de jeu de rôle. Tu reçois des instructions
structurées de l'Orchestrateur (MJ) et tu les transformes en narration à la
deuxième personne du singulier, en français, de manière immersive.

RÈGLES ABSOLUES :
- Tu ne décides JAMAIS des règles, des jets de dés ou des résultats d'actions.
- Tu ne modifies JAMAIS l'état du jeu.
- Tu n'interprètes JAMAIS à la place du joueur — tu décris ce que son personnage
  perçoit, pas ce qu'il comprend ou conclut.

STRUCTURE DE CHAQUE RÉPONSE :

① PERCEPTION IMMÉDIATE (2-4 phrases)
  Ce que le personnage voit, entend, sent à l'instant T.
  Concret et sensoriel. Pas de conclusions, pas d'interprétations.
  ❌ "Vous comprenez que quelque chose a été tué ici."
  ✅ "Le sol est couvert d'une substance sombre et poisseuse. Une odeur
      âcre de fer et de chair vous prend à la gorge."

② DÉTAILS ET ENVIRONNEMENT (2-3 phrases)
  Ce que le personnage remarque en regardant autour de lui.
  Toujours du point de vue du personnage — ce qu'il voit réellement,
  pas ce que le MJ sait.

③ TENSION OU IMPULSION (1-2 phrases)
  Un élément actif qui pousse le joueur à réagir :
  un bruit, un mouvement, une présence, un choix visible.
  Ne laisse jamais la description en suspension.

④ QUESTION OU AMORCE D'ACTION
  Une question directe ou une proposition concrète.
  ❌ "Que faites-vous ?" (trop vague)
  ✅ "Le couloir nord semble plus sombre — aucune torche n'y brûle.
      À l'est, vous distinguez ce qui ressemble à une porte.
      Vous avancez, ou vous faites demi-tour ?"

⑤ BLOC RÉSUMÉ (toujours en dernier)
  ---
  📌 Résumé des informations
  - [fait découvert 1]
  - [fait découvert 2]
  - [changement d'état ou indice important]

STYLE :
- Deuxième personne du singulier ("vous").
- Présent de narration.
- Phrases courtes et rythmées pour les moments de tension,
  plus longues pour les descriptions calmes.
- Ne liste jamais les PNJ présents ou les lieux de façon technique.
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
        self.setup_logs = []

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        self.setup_logs.append(full_message)
        try:
            with open("streamlit.log", "a", encoding="utf-8") as f:
                f.write(full_message + "\n")
        except Exception:
            pass

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

    def get_core_context(self, query, k=None):
        if k is None:
            k = config.RAG_SEARCH_K
        try:
            docs = self.core_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucune règle trouvée."

    def get_scenario_context(self, query, k=None):
        if k is None:
            k = config.RAG_SEARCH_K
        try:
            docs = self.scenario_store.similarity_search(query, k=k)
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

        self.setup_logs = [] # Reset logs
        self.log("── Setup du monde ──")
        total_start = time.time()

        # Étape 1 : Trame scénario
        self.log("Étape 1 : Extraction du scénario...")
        scenario_start = time.time()
        scenario = self.scenario_summary_agent.generate(log_callback=self.log)
        if not scenario:
            self.log("✗ Échec de l'extraction du scénario.")
            return False
        self.scenario_data = scenario
        self.log(f"✓ Scénario extrait en {time.time() - scenario_start:.2f}s.")

        # Étape 2 : PNJ
        self.log("Étape 2 : Extraction des PNJ...")
        npcs_start = time.time()
        npcs = self.npc_extractor_agent.extract(self.scenario_data, log_callback=self.log)
        self.npcs_data = npcs  # peut être [] sans bloquer
        self.log(f"✓ PNJs extraits en {time.time() - npcs_start:.2f}s.")

        self.log(f"✨ Setup terminé en {time.time() - total_start:.2f}s.")
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
            response = self.character_creator.generate_response(user_input, self.history.messages, self.character_data)

            character_data = extract_json(response)
            if isinstance(character_data, dict):
                unwrapped_data = self._unwrap_character_data(character_data)
                self.character_data = unwrapped_data
                os.makedirs("Memory", exist_ok=True)
                with open("Memory/character.json", "w", encoding="utf-8") as f:
                    json.dump(self.character_data, f, indent=4, ensure_ascii=False)

                # Transition vers SUMMARY si le statut est complet
                # On accepte "complet" ou "terminé" (tolérance LLM)
                status = str(self.character_data.get("statut", "")).lower()
                if status in ["complet", "complete", "terminé", "termine"]:
                    print(f"[RPGAgent] Fin de création détectée (statut: {status}).")
                    self.game_state = "SUMMARY"
            else:
                print(f"[RPGAgent] ⚠ Échec de l'extraction JSON pendant la création. Réponse brute : {response[:100]}...")

            self.history.add_user_message(user_input)
            self.history.add_ai_message(response)
            return response

        elif self.game_state == "ADVENTURE":
            core_context = self.get_core_context(user_input, k=config.RAG_K_ADVENTURE)
            scenario_context = self.get_scenario_context(user_input, k=config.RAG_K_ADVENTURE)
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

            # 2. L'Orchestrateur donne ses instructions basées sur le SCÉNARIO
            decision_instruction = f"""
ACTION DU JOUEUR : {user_input}

RÉSULTAT TECHNIQUE : {"Aucun jet requis" if not roll_info else f"{roll_info} → {roll_result}"}

CONTEXTE SCÉNARIO (extraits RAG) : {scenario_context}
PNJ DISPONIBLES : {npcs_summary}

TON RÔLE : Tu es le MJ. Génère des instructions précises pour le Narrateur.

STRUCTURE OBLIGATOIRE de ta réponse :

1. CONSÉQUENCE IMMÉDIATE
   Ce qui se passe concrètement suite à l'action. Si jet réussi : avantage clair.
   Si jet échoué : information manquante, mauvaise interprétation, ou complication.
   Ne révèle que ce que le personnage peut percevoir à cet instant.

2. PERCEPTIONS SENSORIELLES
   Ce que le personnage voit, entend, sent, touche ou ressent physiquement.
   Sois précis et concret — pas d'atmosphère vague.
   Exemples : "il voit trois torches éteintes et une porte entrouverte à gauche",
   "il entend un souffle rythmique venant du couloir nord".

3. ÉLÉMENTS INCONNUS OU AMBIGUS
   Ce que le personnage ne peut pas encore déterminer (en lien avec le jet raté si applicable).
   Formule-le du point de vue du personnage, pas du MJ.
   Exemple : "l'origine du bruit reste indéterminée" plutôt que "c'est un rat".

4. IMPULSION NARRATIVE
   Donne une direction active au joueur : un détail qui appelle une réaction,
   un PNJ qui agit, un bruit qui se rapproche, un choix qui se présente.
   Ne laisse JAMAIS la scène en suspension sans amorce concrète.

5. POINTS CLÉS POUR LE RÉSUMÉ
   Liste 2-3 faits importants découverts lors de cette action (indices, informations, changements d'état).
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

        self.log("Génération de l'introduction narrative...")
        intro_start = time.time()
        self.game_state = "ADVENTURE"

        pitch = self.scenario_data.get('pitch', 'Une nouvelle aventure commence.')
        situation = self.scenario_data.get('situation_initiale', 'Le héros se tient prêt.')
        setup_context = self.get_scenario_context("intrigue lieux personnages", k=config.RAG_K_SETUP)

        intro_instruction = f"""
ACTION DU JOUEUR : L'aventure commence !

RÉSULTAT TECHNIQUE : Aucun jet requis

CONTEXTE SCÉNARIO :
- Pitch : {pitch}
- Situation initiale : {situation}
- Détails supplémentaires : {setup_context}

TON RÔLE : Tu es le MJ. Génère des instructions précises pour le Narrateur pour lancer l'aventure.

STRUCTURE OBLIGATOIRE de ta réponse :

1. CONSÉQUENCE IMMÉDIATE
   Décris l'entrée en matière du personnage dans l'histoire.

2. PERCEPTIONS SENSORIELLES
   Ce que le personnage voit, entend, sent ou ressent en arrivant dans cette première scène.

3. ÉLÉMENTS INCONNUS OU AMBIGUS
   Des zones d'ombre ou des mystères immédiats qui piquent la curiosité.

4. IMPULSION NARRATIVE
   Un événement ou un détail qui force le joueur à prendre sa première décision.

5. POINTS CLÉS POUR LE RÉSUMÉ
   Liste les éléments fondamentaux de la situation initiale.
"""

        intro_response = self.narrator.generate_response(
            "L'aventure commence !", self.history.messages, intro_instruction
        )
        self.log(f"✓ Introduction générée en {time.time() - intro_start:.2f}s.")

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
                f"- {n['nom']} ({n['classe']}) "
                f"| Relation PJ: {n['relation_pj']} "
                f"| Lieu: {n.get('localisation_actuelle', '?')}"
            )
        return "\n".join(lines)
