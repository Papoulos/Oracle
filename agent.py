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
from scenario_agents import ScenarioExtractorAgent
from validation import validate_scenario_structure, validate_character_sheet

class CharacterCreator(BaseAgent):
    def __init__(self, vector_store, verbose=False):
        super().__init__(model=config.CHARACTER_MODEL, temperature=config.CHARACTER_TEMP, verbose=verbose)
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Game Master (GM) expert in role-playing games.
            Your current goal is to guide the player in creating their character.
            Use the CHARACTER CREATION MANUAL as a checklist to ensure nothing is forgotten, but remain interactive and flexible.

            CHARACTER CREATION MANUAL (Global structure):
            {manual}

            DIALOGUE AND MECHANICAL GUIDELINES:
            1. ALWAYS respond to the player in French.
            2. Ask ONLY ONE question at a time.
            3. TECHNICAL PREPARATION: Before asking a question, consult the CODEX (RAG) to know ALL constraints and benefits related to the current choice (e.g., "How many weapons can they master?", "What are the benefits of this race?").
            4. PEDAGOGY: ALWAYS explain to the player the technical consequences of their choice in French (e.g., "En choisissant cette classe, tu as droit à 3 maîtrises d'armes qui te donneront un bonus de +2 aux jets d'attaque").
            5. For statistics/attributes: ALWAYS ask the player if they want you to roll the dice for them (according to the Codex method) or if they prefer to do it themselves.
            6. For Race and Class choices: Query the CODEX (RAG) to get the complete and exact list of available options and present them concisely to the player.
            7. Calculate derived statistics (hit_points, AC, modifiers) scrupulously following the formulas from the CODEX.

            STILL MISSING FIELDS (You must guide the player to fill these):
            {missing_fields}

            CONSIGNES DE FIN DE CRÉATION (In French for player):
            If all steps of the manual are completed, congratulate the player and tell them their character is ready for adventure.
            Use the keyword "CREATION_COMPLETED" in your text only when ALL steps are validated.

            CURRENT CHARACTER STATE (for your information):
            {current_character}

            CODEX (Details of the rules to query for each choice):
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            # Slightly reduce RAG depth for creation to optimize context window
            k = max(1, config.RAG_K_CREATION - 3)
            docs = self.vector_store.similarity_search(query, k=k)
            print(f"[CharacterCreator] DEBUG: Contextual search for '{query}' -> {len(docs)} docs found.")
            if docs:
                print(f"[CharacterCreator] DEBUG: First excerpt: {docs[0].page_content[:200]}...")
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[CharacterCreator] DEBUG: RAG Error: {e}")
            return "No context found."

    def _load_manual(self):
        path = "Memory/creation_manual.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.dumps(json.load(f), ensure_ascii=False, indent=2)
            except Exception:
                return "Manual unavailable (use Codex)."
        return "Manual unavailable (use Codex)."

    def generate_response(self, user_input, history, character_data=None, missing_fields=None):
        # Enrich the RAG query with the previous turn to focus context retrieval
        rag_query = user_input
        if len(history) >= 1:
            last_msg = history[-1].content
            rag_query = f"{last_msg[:100]} {user_input}"

        context = self.get_context(rag_query)
        manual = self._load_manual()

        if missing_fields and isinstance(missing_fields, list):
            missing_fields_str = ", ".join(missing_fields)
        else:
            missing_fields_str = "None (sheet is complete)."

        inputs = {
            "manual": manual,
            "context": context,
            "history": history,
            "input": user_input,
            "current_character": json.dumps(character_data, ensure_ascii=False, indent=2) if character_data else "No data yet.",
            "missing_fields": missing_fields_str
        }
        response = self.chain.invoke(inputs)
        return response.content

class ChronicleAgent(BaseAgent):
    def __init__(self, verbose=False):
        super().__init__(model=config.CHRONICLE_MODEL, temperature=config.CHRONICLE_TEMP, verbose=verbose)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Chronicler of a role-playing adventure.
            Your role is to maintain a factual and concise summary of the story so far in FRENCH.
            You receive the old summary, the player's action, the narrator's response, and potentially an additional fact.
            You must produce a NEW updated summary that integrates these new events.

            CONSIGNES:
            - Be concise and factual.
            - Keep important elements (locations, encounters, obtained items, wounds).
            - Always respond in French.
            - Reply only with the new summary, with no fluff.
            """),
            ("human", """OLD CHRONICLE: {old_chronicle}
            PLAYER ACTION: {user_input}
            NARRATOR RESPONSE: {narrator_response}
            ADDITIONAL FACT TO KEEP (may diverge from source scenario): {notable_deviation}

            New updated summary:"""),
        ])
        self.chain = self.prompt | self.llm

    def update(self, old_chronicle, user_input, narrator_response, notable_deviation=None):
        inputs = {
            "old_chronicle": old_chronicle if old_chronicle else "L'aventure commence à peine.",
            "user_input": user_input,
            "narrator_response": narrator_response,
            "notable_deviation": notable_deviation if notable_deviation else "Aucun fait additionnel."
        }
        response = self.chain.invoke(inputs)
        return response.content

class SheetManagerAgent(BaseAgent):
    def __init__(self, vector_store, verbose=False):
        super().__init__(model=config.SHEET_MANAGER_MODEL, temperature=config.SHEET_MANAGER_TEMP, verbose=verbose)
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Character Sheet Manager.
            Your role is to update the JSON character sheet for NARRATIVE aspects only (equipment, status, description).
            You receive the current sheet, the player's action, the narrator's response, and relevant rules.

            ⚠️ CRITICAL RULE: You must NEVER modify Hit Points (hit_points), XP, level, spells, or resources (rage, etc.). These elements are managed deterministically by the GameStateEngine.

            CONSIGNES:
            - Add or remove items from the equipment list if necessary.
            - Ensure the JSON is valid and complete.
            - Never modify base statistics unless a permanent event demands it.
            - Use the exact keys from the glossary: "name", "race", "class", "statistics", "equipment", "status", "resources", "hit_points" (with "current" and "max" under "hit_points").
            - Reply ONLY with the JSON block enclosed between ```json and ```.

            CODEX RULES (Context):
            {context}
            """),
            ("human", """CURRENT SHEET:
            {character_sheet}

            LATEST EVENTS:
            Player action: {user_input}
            Narrator response: {narrator_response}

            New updated JSON sheet:"""),
        ])
        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=config.RAG_K_ADVENTURE)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "No rules found for update."

    def update_sheet(self, character_sheet, user_input, narrator_response, mode="ADVENTURE"):
        context = self.get_context(f"Rules for: {user_input} {narrator_response}")

        llm = self.llm

        if mode == "CREATION":
            creation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are the Character Sheet Manager in the CREATION phase.
                Your role is to extract information from the conversation to update the JSON character sheet.

                CONSIGNES:
                - Analyze the player's action and the GM's response to identify new choices (name, race, class, statistics, equipment, etc.).
                - Update all necessary fields.
                - If the GM mentions that creation is complete (keyword "CREATION_COMPLETED"), set the "status" field to "complet".
                - Use the exact keys from the glossary: "name", "race", "class", "statistics", "equipment", "status", "resources", "hit_points" (with "current" and "max" under "hit_points").
                - Reply ONLY with the complete JSON block.

                CODEX RULES (Context):
                {context}
                """),
                ("human", """CURRENT SHEET:
                {character_sheet}

                LATEST EVENTS:
                Player action: {user_input}
                GM Response: {narrator_response}

                New updated JSON sheet:"""),
            ])
            chain = creation_prompt | llm
            inputs = {
                "context": context,
                "character_sheet": json.dumps(character_sheet, ensure_ascii=False, indent=2) if character_sheet else "{}",
                "user_input": user_input,
                "narrator_response": narrator_response
            }
            try:
                response = chain.invoke(inputs)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    print(f"[SheetManagerAgent] ⚠ Model '{config.SHEET_MANAGER_MODEL}' not found, fallback to '{config.LLM_MODEL}'.")
                    fallback_llm = get_llm(config.LLM_MODEL, config.SHEET_MANAGER_TEMP)
                    chain = creation_prompt | fallback_llm
                    response = chain.invoke(inputs)
                else:
                    raise e
        else:
            inputs = {
                "context": context,
                "character_sheet": json.dumps(character_sheet, ensure_ascii=False, indent=2),
                "user_input": user_input,
                "narrator_response": narrator_response
            }
            try:
                response = self.chain.invoke(inputs)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    print(f"[SheetManagerAgent] ⚠ Model '{config.SHEET_MANAGER_MODEL}' not found, fallback to '{config.LLM_MODEL}'.")
                    fallback_llm = get_llm(config.LLM_MODEL, config.SHEET_MANAGER_TEMP)
                    fallback_chain = self.prompt | fallback_llm
                    response = fallback_chain.invoke(inputs)
                else:
                    raise e

        new_sheet = extract_json(response.content)
        return new_sheet

    def audit_and_complete(self, character_data: dict, messages: list, schema: dict) -> dict | None:
        context = self.get_context("statistiques, points de vie, équipement, armes, armures, ressources")
        audit_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Character Sheet Manager, performing a FINAL AUDIT of character creation.
Review the ENTIRE creation conversation below and produce the most complete and faithful JSON character sheet possible, based on the provided required fields schema.
Search in particular for values mentioned in the conversation but missing from the current sheet (often forgotten by incremental turn-by-turn updates).
Do not modify fields that are already correct and do not change their existing name/structure.
Use the exact keys from the glossary: "name", "race", "class", "statistics", "equipment", "status", "resources", "hit_points" (with "current" and "max" under "hit_points").
Reply ONLY with the complete JSON block of the sheet.

REQUIRED FIELDS SCHEMA: {schema}
CODEX RULES: {context}
"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "CURRENT SHEET:\n{character_sheet}\n\nProduce the final, complete JSON sheet."),
        ])
        chain = audit_prompt | self.llm
        try:
            response = chain.invoke({
                "schema": json.dumps(schema, ensure_ascii=False),
                "context": context,
                "history": messages,
                "character_sheet": json.dumps(character_data, ensure_ascii=False, indent=2),
            })
            return extract_json(response.content, expected_type=dict)
        except Exception:
            return None

class Narrator(BaseAgent):
    def __init__(self, verbose=False):
        super().__init__(model=config.NARRATOR_MODEL, temperature=config.NARRATOR_TEMP, verbose=verbose)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Narrator of a role-playing adventure. You receive structured instructions
from the Orchestrator (GM) and you transform them into immersive narration in the second person singular, in FRENCH.

ABSOLUTE RULES:
- You NEVER decide on rules, dice rolls, or action outcomes.
- You NEVER modify the game state.
- You NEVER interpret on behalf of the player — you describe what their character perceives, not what they understand or conclude.
- You NEVER directly reveal the name of the final antagonist, the exact nature of the threat, or the scenario resolution condition until the player has discovered them through gameplay - even if this information is given to you in context by the Orchestrator.

STRUCTURE of each response:

Write a continuous and fluid narration in FRENCH, WITHOUT titles, WITHOUT numbers, WITHOUT bullet points in the narrative text itself. Advance your text through these movements, without ever announcing them explicitly to the player:

- Immediate perception: what the character sees, hears, smells at the current moment. Concrete and sensory. No conclusions, no interpretations.
  ❌ "Vous comprenez que quelque chose a été tué ici."
  ✅ "Le sol est couvert d'une substance sombre et poisseuse. Une odeur âcre de fer et de chair vous prend à la gorge."
- Details and environment: what the character notices looking around, always from their own point of view - what they actually see, never what the GM knows in addition.
- Tension or impulse: an active element that pushes the player to react (a noise, a movement, a presence, a visible choice). Never leave the description in suspension.
- Question or action prompt, to finish: a direct question or concrete proposal in French.
  ❌ "Que faites-vous ?" (too vague)
  ✅ "Le couloir nord semble plus sombre — aucune torche n'y brûle. À l'est, vous distinguez ce qui ressemble à une porte. Vous avancez, ou vous faites demi-tour ?"

After this narrative text, add a separate summary block, introduced by "---" and the header "📌 Résumé des informations" (see strict rules below).

STRICT RULES OF THE SUMMARY BLOCK:
- List ONLY facts that you have just explicitly stated in the narrative text above, in this same response.
- NEVER add a fact, name, meaning, or interpretation that does not appear (word for word or in very close substance) in the text you have just written. If an object or clue exists but its content or meaning has not yet been revealed to the player in the narration, do NOT mention this content in the summary - only mention its physical existence if it has been described.
  ❌ Narration: "des gravures anciennes ornent les parois." / Summary: "Indice : des gravures décrivant la gloire d'Aethelgard." (information not revealed in narration → FORBIDDEN)
  ✅ Narration: "des gravures anciennes ornent les parois." / Summary: "Des gravures anciennes ornent les parois, non encore examinées."

STYLE:
- Second person singular ("vous").
- Present tense of narration.
- Short, rhythmic sentences for moments of tension, longer ones for calm descriptions.
- Never list NPCs present or locations technically.
"""),
            MessagesPlaceholder(variable_name="history"),
            ("system", "ORCHESTRATOR INSTRUCTIONS: {instructions}"),
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
    def __init__(self, verbose=False):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=config.ORCHESTRATOR_TEMP, verbose=verbose)
        self.embeddings = get_embeddings()
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)

        # Core Rules collection
        self.core_store = Chroma(
            client=self.client,
            collection_name=config.CORE_COLLECTION_NAME,
            embedding_function=self.embeddings
        )

        # Scenario collection
        self.scenario_store = Chroma(
            client=self.client,
            collection_name=config.SCENARIO_COLLECTION_NAME,
            embedding_function=self.embeddings
        )

        self.character_creator = CharacterCreator(self.core_store, verbose=verbose)
        self.narrator = Narrator(verbose=verbose)
        self.chronicle_agent = ChronicleAgent(verbose=verbose)
        self.sheet_manager = SheetManagerAgent(self.core_store, verbose=verbose)

        # Setup Agent
        self.scenario_extractor_agent = ScenarioExtractorAgent(self.scenario_store, verbose=verbose)

        from game_state_engine import GameStateEngine
        self.gse = GameStateEngine()

        # Scenario data & progression
        self.scenario_structure = None
        self.progression = None
        self._pnj_by_id = {}
        self._lieux_by_id = {}
        self._scenes_by_id = {}
        self._actes_by_id = {}

        self.npcs_data = None
        self.current_scene_id = None

        self.history = ChatMessageHistory()
        self.game_state = "CREATION" # CREATION, SUMMARY, ADVENTURE
        self.character_data = None
        self.scenario_data = None
        self.chronicle_data = None
        self.setup_logs = []
        self._missing_character_fields = None

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
            (config.CORE_COLLECTION_NAME,     "Rules"),
            (config.SCENARIO_COLLECTION_NAME, "Scenario"),
        ]
        all_ok = True
        for coll_name, label in checks:
            try:
                collection = self.client.get_collection(coll_name)
                count = collection.count()
                if count == 0:
                    print(f"⚠ [{label}] Collection '{coll_name}' is empty — run 'python indexer.py'")
                    all_ok = False
                else:
                    print(f"✓ [{label}] {count} chunks available.")
            except Exception:
                print(f"✗ [{label}] Collection '{coll_name}' not found.")
                all_ok = False
        return all_ok

    def get_core_context(self, query, k=None):
        if k is None:
            k = config.RAG_SEARCH_K
        try:
            docs = self.core_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "No rules found."

    def get_scenario_context(self, query, k=None):
        if k is None:
            k = config.RAG_SEARCH_K
        try:
            docs = self.scenario_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "No scenario details found."

    def get_current_scene(self) -> str:
        """
        Retrieves current scene by direct lookup (no RAG or Chroma filtering).
        """
        if not self.scenario_structure or not self.current_scene_id:
            return ""
        scene = self._scenes_by_id.get(self.current_scene_id)
        if not scene:
            return ""
        return json.dumps(scene, ensure_ascii=False, indent=2)

    def get_current_context(self) -> str:
        """
        Generates clean context from progression and static lookups (no RAG).
        """
        if not self.progression or not self.scenario_structure:
            return ""
        scene_id = self.progression.get("current_scene")
        scene = self._scenes_by_id.get(scene_id)
        if not scene:
            return ""
        location = self._lieux_by_id.get(scene.get("location_id"))
        npcs = [self._pnj_by_id[pid] for pid in scene.get("present_npcs", []) if pid in self._pnj_by_id]
        act = self._actes_by_id.get(scene.get("act_id"))

        context_lines = []
        context_lines.append(f"CURRENT SCENE : {scene.get('title')} (ID: {scene_id})")
        if location:
            context_lines.append(f"Location : {location.get('full_name')} ({location.get('sensory_atmosphere')})")
            if location.get("interactive_elements"):
                context_lines.append(f"Interactive elements: {location.get('interactive_elements')}")
        if npcs:
            context_lines.append("NPCs present:")
            for p in npcs:
                context_lines.append(f"- {p.get('full_name')} (Motivation: {p.get('agenda_and_motivation')}, Attitude: {p.get('initial_attitude')}, Stats/Abilities: {p.get('stats_and_abilities')})")
        if act:
            context_lines.append(f"Parent Act: {act.get('title')} (Completion Condition: {act.get('completion_condition')})")

        return "\n".join(context_lines)

    def roll_dice(self, sides=20):
        return random.randint(1, sides)

    def _extract_and_add_resources(self):
        """Extracts class resources (spells, rage, etc.) and adds them to the character sheet."""
        self.log("Extracting class resources and abilities...")
        classe = self.character_data.get("class", "Inconnu")
        niveau = self.character_data.get("level", 1)
        race = self.character_data.get("race", "Inconnu")

        query = f"Ressources de classe pour {classe} niveau {niveau}, race {race}. Emplacements de sorts, points de rage, capacités limitées par jour, points de vie."
        context = self.get_core_context(query, k=10)

        prompt = f"""You are an expert in RPG rules.
Based on the following CODEX excerpts, identify ALL consumable resources (spell slots per day, limited use abilities, hit points, etc.) for a character of level {niveau}, class {classe} and race {race}.

CODEX EXCERPTS:
{context}

CURRENT SHEET:
{json.dumps(self.character_data, ensure_ascii=False, indent=2)}

Produce a JSON 'resources' object that can be integrated into the sheet.
Each resource must have a 'total' and a 'current' equal to total.
Use clear, lowercase technical French key names for custom classes resources (e.g. 'emplacements_sorts_niv1', 'points_de_rage') except for hit_points which is already 'hit_points'. Always keep the main wrapper as 'resources'.
If the character has spells, also list known spells if mentioned or suggested for this level.

Reply ONLY with the JSON block:
{{
  "resources": {{
    "nom_ressource": {{ "total": X, "current": X }},
    ...
  }},
  "spells": ["spell_name_1", "spell_name_2"]
}}
"""
        try:
            response = self.llm.invoke(prompt).content
            data = extract_json(response)
            if data:
                if "resources" in data and data["resources"]:
                    if "resources" not in self.character_data:
                        self.character_data["resources"] = {}
                    self.character_data["resources"].update(data["resources"])

                if "spells" in data and data["spells"]:
                    existing_spells = self.character_data.get("spells", [])
                    if not existing_spells or len(existing_spells) < len(data["spells"]):
                        self.character_data["spells"] = data["spells"]

                os.makedirs("Memory", exist_ok=True)
                with open("Memory/character.json", "w", encoding="utf-8") as f:
                    json.dump(self.character_data, f, indent=4, ensure_ascii=False)
                self.log("✓ Resources and spells updated in character sheet.")
        except Exception as e:
            self.log(f"⚠ Error during resources extraction: {e}")

    def setup_world(self) -> bool:
        """
        Complete setup pipeline (one-shot, executed after character creation).
        Generates scenario_structure.json and initializes progression.json.
        """
        if not self._check_collections():
            return False

        self.setup_logs = [] # Reset logs
        self.log("── World Setup ──")
        total_start = time.time()

        json_files = [f for f in os.listdir(config.SCENARIO_DATA_PATH) if f.endswith(".json")]

        try:
            if len(json_files) > 1:
                raise ValueError(
                    f"Multiple scenario JSON files found in {config.SCENARIO_DATA_PATH}: "
                    f"{json_files}. Only one structured scenario is supported at a time."
                )
            elif len(json_files) == 1:
                # Direct load of pre-structured scenario
                self.log(f"Direct loading of structured scenario: {json_files[0]}")
                with open(os.path.join(config.SCENARIO_DATA_PATH, json_files[0]), encoding="utf-8") as f:
                    raw = json.load(f)
                self.scenario_structure, warnings, errors = validate_scenario_structure(raw)
            else:
                # Extract from PDFs
                self.log("Extracting scenario structure via ScenarioExtractorAgent in 5 passes...")
                raw = self.scenario_extractor_agent.generate(log_callback=self.log)
                self.scenario_structure, warnings, errors = validate_scenario_structure(raw)

            for w in warnings:
                self.log(f"[Validation] {w}")

            if errors:
                for e in errors:
                    self.log(f"[Validation][ERROR] {e}")
                raise ValueError(
                    "The scenario contains blocking errors that cannot be repaired automatically "
                    "(see logs) - manual correction or re-extraction is required before starting."
                )

            os.makedirs("Memory", exist_ok=True)
            with open("Memory/scenario_structure.json", "w", encoding="utf-8") as f:
                json.dump(self.scenario_structure, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.log(f"✗ Error during setup or structure validation: {e}")
            raise e

        # Build lookups
        self._build_lookups()

        # Initialize progression state
        starting_scene = self.scenario_structure.get("metadata", {}).get("starting_scene")
        if not starting_scene and self.scenario_structure.get("scene_nodes"):
            starting_scene = self.scenario_structure["scene_nodes"][0]["scene_id"]
        elif not starting_scene:
            starting_scene = "Inconnu"

        current_act = "Inconnu"
        if starting_scene in self._scenes_by_id:
            current_act = self._scenes_by_id[starting_scene].get("act_id", "Inconnu")

        self.progression = {
            "current_act": current_act,
            "current_scene": starting_scene,
            "resolved_scenes": [],
            "bypassed_scenes": [],
            "clocks": {
                h["name"]: {"segments": 0, "declenchee": False} for h in self.scenario_structure.get("global_clocks", [])
            },
            "notable_deviations": []
        }

        # Save progression
        try:
            os.makedirs("Memory", exist_ok=True)
            with open("Memory/progression.json", "w", encoding="utf-8") as f:
                json.dump(self.progression, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log(f"⚠ Error saving progression.json: {e}")

        self.current_scene_id = starting_scene

        # Compatibility with app.py using translated keys
        self.scenario_data = {
            "title": self.scenario_structure["metadata"]["title"],
            "pitch": self.scenario_structure["metadata"]["global_pitch"],
        }
        try:
            with open("Memory/scenario.json", "w", encoding="utf-8") as f:
                json.dump(self.scenario_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log(f"⚠ Error saving scenario.json: {e}")

        # NPCs for narrator and orchestrator
        self.npcs_data = self.scenario_structure.get("entities", {}).get("npcs", [])

        self.log(f"✨ Setup completed in {time.time() - total_start:.2f}s.")
        return True

    def _build_lookups(self):
        if not self.scenario_structure:
            return
        self._pnj_by_id = {p["id"]: p for p in self.scenario_structure.get("entities", {}).get("npcs", []) if "id" in p}
        self._lieux_by_id = {l["id"]: l for l in self.scenario_structure.get("entities", {}).get("locations", []) if "id" in l}
        self._scenes_by_id = {s["scene_id"]: s for s in self.scenario_structure.get("scene_nodes", []) if "scene_id" in s}
        self._actes_by_id = {a["act_id"]: a for a in self.scenario_structure.get("acts", []) if "act_id" in a}

    def _unwrap_character_data(self, data):
        """Unwraps nested character sheet if root keys are present."""
        if not isinstance(data, dict):
            return data

        root_keys = ["personnage", "character", "pj", "sheet", "fiche"]

        if len(data) == 1:
            key = list(data.keys())[0]
            if key.lower() in root_keys and isinstance(data[key], dict):
                return data[key]

        for key in root_keys:
            if key in data and isinstance(data[key], dict) and len(data[key]) > 2:
                return data[key]

        return data

    def _load_character_schema(self):
        path = "Memory/character_schema.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"required_fields": [{"path": "name", "type": "string"}]}

    def _load_action_catalog(self) -> dict:
        path = "Memory/action_catalog.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"common_actions": []}

    def _check_resources(self, action):
        """Checks if player's action consumes any limited resource and whether it is available."""
        if not self.character_data or "resources" not in self.character_data:
            return {"ok": True}

        prompt = f"""You are an RPG referee. Analyze the player's action and determine if it consumes a limited resource from their sheet.
ACTION: "{action}"
AVAILABLE RESOURCES: {json.dumps(self.character_data['resources'], ensure_ascii=False)}

Reply ONLY with this JSON:
{{
  "consumes": boolean,
  "resource_name": "name_of_the_key_in_the_json_or_null",
  "quantity": integer_or_null,
  "reason": "short explanation"
}}
"""
        try:
            response = self.llm.invoke(prompt).content
            analysis = extract_json(response)
            if analysis and analysis.get("consumes"):
                res_name = analysis.get("resource_name")
                qty = analysis.get("quantity") or 1

                actual_res_name = None
                if res_name in self.character_data["resources"]:
                    actual_res_name = res_name
                else:
                    for k in self.character_data["resources"].keys():
                        if res_name.lower() in k.lower() or k.lower() in res_name.lower():
                            actual_res_name = k
                            break

                if actual_res_name:
                    res = self.character_data["resources"][actual_res_name]
                    if res.get("current", 0) >= qty or res.get("restants", 0) >= qty:
                        return {"ok": True, "resource": actual_res_name, "cost": qty}
                    else:
                        cur_val = res.get("current") if "current" in res else res.get("restants", 0)
                        max_val = res.get("max", res.get("total", 0))
                        return {"ok": False, "reason": f"Ressource insuffisante : {actual_res_name} ({cur_val}/{max_val})"}

            return {"ok": True}
        except Exception as e:
            print(f"[RPGAgent] Error check_resources: {e}")
            return {"ok": True}

    def chat(self, user_input):
        if self.game_state == "CREATION":
            # 1. Narrative Step: Interact with player
            try:
                response = self.character_creator.generate_response(
                    user_input, self.history.messages, self.character_data,
                    missing_fields=getattr(self, "_missing_character_fields", None)
                )
            except TypeError:
                response = self.character_creator.generate_response(
                    user_input, self.history.messages, self.character_data
                )

            # 2. Technical Step (silent): Update sheet via SheetManager
            try:
                new_sheet = self.sheet_manager.update_sheet(
                    self.character_data,
                    user_input,
                    response,
                    mode="CREATION"
                )
                if new_sheet and isinstance(new_sheet, dict):
                    self.character_data = self._unwrap_character_data(new_sheet)
                    self.gse.state = self.character_data
                    self.gse.synchronize_and_recalculate()
                    self.character_data = self.gse.state

                schema = self._load_character_schema()
                is_complete, missing_fields = validate_character_sheet(self.character_data, schema)

                if not is_complete and any(
                    kw in response.upper() for kw in ["TERMIN", "PRET POUR L'AVENTURE", "PRÊT POUR L'AVENTURE", "CREATION_TERMIN_E", "CREATION_COMPLETED"]
                ):
                    audited = self.sheet_manager.audit_and_complete(
                        self.character_data, self.history.messages, schema
                    )
                    if audited and isinstance(audited, dict):
                        self.character_data = self._unwrap_character_data(audited)
                        self.gse.state = self.character_data
                        self.gse.synchronize_and_recalculate()
                        self.character_data = self.gse.state
                        is_complete, missing_fields = validate_character_sheet(self.character_data, schema)

                os.makedirs("Memory", exist_ok=True)
                with open("Memory/character.json", "w", encoding="utf-8") as f:
                    json.dump(self.character_data, f, indent=4, ensure_ascii=False)

                if is_complete:
                    if not self.character_data:
                        self.character_data = {}
                    self.character_data["status"] = "complet"
                    self._missing_character_fields = None
                    self._extract_and_add_resources()
                    self.game_state = "SUMMARY"
                else:
                    self._missing_character_fields = missing_fields

            except Exception as e:
                print(f"[RPGAgent] ⚠ Error during technical character sheet update: {e}")

            self.history.add_user_message(user_input)
            self.history.add_ai_message(response)
            return response

        elif self.game_state == "ADVENTURE":
            # 0. Resources validation
            res_check = self._check_resources(user_input)
            if not res_check["ok"]:
                msg = f"Action impossible : {res_check['reason']}"
                self.history.add_user_message(user_input)
                self.history.add_ai_message(msg)
                return msg

            scenario_context = self.get_scenario_context(user_input, k=config.RAG_K_ADVENTURE)
            npcs_summary = self.get_npcs_context()
            chronicle_text = self.chronicle_data.get("summary", "L'aventure commence.") if self.chronicle_data else "L'aventure commence."

            # Mechanical detection and validation (Proactive actions)
            action_type = self.gse.detect_action_type(user_input)
            mechanical_result = None

            if action_type == "spell":
                spell_level = 1
                mechanical_result = self.gse.consume_spell_slot(spell_level)
            elif action_type == "rage":
                mechanical_result = self.gse.consume_resource("points_de_rage")
            elif action_type and action_type.startswith("rest:"):
                palier_id = action_type.split(":", 1)[1]
                mechanical_result = self.gse.rest(palier_id)
            elif action_type == "rest":
                rest_type = "long" if any(w in user_input.lower() for w in ["long", "nuit", "camp"]) else "short"
                mechanical_result = self.gse.rest(rest_type)

            mechanical_context = ""
            if mechanical_result:
                if not mechanical_result.success:
                    mechanical_context = f"\n⚠ ACTION BLOQUÉE : {mechanical_result.blocked_reason} — {mechanical_result.message}"
                else:
                    mechanical_context = f"\nMECANIQUE : {mechanical_result.message}"

            state_summary = self.gse.get_state_summary()
            self.character_data = self.gse.state # Sync before analysis

            # Catalog lookups before system-wide RAG
            action_catalog = self._load_action_catalog()
            analysis_json = None

            if action_catalog.get("common_actions"):
                catalog_prompt = f"""Analyze the player's action: "{user_input}"

COMMON ACTIONS CATALOG FOR THIS SYSTEM:
{json.dumps(action_catalog, ensure_ascii=False, indent=2)}

Adventure history (Chronicle):
{chronicle_text}

Present NPCs and their secrets (if relevant):
{json.dumps(self.npcs_data, ensure_ascii=False, indent=2)}

MECHANICAL STATE: {state_summary}
{mechanical_context}

Does this action correspond to an entry in the catalog above? If yes, apply its resolution to the character sheet ({json.dumps(self.character_data, ensure_ascii=False)}).
If the action does not correspond to ANY entry, reply "covered_by_catalog": false WITHOUT guessing - the system will look up the complete rules.

Reply in JSON format:
{{
    "covered_by_catalog": boolean,
    "need_roll": boolean, "stat": "stat_name_or_null", "bonus": integer_or_null,
    "calculation_breakdown": "...", "dc": integer_or_null, "reason": "...",
    "mechanical_decision": {{"action": "damage" | "heal" | "xp" | null, "amount": integer_or_null}}
}}
"""
                try:
                    analysis_response = self.llm.invoke(catalog_prompt).content
                    analysis_json = extract_json(analysis_response, expected_type=dict)
                except Exception as e:
                    print(f"[RPGAgent] Error during catalog analysis: {e}")

            if not analysis_json or analysis_json.get("covered_by_catalog") is False:
                core_context = self.get_core_context(user_input, k=config.RAG_K_ADVENTURE)
                analysis_prompt = f"""Analyze the player's action: "{user_input}"
                Based on the following CODEX RULES:
                {core_context}

                Adventure history (Chronicle):
                {chronicle_text}

                Present NPCs and their secrets (if relevant):
                {json.dumps(self.npcs_data, ensure_ascii=False, indent=2)}

                MECHANICAL STATE: {state_summary}
                {mechanical_context}

                According to the character ({json.dumps(self.character_data, ensure_ascii=False)}), is a dice roll required?
                If so, identify the appropriate bonus by RIGOROUSLY applying the CODEX rules above to the player's character sheet.
                Also determine if this action or its immediate consequences result in damage, healing, or XP gain.

                Reply in JSON format:
                {{
                    "need_roll": boolean,
                    "stat": "stat_name_or_null",
                    "bonus": integer_or_null,
                    "calculation_breakdown": "bonus explanation (e.g., +3 Strength, +2 Athletics)",
                    "dc": integer_or_null,
                    "reason": "short explanation",
                    "mechanical_decision": {{
                        "action": "damage" | "heal" | "xp" | null,
                        "amount": integer_or_null
                    }}
                }}
                """
                try:
                    analysis_response = self.llm.invoke(analysis_prompt).content
                    analysis_json = extract_json(analysis_response, expected_type=dict)
                except Exception as e:
                    print(f"[RPGAgent] Error during Codex analysis: {e}")

            roll_info = ""
            roll_result = None

            try:
                if not analysis_json:
                    analysis_json = {"need_roll": False}

                m_decision = analysis_json.get("mechanical_decision")
                if m_decision and m_decision.get("action"):
                    m_res = self.gse.apply_orchestrator_decision(m_decision)
                    mechanical_context += f"\nDECISION MJ : {m_res.message}"
                    self.character_data = self.gse.state # Sync after decision

                if analysis_json.get("need_roll"):
                    die_roll = self.roll_dice(20)
                    bonus = analysis_json.get("bonus")
                    if bonus is None:
                        bonus = 0
                    total = die_roll + bonus
                    stat_name = analysis_json.get("stat", "Inconnu")
                    dc = analysis_json.get("dc", 10)
                    breakdown = analysis_json.get("calculation_breakdown", f"bonus +{bonus}")

                    roll_info = f"Jet de {stat_name} (DC {dc}) : {die_roll} + {bonus} ({breakdown}) = {total}"
                    roll_result = "Succès" if total >= dc else "Échec"
            except Exception:
                analysis_json = {"need_roll": False}

            # 2. Scene analysis from the Orchestrator (Transition / Improvisation / Bypassed)
            scene_analysis_result = {
                "category": "improvisation",
                "next_scene": None,
                "impacted_clocks": [],
                "notable_deviation": None
            }

            if self.scenario_structure and self.current_scene_id:
                current_scene_dict = self._scenes_by_id.get(self.current_scene_id, {})

                scene_classification_prompt = f"""Analyze the player's action relative to the current scene.

CURRENT SCENE (id {self.current_scene_id}):
Title: {current_scene_dict.get('title', 'Unknown')}
GM Objective (atmosphere): {current_scene_dict.get('gm_objective', '')}
Resolution Condition (the REAL exit criterion): {current_scene_dict.get('resolution_condition', '')}
Anticipated Logical Exits (indicative): {json.dumps(current_scene_dict.get('logical_exits', []), ensure_ascii=False)}
Anticipated Challenges (indicative): {json.dumps(current_scene_dict.get('challenges_and_encounters', []), ensure_ascii=False)}

PLAYER ACTION: {user_input}

Classify the situation into ONE of the three categories. NEVER block a player action, whatever the category — this classification serves only for internal tracking.
- "transition": the player resolves the narrative objective of this scene in one way or another, allowing progression to a subsequent scene.
- "improvisation": the player acts within the current scene without yet reaching the objective or radically shifting the scenario. We stay in the current scene.
- "bypassed": the player completely exits the planned framework, ignores the goal, destroys the opportunity to reach it (e.g., kills a key quest giver, flees the area), or improvises an unanticipated solution that bypasses the current scene without a normal transition.

Reply ONLY in JSON:
{{
  "category": "transition" | "improvisation" | "bypassed",
  "next_scene": "id or null",
  "impacted_clocks": [{{"name": "Clock name", "segments_added": 1}}],
  "notable_deviation": "fact to remember for the future, in one sentence, even if not planned by the source scenario, or null"
}}
"""
                try:
                    classification_response = self.llm.invoke(scene_classification_prompt).content
                    extracted_classification = extract_json(classification_response)
                    if extracted_classification:
                        scene_analysis_result = extracted_classification
                except Exception as e:
                    print(f"[RPGAgent] Error during scene classification: {e}")

            # Deterministic processing of classification
            cat = scene_analysis_result.get("category", "improvisation")
            next_scene_id = scene_analysis_result.get("next_scene")
            impacted_clocks = scene_analysis_result.get("impacted_clocks", [])
            notable_deviation = scene_analysis_result.get("notable_deviation")

            if cat not in ["transition", "improvisation", "bypassed"]:
                cat = "improvisation"

            if cat == "transition":
                if next_scene_id and next_scene_id in self._scenes_by_id:
                    if self.current_scene_id not in self.progression["resolved_scenes"]:
                        self.progression["resolved_scenes"].append(self.current_scene_id)
                    self.current_scene_id = next_scene_id
                    self.progression["current_scene"] = next_scene_id
                    next_scene = self._scenes_by_id[next_scene_id]
                    self.progression["current_act"] = next_scene.get("act_id", "Inconnu")
                    print(f"[RPGAgent] Deterministic transition to scene {next_scene_id}.")
                else:
                    print(f"[RPGAgent] Aborted transition: next scene ID '{next_scene_id}' is invalid or null.")
                    cat = "improvisation"
            elif cat == "bypassed":
                if self.current_scene_id not in self.progression["bypassed_scenes"]:
                    self.progression["bypassed_scenes"].append(self.current_scene_id)
                if next_scene_id and next_scene_id in self._scenes_by_id:
                    self.current_scene_id = next_scene_id
                    self.progression["current_scene"] = next_scene_id
                    next_scene = self._scenes_by_id[next_scene_id]
                    self.progression["current_act"] = next_scene.get("act_id", "Inconnu")
                    print(f"[RPGAgent] Scene bypassed, transitioning to scene {next_scene_id}.")
                else:
                    print(f"[RPGAgent] Scene bypassed {self.current_scene_id} without next scene.")

            # Clocks management
            clock_consequences_to_inject = []
            if "clocks" not in self.progression:
                self.progression["clocks"] = {}

            clocks_meta_dict = {h["name"]: h for h in self.scenario_structure.get("global_clocks", [])}

            for name, h_meta in clocks_meta_dict.items():
                if name not in self.progression["clocks"]:
                    self.progression["clocks"][name] = {"segments": 0, "declenchee": False}

            for h_impacted in impacted_clocks:
                h_name = h_impacted.get("name")
                added_segments = h_impacted.get("segments_added", 0)
                if h_name and h_name in self.progression["clocks"]:
                    meta = clocks_meta_dict[h_name]
                    threshold = meta.get("threshold", 6)
                    consequence = meta.get("consequence", "")

                    current_data = self.progression["clocks"][h_name]
                    old_segments = current_data.get("segments", 0)
                    new_segments = min(old_segments + added_segments, threshold)
                    current_data["segments"] = new_segments

                    if new_segments >= threshold and not current_data.get("declenchee", False):
                        current_data["declenchee"] = True
                        clock_consequences_to_inject.append((h_name, consequence))

            if notable_deviation and str(notable_deviation).lower() != "null":
                if "notable_deviations" not in self.progression:
                    self.progression["notable_deviations"] = []
                self.progression["notable_deviations"].append(notable_deviation)
            else:
                notable_deviation = None

            for h_name, consequence in clock_consequences_to_inject:
                if "notable_deviations" not in self.progression:
                    self.progression["notable_deviations"] = []
                self.progression["notable_deviations"].append(consequence)
                if notable_deviation:
                    notable_deviation += f" | {consequence}"
                else:
                    notable_deviation = consequence

            try:
                os.makedirs("Memory", exist_ok=True)
                with open("Memory/progression.json", "w", encoding="utf-8") as f:
                    json.dump(self.progression, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"[RPGAgent] Error saving progression.json: {e}")

            # 3. GM Orchestrator builds instructions based on SCENARIO and classification
            current_scene_dict = self._scenes_by_id.get(self.current_scene_id, {})
            current_context = self.get_current_context()

            clock_instruction = ""
            for h_name, consequence in clock_consequences_to_inject:
                clock_instruction += f"\n\nÉVÉNEMENT DÉCLENCHÉ (horloge '{h_name}') : {consequence} — à intégrer immédiatement dans la narration de ce tour."

            decision_instruction = f"""
PLAYER ACTION: {user_input}

TECHNICAL RESULT: {"Aucun jet requis" if not roll_info else f"{roll_info} → {roll_result}"}
MECHANICAL STATE: {self.gse.get_state_summary()}
{mechanical_context}

SCENARIO CONTEXT (RAG excerpts): {scenario_context}

SCENARIO CONTEXT STRUCTURED (Lookup):
{current_context}

CURRENT SCENE:
Id: {self.current_scene_id}
Title: {current_scene_dict.get('title', 'Unknown')}
Vibe: {current_scene_dict.get('esprit_de_la_scene', '')}
Goal: {current_scene_dict.get('objectif_atteint_si', '')}
Preserved elements: {current_scene_dict.get('elements_a_preserver', [])}

PLAYER ACTION CLASSIFICATION:
Category: {cat}
Additional Fact (Chronicle): {notable_deviation}{clock_instruction}

NPCs AVAILABLE: {npcs_summary}
{self._build_structure_instructions()}"""

            final_response = self.narrator.generate_response(user_input, self.history.messages, decision_instruction)

            if roll_info:
                final_response += f"\n\n---\n*🎲 {roll_info} ({roll_result})*"

            try:
                new_sheet = self.sheet_manager.update_sheet(self.character_data, user_input, final_response)
                if new_sheet and isinstance(new_sheet, dict):
                    self.character_data = self._unwrap_character_data(new_sheet)
                    os.makedirs("Memory", exist_ok=True)
                    with open("Memory/character.json", "w", encoding="utf-8") as f:
                        json.dump(self.character_data, f, indent=4, ensure_ascii=False)
                    self.gse.reload()
                    print("[RPGAgent] Character sheet updated.")
            except Exception as e:
                print(f"[RPGAgent] ⚠ Error updating character sheet: {e}")

            self.update_chronicle(user_input, final_response, notable_deviation)

            self.history.add_user_message(user_input)
            self.history.add_ai_message(final_response)
            return final_response

    def start_adventure(self):
        """Launches the world setup pipeline then generates narrative introduction."""
        if not self.setup_world():
            return "Erreur lors de la génération du monde."

        self.log("Generating narrative introduction...")
        intro_start = time.time()
        self.game_state = "ADVENTURE"

        pitch = self.scenario_data.get('pitch', 'Une nouvelle aventure commence.')
        situation = "Le héros se tient prêt."
        starting_scene_id = self.scenario_structure.get("metadata", {}).get("starting_scene")
        if starting_scene_id in self._scenes_by_id:
            scene_init = self._scenes_by_id[starting_scene_id]
            situation = f"Scène initiale : {scene_init.get('title')}. Objectif MJ : {scene_init.get('gm_objective')}."

        setup_context = self.get_scenario_context("intrigue lieux personnages", k=config.RAG_K_SETUP)
        current_context = self.get_current_context()

        intro_instruction = f"""
PLAYER ACTION: L'aventure commence !

TECHNICAL RESULT: Aucun jet requis

SCENARIO CONTEXT (RAG excerpts):
- Pitch: {pitch}
- Starting situation: {situation}
- Additional details: {setup_context}

SCENARIO CONTEXT STRUCTURED (Lookup):
{current_context}
{self._build_structure_instructions()}"""

        intro_response = self.narrator.generate_response(
            "L'aventure commence !", self.history.messages, intro_instruction
        )
        self.log(f"✓ Introduction generated in {time.time() - intro_start:.2f}s.")

        full_response = f"**{self.scenario_data.get('title', 'Aventure')}**\n\n*{pitch}*\n\n{intro_response}"

        self.update_chronicle("L'aventure commence !", full_response)
        self.history.add_ai_message(full_response)
        return full_response

    def update_chronicle(self, user_input, response, notable_deviation=None):
        old_summary = ""
        if self.chronicle_data and isinstance(self.chronicle_data, dict):
            old_summary = self.chronicle_data.get("summary", "")

        new_summary = self.chronicle_agent.update(old_summary, user_input, response, notable_deviation)
        self.chronicle_data = {"summary": new_summary}

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/Chronicle.json", "w", encoding="utf-8") as f:
            json.dump(self.chronicle_data, f, indent=4, ensure_ascii=False)

    def load_character(self):
        """Loads only the character and sets game state to SUMMARY."""
        try:
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.character_data = self._unwrap_character_data(data)
                self.gse.reload()
                self.game_state = "SUMMARY"
                nom = self.character_data.get('name') if self.character_data else "Inconnu"
                print(f"[RPGAgent] Character loaded: {nom}")

            if os.path.exists("Memory/scenario_structure.json"):
                with open("Memory/scenario_structure.json", "r", encoding="utf-8") as f:
                    self.scenario_structure = json.load(f)
                self._build_lookups()
                self.scenario_data = {
                    "title": self.scenario_structure["metadata"]["title"],
                    "pitch": self.scenario_structure["metadata"]["global_pitch"],
                }
                self.npcs_data = self.scenario_structure.get("entities", {}).get("npcs", [])

            if os.path.exists("Memory/progression.json"):
                with open("Memory/progression.json", "r", encoding="utf-8") as f:
                    self.progression = json.load(f)
                self.current_scene_id = self.progression.get("current_scene")

                return True
        except Exception as e:
            print(f"[RPGAgent] Error loading character: {e}")
        return False

    def load_game(self):
        """Loads full game save (PJ + PNJ + Scenario + Chronicle)."""
        try:
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.character_data = self._unwrap_character_data(data)
                self.gse.reload()

            if os.path.exists("Memory/scenario_structure.json"):
                with open("Memory/scenario_structure.json", "r", encoding="utf-8") as f:
                    self.scenario_structure = json.load(f)
                self._build_lookups()
                self.scenario_data = {
                    "title": self.scenario_structure["metadata"]["title"],
                    "pitch": self.scenario_structure["metadata"]["global_pitch"],
                }
                self.npcs_data = self.scenario_structure.get("entities", {}).get("npcs", [])

            if os.path.exists("Memory/progression.json"):
                with open("Memory/progression.json", "r", encoding="utf-8") as f:
                    self.progression = json.load(f)
                self.current_scene_id = self.progression.get("current_scene")

            if os.path.exists("Memory/Chronicle.json"):
                with open("Memory/Chronicle.json", "r", encoding="utf-8") as f:
                    self.chronicle_data = json.load(f)

            if self.character_data and self.scenario_structure and self.progression:
                self.game_state = "ADVENTURE"
                nb_npcs = len(self.npcs_data) if self.npcs_data else 0
                print(f"[RPGAgent] Game loaded — {nb_npcs} NPCs available.")
                return True
            elif self.character_data:
                self.game_state = "SUMMARY"
                print(f"[RPGAgent] Character loaded (incomplete game state).")
                return True

        except Exception as e:
            print(f"[RPGAgent] Error loading game: {e}")

        return False

    def clear_history(self):
        """Resets the game state and conversation history completely."""
        self.history.clear()
        self.game_state = "CREATION"
        self.character_data = None
        self.scenario_data = None
        self._missing_character_fields = None
        self.chronicle_data = None
        self.npcs_data = None
        self.current_scene_id = None
        self.progression = None
        self.scenario_structure = None

        from game_state_engine import GameStateEngine
        self.gse = GameStateEngine()

        for file in ["character.json", "Chronicle.json", "progression.json"]:
            path = os.path.join("Memory", file)
            if os.path.exists(path):
                os.remove(path)
                print(f"[RPGAgent] Deleted: {path}")

    def get_npc(self, npc_id: str) -> dict | None:
        if not self.npcs_data:
            return None
        return next((n for n in self.npcs_data if n.get("id") == npc_id), None)

    def get_npcs_context(self) -> str:
        """Returns a summary of NPCs (WITHOUT secrets) for the Narrator."""
        if not self.npcs_data:
            return "Aucun PNJ disponible."
        lines = []
        for n in self.npcs_data:
            lines.append(
                f"- {n['full_name']} "
                f"| Attitude: {n.get('initial_attitude', 'Inconnue')} "
                f"| Lieu: {n.get('usual_location', '?')}"
            )
        return "\n".join(lines)

    def _build_structure_instructions(self) -> str:
        """Mandatory structure instructions for narrative GM choices."""
        return """
YOUR ROLE: You are the GM. Generate precise instructions for the Narrator based on the context provided above.

MANDATORY STRUCTURE of your response:

1. IMMEDIATE CONSEQUENCE
   What happens concretely following the action. If roll succeeded: clear advantage.
   If roll failed: complication or false lead.
   Reveal only what the character can perceive at this moment.

2. SENSORY PERCEPTIONS
   What the character physically sees, hears, smells, touches, or feels.
   Be precise and concrete — no vague atmosphere.

3. UNKNOWN OR AMBIGUOUS ELEMENTS
   What the character cannot yet determine.

4. NARRATIVE IMPULSE
   Give an active direction to the player: a detail that calls for a reaction.

5. KEY POINTS FOR THE SUMMARY
   The Narrator's summary must only include facts explicitly stated in the narrative text of this same response - never additional information, interpretation, or clue content not yet discovered by the player.
"""
