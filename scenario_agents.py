import json
import re
import os
import time
import config
from langchain_core.prompts import ChatPromptTemplate
from base_utils import BaseAgent, extract_json, get_full_store_text, get_relevant_context

class ScenarioExtractorAgent(BaseAgent):
    """
    Unified agent to extract the complete scenario structure in 5 passes.
    Produces: Memory/scenario_structure.json
    """

    def __init__(self, scenario_store, verbose=False):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1, verbose=verbose)
        self.scenario_store = scenario_store

    def generate(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[ScenarioExtractor] {msg}")
            else:
                print(f"[ScenarioExtractorAgent] {msg}")

        log("Starting scenario extraction pipeline in 5 passes...")
        start_time = time.time()

        # Pass 1: Entities (npcs, locations)
        entities = self._extract_entities(log)

        # Pass 2: Scene nodes (scene_nodes)
        nodes = self._extract_scene_nodes(entities, log)

        # Pass 3: Acts (acts / macro-structure)
        acts = self._extract_acts(nodes, log)

        # Pass 4: Global clocks (global_clocks)
        clocks = self._extract_global_clocks(log)

        # Pass 5: Metadata (metadata)
        metadata = self._extract_metadata(acts, log)

        # Consolidating the final result
        structure = {
            "metadata": metadata.get("metadata", {}),
            "acts": acts.get("acts", []),
            "global_clocks": clocks.get("global_clocks", []),
            "entities": {
                "npcs": entities.get("npcs", []),
                "locations": entities.get("locations", [])
            },
            "scene_nodes": nodes.get("scene_nodes", [])
        }

        # Ensure Memory directory exists
        os.makedirs("Memory", exist_ok=True)
        with open("Memory/scenario_structure.json", "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=4, ensure_ascii=False)

        total_time = time.time() - start_time
        log(f"Consolidated extraction completed successfully in {total_time:.2f}s.")
        return structure

    def _get_context(self, queries, log, k=15) -> str:
        return get_relevant_context(
            self.scenario_store, queries, log, config.SCENARIO_FULLTEXT_THRESHOLD_CHARS, k=k
        )

    def _extract_entities(self, log) -> dict:
        log("Pass 1: Extracting entities (NPCs and Locations)...")
        queries = [
            "personnages importants, personnages nommés, PNJ, main characters, named NPCs, important figures",
            "lieux de l'aventure, villes, pièces, donjons, locations, regions, places of interest, environments"
        ]
        context = self._get_context(queries, log, k=15)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tabletop role-playing game preparation assistant.
From the following scenario excerpts (in French or English), extract the non-player characters (NPCs) and main locations in FRENCH.
Always write descriptions, atmosphere, and motivations in French.
Do not invent or complete any details not present in the excerpts.

Reply ONLY with a JSON in the following format:
{{
  "npcs": [
    {{
      "id": "NPC_ID_UPPERCASE_NO_ACCENTS (e.g., MAITRE_ELROND)",
      "full_name": "Full name and title in French",
      "usual_location": "LOCATION_ID_UPPERCASE_NO_ACCENTS (e.g., FONDCOMBE)",
      "agenda_and_motivation": "What the NPC is trying to achieve",
      "fears_and_weaknesses": "What makes them give up or flee",
      "initial_attitude": "Initial attitude when met",
      "stats_and_abilities": "Threat level, HP, key attacks or unknown"
    }}
  ],
  "locations": [
    {{
      "id": "LOCATION_ID_UPPERCASE_NO_ACCENTS (e.g., FONDCOMBE)",
      "full_name": "Full name of the location",
      "sensory_atmosphere": "Sight, sound, smell, atmosphere in French",
      "interactive_elements": "Objects, levers, containers, scenery features in French"
    }}
  ]
}}
"""),
            ("human", "SCENARIO EXCERPTS:\n{context}")
        ])

        response = self._invoke_logged(prompt, {"context": context}, label="extract_entities")
        res = extract_json(response.content, expected_type=dict)
        if not res:
            log(f"⚠ JSON parsing failed for extract_entities. Raw response start: {response.content[:300]!r}")
            res = {"npcs": [], "locations": []}
        return res

    def _extract_scene_nodes(self, entities, log) -> dict:
        log("Pass 2: Extracting scene nodes...")
        queries = [
            "déroulement de l'aventure, scènes, chapitres, actes, structure narrative",
            "rencontres, défis, combats, énigmes, pièges, obstacles"
        ]
        context = self._get_context(queries, log, k=15)

        npc_ids = [p["id"] for p in entities.get("npcs", [])]
        location_ids = [l["id"] for l in entities.get("locations", [])]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tabletop role-playing game preparation assistant.
From the following scenario excerpts, extract the list of all scene nodes in FRENCH.
Always write titles, descriptions, and objectives in French.
Do not invent or complete any details not present.

VALID NPC IDs: {npc_ids}
VALID LOCATION IDs: {location_ids}

FIELDS GUIDELINES:
- "act_id": ID of the parent act this scene belongs to (e.g., ACTE_1, ACTE_2).
- "location_id": ID of the associated location. You MUST choose an ID from the VALID LOCATION IDs list above if possible.
- "present_npcs": list of NPC IDs present. You MUST only use IDs from the VALID NPC IDs list above.
- "resolution_condition": Condition that closes this node, formulated as a GOAL reached by any plausible means, regardless of the logical exit methods.
- "logical_exits": for each exit, "destination_scene_id" must correspond to the ID of another scene (e.g., SCENE_02_ROUTE).

Reply ONLY with a JSON in the following format:
{{
  "scene_nodes": [
    {{
      "scene_id": "SCENE_NUMBER_NAME (e.g., SCENE_01_AUBERGE)",
      "act_id": "ACTE_1",
      "location_id": "LOCATION_ID",
      "title": "Title of the scene in French",
      "present_npcs": ["NPC_ID"],
      "gm_objective": "What the GM must convey or make the player feel in French (vibe, not exit condition)",
      "resolution_condition": "Goal that closes this node in French...",
      "local_rules_and_limits": "Physical, magical or behavioral constraints of this node in French",
      "challenges_and_encounters": [
        {{
          "type": "Combat / Enigme / Piège / Obstacle physique",
          "description": "Concrete description of the challenge in French",
          "possible_resolution": "Anticipated logical ways of overcoming the challenge in French"
        }}
      ],
      "logical_exits": [
        {{
          "action_or_direction": "What the player does in French",
          "destination_scene_id": "ID_OF_THE_DESTINATION_SCENE"
        }}
      ]
    }}
  ]
}}
"""),
            ("human", "SCENARIO EXCERPTS:\n{context}")
        ])

        response = self._invoke_logged(prompt, {
            "npc_ids": json.dumps(npc_ids),
            "location_ids": json.dumps(location_ids),
            "context": context
        }, label="extract_scene_nodes")
        res = extract_json(response.content, expected_type=dict)
        if not res or "scene_nodes" not in res:
            log(f"⚠ JSON parsing failed for extract_scene_nodes. Raw response start: {response.content[:300]!r}")
            res = {"scene_nodes": []}

        # Pass 2 Validation
        validated_scenes = []
        for scene in res.get("scene_nodes", []):
            scene_id = scene.get("scene_id")
            if not scene_id:
                continue

            # Location validation
            loc_id = scene.get("location_id")
            if loc_id and loc_id not in location_ids:
                log(f"[Validation] Scene '{scene_id}': location_id '{loc_id}' invalid. Set to null.")
                scene["location_id"] = None

            # NPC validation
            presents = scene.get("present_npcs", [])
            valid_presents = []
            for pid in presents:
                if pid in npc_ids:
                    valid_presents.append(pid)
                else:
                    log(f"[Validation] Scene '{scene_id}': present_npc '{pid}' invalid. Removed.")
            scene["present_npcs"] = valid_presents

            validated_scenes.append(scene)

        res["scene_nodes"] = validated_scenes
        return res

    def _extract_acts(self, nodes, log) -> dict:
        log("Pass 3: Extracting acts/macro-structure...")
        queries = [
            "structure globale, actes, chapitres majeurs, grandes étapes, main plot points, story structure"
        ]
        context = self._get_context(queries, log, k=15)

        scene_ids = [s["scene_id"] for s in nodes.get("scene_nodes", [])]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tabletop role-playing game preparation assistant.
From the following scenario excerpts, structure the story into major ACTS in FRENCH.
Always write titles and conditions in French.
Do not invent or complete any details not present.

VALID SCENE IDs: {scene_ids}

CONSIGNES:
- Each act has a unique "act_id" (e.g., ACTE_1, ACTE_2).
- "included_scenes" must ONLY contain scene IDs from the VALID SCENE IDs list above.

Reply ONLY with a JSON in the following format:
{{
  "acts": [
    {{
      "act_id": "ACTE_1",
      "title": "Title of the act in French",
      "entry_condition": "Event or choice that triggers this act in French",
      "completion_condition": "Strict condition to complete this act and pass to the next in French",
      "included_scenes": ["SCENE_01_ID", "SCENE_02_ID"]
    }}
  ]
}}
"""),
            ("human", "SCENARIO EXCERPTS:\n{context}")
        ])

        response = self._invoke_logged(prompt, {
            "scene_ids": json.dumps(scene_ids),
            "context": context
        }, label="extract_acts")
        res = extract_json(response.content, expected_type=dict)
        if not res or "acts" not in res:
            log(f"⚠ JSON parsing failed for extract_acts. Raw response start: {response.content[:300]!r}")
            res = {"acts": []}

        # Pass 3 Validation
        validated_acts = []
        for act in res.get("acts", []):
            act_id = act.get("act_id")
            if not act_id:
                continue

            # Included scenes validation
            incluses = act.get("included_scenes", [])
            valid_incluses = []
            for sid in incluses:
                if sid in scene_ids:
                    valid_incluses.append(sid)
                else:
                    log(f"[Validation] Act '{act_id}': orphan included_scene '{sid}' removed.")
            act["included_scenes"] = valid_incluses
            validated_acts.append(act)

        res["acts"] = validated_acts

        # Bidirectional check and correction
        acts_dict = {a["act_id"]: a for a in validated_acts}
        for scene in nodes.get("scene_nodes", []):
            scene_id = scene.get("scene_id")
            scene_act_id = scene.get("act_id")

            if scene_act_id in acts_dict:
                target_act = acts_dict[scene_act_id]
                if scene_id not in target_act["included_scenes"]:
                    log(f"[Validation] Bidirectional correction: Adding scene '{scene_id}' to 'included_scenes' of act '{scene_act_id}'.")
                    target_act["included_scenes"].append(scene_id)
            else:
                log(f"[Validation] Warning: Scene '{scene_id}' references non-existent act_id '{scene_act_id}'.")

        return res

    def _extract_global_clocks(self, log) -> dict:
        log("Pass 4: Extracting global clocks...")
        queries = [
            "menaces temporelles, dangers qui progressent, horloges, comptes à rebours, clocks, timers, consequences"
        ]
        context = self._get_context(queries, log, k=10)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tabletop role-playing game preparation assistant.
From the following scenario excerpts, extract the progressive threats (clocks or timers) in FRENCH.
Always write name, trigger, and consequence in French.
Do not invent or complete any details not present.

CONSIGNES:
- "threshold": number of segments to trigger the consequence. If not specified, default to 6.

Reply ONLY with a JSON in the following format:
{{
  "global_clocks": [
    {{
      "name": "Name of the global or temporal threat in French",
      "trigger": "Player action or passing time in French",
      "consequence": "Impact on the world or locked paths in French",
      "threshold": 6
    }}
  ]
}}
"""),
            ("human", "SCENARIO EXCERPTS:\n{context}")
        ])

        response = self._invoke_logged(prompt, {"context": context}, label="extract_global_clocks")
        res = extract_json(response.content, expected_type=dict)
        if not res or "global_clocks" not in res:
            log(f"⚠ JSON parsing failed for extract_global_clocks. Raw response start: {response.content[:300]!r}")
            res = {"global_clocks": []}

        # Threshold validation
        for clock in res.get("global_clocks", []):
            threshold = clock.get("threshold")
            try:
                clock["threshold"] = int(threshold) if threshold is not None else 6
            except ValueError:
                clock["threshold"] = 6

        return res

    def _extract_metadata(self, acts, log) -> dict:
        log("Pass 5: Extracting metadata...")
        queries = [
            "titre de l'aventure, adventure title, name of the module",
            "pitch résumé introduction début, synopsis, adventure hook, background, plot summary"
        ]
        context = self._get_context(queries, log, k=10)

        # Find first scene of first act
        default_starting_scene = None
        if acts.get("acts"):
            first_act = acts["acts"][0]
            if first_act.get("included_scenes"):
                default_starting_scene = first_act["included_scenes"][0]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tabletop role-playing game preparation assistant.
From the following scenario excerpts, extract the title, the starting pitch, and the starting scene of the adventure in FRENCH.
Always write title and global_pitch in French.
Do not invent or complete any details not present.

DEFAULT STARTING SCENE: "{default_starting_scene}"

CONSIGNES:
- "global_pitch": Vibe/summary of the adventure in 2 sentences in French with full proper names.
- "starting_scene": ID of the opening scene. If not explicitly specified, use the default "{default_starting_scene}".

Reply ONLY with a JSON in the following format:
{{
  "metadata": {{
    "title": "Name of the scenario in French",
    "global_pitch": "Vibe/summary of the adventure in 2 sentences in French with full proper names.",
    "starting_scene": "SCENE_NUMBER_NAME"
  }}
}}
"""),
            ("human", "SCENARIO EXCERPTS:\n{context}")
        ])

        response = self._invoke_logged(prompt, {
            "default_starting_scene": default_starting_scene or "",
            "context": context
        }, label="extract_metadata")
        res = extract_json(response.content, expected_type=dict)
        if not res or "metadata" not in res:
            log(f"⚠ JSON parsing failed for extract_metadata. Raw response start: {response.content[:300]!r}")
            res = {"metadata": {}}

        # Metadata validation
        meta = res.get("metadata", {})
        if not meta.get("title"):
            meta["title"] = "Inconnu"
        if not meta.get("global_pitch"):
            meta["global_pitch"] = "Inconnu"
        if not meta.get("starting_scene"):
            meta["starting_scene"] = default_starting_scene or "Inconnu"

        res["metadata"] = meta
        return res


SCHEMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in tabletop role-playing game systems design.
Based on the provided rule excerpts, produce a SCHEMA describing the fields
a character sheet MUST contain to be considered complete in THIS specific game system.

CRITICAL INSTRUCTIONS:
1. Do not invent any field not mentioned in the provided rules.
2. Use consistent, lowercase, unaccented technical keys reflecting the name given BY THIS SYSTEM to each resource (e.g., "hit_points" if the game has it, but "vigor"/"celerity"/"intellect" for multiple pool systems, or "sanity" if a separate gauge exists). Never use "hit_points" by default if it is not the actual name of the resource in this system.
3. A field of type "object" (such as attributes, resources...) must list its expected sub_fields with the exact names used by THIS system (which can be very different across systems: "Strength/Dexterity" or "STR/DEX/POW", etc.).
4. A field of type "list" must specify whether an empty list is acceptable ("non_empty": false) or not ("non_empty": true).
5. Include any vitality resource mentioned in the rules, even if there are several (e.g., hit points AND sanity, or wound boxes AND stress).
6. Do NOT include purely narrative fields (backstory, appearance, player name) unless the rules make them strictly necessary to play.

Reply ONLY with a JSON of this form, enclosed in ```json tags:
```json
{{
  "required_fields": [
    {{"path": "name", "type": "string"}},
    {{"path": "statistics", "type": "object", "sub_fields": ["...exact system names..."]}},
    {{"path": "resources.<resource_name>", "type": "object", "sub_fields": ["current", "max"]}},
    {{"path": "equipment", "type": "list", "non_empty": true}}
  ]
}}
```"""),
    ("human", "CODEX EXCERPTS (Rules):\n{context}"),
])

DISCOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in analyzing tabletop role-playing game systems.
Based on the rule excerpts below, produce an EXHAUSTIVE list of all concrete character
creation steps and procedures described by THIS text, at the SAME level of granularity
as the source. If the source enumerates detailed numbered steps (e.g., 14 distinct steps),
list them separately - NEVER group several numbered steps from the source into a single
generic category like "method A" / "method B".
If the system offers multiple methods or paths for creation, list each step of each method separately.

Do not invent any steps not in the text. Do not omit any steps present, even minor ones
(e.g., "write down attack values", "choose an alignment", "buy equipment" are separate steps if the text mentions them separately).

Use EXCLUSIVELY the vocabulary unique to this system - never generic synonyms like "race/class" if the system does not use them.

Reply ONLY with JSON in this form:
```json
{{
  "components": ["exact step/procedure 1", "exact step/procedure 2", "..."]
}}
```"""),
    ("human", "CODEX EXCERPTS (Rules):\n{context}"),
])


class ManualGeneratorAgent(BaseAgent):
    """
    One-shot agent: extracts character creation steps from the Core RAG.
    Produces: Memory/creation_manual.json
    """

    def __init__(self, core_store, verbose=False):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1, verbose=verbose)
        self.core_store = core_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in tabletop role-playing game systems design.
Your mission is to write a structured CHARACTER CREATION MANUAL in FRENCH, based ONLY on the provided rule excerpts.

This manual will serve as a master guide for another AI agent that will accompany the player during character creation.
It must be COMPLETE on all steps required by the system, but remain PURELY STRUCTURAL.

CRITICAL INSTRUCTIONS:
1. List ALL creation steps in the logical order required by THIS system, at the SAME granularity as the provided source text - if the source enumerates detailed numbered steps, reproduce this same fine breakdown, NEVER summarize several numbered steps from the source into a single generic step. Use EXCLUSIVELY the terminology specific to this system. NEVER invent a concept absent from the provided rules (e.g., do not mention traditional races or profiles if the system does not have them). On the other hand, if a concept does exist in the rules (attributes, hit points, armor/protection, or any other mechanics), it MUST be detailed completely and faithfully - this agnosticism guideline never justifies reducing the level of detail on mechanics that actually exist in this system.
2. For each step, give a description of the procedure in French.
3. DO NOT list individual specific options of THIS system (e.g., for a system with professions, do not list specific jobs). Simply indicate that a choice must be made in each identified category, whatever it may be in this specific system.
4. The final agent will use RAG to find lists of options. Your role is to tell it WHEN and HOW to make choices.
5. EXPLANATION OF RULES: Specify for each step if numerical limits apply (e.g., "Choose 2 skills", "Choose 1 melee weapon and 1 ranged weapon") in French so the agent can explain them to the player.
6. Clearly indicate any calculation methods mentioned (e.g., "Roll 3d6", "Distribute 15 points").

Reply ONLY with a JSON block enclosed in ```json tags.

EXPECTED JSON FORMAT:
```json
{{
  "steps": [
    {{
      "step": 1,
      "name": "Step 1 name in French",
      "description": "Procedure description in French"
    }}
  ],
  "general_rules": "Global notes in French (e.g., importance of checking prerequisites before choosing equipment)"
}}
```"""),
            ("human", "CODEX EXCERPTS (Rules):\n{context}"),
        ])

    def generate(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[ManualGenerator] {msg}")
            else:
                print(f"[ManualGeneratorAgent] {msg}")

        log("Extracting character creation steps...")
        start_time = time.time()

        full_core_text = get_full_store_text(self.core_store, log)

        if full_core_text and len(full_core_text) <= config.CORE_FULLTEXT_THRESHOLD_CHARS:
            log("Core source text fits in context, direct use (no discovery needed).")
            deduplicated_context = full_core_text
        else:
            log("Core too large for context - discovering system components...")
            discovery_context = get_relevant_context(
                self.core_store,
                [
                    "création de personnage, character creation, comment créer un personnage",
                    "construire un personnage, personnage joueur, feuille de personnage",
                ],
                log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )

            components = []
            if discovery_context.strip():
                try:
                    discovery_response = self._invoke_logged(DISCOVERY_PROMPT, {"context": discovery_context}, label="discovery_manual")
                    discovery_result = extract_json(discovery_response.content, expected_type=dict)
                    components = discovery_result.get("components", []) if discovery_result else []
                except Exception as e:
                    log(f"⚠ Error during components discovery: {e}")

            if components:
                log(f"Discovered components: {components}")
                queries = [f"{c}, création de personnage" for c in components]
                if len(components) < config.MIN_COMPOSANTES_DECOUVERTES:
                    log(f"⚠ Only {len(components)} component(s) discovered - adding fallback generic queries for completeness.")
                    queries += [
                        "étapes numérotées de création de personnage, procédure complète",
                        "caractéristiques, points de vie, équipement, capacités de classe",
                    ]
            else:
                log("⚠ No components discovered - falling back to generic queries.")
                queries = [
                    "création de personnage, caractéristiques, capacités spéciales",
                    "équipement, ressources de départ, progression du personnage",
                ]

            deduplicated_context = get_relevant_context(
                self.core_store, queries, log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )

        rag_time = time.time() - start_time
        log(f"Context retrieval completed in {rag_time:.2f}s.")

        if not deduplicated_context.strip():
            log("⚠ No excerpts found in Core RAG. Manual will be empty.")
            return {}

        llm_start = time.time()
        try:
            response = self._invoke_logged(self.prompt, {"context": deduplicated_context}, label="generate_manual")
            content = response.content
        except Exception as e:
            log(f"✗ Error during LLM call for manual generation: {e}")
            return {}
        llm_time = time.time() - llm_start
        manual = extract_json(content, expected_type=dict)
        log(f"LLM finished in {llm_time:.2f}s.")

        if not manual:
            log(f"⚠ JSON parsing failed for generate_manual. Raw response start: {content[:300]!r}")
            return {}

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/creation_manual.json", "w", encoding="utf-8") as f:
            json.dump(manual, f, indent=4, ensure_ascii=False)

        log("✓ Character creation manual generated in Memory/creation_manual.json.")

        schema_response = None
        try:
            schema_response = self._invoke_logged(SCHEMA_PROMPT, {"context": deduplicated_context}, label="generate_schema")
            schema = extract_json(schema_response.content, expected_type=dict)
        except Exception as e:
            log(f"✗ Error during character sheet schema generation: {e}")
            schema = None

        if not schema or not schema.get("required_fields"):
            if schema_response:
                log(f"⚠ JSON parsing failed for generate_schema. Raw response start: {schema_response.content[:300]!r}")
            schema = schema or {"required_fields": []}

        with open("Memory/character_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=4, ensure_ascii=False)
        log("✓ Character sheet schema generated in Memory/character_schema.json.")

        if len(manual.get("steps", [])) < 4:
            log(f"⚠ The generated manual only contains {len(manual.get('steps', []))} step(s) - possible detail loss, check Memory/creation_manual.json manually.")

        return manual


class SceneGraphAgent(BaseAgent):
    """
    Scene Graph Agent: extracts scenes, plot structure, and generates a scene graph.
    Produces: Memory/scenes.json
    """

    def __init__(self, scenario_store, verbose=False):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1, verbose=verbose)
        self.scenario_store = scenario_store

    def generate(self, scenario_summary: dict, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[SceneGraph] {msg}")
            else:
                print(f"[SceneGraphAgent] {msg}")

        log("Extracting scenes and scenario structure...")
        start_time = time.time()

        queries = [
            "déroulement de l'aventure, scènes, chapitres, actes, structure narrative",
            "lieux, salles, zones, rencontres",
            "conditions, ce qui se passe si, déclencheurs, réactions des PNJ"
        ]

        all_docs = []
        for query in queries:
            docs = self.scenario_store.similarity_search(query, k=15)
            all_docs.extend(docs)

        unique_contents = {doc.page_content: doc for doc in all_docs}
        deduplicated_context = "\n\n---\n\n".join(unique_contents.keys())
        rag_time = time.time() - start_time
        log(f"RAG completed in {rag_time:.2f}s ({len(unique_contents)} excerpts).")

        if not deduplicated_context.strip():
            log("⚠ No excerpt found in scenario to extract scenes.")
            return {}

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tabletop role-playing game preparation assistant.
Based on these scenario excerpts (which may be in French or English), produce a logical scene structure in FRENCH.
Always write titles, atmosphere, and objectives in French.
Do not invent or complete any details not present in the excerpts.

Reply ONLY with a valid JSON exactly following this schema:
{{
  "scene_initiale": "1.1",
  "scenes": [
    {{
      "id": "1.1",
      "titre": "string in French",
      "lieu": "string in French",
      "pnjs": ["npc_id"],
      "esprit_de_la_scene": "what this scene must bring to the plot, in one sentence in French",
      "elements_a_preserver": ["facts or info that must remain true even if the scene unfolds differently, in French"],
      "reactions_anticipees": [
        {{"action_probable": "string in French", "consequence": "string in French"}}
      ],
      "objectif_atteint_si": "condition formulated as a GOAL in French, not a literal action",
      "statut": "a_venir"
    }}
  ]
}}
"""),
            ("human", "SCENARIO EXCERPTS:\n{context}")
        ])

        llm_start = time.time()
        response = self._invoke_logged(prompt, {"context": deduplicated_context}, label="generate_scenegraph")
        llm_time = time.time() - llm_start
        scenes_data = extract_json(response.content, expected_type=dict)
        log(f"LLM completed in {llm_time:.2f}s.")

        if not scenes_data:
            log(f"⚠ JSON parsing failed for generate_scenegraph. Raw response start: {response.content[:300]!r}")
            return {}

        # Minimal validation
        if "scene_initiale" not in scenes_data:
            scenes_data["scene_initiale"] = "1.1"
        if "scenes" not in scenes_data or not isinstance(scenes_data["scenes"], list):
            scenes_data["scenes"] = []

        # Ensure starting scene has status 'en_cours' and others 'a_venir'
        initial_id = scenes_data["scene_initiale"]
        for scene in scenes_data["scenes"]:
            if scene.get("id") == initial_id:
                scene["statut"] = "en_cours"
            else:
                scene["statut"] = "a_venir"

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/scenes.json", "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, indent=4, ensure_ascii=False)

        log(f"✓ Scene graph generated ({len(scenes_data['scenes'])} scenes) in Memory/scenes.json.")
        return scenes_data


DISCOVERY_RECOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in tabletop role-playing game systems.
Based on the rule excerpts below, identify the RESOURCE RECOVERY MECHANISMS
(rests, downtime, healing between encounters) specific to THIS system,
with their EXACT names. Some systems have two tiers (short/long rest), others
have several different tiers (e.g., recovery at 10 minutes / 1 hour / 10 hours /
24 hours), others have almost no automatic recovery (slow healing over several days for example).
Never assume a short/long model by default.

Reply ONLY with:
```json
{{"tiers": ["exact name of tier 1", "exact name of tier 2"]}}
```
Empty list if the system has no formalized recovery mechanism."""),
    ("human", "CODEX EXCERPTS (Rules):\n{context}"),
])

EXTRACTION_RECOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """For EACH recovery tier listed, specify its trigger and the resources it restores.
Use ONLY these three types of effects:
- "restore_full": the resource returns to its maximum.
- "restore_percentage": increases by a percentage of the maximum ("value": integer 1-100).
- "restore_fixed_value": increases by a fixed value ("value": integer). If the rules indicate a dice roll, use a reasonable estimate of its average.

The "resource" field of each effect must be a consistent, lowercase, unaccented technical name reflecting the resource as described in the rules. If a list of already known resources is provided below and is not empty, prioritize those exact names for consistency; otherwise, choose a clear and consistent name from the rules themselves.

ALREADY KNOWN RESOURCES (may be empty - in this case, ignore this section): {known_resources}

TIERS TO DETAIL: {tiers}

Reply ONLY with:
```json
{{
  "recovery_tiers": [
    {{
      "id": "short_identifier_no_spaces",
      "name": "exact name of the tier",
      "text_triggers": ["variant 1", "variant 2"],
      "effects": [{{"resource": "exact_resource_name", "action": "restore_full" | "restore_percentage" | "restore_fixed_value", "value": null_or_integer}}]
    }}
  ]
}}
```"""),
    ("human", "CODEX EXCERPTS (Rules):\n{context}"),
])

DISCOVERY_ACTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Identify the COMMON ACTIONS a character can undertake in the game (combat, stealth, persuasion, perception, etc.) and which require resolution by the rules (dice roll, comparison to a threshold). Use the EXACT names of this system.

Reply ONLY with:
```json
{{"actions": ["exact name of action 1", "exact name of action 2"]}}
```"""),
    ("human", "CODEX EXCERPTS (Rules):\n{context}"),
])

EXTRACTION_ACTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """For EACH action listed, specify how it is resolved in this system: which roll, which attribute/skill, against what, and the consequences of success/failure.

ACTIONS TO DETAIL: {actions}

Reply ONLY with:
```json
{{
  "common_actions": [
    {{
      "name": "exact name of the action",
      "triggers": ["variant 1", "variant 2"],
      "resolution": "description of the resolution mechanics",
      "on_success": "description of success consequences",
      "on_failure": "description of failure consequences"
    }}
  ]
}}
```"""),
    ("human", "CODEX EXCERPTS (Rules):\n{context}"),
])


class GameplayRulesAgent(BaseAgent):
    def __init__(self, core_store, verbose=False):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1, verbose=verbose)
        self.core_store = core_store

    def _load_character_schema(self) -> dict:
        path = "Memory/character_schema.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"required_fields": []}

    def generate_recovery_rules(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[GameplayRules] {msg}")
            else:
                print(f"[GameplayRulesAgent] {msg}")

        character_schema = self._load_character_schema()

        full_core = get_full_store_text(self.core_store, log)
        if full_core and len(full_core) <= config.CORE_FULLTEXT_THRESHOLD_CHARS:
            discovery_context = full_core
        else:
            discovery_context = get_relevant_context(
                self.core_store,
                ["repos, récupération, guérison entre les rencontres, downtime, recovery, healing"],
                log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )

        tiers = []
        if discovery_context.strip():
            try:
                resp = self._invoke_logged(DISCOVERY_RECOVERY_PROMPT, {"context": discovery_context}, label="discovery_recovery_tiers")
                result = extract_json(resp.content, expected_type=dict)
                tiers = result.get("tiers", []) if result else []
            except Exception as e:
                log(f"⚠ Error discovering recovery tiers: {e}")

        if not tiers:
            log("No recovery tiers detected - the system might not have formalized recovery (e.g., slow natural healing).")
            recovery_rules = {"recovery_tiers": []}
        else:
            log(f"Discovered recovery tiers: {tiers}")
            queries = [f"{p}, récupération, repos" for p in tiers]
            context_extraction = get_relevant_context(
                self.core_store, queries, log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )
            known_resources = [
                f["path"] for f in character_schema.get("required_fields", [])
                if f["path"].startswith("resources.")
            ]
            try:
                resp = self._invoke_logged(EXTRACTION_RECOVERY_PROMPT, {
                    "tiers": json.dumps(tiers, ensure_ascii=False),
                    "known_resources": json.dumps(known_resources, ensure_ascii=False),
                    "context": context_extraction,
                }, label="extraction_recovery")
                recovery_rules = extract_json(resp.content, expected_type=dict)
                if not recovery_rules:
                    log(f"⚠ JSON parsing failed for extraction_recovery. Raw response start: {resp.content[:300]!r}")
                    recovery_rules = {"recovery_tiers": []}
            except Exception as e:
                log(f"⚠ Error extracting recovery rules: {e}")
                recovery_rules = {"recovery_tiers": []}

        with open("Memory/recovery_rules.json", "w", encoding="utf-8") as f:
            json.dump(recovery_rules, f, indent=4, ensure_ascii=False)
        log(f"✓ {len(recovery_rules.get('recovery_tiers', []))} recovery tier(s) saved.")
        return recovery_rules

    def generate_action_catalog(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[GameplayRules] {msg}")
            else:
                print(f"[GameplayRulesAgent] {msg}")

        full_core = get_full_store_text(self.core_store, log)
        if full_core and len(full_core) <= config.CORE_FULLTEXT_THRESHOLD_CHARS:
            discovery_context = full_core
        else:
            discovery_context = get_relevant_context(
                self.core_store,
                ["actions de combat, tests de compétence, résolution d'action, combat actions, skill checks"],
                log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )

        actions = []
        if discovery_context.strip():
            try:
                resp = self._invoke_logged(DISCOVERY_ACTIONS_PROMPT, {"context": discovery_context}, label="discovery_actions")
                result = extract_json(resp.content, expected_type=dict)
                actions = result.get("actions", []) if result else []
            except Exception as e:
                log(f"⚠ Error discovering common actions: {e}")

        if not actions:
            log("⚠ No common actions detected - the catalog will remain empty, RAG will be used systematically.")
            action_catalog = {"common_actions": []}
        else:
            log(f"Discovered actions: {actions}")
            queries = [f"{a}, résolution, jet de dé" for a in actions]
            context_extraction = get_relevant_context(
                self.core_store, queries, log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )
            try:
                resp = self._invoke_logged(EXTRACTION_ACTIONS_PROMPT, {
                    "actions": json.dumps(actions, ensure_ascii=False),
                    "context": context_extraction,
                }, label="extraction_actions")
                action_catalog = extract_json(resp.content, expected_type=dict)
                if not action_catalog:
                    log(f"⚠ JSON parsing failed for extraction_actions. Raw response start: {resp.content[:300]!r}")
                    action_catalog = {"common_actions": []}
            except Exception as e:
                log(f"⚠ Error extracting action catalog: {e}")
                action_catalog = {"common_actions": []}

        with open("Memory/action_catalog.json", "w", encoding="utf-8") as f:
            json.dump(action_catalog, f, indent=4, ensure_ascii=False)
        log(f"✓ {len(action_catalog.get('common_actions', []))} common action(s) saved.")
        return action_catalog
