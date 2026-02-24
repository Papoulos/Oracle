import random
import re
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from agents.agent_regles import simulate_dice_roll
import config

class AgentPersonnage:
    def __init__(self, codex_db=None, intrigue_db=None):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7
        )
        self.llm_json = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )
        self.codex_db = codex_db
        self.intrigue_db = intrigue_db

    def generer_guide_creation(self):
        """
        Analyzes the CODEX to extract character creation steps and rules.
        Used internally to set up the creation steps.
        """
        queries = [
            "steps for character creation",
            "available races",
            "classes occupations professions",
            "calculating characteristics attributes stats",
            "starting equipment"
        ]
        context_text = ""
        if self.codex_db:
            for q in queries:
                docs = self.codex_db.similarity_search(q, k=3)
                context_text += f"\n\n--- CODEX EXCERPT ON '{q}' ---\n"
                context_text += "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_template("""
        You are the System Architect. Your goal is to extract character creation rules from the CODEX for an internal state machine.

        CODEX DOCUMENTS:
        {context}

        MISSION:
        Extract the structured data needed to guide a player through character creation.

        EXPECTED JSON STRUCTURE:
        {{
            "steps": ["nom", "race", "classe", "statistiques", "equipement", ...],
            "rules_summary": {{
                "races": ["list of race names"],
                "classes": ["list of class names"],
                "stats_method": "e.g., 3d6, point buy...",
                "starting_equipment": "summary"
            }},
            "internal_notes": "Technical summary of rules detected."
        }}

        Respond ONLY in JSON.
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            return chain.invoke({"context": context_text})
        except Exception as e:
            return {
                "steps": ["nom", "race", "classe", "statistiques", "equipement"],
                "rules_summary": {},
                "internal_notes": f"Error during generation: {e}"
            }

    def interagir_creation(self, query, memory, journal=[]):
        """
        Main interaction loop for character creation with a robust two-pass logic.
        """
        pdp = memory.get("personnage", {}).get("points_de_passage", {})
        char_sheet = memory.get("personnage", {})

        # Step 0: Initialization
        if not pdp:
            guide = self.generer_guide_creation()
            pdp = self.extraire_pdp_du_guide(guide)
            char_sheet["points_de_passage"] = pdp

        # Determine current step (the one we are asking about)
        current_step = next((k for k, v in pdp.items() if not v), "fin")

        # Step 1: Search relevant CODEX info for CURRENT step to validate input
        context_docs = self.codex_db.similarity_search(f"character creation {current_step} options rules", k=10) if self.codex_db else []
        current_context = "\n\n".join([doc.page_content for doc in context_docs])

        # Step 2: Pass 1 - Analysis and Rule Extraction
        analysis_prompt = ChatPromptTemplate.from_template("""
        You are the Character Creation Analyst. Your job is to update the character state based on the player's input.

        CURRENT CHECKLIST: {pdp_values}
        CURRENT CHARACTER SHEET: {char_sheet}
        CURRENT STEP: {current_step}
        CODEX CONTEXT: {context}
        PLAYER'S MESSAGE: {query}
        HISTORY: {journal}

        TASKS:
        1. INFO REQUEST: If the player is asking for a list of options, details about a class/race, or says "I don't know", set 'is_info_request' to true.
        2. DATA EXTRACTION: Extract information provided by the player for the CURRENT STEP ({current_step}) or other steps.
        3. VALIDATION: Validate choices against the CODEX CONTEXT.
        4. CHECKLIST:
           - ONLY set a step to true in 'completed_steps' if it's newly completed.
           - NEVER set a step to false if it was already true in the CURRENT CHECKLIST.
        5. DICE ROLL:
           - If {current_step} or the next step requires a dice roll, and the player hasn't agreed yet, set 'ask_for_roll' to the dice format (e.g., "3d6").
           - If the player explicitly agreed ("Oui", "Vas-y", "Fais-le"), set 'perform_roll' to the dice format.

        Respond ONLY in JSON:
        {{
            "detected_updates": {{ "field": "value", ... }},
            "completed_steps": {{ "etape_name": true, ... }},
            "is_info_request": bool,
            "ask_for_roll": "NdM+K" or null,
            "perform_roll": "NdM+K" or null,
            "internal_thought": "..."
        }}
        """)

        analysis_chain = analysis_prompt | self.llm_json | JsonOutputParser()
        # Filter journal to avoid passing massive technical dumps back into context
        filtered_journal = []
        for msg in journal[-5:]:
            if len(msg) < 500: # Skip very long messages (likely old technical dumps)
                filtered_journal.append(msg)

        analysis = analysis_chain.invoke({
            "pdp_values": json.dumps(pdp),
            "char_sheet": json.dumps(char_sheet),
            "current_step": current_step,
            "context": current_context,
            "query": query,
            "journal": json.dumps(filtered_journal)
        })

        # Step 3: Update local state to find the REAL next step
        temp_pdp = pdp.copy()
        temp_pdp.update(analysis.get("completed_steps", {}))
        real_next_step = next((k for k, v in temp_pdp.items() if not v), "fin")

        # Step 4: Handle Dice Rolls
        roll_result = None
        if analysis.get("perform_roll"):
            roll_data = simulate_dice_roll(analysis["perform_roll"])
            if roll_data:
                roll_result = roll_data["texte"]

        # Step 5: Search CODEX info for the NEXT step to provide options
        next_context = ""
        if real_next_step != "fin":
            # Search for ALL options to avoid skipping any
            next_docs = self.codex_db.similarity_search(f"character creation list of all {real_next_step} options available", k=10) if self.codex_db else []
            next_context = "\n\n".join([doc.page_content for doc in next_docs])

        # Step 6: Pass 2 - Response Generation (MJ Persona)
        generation_prompt = ChatPromptTemplate.from_template("""
        You are the Game Master (MJ). Your goal is to guide the player in creating their character.
        Be immersive, encouraging, and EXTREMELY CONCISE.
        You speak FRENCH.

        ANALYSIS:
        - Updates: {updates}
        - Completed: {completed}
        - Dice Roll: {roll_result}
        - Ask for Roll: {ask_for_roll}
        - Is Info Request: {is_info_request}

        NEXT STEP: {next_step}
        CODEX FOR NEXT STEP: {next_context}
        PLAYER_QUERY: {query}

        CRITICAL INSTRUCTIONS:
        0. IGNORE GUIDE: If the CODEX context above contains a "Guide" or "List of steps", IGNORE IT. Only look for specific options for {next_step}.
        1. NO TECHNICAL DUMP: Never list all the steps of creation. Never copy-paste technical tables.
        2. INFO REQUEST: If 'is_info_request' is true, your priority is to provide the list or details the player asked for (using CODEX FOR NEXT STEP). Then ask them to make a choice.
        3. LIST ALL OPTIONS: For {next_step}, you MUST list ALL available names found in the CODEX. Do not get lazy.
        4. NO SPOILERS: Only talk about {next_step}.
        5. WELCOME: If PLAYER_QUERY is "Début de l'aventure", give a 1-sentence welcome.
        6. CONFIRM: Briefly confirm what was saved.
        7. QUESTION: You MUST end your message with a clear, direct question in French.
        8. DICE: If 'ask_for_roll' is set, ask "Voulez-vous que je lance les dés pour vos statistiques ?".

        Respond ONLY in JSON format:
        {{
            "message": "Your response in FRENCH",
            "reflexion": "Internal thought in English",
            "additional_updates": {{ ... }}
        }}
        """)

        generation_chain = generation_prompt | self.llm_json | JsonOutputParser()
        gen_res = generation_chain.invoke({
            "updates": json.dumps(analysis.get("detected_updates", {})),
            "completed": json.dumps(analysis.get("completed_steps", {})),
            "roll_result": roll_result or "None",
            "ask_for_roll": analysis.get("ask_for_roll") or "None",
            "is_info_request": analysis.get("is_info_request", False),
            "next_step": real_next_step,
            "next_context": next_context,
            "query": query
        })

        # Step 7: Consolidate updates
        personnage_updates = analysis.get("detected_updates", {})
        if gen_res.get("additional_updates"):
            personnage_updates.update(gen_res["additional_updates"])

        # Merge checklist updates - ONLY set true, NEVER reset to false
        new_pdp = pdp.copy()
        for k, v in analysis.get("completed_steps", {}).items():
            if v is True:
                new_pdp[k] = True
        personnage_updates["points_de_passage"] = new_pdp

        # Check if finished
        creation_terminee = all(new_pdp.values())

        res = {
            "reflexion": f"Analysis: {analysis.get('internal_thought')} | Generation: {gen_res.get('reflexion')}",
            "message": gen_res.get("message"),
            "personnage_updates": personnage_updates,
            "creation_terminee": creation_terminee
        }

        # Special handling for initialization: ensure the pdp we just created is included in updates
        if not memory.get("personnage", {}).get("points_de_passage"):
            res["personnage_updates"]["points_de_passage"] = new_pdp

        return res

    def extraire_pdp_du_guide(self, guide_data):
        """
        Extracts a checklist of steps from the guide data.
        """
        steps = guide_data.get("steps", ["nom", "race", "classe", "statistiques", "equipement"])
        return {step: False for step in steps}

    def calculer_xp(self, action_query, narration, regles_info, world_info):
        context_query = f"XP reward for action {action_query}"
        codex_docs = self.codex_db.similarity_search(context_query, k=2) if self.codex_db else []
        intrigue_docs = self.intrigue_db.similarity_search(context_query, k=2) if self.intrigue_db else []

        context_text = "CODEX:\n" + "\n".join([d.page_content for d in codex_docs])
        context_text += "\n\nINTRIGUE:\n" + "\n".join([d.page_content for d in intrigue_docs])

        prompt = ChatPromptTemplate.from_template("""
        You are the Character Agent. Determine XP gain for the last action.

        CONTEXT:
        {context}

        ACTION: {query}
        RULES EVALUATION: {regles}
        NARRATION: {narration}

        Return JSON:
        {{
            "xp_gagne": int,
            "raison": "Reason in French"
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({
            "context": context_text,
            "query": action_query,
            "regles": regles_info,
            "narration": narration
        })

    def verifier_niveau(self, personnage_memory):
        xp = personnage_memory.get("xp", 0)
        niveau = personnage_memory.get("niveau", 1)

        context_docs = self.codex_db.similarity_search("level up progression table XP requirements", k=3) if self.codex_db else []
        context_text = "\n".join([d.page_content for d in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Check for level up.

        RULES:
        {context}

        CURRENT: Level {niveau}, XP {xp}

        Return JSON:
        {{
            "passage_niveau": boolean,
            "nouveau_niveau": int
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({
            "context": context_text,
            "niveau": niveau,
            "xp": xp
        })

    def gerer_evolution(self, query, memory, historique=[]):
        context_docs = self.codex_db.similarity_search("level up bonus characteristics skills attributes", k=5) if self.codex_db else []
        context_text = "\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        You are the Character Agent, guiding the player through leveling up.

        HISTORY:
        {historique}

        CHARACTER SHEET:
        {char_sheet}

        PLAYER RESPONSE:
        {query}

        LEVEL UP RULES (CODEX):
        {context}

        INSTRUCTIONS:
        1. Analyze player choices based on CODEX rules.
        2. Explicitly confirm changes.
        3. List remaining choices clearly if any.
        4. Update 'personnage_updates' with bonuses (stats, new skills, etc.).
        5. Set 'evolution_terminee' to true when all bonuses are applied.
        6. Respond in FRENCH to the player.

        Return JSON:
        {{
            "reflexion": "Internal reasoning in English.",
            "message": "Your response to the player in FRENCH.",
            "personnage_updates": {{...}},
            "evolution_terminee": boolean
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        return chain.invoke({
            "context": context_text,
            "char_sheet": json.dumps(memory.get("personnage", {})),
            "historique": json.dumps(historique),
            "query": query
        })
