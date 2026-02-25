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
        Main interaction loop for character creation in DISCUSSION MODE using a robust two-pass logic.
        """
        char_sheet = memory.get("personnage", {})

        # Step 0: Search Context
        search_queries = [
            f"character creation rules {query}",
            "available races and classes list",
            "how to calculate stats attributes 3d6"
        ]
        context_text = ""
        if self.codex_db:
            for q in search_queries:
                docs = self.codex_db.similarity_search(q, k=4)
                context_text += f"\n--- Results for '{q}' ---\n"
                context_text += "\n\n".join([d.page_content for d in docs])

        filtered_journal = [m for m in journal[-6:] if len(m) < 600]

        # Step 1: Pass 1 - The Analyst (Strict Data Extraction)
        analysis_prompt = ChatPromptTemplate.from_template("""
        You are the Character Analyst. Your job is to update the character sheet state based on the conversation.
        Focus on facts. English instructions, JSON output.

        GOALS: We need final values for: nom (name), race, classe, stats (attributes), equipement.

        CURRENT SHEET:
        {char_sheet}

        HISTORY:
        {history}

        LATEST PLAYER MESSAGE:
        {query}

        TASKS:
        1. DATA EXTRACTION: Extract any NEW information provided by the player. Look at the conversation history to see what was just discussed.
        2. PERSISTENCE: If a value was already set in the CURRENT SHEET, keep it UNLESS the player explicitly changed it.
        3. NO INSTRUCTIONS: Never store things like "Roll 3d6" or "Lancer les dés" as values. Only store final names or numbers.
        4. DICE ROLL AGREEMENT: Set 'player_agreed_to_roll' to true ONLY if the player just said "Yes", "Ok", "Fais-le", etc., in response to a roll proposal.
        5. STATS: If you see dice results in the history (e.g. [MJ] J'ai lancé les dés...), extract the values into the 'stats' field if appropriate.

        Respond ONLY in JSON:
        {{
            "updates": {{ "nom": "...", "race": "...", "classe": "...", "stats": {{...}}, "equipement": "..." }},
            "player_agreed_to_roll": bool,
            "internal_thought": "Explain your data extraction logic here."
        }}
        """)

        analysis_chain = analysis_prompt | self.llm_json | JsonOutputParser()
        analysis = analysis_chain.invoke({
            "char_sheet": json.dumps(char_sheet),
            "history": json.dumps(filtered_journal),
            "query": query
        })

        # Apply updates to a local sheet for the next pass
        updated_sheet = char_sheet.copy()
        if analysis.get("updates"):
            for k, v in analysis["updates"].items():
                if v not in ["À définir", "...", None, ""]:
                    updated_sheet[k] = v

        # Step 2: Handle Dice Rolls
        roll_result = None
        if analysis.get("player_agreed_to_roll"):
            # DM logic will decide the format, for now let's assume 3d6 if stats are missing
            roll_data = simulate_dice_roll("3d6")
            if roll_data:
                roll_result = roll_data["texte"]

        # Step 3: Pass 2 - The DM (Conversational Response)
        dm_prompt = ChatPromptTemplate.from_template("""
        You are the Character Creation DM. You are having a natural DISCUSSION in French with the player.

        UPDATED SHEET (Truth):
        {updated_sheet}

        HISTORY:
        {history}

        CODEX CONTEXT:
        {context}

        DICE ROLL RESULT (If performed): {roll_result}

        MISSION:
        1. DISCUSSION: Acknowledge the player's last message.
        2. CONTINUITY: Check the UPDATED SHEET. If a field is already filled (not 'À définir'), move to the next logical step (Race -> Classe -> Stats -> Equipement).
        3. NO REPETITION: Do not ask for information that is ALREADY in the UPDATED SHEET.
        4. OPTIONS: List names of options from CODEX concisely.
        5. DICE: If you are at the 'stats' step and DICE ROLL RESULT is None, suggest rolling 3d6.
        6. QUESTION: Always end with a clear question.

        Respond ONLY in JSON:
        {{
            "message": "Your response in FRENCH",
            "reflexion": "Internal thought in English",
            "creation_terminee": bool
        }}
        """)

        dm_chain = dm_prompt | self.llm_json | JsonOutputParser()
        dm_res = dm_chain.invoke({
            "updated_sheet": json.dumps(updated_sheet),
            "history": json.dumps(filtered_journal),
            "context": context_text,
            "roll_result": roll_result or "None"
        })

        # Step 4: Final Consolidation
        res = {
            "reflexion": f"Analyst: {analysis.get('internal_thought')} | DM: {dm_res.get('reflexion')}",
            "message": dm_res.get("message"),
            "personnage_updates": analysis.get("updates", {}),
            "creation_terminee": dm_res.get("creation_terminee", False)
        }

        if roll_result:
            res["message"] += f"\n\n[MJ] J'ai lancé les dés pour vous : {roll_result}"

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
