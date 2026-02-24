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
        Main interaction loop for character creation in DISCUSSION MODE.
        """
        char_sheet = memory.get("personnage", {})

        # Step 0: Ensure minimal points_de_passage exist
        if not char_sheet.get("points_de_passage"):
            char_sheet["points_de_passage"] = {
                "nom": False, "race": False, "classe": False, "stats": False, "equipement": False
            }

        # 1. Multi-faceted Search for CODEX context
        # We search for rules, but also specifically for races, classes and stats to give the DM a full view
        search_queries = [
            f"character creation rules {query}",
            "available races and classes list",
            "how to calculate stats attributes 3d6"
        ]
        context = ""
        if self.codex_db:
            for q in search_queries:
                docs = self.codex_db.similarity_search(q, k=4)
                context += f"\n--- Results for '{q}' ---\n"
                context += "\n\n".join([d.page_content for d in docs])

        # 2. Discussion Prompt
        prompt = ChatPromptTemplate.from_template("""
        You are the Character Creation DM. Your goal is to build a character through a natural DISCUSSION in French.

        GOALS: We need to define: Nom (Name), Race, Classe, Statistiques (Stats), and Équipement.

        CURRENT CHARACTER SHEET:
        {char_sheet}

        RECENT CONVERSATION HISTORY:
        {history}

        CODEX CONTEXT:
        {context}

        PLAYER'S MESSAGE:
        {query}

        MISSION:
        1. DISCUSSION MODE: Talk like a human DM. Acknowledge the player's input. Suggest options if they are stuck.
        2. CONTINUITY: Look at the HISTORY and SHEET. If a field is already filled, move to the next logical step.
        3. DATA EXTRACTION:
           - Extract ANY valid information (name, race, class, stats) provided by the player into 'personnage_updates'.
           - NEVER store instructions like "Roll 3d6" as values.
        4. DICE ROLLS:
           - If stats are missing, EXPLAIN the CODEX rule (e.g., 3d6) and ASK the player if they want you to roll.
           - If the player explicitly agreed ("Yes", "Ok", "Fais-le"), set 'perform_roll' to the dice format (e.g., "3d6").
        5. OPTIONS: When presenting options for race or class, provide a CONCISE list of names found in the CODEX. Do not list technical progression tables.
        6. QUESTION: You MUST end your message with a direct, clear question in French to move the process forward.
        7. COMPLETION: Set 'creation_terminee' to true ONLY when all main fields (Nom, Race, Classe, Stats, Équipement) are filled.

        Respond ONLY in JSON format:
        {{
            "reflexion": "Internal thought in English about progress and next step.",
            "message": "Your response in FRENCH. Always ends with a question.",
            "personnage_updates": {{ "nom": "...", "race": "...", "classe": "...", "stats": {{...}} }},
            "perform_roll": "NdM+K" or null,
            "creation_terminee": bool
        }}
        """)

        # Filter journal
        filtered_journal = [m for m in journal[-6:] if len(m) < 600]

        chain = prompt | self.llm_json | JsonOutputParser()
        res = chain.invoke({
            "char_sheet": json.dumps(char_sheet),
            "history": json.dumps(filtered_journal),
            "context": context,
            "query": query
        })

        # 3. Automated Dice Rolling
        if res.get("perform_roll"):
            roll_data = simulate_dice_roll(res["perform_roll"])
            if roll_data:
                # Inject the roll result into the message
                res["message"] = f"{res['message']}\n\n[MJ] J'ai lancé les dés pour vous : {roll_data['texte']}"

                # We also add the raw result to reflexion to help the next turn
                res["reflexion"] += f" | Dice Roll Performed: {roll_data['texte']}"

                # We ensure points_de_passage for stats is NOT true yet,
                # because the player might need to assign them or the DM needs to confirm.
                # UNLESS the DM already assigned them in personnage_updates.

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
