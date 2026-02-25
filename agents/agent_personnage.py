import random
import re
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from agents.agent_regles import simulate_dice_roll
from agents.models import (
    XPGain, LevelUpCheck, CharacterCreationAnalysis,
    CharacterCreationResponse, CharacterEvolutionResponse,
    CreationGuide
)
from llm_utils import safe_chain_invoke, handle_llm_error
import config

class AgentPersonnage:
    def __init__(self, codex_db=None, intrigue_db=None):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7,
            timeout=20
        )
        self.llm_json = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            timeout=20
        )
        self.codex_db = codex_db
        self.intrigue_db = intrigue_db

    @staticmethod
    def _normalize_character_sheet(sheet):
        """Normalize character sheet shape to keep step progression deterministic."""
        normalized = dict(sheet or {})
        normalized.setdefault("nom", "À définir")
        normalized.setdefault("race", "À définir")
        normalized.setdefault("classe", "À définir")
        normalized.setdefault("stats", {})

        equipement = normalized.get("equipement")
        inventaire = normalized.get("inventaire", [])
        if (not equipement or equipement == "À définir") and inventaire:
            normalized["equipement"] = ", ".join(inventaire)
        elif "equipement" not in normalized:
            normalized["equipement"] = "À définir"

        return normalized

    @staticmethod
    def _is_stats_complete(stats):
        if not isinstance(stats, dict) or not stats:
            return False
        invalid_values = {"À définir", "...", "", None, "Roll 3d6", "Lancer 3d6"}
        valid_stats = [v for v in stats.values() if v not in invalid_values]
        # Most OSR games use 6 stats.
        return len(valid_stats) >= 6

    def _compute_missing_fields(self, sheet, analyst_missing=None):
        """
        Computes missing fields by combining programmatic checks and analyst feedback.
        The character sheet content always takes precedence over analyst hallucinations.
        """
        missing = []
        # Primary fields check
        if sheet.get("nom") in ["À définir", "...", None, ""]:
            missing.append("nom")
        if sheet.get("race") in ["À définir", "...", None, ""]:
            missing.append("race")
        if sheet.get("classe") in ["À définir", "...", None, ""]:
            missing.append("classe")
        if not self._is_stats_complete(sheet.get("stats")):
            missing.append("stats")
        if sheet.get("equipement") in ["À définir", "...", None, ""]:
            missing.append("equipement")

        # We can append extra fields identified by the analyst if they are truly missing in the sheet
        if analyst_missing:
            for field in analyst_missing:
                if field not in missing:
                    # Only add it if it's not already set in the sheet
                    val = sheet.get(field)
                    if val in ["À définir", "...", None, ""]:
                        missing.append(field)

        return missing

    def generer_guide_creation(self):
        """
        Analyzes the CODEX to extract character creation steps and rules.
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
        You are the System Architect. Your goal is to extract character creation rules from the CODEX.

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
            res_dict = safe_chain_invoke(chain, {"context": context_text})
            validated = CreationGuide(**res_dict)
            return validated.model_dump()
        except Exception as e:
            return {
                "steps": ["nom", "race", "classe", "statistiques", "equipement"],
                "rules_summary": {},
                "internal_notes": f"Error during generation: {e}"
            }

    def interagir_creation(self, query, memory, journal=None):
        """
        Main interaction loop for character creation in DISCUSSION MODE using a robust two-pass logic.
        """
        char_sheet = self._normalize_character_sheet(memory.get("personnage", {}))
        journal = journal or []

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

        CODEX CONTEXT (for rules):
        {context}

        CURRENT SHEET:
        {char_sheet}

        HISTORY:
        {history}

        LATEST PLAYER MESSAGE:
        {query}

        TASKS:
        1. DATA EXTRACTION: Extract any NEW information provided by the player in French.
        2. PERSISTENCE: If a value was already set in the CURRENT SHEET, keep it UNLESS the player explicitly changed it.
        3. NO INSTRUCTIONS: Never store things like "Roll 3d6" or "Lancer les dés" as values. Only store final names or numbers.
        4. DICE ROLL AGREEMENT: Set 'player_agreed_to_roll' to true ONLY if the player just said "Yes", "Ok", "Fais-le", "Prêt", etc., in response to a roll proposal or when prompted to start stats/attribute generation.
        5. STATS: If you see dice results in the history (e.g. [MJ] J'ai lancé les dés...), extract the values into the 'stats' field.
        6. MISSING FIELDS: Identify which fields from [nom, race, classe, stats, equipement] are still missing or incomplete.
           'stats' is incomplete if not all required attributes (Force, Intelligence, Sagesse, Dextérité, Constitution, Charisme) are determined.
        7. STATS TO ROLL: If 'stats' is missing and player agreed to roll, list the names of the stats that still need a roll (e.g. ["Force", "Intelligence", ...]).

        Respond ONLY in JSON matching this structure:
        {{
            "updates": {{ "nom": "...", "race": "...", "classe": "...", "stats": {{...}}, "equipement": "..." }},
            "player_agreed_to_roll": boolean,
            "missing_fields": ["field1", "field2", ...],
            "stats_to_roll": ["Stat1", "Stat2", ...],
            "internal_thought": "English explanation"
        }}
        """)

        analysis_chain = analysis_prompt | self.llm_json | JsonOutputParser()
        try:
            analysis_dict = safe_chain_invoke(analysis_chain, {
                "context": context_text,
                "char_sheet": json.dumps(char_sheet),
                "history": json.dumps(filtered_journal),
                "query": query
            })
            analysis = CharacterCreationAnalysis(**analysis_dict)
        except Exception as e:
            analysis = CharacterCreationAnalysis(internal_thought=f"Error: {e}")

        # Apply updates to a local sheet for the next pass
        updated_sheet = char_sheet.copy()
        if analysis.updates:
            for k, v in analysis.updates.items():
                if v not in ["À définir", "...", None, ""]:
                    updated_sheet[k] = v

        updated_sheet = self._normalize_character_sheet(updated_sheet)
        missing_fields = self._compute_missing_fields(updated_sheet, analysis.missing_fields)
        next_step = missing_fields[0] if missing_fields else "termine"

        # Step 2: Handle Dice Rolls
        roll_results = []
        if analysis.player_agreed_to_roll:
            stats_to_gen = analysis.stats_to_roll if analysis.stats_to_roll else (["Jet"] if "stats" in missing_fields else [])
            for stat in stats_to_gen:
                roll_data = simulate_dice_roll("3d6")
                if roll_data:
                    roll_results.append(f"{stat}: {roll_data['texte']}")

        roll_result_str = "\n".join(roll_results) if roll_results else None

        # Step 3: Pass 2 - The DM (Conversational Response)
        dm_prompt = ChatPromptTemplate.from_template("""
        You are the Character Creation DM. You are having a natural DISCUSSION in French with the player.

        UPDATED SHEET (Truth):
        {updated_sheet}

        NEXT STEP TO HANDLE:
        {next_step}

        MISSING FIELDS:
        {missing_fields}

        HISTORY:
        {history}

        CODEX CONTEXT:
        {context}

        DICE ROLL RESULT (If performed): {roll_result}

        MISSION:
        1. DISCUSSION: Acknowledge the player's last message in French.
        2. CONTINUITY: Follow NEXT STEP TO HANDLE. Never ask a question about a field that is not in MISSING FIELDS.
        3. NO REPETITION: Do not ask for information that is ALREADY in the UPDATED SHEET.
        4. OPTIONS: List names of options from CODEX concisely in French.
        5. DICE: If you are at the 'stats' step and DICE ROLL RESULT is None, suggest rolling 3d6 for all missing attributes in French.
        6. COMPLETION: If NEXT STEP TO HANDLE is 'termine', summarize the full character sheet and set creation_terminee to true.
        7. QUESTION: If creation is not complete, always end with a clear question in French.

        Respond ONLY in JSON:
        {{
            "message": "Your response in FRENCH",
            "reflexion": "Internal thought in English",
            "creation_terminee": boolean
        }}
        """)

        dm_chain = dm_prompt | self.llm_json | JsonOutputParser()
        try:
            dm_dict = safe_chain_invoke(dm_chain, {
                "updated_sheet": json.dumps(updated_sheet),
                "history": json.dumps(filtered_journal),
                "context": context_text,
                "roll_result": roll_result_str or "None",
                "next_step": next_step,
                "missing_fields": json.dumps(missing_fields)
            })
            dm_res = CharacterCreationResponse(**dm_dict)
        except Exception as e:
            dm_res = CharacterCreationResponse(
                message=f"Désolé, j'ai eu un petit problème technique : {handle_llm_error(e)}",
                reflexion=f"Error: {e}"
            )

        creation_terminee = dm_res.creation_terminee or not missing_fields

        # Step 4: Final Consolidation
        res = {
            "reflexion": f"Analyst: {analysis.internal_thought} | DM: {dm_res.reflexion}",
            "message": dm_res.message,
            "personnage_updates": analysis.updates,
            "creation_terminee": creation_terminee
        }

        if roll_result_str:
            res["message"] += f"\n\n[MJ] J'ai lancé les dés pour vous :\n{roll_result_str}"

        return res

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
            "xp_gagne": integer,
            "raison": "Reason in French"
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            xp_dict = safe_chain_invoke(chain, {
                "context": context_text,
                "query": action_query,
                "regles": regles_info,
                "narration": narration
            })
            validated = XPGain(**xp_dict)
            return validated.model_dump()
        except Exception as e:
            return {"xp_gagne": 0, "raison": f"Erreur lors du calcul XP : {handle_llm_error(e)}"}

    def verifier_niveau(self, personnage_memory):
        xp = personnage_memory.get("xp", 0)
        niveau = personnage_memory.get("niveau", 1)

        context_docs = self.codex_db.similarity_search("level up progression table XP requirements", k=3) if self.codex_db else []
        context_text = "\n".join([d.page_content for d in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Check for level up based on the rules.

        RULES:
        {context}

        CURRENT: Level {niveau}, XP {xp}

        Return JSON:
        {{
            "passage_niveau": boolean,
            "nouveau_niveau": integer
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            lvl_dict = safe_chain_invoke(chain, {
                "context": context_text,
                "niveau": niveau,
                "xp": xp
            })
            validated = LevelUpCheck(**lvl_dict)
            return validated.model_dump()
        except Exception:
            return {"passage_niveau": False, "nouveau_niveau": niveau}

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
        2. Explicitly confirm changes in French.
        3. List remaining choices clearly in French if any.
        4. Update 'personnage_updates' with bonuses (stats, new skills, etc.).
        5. Set 'evolution_terminee' to true when all bonuses are applied.
        6. Respond in FRENCH to the player.

        Return JSON matching this structure:
        {{
            "reflexion": "English reasoning",
            "message": "Response in FRENCH",
            "personnage_updates": {{...}},
            "evolution_terminee": boolean
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            evol_dict = safe_chain_invoke(chain, {
                "context": context_text,
                "char_sheet": json.dumps(memory.get("personnage", {})),
                "historique": json.dumps(historique),
                "query": query
            })
            validated = CharacterEvolutionResponse(**evol_dict)
            return validated.model_dump()
        except Exception as e:
            return {
                "reflexion": f"Error: {e}",
                "message": f"Désolé, j'ai eu un problème technique pendant l'évolution : {handle_llm_error(e)}",
                "personnage_updates": {},
                "evolution_terminee": False
            }
