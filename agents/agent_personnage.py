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
        Analyzes the CODEX to generate a complete character creation guide.
        Used internally to set up the creation steps.
        """
        queries = [
            "steps for character creation",
            "available races and bonuses",
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
        You are the Lore Keeper (Agent Personnage). Your goal is to extract character creation rules from the CODEX.

        CODEX DOCUMENTS:
        {context}

        MISSION:
        Generate a comprehensive character creation guide. It must be structured, clear, and exhaustive based on the CODEX.

        EXPECTED CONTENT:
        1. STEPS: List steps in order (e.g., 1. Name, 2. Race...).
        2. OPTIONS: For each step (Race, Class, etc.), list ALL options from the CODEX with their specifics.
        3. MECHANICS: Explain how stats are defined (dice rolls, point buy, etc.).
        4. EQUIPMENT: Detail starting items.

        Respond ONLY in JSON format:
        {{
            "reflexion": "Technical summary of your findings in the CODEX.",
            "message": "The full guide in Markdown, written in French for the player."
        }}
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            return chain.invoke({"context": context_text})
        except Exception as e:
            return {
                "reflexion": f"Error during generation: {e}",
                "message": "Je n'ai pas pu compiler le guide de création. Vérifiez que le CODEX est bien indexé."
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
            pdp = self.extraire_pdp_du_guide(guide["message"])
            char_sheet["points_de_passage"] = pdp

        # Determine current step
        prochaine_etape = next((k for k, v in pdp.items() if not v), "fin")

        # Step 1: Search relevant CODEX info
        context_docs = self.codex_db.similarity_search(f"character creation {prochaine_etape} options rules", k=8) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Step 2: Pass 1 - Analysis and Rule Extraction
        analysis_prompt = ChatPromptTemplate.from_template("""
        You are the Character Creation Analyst. Your job is to update the character state based on the player's input.

        CURRENT CHECKLIST: {pdp_values}
        CURRENT CHARACTER SHEET: {char_sheet}
        CURRENT STEP: {prochaine_etape}
        CODEX CONTEXT: {context}
        PLAYER'S MESSAGE: {query}
        HISTORY: {journal}

        TASKS:
        1. Extract any new information provided by the player (name, race, class, etc.).
        2. Validate it against the CODEX.
        3. Check if the current step requires a dice roll (e.g., stats).
        4. If the player agreed to a roll OR if the CODEX says you should do it, specify the dice format (e.g., "3d6").
        5. If a step is completed, mark it as true in the checklist.

        Respond ONLY in JSON:
        {{
            "detected_updates": {{ "field": "value", ... }},
            "completed_steps": {{ "etape_name": true, ... }},
            "needs_dice_roll": "NdM+K" or null,
            "roll_reason": "Why we are rolling" or null,
            "internal_thought": "..."
        }}
        """)

        analysis_chain = analysis_prompt | self.llm_json | JsonOutputParser()
        analysis = analysis_chain.invoke({
            "pdp_values": json.dumps(pdp),
            "char_sheet": json.dumps(char_sheet),
            "prochaine_etape": prochaine_etape,
            "context": context_text,
            "query": query,
            "journal": json.dumps(journal[-5:]) # Last 5 exchanges for context
        })

        # Step 3: Handle Dice Rolls if requested
        roll_result = None
        if analysis.get("needs_dice_roll"):
            roll_data = simulate_dice_roll(analysis["needs_dice_roll"])
            if roll_data:
                roll_result = roll_data["texte"]

        # Step 4: Pass 2 - Response Generation
        generation_prompt = ChatPromptTemplate.from_template("""
        You are the Game Master (MJ). Write an immersive response in FRENCH to the player.

        ANALYSIS RESULTS:
        - Updates detected: {updates}
        - Steps completed: {completed}
        - Dice Roll Result: {roll_result}

        CURRENT CHARACTER SHEET: {char_sheet}
        CURRENT CHECKLIST: {pdp_values}
        CODEX CONTEXT: {context}
        PLAYER'S MESSAGE: {query}

        INSTRUCTIONS:
        1. Confirm what has been saved/updated.
        2. If a dice roll was performed, announce it and explain the results.
        3. Present the options for the NEXT step (from CODEX).
        4. Ask the next question.
        5. Be immersive and concise.
        6. Do not repeat the whole guide.
        7. If everything is done, say goodbye and prepare for adventure.
        8. UPDATES: If you performed a dice roll or made a decision, include it in 'additional_updates' so it can be saved to the character sheet.

        Respond ONLY in JSON:
        {{
            "message": "Your response in FRENCH",
            "reflexion": "Internal reasoning in English",
            "additional_updates": {{ "field": "value", ... }}
        }}
        """)

        generation_chain = generation_prompt | self.llm_json | JsonOutputParser()
        gen_res = generation_chain.invoke({
            "updates": json.dumps(analysis.get("detected_updates", {})),
            "completed": json.dumps(analysis.get("completed_steps", {})),
            "roll_result": roll_result or "None",
            "char_sheet": json.dumps(char_sheet),
            "pdp_values": json.dumps(pdp),
            "context": context_text,
            "query": query
        })

        # Step 5: Consolidate updates
        personnage_updates = analysis.get("detected_updates", {})
        if gen_res.get("additional_updates"):
            personnage_updates.update(gen_res["additional_updates"])

        # Merge checklist updates
        new_pdp = pdp.copy()
        new_pdp.update(analysis.get("completed_steps", {}))
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

    def extraire_pdp_du_guide(self, guide_text):
        """
        Extracts a checklist of steps from the guide text.
        """
        prompt = ChatPromptTemplate.from_template("""
        Analyze this character creation guide and extract the mandatory steps (e.g., name, race, class, attributes, equipment).

        GUIDE:
        {guide}

        Return a JSON object with each step as a key and 'false' as the value.
        Always start with 'nom' (name).
        Be concise with key names (e.g., "nom", "race", "classe", "statistiques", "equipement").

        JSON:
        """)
        chain = prompt | self.llm_json | JsonOutputParser()
        try:
            return chain.invoke({"guide": guide_text})
        except:
            return {"nom": False, "race": False, "classe": False, "statistiques": False, "equipement": False}

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
