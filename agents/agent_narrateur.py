from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_utils import safe_chain_invoke, handle_llm_error
import config

class AgentNarrateur:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.8,
            timeout=20
        )

    def narrate(self, query, rules_info, world_info, garde_info, memory):
        prompt = ChatPromptTemplate.from_template("""
        You are the GM Narrator (MJ Narrateur). Your role is to describe the resolution of the player's action and play the NPCs.
        You transform technical information from other agents into an immersive narration in French.
        **CONTEXT: This is a SOLO roleplaying game. There is only one player.**

        PLAYER ACTION:
        {query}

        GUARD VALIDATION (Agent Garde):
        {garde_info}

        RULES INFO (Agent Règles):
        {rules_info}

        WORLD/SCENARIO INFO (Agent Monde):
        {world_info}

        GAME MEMORY:
        {memory}

        INSTRUCTIONS:
        - Focus on resolving the current action.
        - **IMPORTANT: Player Autonomy**: NEVER make the player character (PC) act. Do not describe their thoughts, feelings, movements, or words autonomously.
          - Forbidden: "You advance cautiously", "You think that...", "You then say...".
          - Allowed: Describe the result of the action requested by the player.
        - If Agent Monde signals an "ERREUR: Aucune information trouvée", do not invent anything. Explain to the player in French (staying in character) that the GM needs the scenario to be correctly loaded/indexed to continue.
        - If Agent Garde indicates the action is IMPOSSIBLE, explain it narratively and immersively in French (without saying "Agent Garde said"). Tell why the action fails or is blocked.
        - If the action is possible, use RULES and WORLD INFO to build your narrative in French.
        - Do not unnecessarily redescribe the location if the player is already there, unless it changed.
        - Incorporate rule test results (success/failure) fluidly.
        - Make NPCs speak if necessary.
        - Present a choice or ask for an action at the end.
        - Stay faithful to the atmosphere and plot.

        NARRATE THE RESPONSE TO THE PLAYER IN FRENCH:
        """)

        chain = prompt | self.llm | StrOutputParser()
        try:
            return safe_chain_invoke(chain, {
                "query": query,
                "rules_info": rules_info,
                "world_info": world_info,
                "garde_info": garde_info,
                "memory": memory
            })
        except Exception as e:
            return f"Le MJ Narrateur est momentanément aphone : {handle_llm_error(e)}"

    def narrer_introduction(self, world_info):
        prompt = ChatPromptTemplate.from_template("""
        You are the GM Narrator (MJ Narrateur). Your role is to introduce the adventure to the player.
        You must describe the initial scene based ONLY on info from Agent Monde.
        **CONTEXT: SOLO game. One player.**

        SCENARIO INFO (Agent Monde):
        {world_info}

        INSTRUCTIONS:
        1. Briefly introduce yourself as the GM in French.
        2. Describe the setting, atmosphere, and NPCs present according to the scenario in French.
        3. **IMPORTANT: PC Immobile**: DO NOT make the PC act. Do not describe their thoughts, feelings, movements, or starting position actively. The PC should be a "camera" discovering the scene. The player must be totally free for their first action.
        4. If Agent Monde signals an ERROR (no intro found), explain to the player in French (staying in character) that fate is still blurry because the scenario is not ready.
        5. End with an open question inviting the player to act.

        NARRATE THE INTRODUCTION IN FRENCH:
        """)

        chain = prompt | self.llm | StrOutputParser()
        try:
            return safe_chain_invoke(chain, {
                "world_info": world_info
            })
        except Exception as e:
            return f"Le MJ n'arrive pas à lancer l'histoire : {handle_llm_error(e)}"
