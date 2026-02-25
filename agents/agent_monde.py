from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_utils import safe_chain_invoke, handle_llm_error
import config
import json

class AgentMonde:
    def __init__(self, intrigue_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            timeout=20
        )
        self.intrigue_db = intrigue_db

    def consult(self, query, memory_dict):
        # CONTEXTE : Solo Game
        lieu_actuel = memory_dict.get("monde", {}).get("lieu_actuel", "")
        search_query = f"{lieu_actuel} {query}" if lieu_actuel else query

        context_docs = self.intrigue_db.similarity_search(search_query, k=5) if self.intrigue_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        You are the World and Scenario Expert (Agent Monde).
        Your role is to extract necessary elements from the INTRIGUE for the Narrator to describe the scene.
        **CONTEXT: SOLO Game (One player).**

        CURRENT MEMORY:
        {memory}

        INTRIGUE CONTEXT (Scenario, NPCs, Locations):
        {context}

        PLAYER ACTION:
        {query}

        YOUR MISSIONS:
        1. Identify the location, present NPCs, and atmosphere described in the INTRIGUE matching the action or current location.
        2. If the player's action triggers a scenario event, describe it.
        3. Provide concrete details (descriptions, possible dialogues, revealable secrets) to help the Narrator.
        4. If memory is empty, search for introduction or adventure start elements.

        CRITICAL RULES:
        - NEVER say "RAS". Always provide narrative context from the INTRIGUE.
        - DO NOT REVEAL future secrets, only what's relevant HERE and NOW.
        - Respond with raw facts and technical scenario descriptions in French, no narration (the Narrator will handle it).
        - **IMPORTANT**: If no relevant information is found in the INTRIGUE (empty context), explicitly respond in French: "ERREUR : Aucune information trouvée dans le scénario pour ce contexte. Veuillez vérifier l'indexation de l'INTRIGUE." NEVER invent locations or NPCs not in the context.

        RESPONSE (Facts & Scenario details in French):
        """)

        chain = prompt | self.llm | StrOutputParser()
        try:
            return safe_chain_invoke(chain, {
                "memory": json.dumps(memory_dict),
                "context": context_text,
                "query": query
            })
        except Exception as e:
            return f"L'Oracle ne parvient pas à consulter le monde : {handle_llm_error(e)}"

    def chercher_introduction(self):
        # CONTEXTE : Solo Game
        queries = [
            "introduction début aventure scène initiale point de départ",
            "chapitre 1 prologue",
            "personnage joueur commence situation initiale",
            " " # Broad search
        ]

        all_docs = []
        if self.intrigue_db:
            for q in queries:
                all_docs.extend(self.intrigue_db.similarity_search(q, k=5))

        seen = set()
        context_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                context_docs.append(doc)
                seen.add(doc.page_content)

        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        if not context_text:
            return "ERREUR CRITIQUE : Aucune donnée disponible dans la base INTRIGUE. Assurez-vous d'avoir indexé vos fichiers PDF."

        prompt = ChatPromptTemplate.from_template("""
        You are the World and Scenario Expert (Agent Monde).
        Your role is to extract elements from the INTRIGUE to START the adventure.
        **CONTEXT: SOLO Game (One player).**

        INTRIGUE CONTEXT:
        {context}

        YOUR MISSIONS:
        1. Describe the initial situation based on the provided information in French.
        2. Identify the location, atmosphere, and present characters.
        3. Provide all necessary factual details for the Narrator to launch the game.

        CRITICAL RULES:
        - Use the provided context as the absolute source for the adventure start.
        - NEVER say "RAS" or "ERREUR" if context is available. Make the best use of extracted info.
        - DO NOT INVENT anything not suggested by the context.
        - Stay purely factual.

        RESPONSE (Facts & Intro details in French):
        """)

        chain = prompt | self.llm | StrOutputParser()
        try:
            return safe_chain_invoke(chain, {
                "context": context_text
            })
        except Exception as e:
            return f"L'Oracle ne parvient pas à générer l'introduction : {handle_llm_error(e)}"
