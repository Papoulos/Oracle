import random
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

def simulate_dice_roll(dice_expr):
    # Simple regex for NdM (+/-) K
    match = re.match(r"(\d+)d(\d+)([+-]\d+)?", dice_expr.replace(" ", ""))
    if match:
        n = int(match.group(1))
        m = int(match.group(2))
        mod = int(match.group(3)) if match.group(3) else 0
        rolls = [random.randint(1, m) for _ in range(n)]
        total = sum(rolls) + mod
        return f"Jet: {n}d{m}{mod:+} = ({'+'.join(map(str, rolls))}){mod:+} = {total}"
    return None

class AgentRegles:
    def __init__(self, codex_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )
        self.codex_db = codex_db

    def consult(self, query, char_sheet):
        context_docs = self.codex_db.similarity_search(query, k=3) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert des Règles (Agent Règles).
        Ton rôle est de vérifier si l'action du joueur est possible selon les règles du CODEX et de déterminer les jets de dés nécessaires.

        FICHE PERSONNAGE:
        {char_sheet}

        CONTEXTE DU CODEX:
        {context}

        ACTION DU JOUEUR:
        {query}

        INSTRUCTIONS:
        1. Analyse si l'action est possible.
        2. Si un jet de dés est nécessaire, indique le format exact "JET: NdM+K" (ex: JET: 1d20+2).
        3. Explique brièvement la règle appliquée.
        4. Sois concis et technique. Ne fais pas de narration.
        5. S'il s'agit d'un début de partie ou d'une action purement narrative sans enjeu de règle, indique simplement "Aucune règle spécifique applicable."

        RÉPONSE:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "char_sheet": char_sheet,
            "context": context_text,
            "query": query
        })

        # Extraire et simuler le jet de dés si présent
        dice_match = re.search(r"JET:\s*(\d+d\d+[+-]?\d*)", response)
        roll_result = ""
        if dice_match:
            roll_expr = dice_match.group(1)
            roll_result = simulate_dice_roll(roll_expr)
            response += f"\n\n[RESULTAT DE DÉ] {roll_result}"

        return response
