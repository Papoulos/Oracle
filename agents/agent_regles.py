import random
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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
        return {
            "expression": dice_expr,
            "details": f"({'+'.join(map(str, rolls))}){mod:+}",
            "total": total,
            "texte": f"Jet: {n}d{m}{mod:+} = ({'+'.join(map(str, rolls))}){mod:+} = {total}"
        }
    return None

class AgentRegles:
    def __init__(self, codex_db):
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0
        )
        self.llm_json = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )
        self.codex_db = codex_db

    def evaluer_besoin_jet(self, query, char_sheet, world_info):
        context_docs = self.codex_db.similarity_search(query, k=3) if self.codex_db else []
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert des Règles (Agent Règles).
        Ton rôle est de déterminer si l'action du joueur nécessite un jet de dés selon le CODEX.

        FICHE PERSONNAGE:
        {char_sheet}

        INFOS DU MONDE (Agent Monde):
        {world_info}

        CONTEXTE DU CODEX:
        {context}

        ACTION DU JOUEUR:
        {query}

        INSTRUCTIONS:
        1. Analyse si l'action nécessite un test de compétence ou une résolution par les dés.
        2. Si l'Agent Monde a signalé une impossibilité majeure, aucun jet n'est nécessaire.
        3. Réponds au format JSON avec les champs suivants:
           - "besoin_jet": boolean
           - "jet_format": "NdM+K" (ex: "1d20+2") ou null si aucun jet
           - "explication_regle": courte explication de la règle appliquée
           - "seuil": le score à atteindre ou la difficulté (si applicable)

        JSON:
        """)

        chain = prompt | self.llm_json | JsonOutputParser()
        response = chain.invoke({
            "char_sheet": char_sheet,
            "world_info": world_info,
            "context": context_text,
            "query": query
        })
        return response, context_text

    def interpreter_reussite(self, query, roll_result, explication_regle, context_text):
        prompt = ChatPromptTemplate.from_template("""
        Tu es l'Expert des Règles (Agent Règles).
        Tu dois déterminer si l'action est une réussite en fonction du résultat du dé et des règles.

        ACTION: {query}
        RÈGLE APPLIQUÉE: {explication_regle}
        CONTEXTE CODEX: {context}
        RÉSULTAT DU JET: {roll_result}

        INSTRUCTIONS:
        1. Compare le résultat du jet aux règles du CODEX.
        2. Déclare si c'est une RÉUSSITE ou un ÉCHEC.
        3. Explique brièvement les conséquences techniques.
        4. Sois concis et technique. Pas de narration.

        RÉPONSE:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "explication_regle": explication_regle,
            "context": context_text,
            "roll_result": roll_result
        })
        return response

    def consult(self, query, char_sheet, world_info="RAS"):
        # Version monolithique pour compatibilité si besoin, mais on va préférer l'usage séparé dans l'orchestrateur
        analyse, context_text = self.evaluer_besoin_jet(query, char_sheet, world_info)

        if not analyse.get("besoin_jet"):
            return analyse.get("explication_regle", "Aucune règle spécifique applicable.")

        roll_data = simulate_dice_roll(analyse["jet_format"])
        if not roll_data:
            return f"Erreur lors du jet de dé pour : {analyse['jet_format']}"

        verdict = self.interpreter_reussite(query, roll_data["texte"], analyse["explication_regle"], context_text)
        return f"{verdict}\n\n[DÉTAIL DU JET] {roll_data['texte']}"
