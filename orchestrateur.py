from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.agent_regles import AgentRegles
from agents.agent_monde import AgentMonde
from agents.agent_narrateur import AgentNarrateur
from agents.agent_memoire import AgentMemoire
import memory_manager
import json

class AgentState(TypedDict):
    query: str
    memory: dict
    regles_info: str
    world_info: str
    narration: str
    updates: dict

class Orchestrateur:
    def __init__(self, codex_db, intrigue_db):
        self.agent_regles = AgentRegles(codex_db)
        self.agent_monde = AgentMonde(intrigue_db)
        self.agent_narrateur = AgentNarrateur()
        self.agent_memoire = AgentMemoire()

        self.graph = self._build_graph()

    def _consult_regles(self, state: AgentState):
        res = self.agent_regles.consult(state["query"], json.dumps(state["memory"].get("personnage", {})))
        return {"regles_info": res}

    def _consult_monde(self, state: AgentState):
        res = self.agent_monde.consult(state["query"], json.dumps(state["memory"].get("monde", {})))
        return {"world_info": res}

    def _narrate(self, state: AgentState):
        res = self.agent_narrateur.narrate(
            state["query"],
            state["regles_info"],
            state["world_info"],
            json.dumps(state["memory"])
        )
        return {"narration": res}

    def _update_memory_state(self, state: AgentState):
        updates = self.agent_memoire.extract_updates(
            state["query"],
            state["regles_info"],
            state["world_info"],
            state["narration"]
        )

        # Appliquer les mises à jour au fichier physique via memory_manager
        if updates:
            p_up = updates.get("personnage_updates", {})
            if p_up.get("stats"):
                memory_manager.update_stats(p_up["stats"])
            for item in p_up.get("inventaire_ajouts", []):
                memory_manager.add_to_inventory(item)

            m_up = updates.get("monde_updates", {})
            if m_up.get("nouveau_lieu"):
                memory_manager.update_lieu(m_up["nouveau_lieu"])
            if m_up.get("nouvel_evenement"):
                memory_manager.add_evenement(m_up["nouvel_evenement"])

        # Recharger la mémoire pour le prochain tour ou pour l'affichage
        new_memory = memory_manager.load_memory()
        return {"updates": updates, "memory": new_memory}

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("consult_regles", self._consult_regles)
        workflow.add_node("consult_monde", self._consult_monde)
        workflow.add_node("narrate", self._narrate)
        workflow.add_node("update_memory", self._update_memory_state)

        # On peut lancer règles et monde en parallèle
        workflow.set_entry_point("consult_regles")
        workflow.add_edge("consult_regles", "consult_monde")
        workflow.add_edge("consult_monde", "narrate")
        workflow.add_edge("narrate", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def run(self, query):
        initial_state = {
            "query": query,
            "memory": memory_manager.load_memory(),
            "regles_info": "",
            "world_info": "",
            "narration": "",
            "updates": {}
        }
        return self.graph.stream(initial_state)
