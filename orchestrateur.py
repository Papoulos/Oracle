from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.agent_regles import AgentRegles, simulate_dice_roll
from agents.agent_monde import AgentMonde
from agents.agent_garde import AgentGarde
from agents.agent_narrateur import AgentNarrateur
from agents.agent_memoire import AgentMemoire
from agents.agent_chronique import AgentChronique
from agents.agent_personnage import AgentPersonnage
import memory_manager
import json

class AgentState(TypedDict):
    query: str
    memory: dict
    garde_info: dict
    regles_info: str
    world_info: str
    narration: str
    updates: dict
    personnage_info: dict

class Orchestrateur:
    def __init__(self, codex_db, intrigue_db):
        self.agent_garde = AgentGarde(codex_db, intrigue_db)
        self.agent_regles = AgentRegles(codex_db)
        self.agent_monde = AgentMonde(intrigue_db)
        self.agent_narrateur = AgentNarrateur()
        self.agent_memoire = AgentMemoire()
        self.agent_chronique = AgentChronique()
        self.agent_personnage = AgentPersonnage(codex_db, intrigue_db)

        self.graph = self._build_graph()

    def _consult_garde(self, state: AgentState):
        res = self.agent_garde.valider_action(state["query"], json.dumps(state["memory"]))
        return {"garde_info": res}

    def _consult_monde(self, state: AgentState):
        # L'Agent Monde consulte l'état du monde, l'intrigue et l'historique
        res = self.agent_monde.consult(state["query"], state["memory"])
        return {"world_info": res}

    def _consult_regles(self, state: AgentState):
        # L'Agent Règles reçoit l'avis du Garde pour plus de contexte
        char_sheet = json.dumps(state["memory"].get("personnage", {}))
        # On utilise la raison du garde comme info de contexte
        garde_context = state["garde_info"].get("raison", "")

        # 1. Évaluation du besoin de jet
        analyse, context_text = self.agent_regles.evaluer_besoin_jet(state["query"], char_sheet, garde_context)

        if analyse.get("besoin_jet") and analyse.get("jet_format"):
            # 2. Simulation du jet (système)
            roll_data = simulate_dice_roll(analyse["jet_format"])
            if roll_data:
                # 3. Interprétation du résultat par l'agent
                verdict = self.agent_regles.interpreter_reussite(
                    state["query"],
                    roll_data["texte"],
                    analyse["explication_regle"],
                    context_text
                )
                res = f"{verdict}\n\n[DÉTAIL DU JET] {roll_data['texte']}"
            else:
                res = f"Erreur technique lors du jet: {analyse['jet_format']}"
        else:
            # Pas de jet nécessaire
            res = analyse.get("explication_regle", "Aucune règle spécifique applicable.")

        return {"regles_info": res}

    def _narrate(self, state: AgentState):
        res = self.agent_narrateur.narrate(
            state["query"],
            state["regles_info"],
            state["world_info"],
            json.dumps(state["garde_info"]),
            json.dumps(state["memory"])
        )
        return {"narration": res}

    def _consult_personnage_creation(self, state: AgentState):
        historique = state["memory"].get("historique", [])[-5:]
        res = self.agent_personnage.interagir_creation(state["query"], state["memory"], historique)
        return {"personnage_info": res, "narration": res["message"]}

    def _consult_personnage_evolution(self, state: AgentState):
        historique = state["memory"].get("historique", [])[-5:]
        res = self.agent_personnage.gerer_evolution(state["query"], state["memory"], historique)
        return {"personnage_info": res, "narration": res["message"]}

    def _update_memory_state(self, state: AgentState):
        etape = state["memory"].get("etape", "CREATION")

        if etape in ["CREATION", "LEVEL_UP"]:
            # Mise à jour directe via l'Agent Personnage
            res = state.get("personnage_info", {})
            updates = {
                "personnage_updates": res.get("personnage_updates", {}),
                "resume_action": res.get("message", "Mise à jour du personnage")
            }

            if etape == "CREATION" and res.get("creation_terminee"):
                memory_manager.update_etape("AVENTURE")
                updates["etape_change"] = "AVENTURE"
            elif etape == "LEVEL_UP" and res.get("evolution_terminee"):
                memory_manager.update_etape("AVENTURE")
                updates["etape_change"] = "AVENTURE"
        else:
            # Mode Aventure Normal
            updates = self.agent_memoire.extract_updates(
                state["query"],
                state["regles_info"],
                state["world_info"],
                state["narration"]
            )

        # Appliquer les mises à jour au fichier physique via memory_manager
        if updates:
            p_up = updates.get("personnage_updates", {})
            if p_up and isinstance(p_up, dict):
                # Gestion spécifique des ajouts à l'inventaire
                if p_up.get("inventaire_ajouts"):
                    for item in p_up.get("inventaire_ajouts", []):
                        memory_manager.add_to_inventory(item)

                # Mise à jour des autres champs via la nouvelle fonction robuste
                other_updates = {k: v for k, v in p_up.items() if k != "inventaire_ajouts"}
                if other_updates:
                    memory_manager.update_personnage(other_updates)

            m_up = updates.get("monde_updates", {})
            if m_up and isinstance(m_up, dict):
                if m_up.get("nouveau_lieu"):
                    memory_manager.update_lieu(m_up["nouveau_lieu"])
                if m_up.get("nouvel_evenement"):
                    memory_manager.add_evenement(m_up["nouvel_evenement"])

            # Gestion XP et Niveau en mode Aventure
            if etape == "AVENTURE" and state["narration"]:
                xp_res = self.agent_personnage.calculer_xp(
                    state["query"], state["narration"], state["regles_info"], state["world_info"]
                )
                if xp_res.get("xp_gagne"):
                    current_xp = state["memory"]["personnage"].get("xp", 0)
                    memory_manager.update_personnage({"xp": current_xp + xp_res["xp_gagne"]})
                    updates["xp_gain"] = xp_res

                    # Vérification immédiate du passage de niveau
                    new_mem = memory_manager.load_memory()
                    lvl_check = self.agent_personnage.verifier_niveau(new_mem["personnage"])
                    if lvl_check.get("passage_niveau"):
                        # On ne change pas l'étape tout de suite pour laisser le narrateur finir
                        # Mais on prévient l'UI
                        updates["level_up_available"] = True

            # Ajout du résumé à l'historique
            if updates.get("resume_action"):
                memory_manager.add_to_history(updates["resume_action"])

                # Mise à jour du compteur d'actions et de la chronique toutes les 10 actions
                count = memory_manager.increment_action_count()
                if count % 10 == 0:
                    current_memory = memory_manager.load_memory()
                    historique = current_memory.get("historique", [])
                    # On prend les 10 derniers éléments de l'historique
                    derniers_evenements = historique[-10:] if len(historique) >= 10 else historique

                    chapitre = self.agent_chronique.generer_chapitre(derniers_evenements)
                    memory_manager.add_chronique_chapter(chapitre)
                    # On peut ajouter une info dans les updates pour l'UI
                    updates["chronique_update"] = True

        # Recharger la mémoire pour le prochain tour ou pour l'affichage
        new_memory = memory_manager.load_memory()
        return {"updates": updates, "memory": new_memory}

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("consult_garde", self._consult_garde)
        workflow.add_node("consult_regles", self._consult_regles)
        workflow.add_node("consult_monde", self._consult_monde)
        workflow.add_node("narrate", self._narrate)
        workflow.add_node("update_memory", self._update_memory_state)
        workflow.add_node("personnage_creation", self._consult_personnage_creation)
        workflow.add_node("personnage_evolution", self._consult_personnage_evolution)

        def route_entree(state: AgentState):
            if state["query"].strip().lower() == "/levelup":
                memory_manager.update_etape("LEVEL_UP")
                return "evolution"

            etape = state["memory"].get("etape", "CREATION")
            if etape == "CREATION":
                return "creation"
            if etape == "LEVEL_UP":
                return "evolution"
            return "aventure"

        workflow.set_conditional_entry_point(
            route_entree,
            {
                "creation": "personnage_creation",
                "evolution": "personnage_evolution",
                "aventure": "consult_garde"
            }
        )

        # Logique conditionnelle après le Garde
        def check_garde_status(state: AgentState):
            if state["garde_info"].get("possible"):
                return "possible"
            return "impossible"

        workflow.add_conditional_edges(
            "consult_garde",
            check_garde_status,
            {
                "possible": "consult_regles",
                "impossible": "narrate"
            }
        )

        workflow.add_edge("consult_regles", "consult_monde")
        workflow.add_edge("consult_monde", "narrate")
        workflow.add_edge("narrate", "update_memory")
        workflow.add_edge("personnage_creation", "update_memory")
        workflow.add_edge("personnage_evolution", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def run(self, query):
        initial_state = {
            "query": query,
            "memory": memory_manager.load_memory(),
            "garde_info": {},
            "regles_info": "",
            "world_info": "",
            "narration": "",
            "updates": {},
            "personnage_info": {}
        }
        return self.graph.stream(initial_state)

    def initialiser_aventure(self):
        # On vérifie si on doit passer par la création
        memory = memory_manager.load_memory()
        etape = memory.get("etape", "CREATION")

        if etape == "CREATION":
            # On simule un premier échange pour lancer la création
            historique = memory.get("historique", [])[-5:]
            res = self.agent_personnage.interagir_creation("Bonjour", memory, historique)
            yield {"personnage_creation": {"personnage_info": res, "narration": res["message"]}}

            updates = {
                "personnage_updates": res.get("personnage_updates", {}),
                "resume_action": "Début de la création de personnage"
            }
            if updates["personnage_updates"]:
                memory_manager.update_personnage(updates["personnage_updates"])
            memory_manager.add_to_history(updates["resume_action"])

            yield {"update_memory": {"updates": updates}}
        else:
            # Introduction classique au monde
            world_info = self.agent_monde.chercher_introduction()
            yield {"consult_monde": {"world_info": world_info}}

            narration = self.agent_narrateur.narrer_introduction(world_info)
            yield {"narrate": {"narration": narration}}

            updates = self.agent_memoire.extract_updates(
                "Début de l'aventure",
                "N/A",
                world_info,
                narration
            )

            if updates:
                m_up = updates.get("monde_updates", {})
                if m_up and isinstance(m_up, dict):
                    if m_up.get("nouveau_lieu"):
                        memory_manager.update_lieu(m_up["nouveau_lieu"])
                if updates.get("resume_action"):
                    memory_manager.add_to_history(updates["resume_action"])

            yield {"update_memory": {"updates": updates}}
