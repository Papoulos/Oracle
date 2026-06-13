import json
import re
import os
import config
from base_utils import BaseAgent, extract_json

class NPCSetupAgent(BaseAgent):
    """
    Agent one-shot : génère les fiches PNJ structurées depuis le RAG scénario.
    S'exécute une seule fois après la création du personnage joueur.
    Produit : Memory/npcs.json
    """

    def __init__(self, scenario_store):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.2)
        self.scenario_store = scenario_store

    def _get_full_scenario_context(self) -> str:
        try:
            # On cible plus précisément les descriptions de personnages
            docs = self.scenario_store.similarity_search(
                "Description physique et personnalité des personnages non-joueurs PNJ. Liste des membres, alliés et ennemis.",
                k=config.RAG_SEARCH_K
            )
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucun document de scénario disponible."

    def generate_npcs(self, character_data: dict) -> list:
        raw_context = self._get_full_scenario_context()

        prompt = f"""Tu es un Maître du Jeu préparant sa table.
Lire les extraits de scénario ci-dessous et génère une fiche JSON complète
pour CHAQUE PNJ important que tu identifies.

PERSONNAGE JOUEUR :
{json.dumps(character_data, indent=2, ensure_ascii=False)}

EXTRAITS DU SCÉNARIO :
{raw_context}

Pour chaque PNJ, génère un objet JSON avec exactement ces champs :
{{
  "id": "identifiant_snake_case_unique",
  "nom": "Prénom Nom",
  "classe": "Profession ou classe (ex: Guerrier, Mage, Marchand)",
  "niveau": 5,
  "but": "Objectif personnel profond et sincère du PNJ",
  "personnalite": "Traits de caractère, manière d'interagir avec les autres",
  "secret": "Ce qu'il cache absolument — NE JAMAIS révéler sans raison narrative forte",
  "relation_pj": "Inconnu",
  "statut": "Vivant",
  "localisation_actuelle": "Où il se trouve au début",
  "inventaire": [
    {{"nom": "Nom de l'objet", "description": "Description", "valeur_or": 10, "unique": true}}
  ],
  "capacites_notables": ["Capacité 1", "Capacité 2"],
  "notes_mj": "Contexte ou indications pour le MJ"
}}

RÈGLES IMPORTANTES :
- "relation_pj" ∈ ["Inconnu", "Connaissance", "Ami", "Allié", "Ennemi", "Neutre"]
- "secret" est confidentiel — le Narrateur ne le révèle JAMAIS sans déclencheur narratif
- "niveau" reflète la puissance (1 = civil, 20 = légende)
- N'invente de PNJ que si le scénario est lacunaire, reste cohérent avec l'univers
- Inclure uniquement les PNJ avec un rôle narratif réel

Réponds UNIQUEMENT avec un tableau JSON valide entouré de ```json et ```.
Sois concis pour éviter de tronquer la réponse.
"""

        response = self.llm.invoke(prompt)
        raw = response.content

        npcs = extract_json(raw)

        if isinstance(npcs, list):
            # Nettoyage et validation minimale
            valid_npcs = []
            for n in npcs:
                if isinstance(n, dict) and "nom" in n:
                    if "id" not in n:
                        n["id"] = re.sub(r'[^a-z0-9]+', '_', n["nom"].lower()).strip('_')
                    valid_npcs.append(n)

            os.makedirs("Memory", exist_ok=True)
            with open("Memory/npcs.json", "w", encoding="utf-8") as f:
                json.dump({"npcs": valid_npcs, "version": 1}, f, indent=4, ensure_ascii=False)
            print(f"[NPCSetupAgent] ✓ {len(valid_npcs)} PNJ générés → Memory/npcs.json")
            return valid_npcs

        print(f"[NPCSetupAgent] ✗ Aucun JSON valide détecté dans la réponse LLM. Réponse reçue : \n{raw[:500]}...")
        return []


class ScenarioSetupAgent(BaseAgent):
    """
    Agent one-shot : génère la trame narrative structurée en actes.
    Produit : Memory/scenario.json (version enrichie, remplace l'ancien)
    """

    def __init__(self, scenario_store, core_store):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.3)
        self.scenario_store = scenario_store
        self.core_store = core_store

    def _get_context(self) -> str:
        try:
            docs = self.scenario_store.similarity_search(
                "intrigue acte objectif lieu quête antagoniste résolution enjeu",
                k=config.RAG_SEARCH_K
            )
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucun document de scénario disponible."

    def generate_scenario(self, character_data: dict, npcs: list) -> dict:
        raw_context = self._get_context()

        # Résumé PNJ sans secrets pour le prompt
        npc_summary = [
            {
                "id": n.get("id"),
                "nom": n.get("nom"),
                "classe": n.get("classe"),
                "niveau": n.get("niveau"),
                "but": n.get("but"),
                "relation_pj": n.get("relation_pj"),
            }
            for n in npcs
        ]

        prompt = f"""Tu es un Maître du Jeu qui structure son aventure avant la session.
À partir des extraits de scénario, du personnage joueur et des PNJ,
crée une trame narrative complète découpée en actes.

PERSONNAGE JOUEUR :
{json.dumps(character_data, indent=2, ensure_ascii=False)}

PNJ DISPONIBLES :
{json.dumps(npc_summary, indent=2, ensure_ascii=False)}

EXTRAITS DU SCÉNARIO :
{raw_context}

Génère un objet JSON avec EXACTEMENT ce format :
{{
  "titre": "Titre épique et mémorable de l'aventure",
  "pitch": "Ce que le joueur sait au départ — 2-3 phrases accrocheuses",
  "intrigue_complete": "Vérité cachée pour le MJ — enjeux réels, retournements, résolution",
  "situation_initiale": "Description précise de la scène d'ouverture — lieu, ambiance, ce que perçoit le joueur",
  "theme_principal": "Le thème émotionnel ou moral de l'aventure (ex: trahison, rédemption)",
  "lieux_cles": [
    {{
      "nom": "Nom du lieu",
      "description": "Ambiance, architecture, danger",
      "importance_narrative": "Pourquoi ce lieu est crucial à l'histoire"
    }}
  ],
  "actes": [
    {{
      "numero": 1,
      "titre": "Titre court de l'acte",
      "objectif_principal": "Ce que le joueur doit accomplir dans cet acte",
      "evenements_cles": ["Événement déclencheur", "Révélation ou complication", "Point de non-retour"],
      "pnj_impliques": ["id_pnj1", "id_pnj2"],
      "lieux_impliques": ["Nom du lieu A"],
      "resolution_possible": "Comment cet acte peut se terminer (plusieurs issues possibles)"
    }}
  ],
  "quetes_secondaires": [
    {{
      "id": "qs_snake_case",
      "titre": "Nom de la quête",
      "description": "Résumé de l'enjeu",
      "declencheur": "Ce qui initie cette quête",
      "recompense": "XP, objets, ou information obtenus"
    }}
  ],
  "tension_actuelle": 1,
  "acte_en_cours": 1,
  "etat": "actif"
}}

RÈGLES :
- Génère 3 à 5 actes avec une vraie progression dramatique
- "tension_actuelle" : entier 1 (calme) → 5 (crise maximale)
- "pitch" ≠ "intrigue_complete" : le joueur ne sait pas tout
- Les IDs dans "pnj_impliques" doivent correspondre aux "id" des PNJ

Réponds UNIQUEMENT avec le bloc JSON entouré de ```json et ```.
Sois concis et direct pour éviter que la réponse ne soit tronquée.
"""

        response = self.llm.invoke(prompt)
        raw = response.content

        scenario = extract_json(raw)

        if isinstance(scenario, dict):
            # Garantir la présence des champs obligatoires
            if "intrigue_complete" not in scenario:
                scenario["intrigue_complete"] = scenario.get("pitch", "Une mystérieuse aventure commence.")
            if "situation_initiale" not in scenario:
                scenario["situation_initiale"] = "Le héros commence son périple dans un lieu calme."
            if "titre" not in scenario:
                scenario["titre"] = "Une Aventure Sans Nom"
            if "actes" not in scenario or not scenario["actes"]:
                scenario["actes"] = [{
                    "numero": 1,
                    "titre": "Le Commencement",
                    "objectif_principal": "Explorer les environs",
                    "evenements_cles": ["Départ"],
                    "pnj_impliques": []
                }]

            os.makedirs("Memory", exist_ok=True)
            with open("Memory/scenario.json", "w", encoding="utf-8") as f:
                json.dump(scenario, f, indent=4, ensure_ascii=False)
            nb_actes = len(scenario.get("actes", []))
            print(f"[ScenarioSetupAgent] ✓ '{scenario.get('titre')}' — {nb_actes} actes → Memory/scenario.json")
            return scenario

        print(f"[ScenarioSetupAgent] ✗ Aucun JSON valide détecté dans la réponse LLM. Réponse reçue : \n{raw[:500]}...")
        return {}
