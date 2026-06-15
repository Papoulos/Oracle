import json
import re
import os
import config
from base_utils import BaseAgent, extract_json

class ScenarioSummaryAgent(BaseAgent):
    """
    Agent one-shot : extrait les éléments narratifs de la collection scénario.
    Produit : Memory/scenario.json
    """

    def __init__(self, scenario_store):
        # Utilisation de la température basse pour la fidélité aux extraits
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1)
        self.scenario_store = scenario_store

    def generate(self) -> dict:
        print("[ScenarioSummaryAgent] Extraction des éléments du scénario...")

        queries = [
            ("titre de l'aventure", "titre"),
            ("pitch résumé introduction début", "pitch"),
            ("situation initiale scène d'ouverture", "situation"),
        ]

        all_docs = []
        for query, label in queries:
            docs = self.scenario_store.similarity_search(query, k=6)
            all_docs.extend(docs)

        # Déduplication par page_content
        unique_contents = {}
        for doc in all_docs:
            unique_contents[doc.page_content] = doc

        contexte_deduplique = "\n\n---\n\n".join(unique_contents.keys())

        if not contexte_deduplique.strip():
            raise ValueError("Erreur : scenario_collection vide ou sans résultat — vérifie ton indexation avec python indexer.py --verify")

        prompt = f"""Tu es un assistant de préparation de jeu de rôle.
À partir de ces extraits de scénario, produis une présentation structurée.
Ne complète pas et n'invente pas — utilise uniquement ce qui est présent.

EXTRAITS :
{contexte_deduplique}

Réponds UNIQUEMENT avec ce JSON :
{{
  "titre": "Titre de l'aventure (tel qu'il apparaît dans les extraits)",
  "pitch": "Ce que le joueur sait au départ — 2-3 phrases",
  "situation_initiale": "Scène d'ouverture précise — où est le joueur, ce qu'il voit"
}}
"""

        response = self.llm.invoke(prompt)
        scenario = extract_json(response.content, expected_type=dict)

        if not scenario:
            print(f"[ScenarioSummaryAgent] ✗ Échec de l'extraction JSON. Réponse brute :\n{response.content}")
            return {}

        # Validation minimale des champs obligatoires pour éviter des crashs plus tard
        required_fields = ["titre", "pitch", "situation_initiale"]
        for field in required_fields:
            if field not in scenario:
                print(f"[ScenarioSummaryAgent] ⚠ Champ '{field}' manquant dans le JSON.")
                scenario[field] = "Inconnu"

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/scenario.json", "w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=4, ensure_ascii=False)

        print(f"[ScenarioSummaryAgent] ✓ Scénario '{scenario.get('titre')}' généré.")
        return scenario


class NPCExtractorAgent(BaseAgent):
    """
    Agent un par un : génère les fiches PNJ basées sur les noms trouvés dans le scénario.
    Produit : Memory/npcs.json
    """

    def __init__(self, scenario_store):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1)
        self.scenario_store = scenario_store

    def extract(self, scenario_summary: dict) -> list:
        print("[NPCExtractorAgent] Extraction des fiches PNJ depuis le RAG...")

        # Étape 1 : Identifier les PNJ nommés
        identification_queries = [
            "personnages PNJ alliés ennemis antagonistes",
            "rencontres habitants notables",
            "liste des personnages du scénario"
        ]

        all_docs = []
        for q in identification_queries:
            docs = self.scenario_store.similarity_search(q, k=6)
            all_docs.extend(docs)

        unique_contents = {doc.page_content: doc for doc in all_docs}
        contexte_identification = "\n\n---\n\n".join(unique_contents.keys())

        identification_prompt = f"""Tu es un assistant de préparation de jeu de rôle.
Identifie TOUS les personnages nommés (PNJ) mentionnés dans les extraits suivants.
Ignore les créatures génériques sans nom (ex: "un garde", "les loups").
Extrais uniquement leurs noms et leurs rôles apparents.

EXTRAITS :
{contexte_identification}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "pnjs": [
    {{"nom": "Nom du personnage", "role": "Rôle ou fonction"}}
  ]
}}
"""
        id_response = self.llm.invoke(identification_prompt)
        id_data = extract_json(id_response.content, expected_type=dict)

        pnj_list = id_data.get("pnjs", []) if id_data else []

        if not pnj_list:
            print("[NPCExtractorAgent] Aucun PNJ nommé trouvé dans le RAG.")
            self._save_npcs([])
            return []

        print(f"[NPCExtractorAgent] {len(pnj_list)} PNJs identifiés : {[p.get('nom') for p in pnj_list]}")

        # Étape 2 : Extraire les détails pour chaque PNJ
        # Passe 1 — requêtes génériques pour le contexte global
        queries_generiques = [
            ("secret motivation but objectif personnel",       "motivations"),
            ("inventaire objet arme armure trésor unique",     "inventaires"),
        ]

        generic_docs = []
        for q, _ in queries_generiques:
            docs = self.scenario_store.similarity_search(q, k=6)
            generic_docs.extend(docs)

        unique_generic = {doc.page_content: doc for doc in generic_docs}
        contexte_generique = "\n\n".join(unique_generic.keys())

        extracted_npcs = []

        for pnj in pnj_list:
            nom = pnj.get("nom")
            role = pnj.get("role")
            print(f"[NPCExtractorAgent] Extraction des détails pour : {nom}...")

            # Passe 2 — requêtes ciblées
            query = f"{nom} {role}"
            specific_docs = self.scenario_store.similarity_search(query, k=4)

            unique_specific = {doc.page_content: doc for doc in specific_docs}
            chunks_cibles = "\n\n".join(unique_specific.keys())

            prompt = f"""Tu es un assistant de préparation de jeu de rôle.
Génère la fiche détaillée du personnage "{nom}" ({role}) à partir des extraits ci-dessous.
Ne complète pas et n'invente pas — si une information est absente, écris "Inconnu".

CONTEXTE GÉNÉRAL DU SCÉNARIO :
{scenario_summary.get('pitch', 'Inconnu')}

EXTRAITS SPÉCIFIQUES À CE PERSONNAGE :
{chunks_cibles}

EXTRAITS GÉNÉRAUX (motivations, inventaires) :
{contexte_generique}

Réponds UNIQUEMENT avec un objet JSON :
{{
  "id": "{re.sub(r'[^a-z0-9]+', '_', nom.lower()).strip('_')}",
  "nom": "{nom}",
  "classe": "Profession ou classe",
  "niveau": 1,
  "but": "Objectif personnel (Inconnu si absent)",
  "personnalite": "Traits de caractère (Inconnu si absent)",
  "secret": "Ce qu'il cache — NE JAMAIS révéler sans raison narrative",
  "relation_pj": "Inconnu",
  "statut": "Vivant",
  "localisation_actuelle": "Lieu de départ (Inconnu si absent)",
  "inventaire": [
    {{"nom": "...", "description": "...", "valeur_or": 0, "unique": true}}
  ],
  "capacites_notables": ["..."],
  "notes_mj": "Informations contextuelles pour le MJ"
}}
"""
            response = self.llm.invoke(prompt)
            npc_data = extract_json(response.content, expected_type=dict)

            if npc_data:
                extracted_npcs.append(npc_data)
            else:
                print(f"[NPCExtractorAgent] ✗ Échec pour {nom}")

        self._save_npcs(extracted_npcs)
        return extracted_npcs

    def _save_npcs(self, npcs: list):
        os.makedirs("Memory", exist_ok=True)
        data = {"npcs": npcs, "version": 1}
        with open("Memory/npcs.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[NPCExtractorAgent] ✓ {len(npcs)} PNJ sauvegardés dans Memory/npcs.json")
