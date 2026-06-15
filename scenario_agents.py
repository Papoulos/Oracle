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
            ("titre de l'aventure, adventure title, name of the module", "titre"),
            ("pitch résumé introduction début, synopsis, adventure hook, background", "pitch"),
            ("situation initiale scène d'ouverture, starting location, introduction, prologue", "situation"),
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

        # Log de diagnostic
        print(f"[ScenarioSummaryAgent] DEBUG: {len(unique_contents)} extraits uniques récupérés.")
        if len(contexte_deduplique) > 500:
            print(f"[ScenarioSummaryAgent] DEBUG: Début du contexte : {contexte_deduplique[:500]}...")
        else:
            print(f"[ScenarioSummaryAgent] DEBUG: Contexte : {contexte_deduplique}")

        if not contexte_deduplique.strip():
            raise ValueError("Erreur : scenario_collection vide ou sans résultat — vérifie ton indexation avec python indexer.py --verify")

        prompt = f"""Tu es un assistant de préparation de jeu de rôle expert.
À partir de ces extraits de scénario (qui peuvent être en français ou en anglais), produis une présentation structurée en FRANÇAIS.
Ne complète pas et n'invente pas — utilise uniquement ce qui est présent dans les extraits.
Si les extraits sont très courts ou peu clairs, fais au mieux avec les informations disponibles sans halluciner.

EXTRAITS DU SCÉNARIO :
{contexte_deduplique}

Réponds UNIQUEMENT avec ce bloc JSON :
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
        titre = scenario_summary.get("titre", "l'aventure")
        print(f"[NPCExtractorAgent] Recherche de PNJ pour '{titre}'...")

        # Étape 1 : Identifier les PNJ nommés
        # On utilise des requêtes plus larges et liées au titre de l'aventure
        identification_queries = [
            f"Personnages et PNJ de {titre}",
            f"Characters and NPCs in {titre}",
            "Liste des personnages nommés, list of named characters",
            "Protagonistes, alliés et antagonistes, protagonists, allies and antagonists",
            "Habitants, gardes nommés, marchands et chefs, inhabitants, named guards, merchants and leaders",
            "Qui sont les personnages clés de cette histoire? Who are the key characters?"
        ]

        all_docs = []
        for q in identification_queries:
            docs = self.scenario_store.similarity_search(q, k=8)
            all_docs.extend(docs)

        unique_contents = {doc.page_content: doc for doc in all_docs}
        print(f"[NPCExtractorAgent] {len(unique_contents)} extraits uniques trouvés pour l'identification.")

        contexte_identification = "\n\n---\n\n".join(unique_contents.keys())

        # Log de diagnostic
        print(f"[NPCExtractorAgent] DEBUG: {len(unique_contents)} extraits pour identification.")

        identification_prompt = f"""Tu es un assistant MJ expert.
Ta mission est de lister TOUS les personnages nommés (PNJ) présents dans les extraits du scénario "{titre}" ci-dessous (qui peuvent être en anglais).

CONSIGNES :
- Liste chaque individu possédant un NOM PROPRE (ex: "Alaric", "Maître Elrond").
- Inclus les personnages secondaires s'ils ont un nom.
- Ignore les ennemis génériques non nommés (ex: "les gobelins", "les brigands").
- Pour chaque personnage, indique brièvement son rôle (ex: "Aubergiste", "Chef de la garde", "Antagoniste principal").

EXTRAITS DU SCÉNARIO :
{contexte_identification}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "pnjs": [
    {{"nom": "Nom complet", "role": "Fonction ou rôle dans l'intrigue"}}
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
            ("secret motivation but objectif personnel, secret motivation, goal, background",       "motivations"),
            ("inventaire objet arme armure trésor unique, inventory, items, weapons, unique treasure", "inventaires"),
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
            role = pnj.get("role", "Personnage")
            print(f"[NPCExtractorAgent] Extraction des détails pour : {nom} ({role})...")

            # Passe 2 — requêtes ciblées
            # On cherche par nom et par rôle, et aussi des termes généraux liés au personnage
            query = f"Détails sur le personnage {nom} {role}, Details about character {nom} {role}"
            specific_docs = self.scenario_store.similarity_search(query, k=5)

            unique_specific = {doc.page_content: doc for doc in specific_docs}
            chunks_cibles = "\n\n".join(unique_specific.keys())

            # Log de diagnostic
            print(f"[NPCExtractorAgent] DEBUG: {len(unique_specific)} extraits spécifiques pour {nom}.")

            prompt = f"""Tu es un assistant de préparation de jeu de rôle.
Génère la fiche détaillée du personnage "{nom}" ({role}) en FRANÇAIS à partir des extraits ci-dessous (qui peuvent être en anglais).
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
                print(f"[NPCExtractorAgent] ✓ {nom} extrait.")
                extracted_npcs.append(npc_data)
            else:
                print(f"[NPCExtractorAgent] ✗ Échec de l'extraction JSON pour {nom}.")

        self._save_npcs(extracted_npcs)
        return extracted_npcs

    def _save_npcs(self, npcs: list):
        os.makedirs("Memory", exist_ok=True)
        data = {"npcs": npcs, "version": 1}
        with open("Memory/npcs.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[NPCExtractorAgent] ✓ {len(npcs)} PNJ sauvegardés dans Memory/npcs.json")
