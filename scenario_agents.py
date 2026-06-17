import json
import re
import os
import time
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

    def generate(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[ScenarioSummary] {msg}")
            else:
                print(f"[ScenarioSummaryAgent] {msg}")

        log("Extraction des éléments du scénario...")
        start_time = time.time()

        # Multiples requêtes pour couvrir différents aspects sans diluer la pertinence
        queries = [
            "titre de l'aventure, adventure title, name of the module",
            "pitch résumé introduction début, synopsis, adventure hook, background",
            "situation initiale scène d'ouverture, starting location, prologue"
        ]

        all_docs = []
        for query in queries:
            docs = self.scenario_store.similarity_search(query, k=8)
            all_docs.extend(docs)

        # Déduplication par page_content
        unique_contents = {}
        for doc in all_docs:
            unique_contents[doc.page_content] = doc

        contexte_deduplique = "\n\n---\n\n".join(unique_contents.keys())
        rag_time = time.time() - start_time
        log(f"RAG terminé en {rag_time:.2f}s ({len(unique_contents)} extraits).")

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

        llm_start = time.time()
        response = self.llm.invoke(prompt)
        llm_time = time.time() - llm_start
        scenario = extract_json(response.content, expected_type=dict)
        log(f"LLM terminé en {llm_time:.2f}s.")

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

    def extract(self, scenario_summary: dict, log_callback=None) -> list:
        def log(msg):
            if log_callback:
                log_callback(f"[NPCExtractor] {msg}")
            else:
                print(f"[NPCExtractorAgent] {msg}")

        log("Identification des 5 PNJs les plus importants...")
        start_time = time.time()

        # Étape 1 : Identifier les 5 PNJ les plus importants
        # On utilise des requêtes génériques pour trouver les noms les plus cités
        identification_queries = [
            "Personnages nommés et PNJ importants, major characters and named NPCs, key figures",
            "Protagonistes et antagonistes principaux, main characters"
        ]

        all_docs = []
        for q in identification_queries:
            docs = self.scenario_store.similarity_search(q, k=15)
            all_docs.extend(docs)

        unique_contents = {doc.page_content: doc for doc in all_docs}
        contexte_identification = "\n\n---\n\n".join(unique_contents.keys())
        rag_id_time = time.time() - start_time
        log(f"RAG Identification terminé en {rag_id_time:.2f}s ({len(unique_contents)} extraits).")

        identification_prompt = f"""Tu es un assistant MJ expert.
Ta mission est d'identifier les 5 personnages nommés (PNJ) les plus importants dans les extraits du scénario ci-dessous.
L'importance est définie par la fréquence de mention et l'impact sur l'intrigue.

CONSIGNES :
- Liste au maximum 5 individus possédant un NOM PROPRE.
- S'il y a moins de 5 PNJ nommés, liste uniquement ceux présents.
- Pour chaque personnage, indique brièvement son rôle (ex: "Antagoniste principal", "Allié clé").

EXTRAITS DU SCÉNARIO :
{contexte_identification}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "pnjs": [
    {{"nom": "Nom complet", "role": "Fonction ou rôle dans l'intrigue"}}
  ]
}}
"""
        llm_id_start = time.time()
        id_response = self.llm.invoke(identification_prompt)
        llm_id_time = time.time() - llm_id_start
        id_data = extract_json(id_response.content, expected_type=dict)

        pnj_list = id_data.get("pnjs", []) if id_data else []

        if not pnj_list:
            log("Aucun PNJ nommé trouvé.")
            self._save_npcs([])
            return []

        log(f"{len(pnj_list)} PNJs identifiés en {llm_id_time:.2f}s.")

        # Étape 2 : Extraire les détails pour TOUS les PNJ en un seul appel
        log("Extraction des détails pour tous les PNJs...")
        details_start = time.time()

        # RAG pour les détails
        detail_queries = ["détails, secrets, motivations, inventaires, background des PNJ nommés"]
        # Ajouter les noms des PNJ à la requête pour aider le RAG
        detail_queries.append(", ".join([p.get('nom') for p in pnj_list]))

        all_detail_docs = []
        for q in detail_queries:
            docs = self.scenario_store.similarity_search(q, k=15)
            all_detail_docs.extend(docs)

        unique_details = {doc.page_content: doc for doc in all_detail_docs}
        contexte_details = "\n\n---\n\n".join(unique_details.keys())
        rag_details_time = time.time() - details_start
        log(f"RAG Détails terminé en {rag_details_time:.2f}s ({len(unique_details)} extraits).")

        prompt_details = f"""Tu es un assistant de préparation de jeu de rôle expert.
Ta mission est de générer les fiches détaillées en FRANÇAIS pour les PNJs suivants, en utilisant uniquement les extraits fournis.

LISTE DES PNJS À TRAITER :
{json.dumps(pnj_list, ensure_ascii=False, indent=2)}

CONTEXTE GÉNÉRAL DU SCÉNARIO :
{scenario_summary.get('pitch', 'Inconnu')}

EXTRAITS DU SCÉNARIO (contenant les détails) :
{contexte_details}

CONSIGNES :
- Produis une fiche détaillée pour CHAQUE PNJ de la liste.
- Ne complète pas et n'invente pas — si une information est absente, écris "Inconnu".
- "relation_pj" doit être "Inconnu".
- "id" doit être une version simplifiée du nom (ex: "maitre_elrond").
- Réponds UNIQUEMENT avec un JSON au format :
{{
  "npcs": [
    {{
      "id": "...",
      "nom": "...",
      "classe": "...",
      "but": "...",
      "personnalite": "...",
      "secret": "...",
      "relation_pj": "Inconnu",
      "statut": "Vivant",
      "localisation_actuelle": "...",
      "capacites_notables": ["..."]
    }}
  ]
}}
"""
        llm_details_start = time.time()
        response = self.llm.invoke(prompt_details)
        llm_details_time = time.time() - llm_details_start
        log(f"LLM Détails terminé en {llm_details_time:.2f}s.")

        details_data = extract_json(response.content, expected_type=dict)
        extracted_npcs = details_data.get("npcs", []) if details_data else []

        log(f"{len(extracted_npcs)} fiches PNJ générées.")
        self._save_npcs(extracted_npcs)
        return extracted_npcs

    def _save_npcs(self, npcs: list):
        os.makedirs("Memory", exist_ok=True)
        data = {"npcs": npcs, "version": 1}
        with open("Memory/npcs.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[NPCExtractorAgent] ✓ {len(npcs)} PNJ sauvegardés dans Memory/npcs.json")


class ManualGeneratorAgent(BaseAgent):
    """
    Agent one-shot : extrait les étapes de création de personnage du Core RAG.
    Produit : Memory/creation_manual.json
    """

    def __init__(self, core_store):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1)
        self.core_store = core_store

    def generate(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[ManualGenerator] {msg}")
            else:
                print(f"[ManualGeneratorAgent] {msg}")

        log("Extraction des étapes de création de personnage...")
        start_time = time.time()

        # Requêtes pour extraire la structure de création de manière exhaustive
        queries = [
            "étapes de création de personnage, character creation steps, character generation process",
            "caractéristiques, statistiques, scores, ability scores, attribute generation",
            "races, peuples, espèces, character races, species",
            "classes, professions, métiers, character classes",
            "équipement de départ, starting equipment, gold, wealth",
            "sorts, capacités, compétences, skills, spells, feats",
            "calcul des PV et CA, health points and armor class calculation"
        ]

        all_docs = []
        for query in queries:
            docs = self.core_store.similarity_search(query, k=5)
            all_docs.extend(docs)

        # Déduplication
        unique_contents = {doc.page_content: doc for doc in all_docs}
        contexte_deduplique = "\n\n---\n\n".join(unique_contents.keys())
        rag_time = time.time() - start_time
        log(f"RAG terminé en {rag_time:.2f}s ({len(unique_contents)} extraits).")

        if not contexte_deduplique.strip():
            log("⚠ Aucun extrait trouvé dans le Core RAG. Le manuel sera vide.")
            return {}

        prompt = f"""Tu es un expert en conception de systèmes de jeu de rôle.
Ta mission est de rédiger un MANUEL DE CRÉATION DE PERSONNAGE structuré en FRANÇAIS, basé UNIQUEMENT sur les extraits de règles fournis.

Ce manuel servira de guide "maître" à un autre agent IA qui accompagnera le joueur dans sa création.
Il doit être COMPLET sur toutes les étapes requises par le système de jeu, mais rester PUREMENT STRUCTUREL.

EXTRAITS DU CODEX (Règles) :
{contexte_deduplique}

CONSIGNES CRITIQUES :
1. Liste TOUTES les étapes de création dans l'ordre logique requis par le jeu (Caractères, Race, Classe, Équipement, Sorts/Capacités, PV/CA, etc.).
2. Pour chaque étape, donne une description de la procédure à suivre.
3. NE LISTE PAS les options spécifiques (ex: ne liste pas "Elfe", "Nain", "Guerrier"). Indique simplement qu'il faut choisir une race ou une classe.
4. L'agent final utilisera le RAG pour trouver les listes d'options. Ton rôle est de lui dire QUAND et COMMENT faire les choix.
5. Indique clairement les méthodes de calcul mentionnées (ex: "Lancer 3d6", "Répartir 15 points").

Réponds UNIQUEMENT avec un bloc JSON au format :
{{
  "etapes": [
    {{
      "etape": 1,
      "nom": "Nom de l'étape",
      "description": "Procédure détaillée de ce qu'il faut faire à cette étape"
    }}
  ],
  "regles_generales": "Notes sur les calculs globaux, l'ordre des étapes ou points d'attention critiques"
}}
"""

        llm_start = time.time()
        response = self.llm.invoke(prompt)
        llm_time = time.time() - llm_start
        manual = extract_json(response.content, expected_type=dict)
        log(f"LLM terminé en {llm_time:.2f}s.")

        if not manual:
            log("✗ Échec de l'extraction JSON du manuel.")
            return {}

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/creation_manual.json", "w", encoding="utf-8") as f:
            json.dump(manual, f, indent=4, ensure_ascii=False)

        log("✓ Manuel de création généré dans Memory/creation_manual.json.")
        return manual
