import json
import re
import os
import time
import config
from langchain_core.prompts import ChatPromptTemplate
from base_utils import BaseAgent, extract_json, get_full_store_text, get_relevant_context

class ScenarioExtractorAgent(BaseAgent):
    """
    Agent unifié pour extraire le scénario complet en 5 passes.
    Produit : Memory/scenario_structure.json
    """

    def __init__(self, scenario_store):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1)
        self.scenario_store = scenario_store

    def generate(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[ScenarioExtractor] {msg}")
            else:
                print(f"[ScenarioExtractorAgent] {msg}")

        log("Début du pipeline d'extraction du scénario en 5 passes...")
        start_time = time.time()

        # Passe 1 : Entités (pnj, lieux)
        entites = self._extract_entites(log)

        # Passe 2 : Nœuds scéniques (noeuds_sceniques)
        noeuds = self._extract_noeuds_sceniques(entites, log)

        # Passe 3 : Macro-structure (macro_structure)
        macro = self._extract_macro_structure(noeuds, log)

        # Passe 4 : Horloges globales (horloges_globales)
        horloges = self._extract_horloges(log)

        # Passe 5 : Métadonnées (metadata)
        metadata = self._extract_metadata(macro, log)

        # Consolidation du résultat final
        structure = {
            "metadata": metadata.get("metadata", {}),
            "macro_structure": macro.get("macro_structure", []),
            "horloges_globales": horloges.get("horloges_globales", []),
            "entites": {
                "pnj": entites.get("pnj", []),
                "lieux": entites.get("lieux", [])
            },
            "noeuds_sceniques": noeuds.get("noeuds_sceniques", [])
        }

        # S'assurer que le répertoire Memory existe
        os.makedirs("Memory", exist_ok=True)
        with open("Memory/scenario_structure.json", "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=4, ensure_ascii=False)

        total_time = time.time() - start_time
        log(f"Extraction consolidée terminée avec succès en {total_time:.2f}s.")
        return structure

    def _get_context(self, queries, log, k=15) -> str:
        from base_utils import get_relevant_context
        return get_relevant_context(
            self.scenario_store, queries, log, config.SCENARIO_FULLTEXT_THRESHOLD_CHARS, k=k
        )

    def _extract_entites(self, log) -> dict:
        log("Passe 1 : Extraction des entités (PNJs et Lieux)...")
        queries = [
            "personnages importants, personnages nommés, PNJ, main characters, named NPCs, important figures",
            "lieux de l'aventure, villes, pièces, donjons, locations, regions, places of interest, environments"
        ]
        contexte = self._get_context(queries, log, k=15)

        prompt = f"""Tu es un assistant de préparation de jeu de rôle expert.
À partir des extraits de scénario suivants (en français ou en anglais), extrais les personnages non-joueurs (PNJs) et les lieux principaux en FRANÇAIS.
Ne complète pas et n'invente pas d'éléments absents.

EXTRAITS DU SCÉNARIO :
{contexte}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "pnj": [
    {{
      "id": "PNJ_ID_SANS_ACCENT_EN_MAJUSCULES (ex: MAITRE_ELROND)",
      "nom_complet": "Nom complet et titre",
      "localisation_habituelle": "LIEU_ID_SANS_ACCENT_EN_MAJUSCULES (ex: FONDCOMBE)",
      "agenda_et_motivation": "Ce que le PNJ cherche à obtenir",
      "peurs_et_faiblesses": "Ce qui le fait céder ou fuir",
      "attitude_initiale": "Comportement initial lors de la rencontre",
      "stats_et_capacites": "Niveau de menace, PV, attaques clés ou inconnu"
    }}
  ],
  "lieux": [
    {{
      "id": "LIEU_ID_SANS_ACCENT_EN_MAJUSCULES (ex: FONDCOMBE)",
      "nom_complet": "Nom complet du lieu",
      "ambiance_sensorielle": "Vue, ouïe, odeur, atmosphère",
      "elements_interactifs": "Objets, leviers, conteneurs, éléments du décor"
    }}
  ]
}}
"""
        response = self.llm.invoke(prompt)
        res = extract_json(response.content, expected_type=dict)
        if not res:
            res = {"pnj": [], "lieux": []}
        return res

    def _extract_noeuds_sceniques(self, entites, log) -> dict:
        log("Passe 2 : Extraction des nœuds scéniques...")
        queries = [
            "déroulement de l'aventure, scènes, chapitres, actes, structure narrative",
            "rencontres, défis, combats, énigmes, pièges, obstacles"
        ]
        contexte = self._get_context(queries, log, k=15)

        pnj_ids = [p["id"] for p in entites.get("pnj", [])]
        lieu_ids = [l["id"] for l in entites.get("lieux", [])]

        prompt = f"""Tu es un assistant de préparation de jeu de rôle expert.
À partir des extraits de scénario suivants, extrais la liste de tous les nœuds scéniques (scènes) en FRANÇAIS.
Ne complète pas et n'invente pas d'éléments absents.

ID DE PNJS VALIDES : {json.dumps(pnj_ids)}
ID DE LIEUX VALIDES : {json.dumps(lieu_ids)}

CONSIGNES POUR LES CHAMPS :
- "acte_rattache_id" : ID de l'acte parent auquel appartient la scène (ex: ACTE_1, ACTE_2).
- "lieu_rattache_id" : ID du lieu rattaché. Utilise obligatoirement un ID parmi les ID DE LIEUX VALIDES ci-dessus si possible.
- "pnj_presents" : liste d'ID de PNJs présents. Utilise obligatoirement des ID parmi les ID DE PNJS VALIDES ci-dessus.
- "condition_resolution" : Résultat qui clôt ce nœud, formulé comme un BUT atteint par n'importe quel moyen plausible, indépendamment de la méthode listée dans sorties_logiques.
- "sorties_logiques" : pour chaque sortie, "destination_scene_id" doit correspondre à l'ID d'une autre scène (ex: SCENE_02_ROUTE).

EXTRAITS DU SCÉNARIO :
{contexte}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "noeuds_sceniques": [
    {{
      "id_scene": "SCENE_NUMERO_NOM (ex: SCENE_01_AUBERGE)",
      "acte_rattache_id": "ACTE_1",
      "lieu_rattache_id": "LIEU_NOM",
      "titre": "Titre de la scène",
      "pnj_presents": ["PNJ_ID"],
      "objectif_mj": "Ce que le MJ doit transmettre ou faire ressentir (ambiance, pas condition de sortie)",
      "condition_resolution": "Résultat qui clôt ce nœud...",
      "limites_et_regles_locales": "Règles physiques, magiques ou comportementales strictes de ce nœud",
      "defis_et_rencontres": [
        {{
          "type": "Combat / Enigme / Piège / Obstacle physique",
          "description": "Description concrète du défi",
          "resolution_possible": "Moyens logiques ANTICIPÉS de surmonter le défi"
        }}
      ],
      "sorties_logiques": [
        {{
          "action_ou_direction": "Ce que fait le joueur",
          "destination_scene_id": "ID_DE_LA_SCENE_DESTINATION"
        }}
      ]
    }}
  ]
}}
"""
        response = self.llm.invoke(prompt)
        res = extract_json(response.content, expected_type=dict)
        if not res or "noeuds_sceniques" not in res:
            res = {"noeuds_sceniques": []}

        # Validation de Pass 2
        validated_scenes = []
        for scene in res.get("noeuds_sceniques", []):
            scene_id = scene.get("id_scene")
            if not scene_id:
                continue

            # Validation du lieu
            lieu_id = scene.get("lieu_rattache_id")
            if lieu_id and lieu_id not in lieu_ids:
                log(f"[Validation] Scene '{scene_id}' : lieu_rattache_id '{lieu_id}' invalide. Mis à null.")
                scene["lieu_rattache_id"] = None

            # Validation des PNJs
            presents = scene.get("pnj_presents", [])
            valid_presents = []
            for pid in presents:
                if pid in pnj_ids:
                    valid_presents.append(pid)
                else:
                    log(f"[Validation] Scene '{scene_id}' : pnj_present '{pid}' invalide. Supprimé.")
            scene["pnj_presents"] = valid_presents

            validated_scenes.append(scene)

        res["noeuds_sceniques"] = validated_scenes
        return res

    def _extract_macro_structure(self, noeuds, log) -> dict:
        log("Passe 3 : Extraction de la macro-structure...")
        queries = [
            "structure globale, actes, chapitres majeurs, grandes étapes, main plot points, story structure"
        ]
        contexte = self._get_context(queries, log, k=15)

        scene_ids = [s["id_scene"] for s in noeuds.get("noeuds_sceniques", [])]

        prompt = f"""Tu es un assistant de préparation de jeu de rôle expert.
À partir des extraits de scénario suivants, structure l'histoire en grands ACTES (macro-structure) en FRANÇAIS.
Ne complète pas et n'invente pas d'éléments absents.

ID DE SCÈNES VALIDES : {json.dumps(scene_ids)}

CONSIGNES :
- Chaque acte possède un "id_acte" unique (ex: ACTE_1, ACTE_2).
- "scenes_incluses" doit contenir uniquement des ID de scènes parmi la liste des ID DE SCÈNES VALIDES ci-dessus.

EXTRAITS DU SCÉNARIO :
{contexte}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "macro_structure": [
    {{
      "id_acte": "ACTE_1",
      "titre": "Titre de la grande étape",
      "condition_entree": "Événement ou choix qui déclenche cet acte",
      "condition_validation": "Condition stricte pour valider cet acte et passer au suivant",
      "scenes_incluses": ["SCENE_01_ID", "SCENE_02_ID"]
    }}
  ]
}}
"""
        response = self.llm.invoke(prompt)
        res = extract_json(response.content, expected_type=dict)
        if not res or "macro_structure" not in res:
            res = {"macro_structure": []}

        # Validation de Pass 3
        validated_actes = []
        for acte in res.get("macro_structure", []):
            acte_id = acte.get("id_acte")
            if not acte_id:
                continue

            # Validation des scènes incluses
            incluses = acte.get("scenes_incluses", [])
            valid_incluses = []
            for sid in incluses:
                if sid in scene_ids:
                    valid_incluses.append(sid)
                else:
                    log(f"[Validation] Acte '{acte_id}' : scene_incluse '{sid}' invalide. Supprimée.")
            acte["scenes_incluses"] = valid_incluses
            validated_actes.append(acte)

        res["macro_structure"] = validated_actes

        # Vérification bidirectionnelle et corrections
        actes_dict = {a["id_acte"]: a for a in validated_actes}
        for scene in noeuds.get("noeuds_sceniques", []):
            scene_id = scene.get("id_scene")
            scene_acte_id = scene.get("acte_rattache_id")

            if scene_acte_id in actes_dict:
                target_acte = actes_dict[scene_acte_id]
                if scene_id not in target_acte["scenes_incluses"]:
                    log(f"[Validation] Correction bidirectionnelle : Ajout de la scène '{scene_id}' à 'scenes_incluses' de l'acte '{scene_acte_id}'.")
                    target_acte["scenes_incluses"].append(scene_id)
            else:
                log(f"[Validation] Attention : La scène '{scene_id}' référence un acte_rattache_id '{scene_acte_id}' inexistant.")

        return res

    def _extract_horloges(self, log) -> dict:
        log("Passe 4 : Extraction des horloges globales...")
        queries = [
            "menaces temporelles, dangers qui progressent, horloges, comptes à rebours, clocks, timers, consequences"
        ]
        contexte = self._get_context(queries, log, k=10)

        prompt = f"""Tu es un assistant de préparation de jeu de rôle expert.
À partir des extraits de scénario suivants, extrais les menaces progressives (horloges ou comptes à rebours globaux) en FRANÇAIS.
Ne complète pas et n'invente pas d'éléments absents.

CONSIGNES :
- "seuil" : nombre de segments pour déclencher la conséquence. S'il n'est pas spécifié, mettre 6 par défaut.

EXTRAITS DU SCÉNARIO :
{contexte}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "horloges_globales": [
    {{
      "nom": "Nom de la menace globale ou temporelle",
      "declencheur": "Action du joueur ou temps qui passe",
      "consequence": "Impact sur le monde ou fermeture d'accès",
      "seuil": 6
    }}
  ]
}}
"""
        response = self.llm.invoke(prompt)
        res = extract_json(response.content, expected_type=dict)
        if not res or "horloges_globales" not in res:
            res = {"horloges_globales": []}

        # Validation du seuil
        for clock in res.get("horloges_globales", []):
            seuil = clock.get("seuil")
            try:
                clock["seuil"] = int(seuil) if seuil is not None else 6
            except ValueError:
                clock["seuil"] = 6

        return res

    def _extract_metadata(self, macro_structure, log) -> dict:
        log("Passe 5 : Extraction des métadonnées...")
        queries = [
            "titre de l'aventure, adventure title, name of the module",
            "pitch résumé introduction début, synopsis, adventure hook, background, plot summary"
        ]
        contexte = self._get_context(queries, log, k=10)

        # Trouver le premier ID de scène inclus dans le premier acte
        default_scene_initiale = None
        if macro_structure.get("macro_structure"):
            first_act = macro_structure["macro_structure"][0]
            if first_act.get("scenes_incluses"):
                default_scene_initiale = first_act["scenes_incluses"][0]

        prompt = f"""Tu es un assistant de préparation de jeu de rôle expert.
À partir des extraits de scénario suivants, extrais le titre, le pitch de départ et la scène initiale de l'aventure en FRANÇAIS.
Ne complète pas et n'invente pas d'éléments absents.

DÉFAUT SCÈNE INITIALE : "{default_scene_initiale}"

CONSIGNES :
- "pitch_global" : Résumé de l'intrigue en 2 phrases avec noms propres complets.
- "scene_initiale" : ID de la scène d'ouverture. Si non spécifié de manière évidente, utilise la valeur par défaut "{default_scene_initiale}".

EXTRAITS DU SCÉNARIO :
{contexte}

Réponds UNIQUEMENT avec un JSON au format suivant :
{{
  "metadata": {{
    "titre": "Nom du scénario",
    "pitch_global": "Résumé de l'intrigue en 2 phrases avec noms propres complets.",
    "scene_initiale": "SCENE_NUMERO_NOM"
  }}
}}
"""
        response = self.llm.invoke(prompt)
        res = extract_json(response.content, expected_type=dict)
        if not res or "metadata" not in res:
            res = {"metadata": {}}

        # Validation de metadata
        meta = res.get("metadata", {})
        if not meta.get("titre"):
            meta["titre"] = "Inconnu"
        if not meta.get("pitch_global"):
            meta["pitch_global"] = "Inconnu"
        if not meta.get("scene_initiale"):
            meta["scene_initiale"] = default_scene_initiale or "Inconnu"

        res["metadata"] = meta
        return res


SCHEMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert en conception de systèmes de jeu de rôle.
À partir des extraits de règles fournis, produis un SCHÉMA décrivant les champs
qu'une fiche de personnage DOIT contenir pour être considérée comme complète
dans CE système de jeu précis.

CONSIGNES CRITIQUES :
1. N'invente aucun champ absent des règles fournies.
2. Utilise des clés techniques cohérentes en minuscules sans accents, reflétant le nom donné
   PAR CE SYSTÈME à chaque ressource (ex: "points_de_vie" si le jeu en a un, mais "vigueur"/
   "celerite"/"intellect" pour un système à réserves multiples, ou "sante_mentale" si une jauge
   distincte existe). N'utilise JAMAIS "points_de_vie" par défaut si ce n'est pas le nom réel
   de la ressource dans ce système.
3. Un champ de type "object" (bloc de caractéristiques, bloc de ressources...) doit lister
   ses sous_champs attendus, avec les noms exacts utilisés par CE système (peuvent être
   très différents d'un système à l'autre : "Force/Dextérité" ou "FOR/DEX/POU", etc.).
4. Un champ de type "list" doit préciser si une liste vide est acceptable ("non_vide": false)
   ou non ("non_vide": true).
5. Inclue toute ressource de vitalité mentionnée par les règles, même s'il y en a plusieurs
   (ex: points de vie ET santé mentale, ou boîtes de blessure ET stress).
6. N'inclus PAS de champs purement narratifs (historique, apparence, nom du joueur) sauf
   si les règles les rendent strictement nécessaires pour jouer.

Réponds UNIQUEMENT avec un JSON de cette forme, entouré de balises ```json :
```json
{{
  "champs_requis": [
    {{"chemin": "nom", "type": "string"}},
    {{"chemin": "caracteristiques", "type": "object", "sous_champs": ["...noms exacts du système..."]}},
    {{"chemin": "ressources.<nom_ressource>", "type": "object", "sous_champs": ["actuels", "max"]}},
    {{"chemin": "equipement", "type": "list", "non_vide": true}}
  ]
}}
```"""),
    ("human", "EXTRAITS DU CODEX (Règles) :\n{context}"),
])

DISCOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert en analyse de systèmes de jeu de rôle.
À partir des extraits de règles ci-dessous, identifie les GRANDES COMPOSANTES de
la création de personnage dans CE système précis, en utilisant EXCLUSIVEMENT le
vocabulaire propre à ce système - jamais de synonymes génériques type "race/classe"
si le système ne les utilise pas. Certains systèmes utilisent des concepts très
différents (ex: "Type/Descripteur/Focus", "Occupation", "Aspects", "Playbooks",
"Origine/Voie/Vocation"...). Utilise les termes EXACTS des règles fournies.

Réponds UNIQUEMENT avec un JSON de cette forme :
```json
{{
  "composantes": ["terme exact 1", "terme exact 2", "..."]
}}
```"""),
    ("human", "EXTRAITS DU CODEX (Règles) :\n{context}"),
])


class ManualGeneratorAgent(BaseAgent):
    """
    Agent one-shot : extrait les étapes de création de personnage du Core RAG.
    Produit : Memory/creation_manual.json
    """

    def __init__(self, core_store):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1)
        self.core_store = core_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en conception de systèmes de jeu de rôle.
Ta mission est de rédiger un MANUEL DE CRÉATION DE PERSONNAGE structuré en FRANÇAIS, basé UNIQUEMENT sur les extraits de règles fournis.

Ce manuel servira de guide "maître" à un autre agent IA qui accompagnera le joueur dans sa création.
Il doit être COMPLET sur toutes les étapes requises par le système de jeu, mais rester PUREMENT STRUCTUREL.

CONSIGNES CRITIQUES :
1. Liste TOUTES les étapes de création dans l'ordre logique requis par CE système, en utilisant EXCLUSIVEMENT la terminologie et les catégories propres à ce système. Ne suppose JAMAIS l'existence de composantes traditionnelles si les règles fournies n'en parlent pas explicitement - certains systèmes utilisent des concepts complètement différents (types, descripteurs, focus, occupations, aspects, réserves multiples, etc.).
2. Pour chaque étape, donne une description de la procédure à suivre.
3. NE LISTE PAS les options spécifiques individuelles de CE système (par exemple pour un système à professions, ne liste pas de métiers précis). Indique simplement qu'il faut faire un choix dans chaque catégorie identifiée, quelle qu'elle soit dans ce système précis.
4. L'agent final utilisera le RAG pour trouver les listes d'options. Ton rôle est de lui dire QUAND et COMMENT faire les choix.
5. EXPLICATION DES RÈGLES : Précise pour chaque étape si des limites numériques s'appliquent (ex: "Choisir 2 compétences", "Choisir 1 arme de mêlée et 1 de distance") afin que l'agent puisse les expliquer au joueur.
6. Indique clairement les méthodes de calcul mentionnées (ex: "Lancer 3d6", "Répartir 15 points").

Réponds UNIQUEMENT avec un bloc JSON entouré de balises ```json.

FORMAT JSON ATTENDUE :
```json
{{
  "etapes": [
    {{
      "etape": 1,
      "nom": "Étape 1",
      "description": "Description de la procédure"
    }}
  ],
  "regles_generales": "Notes globales (ex: importance de vérifier les prérequis avant de choisir l'équipement)"
}}
```"""),
            ("human", "EXTRAITS DU CODEX (Règles) :\n{context}"),
        ])
        self.chain = self.prompt | self.llm

    def generate(self, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[ManualGenerator] {msg}")
            else:
                print(f"[ManualGeneratorAgent] {msg}")

        log("Extraction des étapes de création de personnage...")
        start_time = time.time()

        full_core_text = get_full_store_text(self.core_store, log)

        if full_core_text and len(full_core_text) <= config.CORE_FULLTEXT_THRESHOLD_CHARS:
            log("Texte source du Core complet, utilisation directe (pas de découverte nécessaire).")
            contexte_deduplique = full_core_text
        else:
            log("Core trop volumineux pour tenir en contexte - découverte des composantes du système...")
            discovery_context = get_relevant_context(
                self.core_store,
                [
                    "création de personnage, character creation, comment créer un personnage",
                    "construire un personnage, personnage joueur, feuille de personnage",
                ],
                log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )

            composantes = []
            if discovery_context.strip():
                try:
                    discovery_chain = DISCOVERY_PROMPT | self.llm
                    discovery_response = discovery_chain.invoke({"context": discovery_context})
                    discovery_result = extract_json(discovery_response.content, expected_type=dict)
                    composantes = discovery_result.get("composantes", []) if discovery_result else []
                except Exception as e:
                    log(f"⚠ Erreur lors de la découverte des composantes : {e}")

            if composantes:
                log(f"Composantes découvertes : {composantes}")
                queries = [f"{c}, création de personnage" for c in composantes]
            else:
                log("⚠ Aucune composante découverte - repli sur des requêtes génériques.")
                queries = [
                    "création de personnage, caractéristiques, capacités spéciales",
                    "équipement, ressources de départ, progression du personnage",
                ]

            contexte_deduplique = get_relevant_context(
                self.core_store, queries, log, config.CORE_FULLTEXT_THRESHOLD_CHARS, k=config.RAG_K_CREATION
            )

        rag_time = time.time() - start_time
        log(f"Récupération du contexte terminée en {rag_time:.2f}s.")

        if not contexte_deduplique.strip():
            log("⚠ Aucun extrait trouvé dans le Core RAG. Le manuel sera vide.")
            return {}

        llm_start = time.time()
        try:
            response = self.chain.invoke({"context": contexte_deduplique})
            content = response.content
        except Exception as e:
            log(f"✗ Erreur lors de l'appel LLM : {e}")
            return {}
        llm_time = time.time() - llm_start
        manual = extract_json(content, expected_type=dict)
        log(f"LLM terminé en {llm_time:.2f}s.")

        if not manual:
            log("✗ Échec de l'extraction JSON du manuel.")
            print(f"[ManualGeneratorAgent] DEBUG: Réponse brute du LLM (len={len(content)}) :\n{repr(content)}")
            return {}

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/creation_manual.json", "w", encoding="utf-8") as f:
            json.dump(manual, f, indent=4, ensure_ascii=False)

        log("✓ Manuel de création généré dans Memory/creation_manual.json.")

        schema_chain = SCHEMA_PROMPT | self.llm
        try:
            schema_response = schema_chain.invoke({"context": contexte_deduplique})
            schema = extract_json(schema_response.content, expected_type=dict)
        except Exception as e:
            log(f"✗ Erreur lors de la génération du schéma de fiche : {e}")
            schema = None

        if not schema or not schema.get("champs_requis"):
            log("⚠ Schéma de fiche vide ou invalide - repli sur un schéma minimal (nom uniquement).")
            schema = {"champs_requis": [{"chemin": "nom", "type": "string"}]}

        with open("Memory/character_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=4, ensure_ascii=False)
        log("✓ Schéma de fiche de personnage généré dans Memory/character_schema.json.")

        return manual


class SceneGraphAgent(BaseAgent):
    """
    Agent un par un / one-shot : extrait les scènes, la structure de l'intrigue et génère un graphe de scènes.
    Produit : Memory/scenes.json
    """

    def __init__(self, scenario_store):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1)
        self.scenario_store = scenario_store

    def generate(self, scenario_summary: dict, log_callback=None) -> dict:
        def log(msg):
            if log_callback:
                log_callback(f"[SceneGraph] {msg}")
            else:
                print(f"[SceneGraphAgent] {msg}")

        log("Extraction des scènes et de la structure du scénario...")
        start_time = time.time()

        queries = [
            "déroulement de l'aventure, scènes, chapitres, actes, structure narrative",
            "lieux, salles, zones, rencontres",
            "conditions, ce qui se passe si, déclencheurs, réactions des PNJ"
        ]

        all_docs = []
        for query in queries:
            docs = self.scenario_store.similarity_search(query, k=15)
            all_docs.extend(docs)

        unique_contents = {doc.page_content: doc for doc in all_docs}
        contexte_deduplique = "\n\n---\n\n".join(unique_contents.keys())
        rag_time = time.time() - start_time
        log(f"RAG terminé en {rag_time:.2f}s ({len(unique_contents)} extraits).")

        if not contexte_deduplique.strip():
            log("⚠ Aucun extrait trouvé dans le scénario pour extraire les scènes.")
            return {}

        prompt = f"""Tu es un assistant de préparation de jeu de rôle expert.
À partir de ces extraits de scénario (qui peuvent être en français ou en anglais), produis une structure de scènes logique en FRANÇAIS.
Ne complète pas et n'invente pas d'éléments absents de ces extraits.

CONSIGNES POUR LE CONTENU :
1. Identifie la scène initiale et les scènes majeures de l'aventure.
2. Initialise TOUS les statuts à "a_venir", SAUF pour la scène pointée par "scene_initiale" qui commence avec le statut "en_cours".
3. L'esprit de la scène doit être résumé en une phrase.
4. "elements_a_preserver" contient les faits ou informations qui doivent rester vrais même si la scène se déroule autrement que prévu.
5. "reactions_anticipees" contient des réactions probables des PNJs/de l'environnement, sous forme d'aide-mémoire indicatif.
6. "objectif_atteint_si" doit être formulé en termes de résultat (ex: "le joueur obtient X"), jamais en termes de méthode/action littérale.

EXTRAITS DU SCÉNARIO :
{contexte_deduplique}

Réponds UNIQUEMENT avec un JSON valide suivant EXACTEMENT ce schéma :
{{
  "scene_initiale": "1.1",
  "scenes": [
    {{
      "id": "1.1",
      "titre": "string",
      "lieu": "string",
      "pnjs": ["id_pnj"],
      "esprit_de_la_scene": "ce que cette scène doit apporter à l'intrigue, en une phrase",
      "elements_a_preserver": ["fait ou info qui doit rester vrai même si la scène se déroule autrement que prévu"],
      "reactions_anticipees": [
        {{"action_probable": "string", "consequence": "string"}}
      ],
      "objectif_atteint_si": "condition formulée comme un BUT, pas une action littérale",
      "statut": "a_venir"
    }}
  ]
}}
"""

        llm_start = time.time()
        response = self.llm.invoke(prompt)
        llm_time = time.time() - llm_start
        scenes_data = extract_json(response.content, expected_type=dict)
        log(f"LLM terminé en {llm_time:.2f}s.")

        if not scenes_data:
            log("✗ Échec de l'extraction JSON du graphe de scènes.")
            return {}

        # Validation minimale
        if "scene_initiale" not in scenes_data:
            scenes_data["scene_initiale"] = "1.1"
        if "scenes" not in scenes_data or not isinstance(scenes_data["scenes"], list):
            scenes_data["scenes"] = []

        # S'assurer que le statut de la scène initiale est en_cours et les autres a_venir
        initial_id = scenes_data["scene_initiale"]
        for scene in scenes_data["scenes"]:
            if scene.get("id") == initial_id:
                scene["statut"] = "en_cours"
            else:
                scene["statut"] = "a_venir"

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/scenes.json", "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, indent=4, ensure_ascii=False)

        log(f"✓ Graphe de scènes généré ({len(scenes_data['scenes'])} scènes) dans Memory/scenes.json.")
        return scenes_data
