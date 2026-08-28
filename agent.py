import json
import re
import random
import os
import time
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import chromadb
import config
from base_utils import BaseAgent, get_llm, get_embeddings, extract_json
from scenario_agents import ScenarioExtractorAgent
from validation import validate_scenario_structure, validate_character_sheet

class CharacterCreator(BaseAgent):
    def __init__(self, vector_store):
        super().__init__(model=config.CHARACTER_MODEL, temperature=config.CHARACTER_TEMP)
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un Maître du Jeu (MJ) expert en jeux de rôle.
            Ton but actuel est de guider le joueur dans la création de son personnage.
            Utilise le MANUEL DE CRÉATION comme une liste de contrôle (checklist) pour ne rien oublier, mais reste interactif et flexible.

            MANUEL DE CRÉATION (Structure globale) :
            {manual}

            CONSIGNES DE DIALOGUE ET MÉCANIQUES :
            1. Réponds TOUJOURS en français.
            2. Pose UNE SEULE QUESTION à la fois.
            3. PRÉPARATION TECHNIQUE : Avant de poser une question, utilise le CODEX (RAG) pour connaître TOUTES les contraintes et bénéfices liés au choix actuel (ex: "Combien d'armes peut-il maîtriser ?", "Quels sont les bonus de cette race ?").
            4. PÉDAGOGIE : Explique TOUJOURS au joueur les conséquences techniques de son choix (ex: "En choisissant cette classe, tu as droit à 3 maîtrises d'armes qui te donneront un bonus de +2 aux jets d'attaque").
            5. Pour les statistiques/caractéristiques : Demande TOUJOURS au joueur s'il souhaite que tu lances les dés pour lui (selon la méthode du Codex) ou s'il préfère le faire lui-même.
            6. Pour les choix de Race et de Classe : Interroge le CODEX (RAG) pour obtenir la liste complète et exacte des options disponibles et présente-les de manière concise au joueur.
            7. Calcule les statistiques dérivées (PV, CA, modificateurs) en suivant scrupuleusement les formules du CODEX.

            CHAMPS ENCORE MANQUANTS (Tu dois impérativement guider le joueur pour remplir ces champs) :
            {champs_manquants}

            CONSIGNES DE FIN DE CRÉATION :
            Si toutes les étapes du manuel sont terminées, félicite le joueur et indique-lui que son personnage est prêt pour l'aventure.
            Utilise le mot-clé "CRÉATION_TERMINÉE" dans ton texte uniquement lorsque TOUTES les étapes sont validées.

            ÉTAT ACTUEL DU PERSONNAGE (pour ton information) :
            {current_character}

            CODEX (Détails des règles à interroger pour chaque choix) :
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            # Réduction légère du nombre de documents pour la création (de 8 à 5)
            # pour éviter de saturer la fenêtre de contexte avec des extraits trop longs.
            k = max(1, config.RAG_K_CREATION - 3)
            docs = self.vector_store.similarity_search(query, k=k)
            print(f"[CharacterCreator] DEBUG: Recherche contextuelle pour '{query}' -> {len(docs)} docs trouvés.")
            if docs:
                print(f"[CharacterCreator] DEBUG: Premier extrait : {docs[0].page_content[:200]}...")
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[CharacterCreator] DEBUG: Erreur RAG : {e}")
            return "Aucun contexte trouvé."

    def _load_manual(self):
        path = "Memory/creation_manual.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.dumps(json.load(f), ensure_ascii=False, indent=2)
            except Exception:
                return "Manuel non disponible (utiliser le Codex)."
        return "Manuel non disponible (utiliser le Codex)."

    def generate_response(self, user_input, history, character_data=None, champs_manquants=None):
        # On enrichit la requête RAG avec les derniers messages pour avoir du contexte sur l'étape de création
        rag_query = user_input
        if len(history) >= 1:
            last_msg = history[-1].content
            # On extrait une partie du dernier message du MJ pour aider le RAG
            rag_query = f"{last_msg[:100]} {user_input}"

        context = self.get_context(rag_query)
        manual = self._load_manual()

        if champs_manquants and isinstance(champs_manquants, list):
            champs_manquants_str = ", ".join(champs_manquants)
        else:
            champs_manquants_str = "Aucun (fiche complète)."

        inputs = {
            "manual": manual,
            "context": context,
            "history": history,
            "input": user_input,
            "current_character": json.dumps(character_data, ensure_ascii=False, indent=2) if character_data else "Aucune donnée pour le moment.",
            "champs_manquants": champs_manquants_str
        }
        response = self.chain.invoke(inputs)
        return response.content

class ChronicleAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=config.CHRONICLE_MODEL, temperature=config.CHRONICLE_TEMP)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Chroniqueur d'une aventure de jeu de rôle.
            Ton rôle est de tenir à jour un résumé factuel et concis de l'histoire jusqu'à présent.
            Tu reçois l'ancien résumé, l'action du joueur, la réponse du narrateur et éventuellement un fait additionnel.
            Tu dois produire un NOUVEAU résumé mis à jour qui intègre ces nouveaux événements.

            CONSIGNES :
            - Sois concis et factuel.
            - Garde les éléments importants (lieux, rencontres, objets obtenus, blessures).
            - Utilise le français.
            - Réponds uniquement avec le nouveau résumé, sans fioritures.
            """),
            ("human", """ANCIEN RÉSUMÉ : {old_chronicle}
            ACTION JOUEUR : {user_input}
            RÉPONSE NARRATEUR : {narrator_response}
            FAIT ADDITIONNEL À CONSERVER (peut diverger du scénario source) : {ecart_notable}

            Nouveau résumé mis à jour :"""),
        ])
        self.chain = self.prompt | self.llm

    def update(self, old_chronicle, user_input, narrator_response, ecart_notable=None):
        inputs = {
            "old_chronicle": old_chronicle if old_chronicle else "L'aventure commence à peine.",
            "user_input": user_input,
            "narrator_response": narrator_response,
            "ecart_notable": ecart_notable if ecart_notable else "Aucun fait additionnel."
        }
        response = self.chain.invoke(inputs)
        return response.content

class SheetManagerAgent(BaseAgent):
    def __init__(self, vector_store):
        super().__init__(model=config.SHEET_MANAGER_MODEL, temperature=config.SHEET_MANAGER_TEMP)
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Gestionnaire de Fiche de Personnage.
            Ton rôle est de mettre à jour la fiche JSON du personnage pour les aspects NARRATIFS uniquement (inventaire, relations, notes, descriptions).
            Tu reçois la fiche actuelle, l'action du joueur, la réponse du narrateur et les règles pertinentes.

            ⚠️ RÈGLE CRITIQUE : Tu ne dois JAMAIS modifier les Points de Vie (PV), l'XP, le niveau, les sorts ou les ressources (rage, magie, etc.). Ces éléments sont gérés de façon déterministe par le GameStateEngine.

            CONSIGNES :
            - Concentre-toi UNIQUEMENT sur les aspects narratifs : ajoute ou retire des objets de l'inventaire si nécessaire.
            - Mets à jour les relations avec les PNJ ou ajoute des notes d'histoire si la narration le justifie.
            - Ne modifie pas les statistiques de base (Force, etc.) sauf si un événement permanent l'exige.
            - Assure-toi que le JSON est valide et complet.
            - Réponds UNIQUEMENT avec le bloc JSON entouré de ```json and ```.

            RÈGLES DU CODEX (Contexte) :
            {context}
            """),
            ("human", """FICHE ACTUELLE :
            {character_sheet}

            DERNIERS ÉVÉNEMENTS :
            Action joueur : {user_input}
            Réponse narrateur : {narrator_response}

            Nouvelle fiche JSON mise à jour :"""),
        ])
        self.chain = self.prompt | self.llm

    def get_context(self, query):
        try:
            docs = self.vector_store.similarity_search(query, k=config.RAG_K_ADVENTURE)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucune règle trouvée pour la mise à jour."

    def update_sheet(self, character_sheet, user_input, narrator_response, mode="ADVENTURE"):
        context = self.get_context(f"Règles pour : {user_input} {narrator_response}")

        # On s'assure d'avoir un LLM valide même si le modèle spécifié a échoué (fallback manuel)
        llm = self.llm

        if mode == "CREATION":
            creation_prompt = ChatPromptTemplate.from_messages([
                ("system", """Tu es le Gestionnaire de Fiche de Personnage en phase de CRÉATION.
                Ton rôle est d'extraire les informations de la conversation pour mettre à jour la fiche JSON.

                CONSIGNES :
                - Analyse l'action du joueur et la réponse du MJ pour identifier les nouveaux choix (nom, race, classe, statistiques, équipement, etc.).
                - Mets à jour TOUS les champs nécessaires.
                - Si le MJ  mentionne que la création est terminée (mot-clé "CRÉATION_TERMINÉE"), mets le champ "statut" à "complet".
                - Réponds UNIQUEMENT avec le bloc JSON complet.

                RÈGLES DU CODEX (Contexte) :
                {context}
                """),
                ("human", """FICHE ACTUELLE :
                {character_sheet}

                DERNIERS ÉVÉNEMENTS :
                Action joueur : {user_input}
                Réponse MJ : {narrator_response}

                Nouvelle fiche JSON mise à jour :"""),
            ])
            chain = creation_prompt | llm
            inputs = {
                "context": context,
                "character_sheet": json.dumps(character_sheet, ensure_ascii=False, indent=2) if character_sheet else "{}",
                "user_input": user_input,
                "narrator_response": narrator_response
            }
            try:
                response = chain.invoke(inputs)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    print(f"[SheetManagerAgent] ⚠ Modèle '{config.SHEET_MANAGER_MODEL}' non trouvé, fallback sur '{config.LLM_MODEL}'.")
                    fallback_llm = get_llm(config.LLM_MODEL, config.SHEET_MANAGER_TEMP)
                    chain = creation_prompt | fallback_llm
                    response = chain.invoke(inputs)
                else:
                    raise e
        else:
            inputs = {
                "context": context,
                "character_sheet": json.dumps(character_sheet, ensure_ascii=False, indent=2),
                "user_input": user_input,
                "narrator_response": narrator_response
            }
            try:
                response = self.chain.invoke(inputs)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    print(f"[SheetManagerAgent] ⚠ Modèle '{config.SHEET_MANAGER_MODEL}' non trouvé, fallback sur '{config.LLM_MODEL}'.")
                    fallback_llm = get_llm(config.LLM_MODEL, config.SHEET_MANAGER_TEMP)
                    fallback_chain = self.prompt | fallback_llm
                    response = fallback_chain.invoke(inputs)
                else:
                    raise e

        new_sheet = extract_json(response.content)
        return new_sheet

    def audit_and_complete(self, character_data: dict, messages: list, schema: dict) -> dict | None:
        context = self.get_context("statistiques, points de vie, équipement, armes, armures, ressources")
        audit_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Gestionnaire de Fiche de Personnage, en AUDIT FINAL de création.
Relis l'INTÉGRALITÉ de la conversation de création ci-dessous et produis la fiche JSON la
plus complète et fidèle possible, en te basant sur le schéma des champs requis fourni.
Cherche en particulier les valeurs mentionnées dans la conversation mais absentes de la
fiche actuelle (souvent oubliées par les mises à jour incrémentales tour par tour).
Ne modifie pas les champs déjà corrects et ne modifie pas leur nom/structure existante.
Réponds UNIQUEMENT avec le bloc JSON complet de la fiche.

SCHÉMA DES CHAMPS REQUIS : {schema}
RÈGLES DU CODEX : {context}
"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "FICHE ACTUELLE :\n{character_sheet}\n\nProduis la fiche JSON finale, complète."),
        ])
        chain = audit_prompt | self.llm
        try:
            response = chain.invoke({
                "schema": json.dumps(schema, ensure_ascii=False),
                "context": context,
                "history": messages,
                "character_sheet": json.dumps(character_data, ensure_ascii=False, indent=2),
            })
            return extract_json(response.content, expected_type=dict)
        except Exception:
            return None

class Narrator(BaseAgent):
    def __init__(self):
        super().__init__(model=config.NARRATOR_MODEL, temperature=config.NARRATOR_TEMP)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es le Narrateur d'une aventure de jeu de rôle. Tu reçois des instructions
structurées de l'Orchestrateur (MJ) et tu les transformes en narration à la
deuxième personne du singulier, en français, de manière immersive.

RÈGLES ABSOLUES :
- Tu ne décides JAMAIS des règles, des jets de dés ou des résultats d'actions.
- Tu ne modifies JAMAIS l'état du jeu.
- Tu n'interprètes JAMAIS à la place du joueur — tu décris ce que son personnage
  perçoit, pas ce qu'il comprend ou conclut.
- Tu ne révèles JAMAIS directement le nom de l'antagoniste final, la nature
  exacte de la menace, ou la condition de résolution du scénario tant que
  le joueur ne les a pas découverts par le jeu - même si cette information
  t'est donnée en contexte par l'Orchestrateur.

STRUCTURE DE CHAQUE RÉPONSE :

Rédige une narration continue et fluide, SANS titres, SANS numéros, SANS
puces dans le texte narratif lui-même. Fais avancer ton texte à travers ces
mouvements, sans jamais les annoncer explicitement au joueur :

- Perception immédiate : ce que le personnage voit, entend, sent à l'instant T.
  Concret et sensoriel. Pas de conclusions, pas d'interprétations.
  ❌ "Vous comprenez que quelque chose a été tué ici."
  ✅ "Le sol est couvert d'une substance sombre et poisseuse. Une odeur
      âcre de fer et de chair vous prend à la gorge."
- Détails et environnement : ce que le personnage remarque en regardant
  alentour, toujours de son propre point de vue - ce qu'il voit
  réellement, jamais ce que le MJ sait en plus.
- Tension ou impulsion : un élément actif qui pousse le joueur à réagir
  (un bruit, un mouvement, une présence, un choix visible). Ne laisse
  jamais la description en suspension.
- Question ou amorce d'action, pour terminer : une question directe ou
  une proposition concrète.
  ❌ "Que faites-vous ?" (trop vague)
  ✅ "Le couloir nord semble plus sombre — aucune torche n'y brûle.
      À l'est, vous distinguez ce qui ressemble à une porte.
      Vous avancez, ou vous faites demi-tour ?"

Après ce texte narratif, ajoute un bloc résumé séparé, introduit par "---"
et l'en-tête "📌 Résumé des informations" (voir règles strictes ci-dessous).

RÈGLES STRICTES DU BLOC RÉSUMÉ :
- Liste UNIQUEMENT des faits que tu viens d'énoncer explicitement dans le
  texte narratif ci-dessus, dans cette même réponse.
- N'ajoute JAMAIS un fait, un nom, une signification ou une interprétation
  qui n'apparaît pas (mot pour mot ou en substance très proche) dans le
  texte que tu viens d'écrire. Si un objet ou indice existe mais que son
  contenu ou sa signification n'a pas encore été révélé au joueur dans la
  narration, ne mentionne PAS ce contenu dans le résumé - mentionne
  seulement son existence physique si elle a été décrite.
  ❌ Narration : "des gravures anciennes ornent les parois." / Résumé :
     "Indice : des gravures décrivant la gloire d'Aethelgard."
     (information non révélée dans la narration → INTERDIT)
  ✅ Narration : "des gravures anciennes ornent les parois." / Résumé :
     "Des gravures anciennes ornent les parois, non encore examinées."

STYLE :
- Deuxième personne du singulier ("vous").
- Présent de narration.
- Phrases courtes et rythmées pour les moments de tension,
  plus longues pour les descriptions calmes.
- Ne liste jamais les PNJ présents ou les lieux de façon technique.
"""),
            MessagesPlaceholder(variable_name="history"),
            ("system", "CONSIGNES DE L'ORCHESTRATEUR : {instructions}"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm

    def generate_response(self, user_input, history, instructions):
        inputs = {
            "history": history,
            "instructions": instructions,
            "input": user_input
        }
        response = self.chain.invoke(inputs)
        return response.content

class RPGAgent(BaseAgent):
    def __init__(self):
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=config.ORCHESTRATOR_TEMP)
        self.embeddings = get_embeddings()
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)

        # Collection pour les règles (Core)
        self.core_store = Chroma(
            client=self.client,
            collection_name=config.CORE_COLLECTION_NAME,
            embedding_function=self.embeddings
        )

        # Collection pour le scénario
        self.scenario_store = Chroma(
            client=self.client,
            collection_name=config.SCENARIO_COLLECTION_NAME,
            embedding_function=self.embeddings
        )

        self.character_creator = CharacterCreator(self.core_store)
        self.narrator = Narrator()
        self.chronicle_agent = ChronicleAgent()
        self.sheet_manager = SheetManagerAgent(self.core_store)

        # Agent de setup unifié
        self.scenario_extractor_agent = ScenarioExtractorAgent(self.scenario_store)

        from game_state_engine import GameStateEngine
        self.gse = GameStateEngine()

        # Données de scénario et progression
        self.scenario_structure = None
        self.progression = None
        self._pnj_by_id = {}
        self._lieux_by_id = {}
        self._scenes_by_id = {}
        self._actes_by_id = {}

        self.npcs_data = None
        self.current_scene_id = None

        self.history = ChatMessageHistory()
        self.game_state = "CREATION" # CREATION, SUMMARY, ADVENTURE
        self.character_data = None
        self.scenario_data = None
        self.chronicle_data = None
        self.setup_logs = []
        self._missing_character_fields = None

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        self.setup_logs.append(full_message)
        try:
            with open("streamlit.log", "a", encoding="utf-8") as f:
                f.write(full_message + "\n")
        except Exception:
            pass

    def _check_collections(self):
        checks = [
            (config.CORE_COLLECTION_NAME,     "Règles"),
            (config.SCENARIO_COLLECTION_NAME, "Scénario"),
        ]
        all_ok = True
        for coll_name, label in checks:
            try:
                collection = self.client.get_collection(coll_name)
                count = collection.count()
                if count == 0:
                    print(f"⚠ [{label}] Collection '{coll_name}' est vide — lance 'python indexer.py'")
                    all_ok = False
                else:
                    print(f"✓ [{label}] {count} chunks disponibles.")
            except Exception:
                print(f"✗ [{label}] Collection '{coll_name}' introuvable.")
                all_ok = False
        return all_ok

    def get_core_context(self, query, k=None):
        if k is None:
            k = config.RAG_SEARCH_K
        try:
            docs = self.core_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucune règle trouvée."

    def get_scenario_context(self, query, k=None):
        if k is None:
            k = config.RAG_SEARCH_K
        try:
            docs = self.scenario_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            return "Aucun élément de scénario trouvé."

    def get_current_scene(self) -> str:
        """
        Récupère la scène courante par lookup (plus de RAG ni de filtre Chroma pour la scène).
        """
        if not self.scenario_structure or not self.current_scene_id:
            return ""
        scene = self._scenes_by_id.get(self.current_scene_id)
        if not scene:
            return ""
        return json.dumps(scene, ensure_ascii=False, indent=2)

    def get_current_context(self) -> str:
        """
        Génère un contexte lisible à partir de la progression et des lookups statiques (Pas de RAG).
        """
        if not self.progression or not self.scenario_structure:
            return ""
        scene_id = self.progression.get("scene_courante")
        scene = self._scenes_by_id.get(scene_id)
        if not scene:
            return ""
        lieu = self._lieux_by_id.get(scene.get("lieu_rattache_id"))
        pnjs = [self._pnj_by_id[pid] for pid in scene.get("pnj_presents", []) if pid in self._pnj_by_id]
        acte = self._actes_by_id.get(scene.get("acte_rattache_id"))

        context_lines = []
        context_lines.append(f"SCÈNE COURANTE : {scene.get('titre')} (ID: {scene_id})")
        if lieu:
            context_lines.append(f"Lieu : {lieu.get('nom_complet')} ({lieu.get('ambiance_sensorielle')})")
            if lieu.get("elements_interactifs"):
                context_lines.append(f"Éléments interactifs du décor : {lieu.get('elements_interactifs')}")
        if pnjs:
            context_lines.append("PNJs présents :")
            for p in pnjs:
                context_lines.append(f"- {p.get('nom_complet')} (Motivation: {p.get('agenda_et_motivation')}, Attitude: {p.get('attitude_initiale')}, Stats/Capacités: {p.get('stats_et_capacites')})")
        if acte:
            context_lines.append(f"Acte rattaché : {acte.get('titre')} (Validation: {acte.get('condition_validation')})")

        return "\n".join(context_lines)

    def roll_dice(self, sides=20):
        return random.randint(1, sides)

    def _extract_and_add_resources(self):
        """Extrait les ressources de classe (sorts, rage, etc.) et les ajoute à la fiche."""
        self.log("Extraction des ressources et capacités de classe...")
        classe = self.character_data.get("classe", "Inconnu")
        niveau = self.character_data.get("niveau", 1)
        race = self.character_data.get("race", "Inconnu")

        query = f"Ressources de classe pour {classe} niveau {niveau}, race {race}. Emplacements de sorts, points de rage, capacités limitées par jour, points de vie."
        context = self.get_core_context(query, k=10)

        prompt = f"""Tu es un expert en règles de JDR.
Basé sur les extraits du CODEX suivants, identifie TOUTES les ressources consommables (sorts par jour, capacités à usages limités, points de vie, etc.) pour un personnage de niveau {niveau}, de classe {classe} et de race {race}.

EXTRAITS DU CODEX :
{context}

FICHE ACTUELLE :
{json.dumps(self.character_data, ensure_ascii=False, indent=2)}

Produis un objet JSON 'ressources' qui pourra être intégré à la fiche.
Chaque ressource doit avoir un 'total' et un 'restants' égal au total.
Utilise des noms de clés clairs en français (ex: 'emplacements_sorts_niv1', 'points_de_rage').
Si le personnage a des sorts, liste également les sorts connus s'ils sont mentionnés ou suggérés pour ce niveau.

Réponds UNIQUEMENT avec le bloc JSON :
{{
  "ressources": {{
    "nom_ressource": {{ "total": X, "restants": X }},
    ...
  }},
  "sorts": ["nom_sort1", "nom_sort2"]
}}
"""
        try:
            response = self.llm.invoke(prompt).content
            data = extract_json(response)
            if data:
                if "ressources" in data and data["ressources"]:
                    if "ressources" not in self.character_data:
                        self.character_data["ressources"] = {}
                    self.character_data["ressources"].update(data["ressources"])

                if "sorts" in data and data["sorts"]:
                    existing_spells = self.character_data.get("sorts", [])
                    if not existing_spells or len(existing_spells) < len(data["sorts"]):
                        self.character_data["sorts"] = data["sorts"]

                os.makedirs("Memory", exist_ok=True)
                with open("Memory/character.json", "w", encoding="utf-8") as f:
                    json.dump(self.character_data, f, indent=4, ensure_ascii=False)
                self.log("✓ Ressources et sorts mis à jour dans la fiche.")
        except Exception as e:
            self.log(f"⚠ Erreur lors de l'extraction des ressources : {e}")

    def setup_world(self) -> bool:
        """
        Pipeline de setup complet (one-shot, exécuté après la création du PJ).
        Génère scenario_structure.json et initialise progression.json.
        """
        if not self._check_collections():
            return False

        self.setup_logs = [] # Reset logs
        self.log("── Setup du monde ──")
        total_start = time.time()

        json_files = [f for f in os.listdir(config.SCENARIO_DATA_PATH) if f.endswith(".json")]

        try:
            if len(json_files) > 1:
                raise ValueError(
                    f"Plusieurs fichiers JSON de scénario trouvés dans {config.SCENARIO_DATA_PATH} : "
                    f"{json_files}. Un seul scénario structuré est supporté à la fois."
                )
            elif len(json_files) == 1:
                # Cas 1 : scénario déjà structuré (généré par l'outil externe)
                self.log(f"Chargement direct du scénario structuré : {json_files[0]}")
                with open(os.path.join(config.SCENARIO_DATA_PATH, json_files[0]), encoding="utf-8") as f:
                    raw = json.load(f)
                self.scenario_structure, warnings, errors = validate_scenario_structure(raw)
            else:
                # Cas 2 : uniquement des PDF -> extraction via ScenarioExtractorAgent
                self.log("Extraction de la structure du scénario via ScenarioExtractorAgent en 5 passes...")
                raw = self.scenario_extractor_agent.generate(log_callback=self.log)
                self.scenario_structure, warnings, errors = validate_scenario_structure(raw)

            for w in warnings:
                self.log(f"[Validation] {w}")

            if errors:
                for e in errors:
                    self.log(f"[Validation][ERREUR] {e}")
                raise ValueError(
                    "Le scénario contient des erreurs bloquantes non réparables automatiquement "
                    "(voir logs) - correction manuelle ou relance de l'extraction nécessaire "
                    "avant de démarrer une partie."
                )

            os.makedirs("Memory", exist_ok=True)
            with open("Memory/scenario_structure.json", "w", encoding="utf-8") as f:
                json.dump(self.scenario_structure, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.log(f"✗ Erreur lors du setup ou de la validation de la structure : {e}")
            raise e

        # Construire les dictionnaires de lookup direct
        self._build_lookups()

        # Initialiser l'état de progression
        scene_initiale = self.scenario_structure.get("metadata", {}).get("scene_initiale")
        if not scene_initiale and self.scenario_structure.get("noeuds_sceniques"):
            scene_initiale = self.scenario_structure["noeuds_sceniques"][0]["id_scene"]
        elif not scene_initiale:
            scene_initiale = "Inconnu"

        acte_courant = "Inconnu"
        if scene_initiale in self._scenes_by_id:
            acte_courant = self._scenes_by_id[scene_initiale].get("acte_rattache_id", "Inconnu")

        self.progression = {
            "acte_courant": acte_courant,
            "scene_courante": scene_initiale,
            "scenes_resolues": [],
            "scenes_contournees": [],
            "horloges": {
                h["nom"]: {"segments": 0, "declenchee": False} for h in self.scenario_structure.get("horloges_globales", [])
            },
            "ecarts_notables": []
        }

        # Sauvegarder la progression
        try:
            os.makedirs("Memory", exist_ok=True)
            with open("Memory/progression.json", "w", encoding="utf-8") as f:
                json.dump(self.progression, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log(f"⚠ Erreur de sauvegarde de progression.json : {e}")

        self.current_scene_id = scene_initiale

        # Compatibilité avec app.py
        self.scenario_data = {
            "titre": self.scenario_structure["metadata"]["titre"],
            "pitch": self.scenario_structure["metadata"]["pitch_global"],
        }
        try:
            with open("Memory/scenario.json", "w", encoding="utf-8") as f:
                json.dump(self.scenario_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log(f"⚠ Erreur de sauvegarde de scenario.json : {e}")

        # Les PNJs pour le narrateur/orchestrateur
        self.npcs_data = self.scenario_structure.get("entites", {}).get("pnj", [])

        self.log(f"✨ Setup terminé en {time.time() - total_start:.2f}s.")
        return True

    def _build_lookups(self):
        if not self.scenario_structure:
            return
        self._pnj_by_id = {p["id"]: p for p in self.scenario_structure.get("entites", {}).get("pnj", []) if "id" in p}
        self._lieux_by_id = {l["id"]: l for l in self.scenario_structure.get("entites", {}).get("lieux", []) if "id" in l}
        self._scenes_by_id = {s["id_scene"]: s for s in self.scenario_structure.get("noeuds_sceniques", []) if "id_scene" in s}
        self._actes_by_id = {a["id_acte"]: a for a in self.scenario_structure.get("macro_structure", []) if "id_acte" in a}

    def _unwrap_character_data(self, data):
        """Désadresse les données du personnage si elles sont imbriquées dans une clé racine."""
        if not isinstance(data, dict):
            return data

        # Liste des clés racines courantes
        root_keys = ["personnage", "character", "pj", "sheet", "fiche"]

        # Si on n'a qu'une seule clé et qu'elle est dans notre liste
        if len(data) == 1:
            key = list(data.keys())[0]
            if key.lower() in root_keys and isinstance(data[key], dict):
                return data[key]

        # Sinon on cherche si une de ces clés existe au premier niveau et contient un dictionnaire significatif
        for key in root_keys:
            if key in data and isinstance(data[key], dict) and len(data[key]) > 2:
                return data[key]

        return data

    def _load_character_schema(self):
        path = "Memory/character_schema.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"champs_requis": [{"chemin": "nom", "type": "string"}]}

    def _load_action_catalog(self) -> dict:
        path = "Memory/action_catalog.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"actions_courantes": []}

    def _check_resources(self, action):
        """Vérifie si l'action consomme une ressource et si elle est disponible."""
        if not self.character_data or "ressources" not in self.character_data:
            return {"ok": True}

        prompt = f"""Tu es un arbitre de JDR. Analyse l'action du joueur et détermine si elle consomme une ressource limitée de sa fiche.
ACTION : "{action}"
RESSOURCES DISPONIBLES : {json.dumps(self.character_data['ressources'], ensure_ascii=False)}

Réponds UNIQUEMENT avec ce JSON :
{{
  "consomme": boolean,
  "nom_ressource": "nom_de_la_cle_dans_le_json_ou_null",
  "quantite": integer_ou_null,
  "raison": "explication courte"
}}
"""
        try:
            response = self.llm.invoke(prompt).content
            analysis = extract_json(response)
            if analysis and analysis.get("consomme"):
                res_name = analysis.get("nom_ressource")
                qty = analysis.get("quantite") or 1

                # Recherche floue de la ressource si le nom exact n'est pas trouvé
                actual_res_name = None
                if res_name in self.character_data["ressources"]:
                    actual_res_name = res_name
                else:
                    # Recherche insensible à la casse ou partielle
                    for k in self.character_data["ressources"].keys():
                        if res_name.lower() in k.lower() or k.lower() in res_name.lower():
                            actual_res_name = k
                            break

                if actual_res_name:
                    res = self.character_data["ressources"][actual_res_name]
                    if res.get("restants", 0) >= qty:
                        return {"ok": True, "ressource": actual_res_name, "cout": qty}
                    else:
                        return {"ok": False, "raison": f"Ressource insuffisante : {actual_res_name} ({res.get('restants')}/{res.get('total')})"}

            return {"ok": True}
        except Exception as e:
            print(f"[RPGAgent] Erreur check_resources : {e}")
            return {"ok": True}

    def chat(self, user_input):
        if self.game_state == "CREATION":
            # 1. Étape Narrative : Dialogue avec le joueur
            try:
                response = self.character_creator.generate_response(
                    user_input, self.history.messages, self.character_data,
                    champs_manquants=getattr(self, "_missing_character_fields", None)
                )
            except TypeError:
                response = self.character_creator.generate_response(
                    user_input, self.history.messages, self.character_data
                )

            # 2. Étape Technique (Masquée) : Mise à jour de la fiche via SheetManager
            try:
                new_sheet = self.sheet_manager.update_sheet(
                    self.character_data,
                    user_input,
                    response,
                    mode="CREATION"
                )
                if new_sheet and isinstance(new_sheet, dict):
                    self.character_data = self._unwrap_character_data(new_sheet)
                    self.gse.state = self.character_data
                    self.gse.synchronize_and_recalculate()
                    self.character_data = self.gse.state

                schema = self._load_character_schema()
                is_complete, missing_fields = validate_character_sheet(self.character_data, schema)

                if not is_complete and any(
                    kw in response.upper() for kw in ["TERMIN", "PRET POUR L'AVENTURE", "PRÊT POUR L'AVENTURE", "CREATION_TERMINEE", "CRÉATION_TERMINÉE"]
                ):
                    audited = self.sheet_manager.audit_and_complete(
                        self.character_data, self.history.messages, schema
                    )
                    if audited and isinstance(audited, dict):
                        self.character_data = self._unwrap_character_data(audited)
                        self.gse.state = self.character_data
                        self.gse.synchronize_and_recalculate()
                        self.character_data = self.gse.state
                        is_complete, missing_fields = validate_character_sheet(self.character_data, schema)

                os.makedirs("Memory", exist_ok=True)
                with open("Memory/character.json", "w", encoding="utf-8") as f:
                    json.dump(self.character_data, f, indent=4, ensure_ascii=False)

                if is_complete:
                    if not self.character_data:
                        self.character_data = {}
                    self.character_data["statut"] = "complet"
                    self._missing_character_fields = None
                    self._extract_and_add_resources()
                    self.game_state = "SUMMARY"
                else:
                    self._missing_character_fields = missing_fields

            except Exception as e:
                print(f"[RPGAgent] ⚠ Erreur lors de la mise à jour technique de la fiche : {e}")

            self.history.add_user_message(user_input)
            self.history.add_ai_message(response)
            return response

        elif self.game_state == "ADVENTURE":
            # 0. Vérification des ressources
            res_check = self._check_resources(user_input)
            if not res_check["ok"]:
                msg = f"Action impossible : {res_check['raison']}"
                self.history.add_user_message(user_input)
                self.history.add_ai_message(msg)
                return msg

            scenario_context = self.get_scenario_context(user_input, k=config.RAG_K_ADVENTURE)
            npcs_summary = self.get_npcs_context()
            chronicle_text = self.chronicle_data.get("summary", "L'aventure commence.") if self.chronicle_data else "L'aventure commence."

            # Détection mécanique et validation (Actions proactives du joueur)
            action_type = self.gse.detect_action_type(user_input)
            mechanical_result = None

            if action_type == "spell":
                spell_level = 1  # Par défaut, pourra être affiné plus tard
                mechanical_result = self.gse.consume_spell_slot(spell_level)
            elif action_type == "rage":
                mechanical_result = self.gse.consume_resource("points_de_rage")
            elif action_type and action_type.startswith("rest:"):
                palier_id = action_type.split(":", 1)[1]
                mechanical_result = self.gse.rest(palier_id)
            elif action_type == "rest":
                rest_type = "long" if any(w in user_input.lower() for w in ["long", "nuit", "camp"]) else "short"
                mechanical_result = self.gse.rest(rest_type)

            # Ajouter au contexte Orchestrateur
            mechanical_context = ""
            if mechanical_result:
                if not mechanical_result.success:
                    mechanical_context = f"\n⚠ ACTION BLOQUÉE : {mechanical_result.blocked_reason} — {mechanical_result.message}"
                else:
                    mechanical_context = f"\nMECANIQUE : {mechanical_result.message}"

            state_summary = self.gse.get_state_summary()
            self.character_data = self.gse.state # Sync avant l'analyse

            # Catalogue d'actions avant RAG systématique
            action_catalog = self._load_action_catalog()
            analysis_json = None

            if action_catalog.get("actions_courantes"):
                catalog_prompt = f"""Analyse l'action du joueur : "{user_input}"

CATALOGUE D'ACTIONS COURANTES DE CE SYSTÈME :
{json.dumps(action_catalog, ensure_ascii=False, indent=2)}

Historique de l'aventure (Chronique) :
{chronicle_text}

PNJs présents et leurs secrets (si pertinent) :
{json.dumps(self.npcs_data, ensure_ascii=False, indent=2)}

ÉTAT MÉCANIQUE : {state_summary}
{mechanical_context}

Cette action correspond-elle à une entrée du catalogue ci-dessus ? Si oui, applique sa
résolution à la fiche du personnage ({json.dumps(self.character_data, ensure_ascii=False)}).
Si l'action ne correspond à AUCUNE entrée, réponds "couvert_par_catalogue": false SANS
deviner - le système ira chercher dans les règles complètes.

Réponds au format JSON :
{{
    "couvert_par_catalogue": boolean,
    "need_roll": boolean, "stat": "nom_stat_ou_null", "bonus": integer_ou_null,
    "calculation_breakdown": "...", "dc": integer_ou_null, "reason": "...",
    "mechanical_decision": {{"action": "damage" | "heal" | "xp" | null, "amount": integer_ou_null}}
}}
"""
                try:
                    analysis_response = self.llm.invoke(catalog_prompt).content
                    analysis_json = extract_json(analysis_response, expected_type=dict)
                except Exception as e:
                    print(f"[RPGAgent] Erreur lors de l'analyse avec le catalogue : {e}")

            if not analysis_json or analysis_json.get("couvert_par_catalogue") is False:
                core_context = self.get_core_context(user_input, k=config.RAG_K_ADVENTURE)
                analysis_prompt = f"""Analyse l'action du joueur : "{user_input}"
                Basé sur les RÈGLES du CODEX suivantes :
                {core_context}

                Historique de l'aventure (Chronique) :
                {chronicle_text}

                PNJs présents et leurs secrets (si pertinent) :
                {json.dumps(self.npcs_data, ensure_ascii=False, indent=2)}

                ÉTAT MÉCANIQUE : {state_summary}
                {mechanical_context}

                Selon le personnage ({json.dumps(self.character_data, ensure_ascii=False)}), un jet de dé est-il nécessaire ?
                Si oui, identifie le bonus approprié en appliquant RIGOUREUSEMENT les règles du CODEX ci-dessus à la fiche de personnage du joueur.
                Détermine aussi si cette action ou ses conséquences immédiates entraînent des dégâts, des soins ou un gain d'XP.

                Réponds au format JSON :
                {{
                    "need_roll": boolean,
                    "stat": "nom_stat_ou_null",
                    "bonus": integer_ou_null,
                    "calculation_breakdown": "explication du bonus (ex: +3 Force, +2 Athlétisme)",
                    "dc": integer_ou_null,
                    "reason": "explication courte",
                    "mechanical_decision": {{
                        "action": "damage" | "heal" | "xp" | null,
                        "amount": integer_ou_null
                    }}
                }}
                """
                try:
                    analysis_response = self.llm.invoke(analysis_prompt).content
                    analysis_json = extract_json(analysis_response, expected_type=dict)
                except Exception as e:
                    print(f"[RPGAgent] Erreur lors de l'analyse avec le Codex : {e}")

            roll_info = ""
            roll_result = None

            try:
                if not analysis_json:
                    analysis_json = {"need_roll": False}

                # Application de la décision mécanique de l'Orchestrateur (Actions réactives)
                m_decision = analysis_json.get("mechanical_decision")
                if m_decision and m_decision.get("action"):
                    m_res = self.gse.apply_orchestrator_decision(m_decision)
                    mechanical_context += f"\nDECISION MJ : {m_res.message}"
                    self.character_data = self.gse.state # Sync après décision

                if analysis_json.get("need_roll"):
                    die_roll = self.roll_dice(20)
                    bonus = analysis_json.get("bonus")
                    if bonus is None:
                        bonus = 0
                    total = die_roll + bonus
                    stat_name = analysis_json.get("stat", "Inconnu")
                    dc = analysis_json.get("dc", 10)
                    breakdown = analysis_json.get("calculation_breakdown", f"bonus +{bonus}")

                    roll_info = f"Jet de {stat_name} (DC {dc}) : {die_roll} + {bonus} ({breakdown}) = {total}"
                    roll_result = "Succès" if total >= dc else "Échec"
            except Exception:
                analysis_json = {"need_roll": False}

            # 2. Analyse de Scène de l'Orchestrateur (Transition / Improvisation / Contournement)
            scene_analysis_result = {
                "categorie": "improvisation",
                "scene_suivante": None,
                "horloges_impactees": [],
                "ecart_notable": None
            }

            if self.scenario_structure and self.current_scene_id:
                current_scene_dict = self._scenes_by_id.get(self.current_scene_id, {})

                scene_classification_prompt = f"""Analyse l'action du joueur par rapport à la scène courante.

SCÈNE COURANTE (id {self.current_scene_id}) :
Titre : {current_scene_dict.get('titre', 'Inconnu')}
Objectif MJ (ambiance) : {current_scene_dict.get('objectif_mj', '')}
Condition de résolution (le VRAI critère de sortie) : {current_scene_dict.get('condition_resolution', '')}
Sorties logiques anticipées (indicatif) : {json.dumps(current_scene_dict.get('sorties_logiques', []), ensure_ascii=False)}
Défis anticipés (indicatif) : {json.dumps(current_scene_dict.get('defis_et_rencontres', []), ensure_ascii=False)}

ACTION DU JOUEUR : {user_input}

Classe la situation en UNE des trois catégories. Ne bloque JAMAIS une action du joueur,
quelle que soit la catégorie — cette classification sert uniquement au suivi interne.
- "transition" : le joueur résout l'objectif narratif de cette scène d'une manière ou d'une autre (qui concrétise l'objectif formulé, ou s'y oppose mais résout l'esprit), permettant de passer à une scène suivante.
- "improvisation" : le joueur agit dans la scène actuelle sans encore atteindre l'objectif ou faire bifurquer radicalement le scénario. On reste dans la scène courante.
- "contournement" : le joueur sort complètement du cadre prévu, ignore le but, détruit l'opportunité de l'atteindre (ex: tue un donneur de quête clé, fuit l'endroit), ou improvise une solution non anticipée qui court-circuite la scène actuelle sans en faire une transition normale.

Réponds UNIQUEMENT en JSON :
{{
  "categorie": "transition" | "improvisation" | "contournement",
  "scene_suivante": "id ou null",
  "horloges_impactees": [{{"nom": "Nom de l'horloge", "segments_ajoutes": 1}}],
  "ecart_notable": "fait à retenir pour la suite, en une phrase, même si non prévu par le scénario source, ou null"
}}
"""
                try:
                    classification_response = self.llm.invoke(scene_classification_prompt).content
                    extracted_classification = extract_json(classification_response)
                    if extracted_classification:
                        scene_analysis_result = extracted_classification
                except Exception as e:
                    print(f"[RPGAgent] Erreur lors de la classification de scène : {e}")

            # Traitement déterministe de la classification
            cat = scene_analysis_result.get("categorie", "improvisation")
            next_scene_id = scene_analysis_result.get("scene_suivante")
            horloges_impactees = scene_analysis_result.get("horloges_impactees", [])
            ecart_notable = scene_analysis_result.get("ecart_notable")

            if cat not in ["transition", "improvisation", "contournement"]:
                cat = "improvisation"

            if cat == "transition":
                if next_scene_id and next_scene_id in self._scenes_by_id:
                    if self.current_scene_id not in self.progression["scenes_resolues"]:
                        self.progression["scenes_resolues"].append(self.current_scene_id)
                    self.current_scene_id = next_scene_id
                    self.progression["scene_courante"] = next_scene_id
                    next_scene = self._scenes_by_id[next_scene_id]
                    self.progression["acte_courant"] = next_scene.get("acte_rattache_id", "Inconnu")
                    print(f"[RPGAgent] Transition déterministe vers la scène {next_scene_id}.")
                else:
                    print(f"[RPGAgent] Transition avortée : ID scène suivante '{next_scene_id}' invalide ou null.")
                    cat = "improvisation"
            elif cat == "contournement":
                if self.current_scene_id not in self.progression["scenes_contournees"]:
                    self.progression["scenes_contournees"].append(self.current_scene_id)
                if next_scene_id and next_scene_id in self._scenes_by_id:
                    self.current_scene_id = next_scene_id
                    self.progression["scene_courante"] = next_scene_id
                    next_scene = self._scenes_by_id[next_scene_id]
                    self.progression["acte_courant"] = next_scene.get("acte_rattache_id", "Inconnu")
                    print(f"[RPGAgent] Contournement menant vers la scène {next_scene_id}.")
                else:
                    print(f"[RPGAgent] Contournement de la scène {self.current_scene_id} sans scène suivante.")

            # Gestion des horloges globales
            clock_consequences_to_inject = []
            if "horloges" not in self.progression:
                self.progression["horloges"] = {}

            horloges_globales_dict = {h["nom"]: h for h in self.scenario_structure.get("horloges_globales", [])}

            # S'assurer de l'initialisation des horloges dans la progression
            for name, h_meta in horloges_globales_dict.items():
                if name not in self.progression["horloges"]:
                    self.progression["horloges"][name] = {"segments": 0, "declenchee": False}

            for h_impacted in horloges_impactees:
                h_name = h_impacted.get("nom")
                added_segments = h_impacted.get("segments_ajoutes", 0)
                if h_name and h_name in self.progression["horloges"]:
                    meta = horloges_globales_dict[h_name]
                    seuil = meta.get("seuil", 6)
                    consequence = meta.get("consequence", "")

                    current_data = self.progression["horloges"][h_name]
                    old_segments = current_data.get("segments", 0)
                    new_segments = min(old_segments + added_segments, seuil)
                    current_data["segments"] = new_segments

                    # Première fois que le seuil est atteint
                    if new_segments >= seuil and not current_data.get("declenchee", False):
                        current_data["declenchee"] = True
                        clock_consequences_to_inject.append((h_name, consequence))

            # Gérer les écarts notables
            if ecart_notable and str(ecart_notable).lower() != "null":
                if "ecarts_notables" not in self.progression:
                    self.progression["ecarts_notables"] = []
                self.progression["ecarts_notables"].append(ecart_notable)
            else:
                ecart_notable = None

            # Fusionner les conséquences d'horloges dans les écarts de ce tour
            for h_name, consequence in clock_consequences_to_inject:
                if "ecarts_notables" not in self.progression:
                    self.progression["ecarts_notables"] = []
                self.progression["ecarts_notables"].append(consequence)
                if ecart_notable:
                    ecart_notable += f" | {consequence}"
                else:
                    ecart_notable = consequence

            # Sauvegarde de self.progression après chaque tour
            try:
                os.makedirs("Memory", exist_ok=True)
                with open("Memory/progression.json", "w", encoding="utf-8") as f:
                    json.dump(self.progression, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"[RPGAgent] Erreur de sauvegarde de progression.json : {e}")

            # 3. L'Orchestrateur donne ses instructions basées sur le SCÉNARIO et la classification
            current_scene_dict = self._scenes_by_id.get(self.current_scene_id, {})
            current_context = self.get_current_context()

            clock_instruction = ""
            for h_name, consequence in clock_consequences_to_inject:
                clock_instruction += f"\n\nÉVÉNEMENT DÉCLENCHÉ (horloge '{h_name}') : {consequence} — à intégrer immédiatement dans la narration de ce tour."

            decision_instruction = f"""
ACTION DU JOUEUR : {user_input}

RÉSULTAT TECHNIQUE : {"Aucun jet requis" if not roll_info else f"{roll_info} → {roll_result}"}
ÉTAT MÉCANIQUE : {self.gse.get_state_summary()}
{mechanical_context}

CONTEXTE SCÉNARIO (extraits RAG) : {scenario_context}

CONTEXTE SCÉNARIO STRUCTURÉ (Lookup) :
{current_context}

SCÈNE COURANTE :
Id : {self.current_scene_id}
Titre : {current_scene_dict.get('titre', 'Inconnu')}
Esprit : {current_scene_dict.get('esprit_de_la_scene', '')}
Objectif : {current_scene_dict.get('objectif_atteint_si', '')}
Éléments à préserver : {current_scene_dict.get('elements_a_preserver', [])}

CLASSIFICATION DE L'ACTION DU JOUEUR :
Catégorie : {cat}
Fait additionnel (Chronique) : {ecart_notable}{clock_instruction}

PNJ DISPONIBLES : {npcs_summary}
{self._build_structure_instructions()}"""

            final_response = self.narrator.generate_response(user_input, self.history.messages, decision_instruction)

            if roll_info:
                final_response += f"\n\n---\n*🎲 {roll_info} ({roll_result})*"

            self.history.add_user_message(user_input)
            self.history.add_ai_message(final_response)

            # Tâches d'arrière-plan (Chronicle & Fiche) pour réduire la latence
            import threading
            def background_tasks(c_data, u_in, f_resp, e_notable):
                # Mise à jour de la fiche de personnage
                try:
                    new_sheet = self.sheet_manager.update_sheet(c_data, u_in, f_resp)
                    if new_sheet and isinstance(new_sheet, dict):
                        self.character_data = self._unwrap_character_data(new_sheet)
                        os.makedirs("Memory", exist_ok=True)
                        with open("Memory/character.json", "w", encoding="utf-8") as f:
                            json.dump(self.character_data, f, indent=4, ensure_ascii=False)
                        self.gse.reload()
                        print("[RPGAgent] Fiche de personnage mise à jour en arrière-plan.")
                except Exception as e:
                    print(f"[RPGAgent] ⚠ Erreur lors de la mise à jour de la fiche en arrière-plan : {e}")

                # Mise à jour de la chronique avec l'intégration de ecart_notable
                try:
                    self.update_chronicle(u_in, f_resp, e_notable)
                except Exception as e:
                    print(f"[RPGAgent] ⚠ Erreur lors de la mise à jour de la chronique en arrière-plan : {e}")

            threading.Thread(
                target=background_tasks,
                args=(self.character_data.copy() if isinstance(self.character_data, dict) else self.character_data, user_input, final_response, ecart_notable),
                daemon=True
            ).start()

            return final_response

    def start_adventure(self):
        """Lance le pipeline de setup puis démarre la narration."""
        if not self.setup_world():
            return "Erreur lors de la génération du monde."

        self.log("Génération de l'introduction narrative...")
        intro_start = time.time()
        self.game_state = "ADVENTURE"

        pitch = self.scenario_data.get('pitch', 'Une nouvelle aventure commence.')
        situation = "Le héros se tient prêt."
        scene_initiale_id = self.scenario_structure.get("metadata", {}).get("scene_initiale")
        if scene_initiale_id in self._scenes_by_id:
            scene_init = self._scenes_by_id[scene_initiale_id]
            situation = f"Scène initiale : {scene_init.get('titre')}. Objectif MJ : {scene_init.get('objectif_mj')}."

        setup_context = self.get_scenario_context("intrigue lieux personnages", k=config.RAG_K_SETUP)
        current_context = self.get_current_context()

        intro_instruction = f"""
ACTION DU JOUEUR : L'aventure commence !

RÉSULTAT TECHNIQUE : Aucun jet requis

CONTEXTE SCÉNARIO (extraits RAG) :
- Pitch : {pitch}
- Situation initiale : {situation}
- Détails supplémentaires : {setup_context}

CONTEXTE SCÉNARIO STRUCTURÉ (Lookup) :
{current_context}
{self._build_structure_instructions()}"""

        intro_response = self.narrator.generate_response(
            "L'aventure commence !", self.history.messages, intro_instruction
        )
        self.log(f"✓ Introduction générée en {time.time() - intro_start:.2f}s.")

        full_response = f"**{self.scenario_data.get('titre', 'Aventure')}**\n\n*{pitch}*\n\n{intro_response}"

        self.update_chronicle("L'aventure commence !", full_response)
        self.history.add_ai_message(full_response)
        return full_response

    def update_chronicle(self, user_input, response, ecart_notable=None):
        old_summary = ""
        if self.chronicle_data and isinstance(self.chronicle_data, dict):
            old_summary = self.chronicle_data.get("summary", "")

        new_summary = self.chronicle_agent.update(old_summary, user_input, response, ecart_notable)
        self.chronicle_data = {"summary": new_summary}

        os.makedirs("Memory", exist_ok=True)
        with open("Memory/Chronicle.json", "w", encoding="utf-8") as f:
            json.dump(self.chronicle_data, f, indent=4, ensure_ascii=False)

    def load_character(self):
        """Charge uniquement le personnage et passe en mode SUMMARY."""
        try:
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.character_data = self._unwrap_character_data(data)
                self.gse.reload()
                self.game_state = "SUMMARY"
                nom = self.character_data.get('nom') if self.character_data else "Inconnu"
                print(f"[RPGAgent] Personnage chargé : {nom}")

            if os.path.exists("Memory/scenario_structure.json"):
                with open("Memory/scenario_structure.json", "r", encoding="utf-8") as f:
                    self.scenario_structure = json.load(f)
                self._build_lookups()
                # Compatibilité pour self.scenario_data
                self.scenario_data = {
                    "titre": self.scenario_structure["metadata"]["titre"],
                    "pitch": self.scenario_structure["metadata"]["pitch_global"],
                }
                self.npcs_data = self.scenario_structure.get("entites", {}).get("pnj", [])

            if os.path.exists("Memory/progression.json"):
                with open("Memory/progression.json", "r", encoding="utf-8") as f:
                    self.progression = json.load(f)
                self.current_scene_id = self.progression.get("scene_courante")

                return True
        except Exception as e:
            print(f"[RPGAgent] Erreur chargement personnage : {e}")
        return False

    def load_game(self):
        """Charge la sauvegarde complète (PJ + PNJ + Scénario + Chronique)."""
        try:
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.character_data = self._unwrap_character_data(data)
                self.gse.reload()

            if os.path.exists("Memory/scenario_structure.json"):
                with open("Memory/scenario_structure.json", "r", encoding="utf-8") as f:
                    self.scenario_structure = json.load(f)
                self._build_lookups()
                # Compatibilité pour self.scenario_data
                self.scenario_data = {
                    "titre": self.scenario_structure["metadata"]["titre"],
                    "pitch": self.scenario_structure["metadata"]["pitch_global"],
                }
                self.npcs_data = self.scenario_structure.get("entites", {}).get("pnj", [])

            if os.path.exists("Memory/progression.json"):
                with open("Memory/progression.json", "r", encoding="utf-8") as f:
                    self.progression = json.load(f)
                self.current_scene_id = self.progression.get("scene_courante")

            if os.path.exists("Memory/Chronicle.json"):
                with open("Memory/Chronicle.json", "r", encoding="utf-8") as f:
                    self.chronicle_data = json.load(f)

            if self.character_data and self.scenario_structure and self.progression:
                self.game_state = "ADVENTURE"
                nb_npcs = len(self.npcs_data) if self.npcs_data else 0
                print(f"[RPGAgent] Partie chargée — {nb_npcs} PNJ disponibles.")
                return True
            elif self.character_data:
                self.game_state = "SUMMARY"
                print(f"[RPGAgent] Personnage chargé (partie incomplète).")
                return True

        except Exception as e:
            print(f"[RPGAgent] Erreur chargement : {e}")

        return False

    def clear_history(self):
        """Réinitialise complètement la partie."""
        self.history.clear()
        self.game_state = "CREATION"
        self.character_data = None
        self.scenario_data = None
        self._missing_character_fields = None
        self.chronicle_data = None
        self.npcs_data = None
        self.current_scene_id = None
        self.progression = None
        self.scenario_structure = None

        from game_state_engine import GameStateEngine
        self.gse = GameStateEngine()  # réinitialise l'état en mémoire

        for file in ["character.json", "Chronicle.json", "progression.json"]:
            path = os.path.join("Memory", file)
            if os.path.exists(path):
                os.remove(path)
                print(f"[RPGAgent] Supprimé : {path}")

    def get_npc(self, npc_id: str) -> dict | None:
        """Retourne la fiche d'un PNJ par son id, ou None."""
        if not self.npcs_data:
            return None
        return next((n for n in self.npcs_data if n.get("id") == npc_id), None)

    def get_npcs_context(self) -> str:
        """
        Retourne un résumé des PNJ (SANS leurs secrets) pour le Narrateur.
        Les secrets restent côté Orchestrateur uniquement.
        """
        if not self.npcs_data:
            return "Aucun PNJ disponible."
        lines = []
        for n in self.npcs_data:
            lines.append(
                f"- {n['nom_complet']} "
                f"| Attitude: {n.get('attitude_initiale', 'Inconnue')} "
                f"| Lieu: {n.get('localisation_habituelle', '?')}"
            )
        return "\n".join(lines)

    def _build_structure_instructions(self) -> str:
        """Bloc d'instructions de structure narrative, identique entre
        start_adventure() et chat() (ADVENTURE) - ne contient aucune donnée
        contextuelle spécifique à l'appelant."""
        return """
TON RÔLE : Tu es le MJ. Génère des instructions précises pour le Narrateur en tenant compte du contexte fourni ci-dessus.

STRUCTURE OBLIGATOIRE de ta réponse :

1. CONSÉQUENCE IMMÉDIATE
   Ce qui se passe concrètement suite à l'action. Si jet réussi : avantage clair.
   Si jet échoué : complication, ou fausse piste.
   Ne révèle que ce que le personnage peut percevoir à cet instant.

2. PERCEPTIONS SENSORIELLES
   Ce que le personnage voit, entend, sent, touche ou ressent physiquement.
   Sois précis et concret — pas d'atmosphère vague.

3. ÉLÉMENTS INCONNUS OU AMBIGUS
   Ce que le personnage ne peut pas encore déterminer.

4. IMPULSION NARRATIVE
   Donne une direction active au joueur : un détail qui appelle une réaction.

5. POINTS CLÉS POUR LE RÉSUMÉ
   Le résumé du Narrateur ne doit reprendre que des faits explicitement énoncés dans le texte narratif de cette même réponse - jamais une information supplémentaire, une interprétation, ou un contenu d'indice pas encore découvert par le joueur.
"""
