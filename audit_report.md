# Rapport d'Audit Complet - RPG Oracle

Ce rapport présente une analyse globale de l'implémentation actuelle du projet, en se concentrant sur trois axes majeurs : l'architecture générale, la pertinence des prompts et des agents, et l'optimisation des performances dans un contexte d'exécution locale (Ollama, RTX 5070, 32 Go RAM).

---

## 1. Architecture et Implémentation Globale

Le projet repose sur une architecture solide, distinguant intelligemment la narration textuelle de la gestion déterministe des mécaniques de jeu.

### Points Forts
- **Séparation des responsabilités (Zero-LLM pour les mathématiques)** : Le `GameStateEngine` gère avec succès les mécaniques de jeu (points de vie, ressources) de manière déterministe en Python. Cela empêche les hallucinations mathématiques communes aux LLMs.
- **Gestion d'état explicite** : L'utilisation de fichiers JSON explicites dans le répertoire `Memory/` (`progression.json`, `character.json`, `scenario_structure.json`) permet un suivi fiable et traçable de la partie.
- **Pipeline de démarrage robuste** : Le système de validation et d'auto-réparation du scénario (`validation.py`) protège efficacement contre des corruptions logiques de la structure narrative.

### Axe d'Amélioration
- L'encapsulation de tous les agents derrière la méthode `_invoke_logged` de `BaseAgent` est très pertinente pour le debug, mais la gestion des JSON mal formattés reste manuelle (via `extract_json` dans `base_utils.py`). Bien que robuste, cela dépend grandement de la qualité de base du LLM.

---

## 2. Pertinence des Prompts et des Agents

L'approche multi-agents (Orchestrateur, Narrateur, SheetManager, CharacterCreator, Chronicle) est très intéressante pour séparer les instructions.

### Points Forts
- **Rôle strict du Narrateur** : Les instructions absolues du `Narrator` (Ne jamais modifier les règles, ne jamais donner de spoil direct) forcent un bon comportement descriptif et immersif.
- **Extraction intelligente en phases** : Le `ScenarioExtractorAgent` est excellemment conçu en procédant par 5 passes séquentielles, ce qui garantit la cohérence des structures massives.

### Points d'Attention et Surcharges
1. **Dissonance cognitive pour le SheetManagerAgent** :
   - Le prompt du `SheetManagerAgent` contient des instructions potentiellement contradictoires. Il est précisé en RÈGLE CRITIQUE : *"Tu ne dois JAMAIS modifier les Points de Vie (PV)..."* mais juste en dessous, dans les CONSIGNES : *"Mets à jour les Points de Vie (PV) si le personnage a été blessé ou soigné."*
   - Cela crée une confusion majeure pour le LLM, gaspillant de l'attention et augmentant les chances d'erreur ou d'hallucination (notamment en écrasant le travail déterministe du `GameStateEngine`).
2. **Prompts très denses en phase de création** :
   - Le prompt du `CharacterCreator` contient beaucoup de consignes sur des mécaniques spécifiques. Si le RAG injecte en plus des manuels très longs, la fenêtre de contexte devient saturée, diluant l'instruction principale : "être conversationnel".
3. **Double vérification de progression** : L'orchestrateur vérifie et génère la structure via un LLM "classificateur" avant d'envoyer l'ordre au narrateur. C'est sécurisant mais lourd en ressources locales.

---

## 3. Optimisation des Performances (Contexte Local : Gemma4:26b)

Dans ta configuration (Ollama, RTX 5070 12Go VRAM, 32 Go RAM système), l'utilisation d'un modèle de taille conséquente (26 Milliards de paramètres) entraîne un *Prompt Processing* très lourd.

### Constats sur le goulot d'étranglement (Temps de réponse du LLM)
Actuellement, pour un seul message du joueur en mode `ADVENTURE`, l'orchestrateur exécute la boucle séquentielle suivante :
1. (Si sort/capacité) Analyse via `GameStateEngine` (Rapide)
2. **Appel LLM 1** : Agent Classificateur (Transition, Improvisation, Contournement).
3. RAG Similarity Search sur Core et Scénario.
4. **Appel LLM 2** : `Narrator` générant la réponse au joueur.
5. **Appel LLM 3** : `SheetManagerAgent` qui lit la réponse pour mettre à jour la fiche.
6. **Appel LLM 4** : `ChronicleAgent` qui résume l'action.

**Impact :** Sur une RTX 5070, traiter 4 requêtes séquentielles avec Gemma4:26b requiert de recharger/évaluer le contexte en VRAM plusieurs fois par tour. Cela entraîne un délai (TTFT - Time To First Token) qui peut facilement atteindre 15 à 30 secondes avant que l'interface Streamlit n'affiche quoi que ce soit.

### Recommandations d'Optimisation

1. **Parallélisation des Tâches en Arrière-plan (Asynchronisme)**
   - Seul le `Narrator` (Appel 2) devrait bloquer la réponse au joueur.
   - Les appels au `SheetManagerAgent` et au `ChronicleAgent` peuvent être envoyés dans un thread asynchrone (via `asyncio` ou un `ThreadPoolExecutor`) en arrière-plan pendant que le joueur commence déjà à lire la réponse dans Streamlit.

2. **Modèles Hétérogènes (Routing)**
   - Utiliser *Gemma4:26b* pour la narration et l'orchestration est pertinent.
   - Cependant, utiliser ce même modèle pour des tâches mineures comme classifier l'action en 1 mot, ou mettre à jour un JSON, est excessif (Overkill).
   - *Action* : Configurer `.env` pour utiliser un modèle ultra-léger et rapide (comme `qwen2.5-coder:7b` ou `llama3.1:8b`) pour le `SHEET_MANAGER_MODEL` et la classification des actions, réservant le grand modèle au `NARRATOR_MODEL`.

3. **Réduction de la Profondeur du RAG (K)**
   - Les variables `RAG_K_ADVENTURE` et `RAG_K_SETUP` sont à 5 et 8 par défaut.
   - Sur un grand modèle, l'injection de 5 à 8 gros chunks dans le prompt alourdit considérablement l'étape de pré-calcul. Réduire ces valeurs à 3 ou filtrer le contexte via un outil de "reranking" pourrait diviser par deux le temps de préparation du prompt du narrateur.

4. **Regroupement des requêtes (Prompt Fusion)**
   - Si l'asynchronisme n'est pas souhaité, il est possible de fusionner les instructions du Classificateur d'Action et du Narrateur en un seul grand prompt structuré demandant une réponse JSON contenant `{"type_action": "transition", "narration": "..."}`. Cela économiserait un tour de chauffe LLM complet.

---

## Conclusion

Le projet est architecturalement excellent sur la séparation narratif / déterministe.
Cependant, sa lourdeur actuelle provient de l'approche séquentielle (jusqu'à 4 appels LLM par tour).

**Prochaines étapes suggérées :**
1. Corriger les contradictions dans le prompt du `SheetManagerAgent`.
2. Implémenter l'exécution asynchrone pour la Chronique et la Fiche de Personnage.
3. Affiner les paramètres du RAG et encourager l'utilisation de modèles plus petits pour les sous-agents d'extraction JSON.