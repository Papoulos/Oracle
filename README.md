# RPG Oracle - Version Simplifiée

Ce projet est une version simplifiée du système RPG Oracle, utilisant un agent unique avec RAG (Retrieval-Augmented Generation) et mémoire en RAM.

## Structure
- `agent.py` : Logique de l'agent RPGAgent (LangChain + Ollama).
- `indexer.py` : Outil CLI pour indexer les PDFs du dossier `data/`.
- `app.py` : Interface utilisateur Streamlit.
- `config.py` : Configuration globale.
- `data/` : Dossier où placer les PDFs à indexer.

## Installation

**Prérequis** : Python 3.9 à 3.13 (Python 3.14+ n'est pas encore supporté par ChromaDB).

1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Assurez-vous qu'Ollama est lancé avec les modèles requis :
   ```bash
   ollama run gemma3
   ollama pull nomic-embed-text
   ```

## Utilisation

1. **Indexation** : Placez vos documents PDF dans le dossier `data/` et lancez :
   ```bash
   python indexer.py
   ```

2. **Lancement de l'interface** :
   ```bash
   python run.py
   ```
   *Note : Vous pouvez configurer l'adresse IP et le port dans `config.py` ou via les variables d'environnement `SERVER_ADDRESS` et `SERVER_PORT`.*

## Architecture des Agents

Le système repose sur une architecture multi-agents coordonnée, où chaque agent possède un rôle spécifique et des paramètres de configuration dédiés.

### 1. Orchestrateur (`RPGAgent`)
- **Rôle fonctionnel** : C'est le cerveau du système. Il gère la logique globale, les transitions d'état (Création, Résumé, Aventure), effectue les jets de dés (D20) et analyse techniquement les actions du joueur.
- **Détails techniques** :
  - **Modèle** : Défini par `ORCHESTRATOR_MODEL` (température `ORCHESTRATOR_TEMP`).
  - **Sources de données** : Accède à `core_collection` (règles) et `scenario_collection` (intrigue).
  - **Responsabilité** : Il donne des instructions au Narrateur basées sur l'analyse technique des règles et du scénario.

### 2. Créateur de Personnage (`CharacterCreator`)
- **Rôle fonctionnel** : Guide le joueur dans la conception de son personnage. Il propose les options (races, classes) et s'assure que toutes les étapes (statistiques, équipement) sont respectées.
- **Détails techniques** :
  - **Modèle** : Défini par `CHARACTER_MODEL` (température `CHARACTER_TEMP`).
  - **Sources de données** : Utilise exclusivement `core_collection` pour garantir le respect des règles.
  - **Sortie** : Génère un bloc JSON final qui verrouille la fiche de personnage.

### 3. Narrateur (`Narrator`)
- **Rôle fonctionnel** : La voix du Maître du Jeu. Il transforme les décisions de l'Orchestrateur en un récit immersif, interprète les PNJs et décrit les environnements.
- **Détails techniques** :
  - **Modèle** : Défini par `NARRATOR_MODEL` (température `NARRATOR_TEMP`).
  - **Fonctionnement** : Reçoit des instructions précises de l'Orchestrateur et s'appuie sur l'historique des échanges.
  - **Format** : Termine chaque intervention par une question et un bloc "📌 Résumé des informations".

### 4. Chroniqueur (`ChronicleAgent`)
- **Rôle fonctionnel** : Historien de l'aventure. Il maintient un résumé factuel et concis des événements au fur et à mesure de la progression.
- **Détails techniques** :
  - **Modèle** : Défini par `CHRONICLE_MODEL` (température `CHRONICLE_TEMP`, par défaut 0.1).
  - **Persistance** : Met à jour le fichier `Memory/Chronicle.json` après chaque interaction.
  - **Objectif** : Fournir une mémoire à long terme résumée pour les sessions prolongées.

## Fonctionnalités de Reprise de Session

Le système vérifie automatiquement la présence de fichiers de sauvegarde dans le dossier `Memory/` lors du lancement :

1. **Reprise de Personnage seul** : Si un fichier `character.json` est détecté sans scénario associé, l'interface affiche la fiche du personnage et propose de lancer une nouvelle aventure avec lui ou d'en créer un nouveau.
2. **Reprise de Partie complète** : Si `character.json` et `scenario.json` sont présents, l'utilisateur peut reprendre la partie là où il s'est arrêté ou démarrer une nouvelle partie.
3. **Résumé de Session** : Un bouton "📋 Afficher le résumé de la partie" permet de prévisualiser les statistiques du personnage et la dernière chronique avant de confirmer la reprise.
