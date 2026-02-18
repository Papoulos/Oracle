# RPG Oracle - Phase 1

Système d'agent conversationnel de jeu de rôle basé sur des agents (RAG).

## Architecture
- **Backend** : Python, LangChain, Ollama (Gemma 3)
- **Base de données** : ChromaDB (Vectorielle)
- **Interface** : Streamlit

## Installation

1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Assurez-vous qu'Ollama est lancé localement avec le modèle gemma3 :
   ```bash
   ollama run gemma3
   ```

3. Générez les fichiers d'exemple et indexez-les :
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   **Important** : L'indexation nécessite qu'Ollama soit en cours d'exécution pour générer les embeddings. Si vous ajoutez de nouveaux PDF dans `data/`, relancez l'indexation.

   (Note : `run.sh` lancera également l'application Streamlit après l'indexation).

## Structure du projet
- `indexer.py` : Outil pour indexer les PDF dans les bases CODEX ou INTRIGUE.
- `app.py` : Interface Streamlit de l'agent de test.
- `config.py` : Paramètres de configuration.
- `data/` : Dossier contenant les sources PDF (répartis en `codex` et `intrigue`).
- `chroma_db/` : Stockage des vecteurs.
