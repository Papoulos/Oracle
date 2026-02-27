# RPG Oracle - Version Simplifiée

Ce projet est une version simplifiée du système RPG Oracle, utilisant un agent unique avec RAG (Retrieval-Augmented Generation) et mémoire en RAM.

## Structure
- `agent.py` : Logique de l'agent RPGAgent (LangChain + Ollama).
- `indexer.py` : Outil CLI pour indexer les PDFs du dossier `data/`.
- `app.py` : Interface utilisateur Streamlit.
- `config.py` : Configuration globale.
- `data/` : Dossier où placer les PDFs à indexer.

## Installation

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
   streamlit run app.py
   ```
