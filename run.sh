#!/bin/bash

# Valeurs par défaut
RESET=false

# Analyse des arguments
for arg in "$@"; do
  case $arg in
    --reset)
      RESET=true
      shift
      ;;
  esac
done

echo "Installation des dépendances..."
pip install -r requirements.txt

if [ "$RESET" = true ]; then
    echo "🚨 Réinitialisation complète demandée..."

    # Réinitialisation de la mémoire via Python
    python3 -c "import memory_manager; memory_manager.reset_memory()"

    # Suppression de la base de données Chroma
    CHROMA_DIR=$(python3 -c "import config; print(config.CHROMA_PATH)")
    if [ -d "$CHROMA_DIR" ]; then
        echo "🗑️ Suppression de la base de données existante : $CHROMA_DIR"
        rm -rf "$CHROMA_DIR"
    fi

    # Assurer que les dossiers de données existent
    mkdir -p data/codex data/intrigue

    # Réindexation
    echo "📚 Indexation du Codex..."
    python3 indexer.py codex

    echo "🗺️ Indexation de l'Intrigue..."
    python3 indexer.py intrigue
fi

echo "🚀 Lancement de l'application Streamlit..."
streamlit run app.py
