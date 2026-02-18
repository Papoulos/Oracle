#!/bin/bash

echo "Installation des dépendances..."
pip install -r requirements.txt

echo "Génération des exemples..."
python3 generate_samples.py

echo "Indexation du Codex..."
python3 indexer.py codex

echo "Indexation de l'Intrigue..."
python3 indexer.py intrigue

echo "Lancement de l'application Streamlit..."
streamlit run app.py
