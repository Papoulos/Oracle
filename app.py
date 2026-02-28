import streamlit as st
import json
import re
import os
from agent import RPGAgent
import config

st.set_page_config(page_title="RPG Oracle - Multi-Agents", page_icon="🎲")

st.title("🎲 RPG Oracle")
st.caption("Votre assistant de jeu de rôle intelligent (Multi-Agents)")

# Initialisation de l'agent dans la session
if "agent" not in st.session_state:
    st.session_state.agent = RPGAgent()

# Sidebar pour les options
with st.sidebar:
    st.header("Options")
    if st.button("Réinitialiser la conversation"):
        st.session_state.agent.clear_history()
        st.rerun()

    st.info(f"Modèle : {config.OLLAMA_MODEL}")

    if st.session_state.agent.character_data:
        st.header("👤 Personnage")
        st.json(st.session_state.agent.character_data)

    if st.session_state.agent.scenario_data:
        st.header("📜 Scénario")
        st.write(f"**{st.session_state.agent.scenario_data.get('titre', 'Aventure')}**")
        st.write(st.session_state.agent.scenario_data.get('intrigue', ''))

# Affichage de l'historique
for message in st.session_state.agent.history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Message de bienvenue automatique (Création de perso)
if not st.session_state.agent.history.messages and st.session_state.agent.game_state == "CREATION":
    welcome_msg = "Bienvenue ! Commençons la création de votre personnage. Quel nom souhaitez-vous lui donner ?"
    with st.chat_message("assistant"):
        st.markdown(welcome_msg)
    st.session_state.agent.history.add_ai_message(welcome_msg)

# Interface spécifique selon l'état du jeu
if st.session_state.agent.game_state == "SUMMARY":
    st.success("La création de votre personnage est terminée !")
    if st.button("🚀 Lancer l'aventure"):
        with st.spinner("Génération du scénario et introduction..."):
            intro = st.session_state.agent.start_adventure()
            st.rerun()

# Zone de saisie (désactivée en mode SUMMARY)
if st.session_state.agent.game_state != "SUMMARY":
    if prompt := st.chat_input("Votre réponse..."):
        # Affichage du message utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)

        # Réponse de l'agent
        with st.chat_message("assistant"):
            with st.spinner("L'Orchestrateur et le Narrateur se concertent..."):
                response = st.session_state.agent.chat(prompt)
                st.markdown(response)
                # Si on vient de finir la création, on force un rerun pour afficher le bouton SUMMARY
                if st.session_state.agent.game_state == "SUMMARY":
                    st.rerun()
