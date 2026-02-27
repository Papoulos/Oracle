import streamlit as st
import json
import re
import os
from agent import RPGAgent
import config

st.set_page_config(page_title="RPG Oracle - Simplified", page_icon="🎲")

st.title("🎲 RPG Oracle")
st.caption("Votre assistant de jeu de rôle intelligent")

# Initialisation de l'agent dans la session
if "agent" not in st.session_state:
    st.session_state.agent = RPGAgent()

def save_character_json(text):
    """Extrait et sauvegarde le JSON du personnage si présent."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            char_data = json.loads(json_match.group(1))
            os.makedirs("Memory", exist_ok=True)
            with open("Memory/character.json", "w", encoding="utf-8") as f:
                json.dump(char_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde du JSON : {e}")
    return False

# Sidebar pour les options
with st.sidebar:
    st.header("Options")
    if st.button("Réinitialiser la conversation"):
        st.session_state.agent.clear_history()
        st.rerun()

    st.info(f"Modèle : {config.OLLAMA_MODEL}")

# Affichage de l'historique
for message in st.session_state.agent.history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Message de bienvenue automatique
if not st.session_state.agent.history.messages:
    welcome_msg = "Bienvenue ! Commençons la création de votre personnage. Quel nom souhaitez-vous lui donner ?"
    with st.chat_message("assistant"):
        st.markdown(welcome_msg)
    st.session_state.agent.history.add_ai_message(welcome_msg)

# Zone de saisie
if prompt := st.chat_input("Votre réponse..."):
    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Le MJ réfléchit..."):
            response = st.session_state.agent.chat(prompt)
            st.markdown(response)
            if save_character_json(response):
                st.success("Personnage sauvegardé dans Memory/character.json !")
