import streamlit as st
from agent import RPGAgent
import config

st.set_page_config(page_title="RPG Oracle - Simplified", page_icon="🎲")

st.title("🎲 RPG Oracle")
st.caption("Votre assistant de jeu de rôle intelligent")

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

# Affichage de l'historique
for message in st.session_state.agent.history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Zone de saisie
if prompt := st.chat_input("Comment puis-je vous aider dans votre aventure ?"):
    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Le MJ réfléchit..."):
            response = st.session_state.agent.chat(prompt)
            st.markdown(response)
