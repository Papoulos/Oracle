import streamlit as st
import json
import re
import os
from agent import RPGAgent
import config

st.set_page_config(page_title="RPG Oracle - Multi-Agents", page_icon="🎲")

def display_character_info(character_data):
    st.markdown(f"### 👤 {character_data.get('nom', 'Aventurier')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Classe :** {character_data.get('classe', 'N/A')}")
        if 'race' in character_data: st.markdown(f"**Race :** {character_data.get('race')}")
        if 'niveau' in character_data: st.markdown(f"**Niveau :** {character_data.get('niveau')}")

    with col2:
        if 'points_de_vie' in character_data: st.markdown(f"**PV :** {character_data.get('points_de_vie')}")
        elif 'pv' in character_data: st.markdown(f"**PV :** {character_data.get('pv')}")
        if 'ca' in character_data: st.markdown(f"**CA :** {character_data.get('ca')}")

    # Essayer de trouver les stats (souvent dans 'statistiques' ou 'stats')
    stats = character_data.get('statistiques') or character_data.get('stats')
    if stats and isinstance(stats, dict):
        st.markdown("**Statistiques :**")
        st.info(" | ".join([f"**{k.capitalize()}**: {v}" for k,v in stats.items()]))

    if 'équipement' in character_data:
        equip = character_data['équipement']
        if isinstance(equip, list):
            st.markdown(f"**Équipement :** {', '.join(equip)}")
        else:
            st.markdown(f"**Équipement :** {equip}")

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
        st.session_state.pop("game_loaded", None)
        st.rerun()

    st.info(f"Modèle : {config.OLLAMA_MODEL}")
    st.info(f"Interface : http://{config.SERVER_ADDRESS}:{config.SERVER_PORT}")

    if st.session_state.agent.character_data:
        st.header("👤 Personnage")
        st.json(st.session_state.agent.character_data)

    if st.session_state.agent.scenario_data:
        st.header("📜 Scénario")
        st.write(f"**{st.session_state.agent.scenario_data.get('titre', 'Aventure')}**")
        st.write(st.session_state.agent.scenario_data.get('pitch', ''))

    if st.session_state.agent.chronicle_data:
        st.header("📖 Chronique")
        st.write(st.session_state.agent.chronicle_data.get('summary', ''))

# Gestion du chargement de la partie
if "game_loaded" not in st.session_state:
    has_character = os.path.exists("Memory/character.json")
    has_scenario = os.path.exists("Memory/scenario.json")

    if has_character and has_scenario:
        st.info("Une partie sauvegardée a été détectée.")

        if st.button("📋 Afficher le résumé de la partie"):
            st.session_state.show_resume = not st.session_state.get("show_resume", False)

        if st.session_state.get("show_resume"):
            with st.expander("Résumé de la partie", expanded=True):
                try:
                    with open("Memory/character.json", "r", encoding="utf-8") as f:
                        char_data = json.load(f)
                    display_character_info(char_data)

                    if os.path.exists("Memory/Chronicle.json"):
                        st.markdown("---")
                        st.markdown("### 📖 Résumé de l'aventure")
                        with open("Memory/Chronicle.json", "r", encoding="utf-8") as f:
                            chronicle = json.load(f)
                            st.write(chronicle.get("summary", "Pas de résumé."))
                except Exception as e:
                    st.error(f"Erreur lors de la lecture des fichiers de sauvegarde : {e}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Reprendre la partie"):
                if st.session_state.agent.load_game():
                    st.session_state.game_loaded = True
                    st.rerun()
                else:
                    st.error("Échec du chargement de la partie.")
        with col2:
            if st.button("🆕 Nouvelle partie"):
                st.session_state.agent.clear_history()
                st.session_state.game_loaded = True
                st.rerun()
        st.stop()

    elif has_character:
        st.info("Un personnage existant a été détecté.")

        with st.expander("Voir la fiche du personnage", expanded=True):
            try:
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    char_data = json.load(f)
                display_character_info(char_data)
            except Exception as e:
                st.error(f"Erreur lors de la lecture de la fiche personnage : {e}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Lancer l'aventure"):
                if st.session_state.agent.load_character():
                    st.session_state.game_loaded = True
                    # Le passage en mode SUMMARY dans load_character affichera le bouton de lancement
                    st.rerun()
                else:
                    st.error("Échec du chargement du personnage.")
        with col2:
            if st.button("🆕 Créer un nouveau personnage"):
                st.session_state.agent.clear_history()
                st.session_state.game_loaded = True
                st.rerun()
        st.stop()

    else:
        st.session_state.game_loaded = True

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

# Message de bienvenue pour partie reprise
if not st.session_state.agent.history.messages and st.session_state.agent.game_state == "ADVENTURE" and st.session_state.agent.chronicle_data:
    pitch = st.session_state.agent.scenario_data.get('pitch', '')
    resume_msg = f"Ravi de vous revoir ! \n\n**Rappel de l'intrigue :** {pitch}\n\n**Où nous en étions :**\n\n{st.session_state.agent.chronicle_data.get('summary', '')}\n\nQue souhaitez-vous faire ?"
    with st.chat_message("assistant"):
        st.markdown(resume_msg)
    st.session_state.agent.history.add_ai_message(resume_msg)

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
