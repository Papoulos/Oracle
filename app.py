import streamlit as st
import json
import re
import os
from agent import RPGAgent
import config

st.set_page_config(page_title="RPG Oracle - Multi-Agents", page_icon="🎲")

def display_character_info(character_data):
    if not character_data:
        st.warning("Aucune donnée de personnage disponible.")
        return

    # Si les données sont imbriquées (ex: {"personnage": {...}})
    if len(character_data) == 1 and isinstance(list(character_data.values())[0], dict):
        key = list(character_data.keys())[0]
        if key.lower() in ["personnage", "character", "pj"]:
            character_data = character_data[key]

    # Liste des clés déjà affichées pour éviter les doublons dans le "catch-all"
    displayed_keys = set()

    def get_val(keys, default="N/A"):
        nonlocal displayed_keys
        if isinstance(keys, str): keys = [keys]

        def search(data, target_keys):
            for k in target_keys:
                if k in data:
                    return k, data[k]
                for actual_key in data.keys():
                    if actual_key.lower() == k.lower():
                        return actual_key, data[actual_key]
            return None, None

        # Recherche récursive (limitée à 2 niveaux pour éviter les boucles)
        def deep_search(data, target_keys, depth=0):
            if depth > 1: return None, None
            k, v = search(data, target_keys)
            if v is not None: return k, v

            for v_child in data.values():
                if isinstance(v_child, dict):
                    k_res, v_res = deep_search(v_child, target_keys, depth + 1)
                    if v_res is not None: return k_res, v_res
            return None, None

        actual_key, value = deep_search(character_data, keys)
        if value is not None:
            if actual_key in character_data:
                displayed_keys.add(actual_key)
            return value
        return default

    nom = get_val(["nom", "name"], "Aventurier Inconnu")
    classe = get_val(["classe", "class"])
    race = get_val("race")
    niveau = get_val(["niveau", "level"])
    pv = get_val(["points_de_vie", "pv", "hp", "health"])
    ca = get_val(["ca", "ac", "armure"])

    st.markdown(f"### 👤 {nom}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Classe :** {classe}")
        if race != "N/A": st.markdown(f"**Race :** {race}")
        if niveau != "N/A": st.markdown(f"**Niveau :** {niveau}")

    with col2:
        if pv != "N/A": st.markdown(f"**PV :** {pv}")
        if ca != "N/A": st.markdown(f"**CA :** {ca}")

    # Essayer de trouver les stats (souvent dans 'statistiques', 'stats' ou 'abilities')
    stats = None
    stats_keys = ["statistiques", "stats", "abilities", "caractéristiques", "caracteristiques", "attributs", "attributes"]
    for k in stats_keys:
        val = get_val(k, None)
        if isinstance(val, dict):
            stats = val
            break

    # Si pas de dictionnaire de stats, on cherche les clés individuelles communes
    if not stats:
        stat_mapping = {
            "Force": ["force", "str", "for"],
            "Dextérité": ["dextérité", "dexterite", "dex"],
            "Constitution": ["constitution", "con"],
            "Intelligence": ["intelligence", "int"],
            "Sagesse": ["sagesse", "wis", "sag"],
            "Charisme": ["charisme", "cha"]
        }
        extracted_stats = {}
        for label, keys in stat_mapping.items():
            val = get_val(keys, None)
            if val is not None and val != "N/A":
                extracted_stats[label] = val
        if extracted_stats:
            stats = extracted_stats

    if stats:
        st.markdown("**Statistiques :**")
        # Affichage en colonnes pour les stats (3 par ligne max)
        num_stats = len(stats)
        cols_per_row = 3
        for i in range(0, num_stats, cols_per_row):
            batch = list(stats.items())[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            for j, (k, v) in enumerate(batch):
                cols[j].metric(label=k, value=str(v))

    # Compétences
    skills = get_val(["compétences", "competences", "skills"], None)
    if skills:
        if isinstance(skills, list):
            st.markdown(f"**Compétences :** {', '.join(skills)}")
        elif isinstance(skills, dict):
            items = [f"{k}: {v}" for k, v in skills.items()]
            st.markdown(f"**Compétences :** {', '.join(items)}")
        else:
            st.markdown(f"**Compétences :** {skills}")

    equip = get_val(["équipement", "equipement", "equipment", "inventaire"], None)
    if equip:
        if isinstance(equip, list):
            st.markdown(f"**Équipement :** {', '.join(equip)}")
        elif isinstance(equip, dict):
            # Si l'équipement est un dict (ex: armes, armure)
            items = []
            for k, v in equip.items():
                if isinstance(v, list): items.extend(v)
                else: items.append(str(v))
            st.markdown(f"**Équipement :** {', '.join(items)}")
        else:
            st.markdown(f"**Équipement :** {equip}")

    # Section Catch-all pour les informations non affichées
    other_info = {k: v for k, v in character_data.items() if k not in displayed_keys and k.lower() not in ["nom", "name", "image", "portrait"]}
    if other_info:
        with st.expander("Autres informations"):
            for k, v in other_info.items():
                if isinstance(v, (str, int, float)):
                    st.write(f"**{k.capitalize()} :** {v}")
                elif isinstance(v, list):
                    st.write(f"**{k.capitalize()} :** {', '.join(map(str, v))}")
                elif isinstance(v, dict):
                    st.write(f"**{k.capitalize()} :**")
                    st.json(v)

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
                    with st.spinner("Génération du scénario et introduction..."):
                        intro = st.session_state.agent.start_adventure()
                        if "Erreur" in intro:
                            st.error(intro)
                        else:
                            if st.session_state.agent.setup_logs:
                                with st.expander("📊 Détails du lancement", expanded=False):
                                    for log_msg in st.session_state.agent.setup_logs:
                                        st.code(log_msg)
                            st.session_state.game_loaded = True
                            # On ne fait pas de rerun immédiat pour laisser l'utilisateur voir les logs s'il le souhaite
                            # Ou on peut utiliser un bouton pour continuer
                            if st.button("Commencer l'aventure"):
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
            try:
                intro = st.session_state.agent.start_adventure()
                if "Erreur" in intro:
                    st.error(intro)
                else:
                    if st.session_state.agent.setup_logs:
                        with st.expander("📊 Détails du lancement", expanded=True):
                            for log_msg in st.session_state.agent.setup_logs:
                                st.code(log_msg)

                    if st.button("Commencer l'aventure"):
                        st.rerun()
            except Exception as e:
                st.error(f"Une erreur critique est survenue : {e}")

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
