import streamlit as st
import json
import os
from agent import RPGAgent
from mechanics.character import build_character_view
from systems.loader import list_available_packs
import config

st.set_page_config(page_title="RPG Oracle - Multi-Agents", page_icon="🎲")


def display_character_info(character_data, pack=None):
    if not character_data:
        st.warning("Aucune donnée de personnage disponible.")
        return

    view = build_character_view(character_data, pack=pack)
    st.markdown(f"### 👤 {view['name']}")

    # 1. Champs scalaires d'identité et de progression (2 colonnes)
    scalars = view["scalars"]
    if scalars:
        col1, col2 = st.columns(2)
        half = (len(scalars) + 1) // 2
        for item in scalars[:half]:
            col1.markdown(f"**{item['label']} :** {item['value']}")
        for item in scalars[half:]:
            col2.markdown(f"**{item['label']} :** {item['value']}")

    # 2. Groupes d'attributs / statistiques / avantages (métriques en grille de 3 colonnes)
    for group in view["stat_groups"]:
        st.markdown(f"**{group['label']} :**")
        items = group["items"]
        cols_per_row = 3
        for i in range(0, len(items), cols_per_row):
            batch = items[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, entry in enumerate(batch):
                cols[j].metric(label=str(entry["label"]), value=str(entry["value"]))

    # 3. Réserves (Pools) et groupes de réserves (Pool Groups) du SystemPack
    if view["pools"] or view["pool_groups"]:
        st.markdown("**Ressources :**")
        for pool in view["pools"]:
            cur = pool["current"]
            mx = pool["max"]
            if cur is not None and mx is not None:
                ratio = cur / mx if mx > 0 else 0
                color = "green" if ratio > 0.5 else "orange" if ratio > 0.2 else "red"
                st.markdown(f"- **{pool['name']}** : :{color}[{cur} / {mx}]")
            elif cur is not None:
                st.markdown(f"- **{pool['name']}** : {cur}")
            elif mx is not None:
                st.markdown(f"- **{pool['name']}** : max {mx}")

        for pgroup in view["pool_groups"]:
            parts = [f"{e['key']}: {e['current']}/{e['max']}" for e in pgroup["entries"]]
            st.markdown(f"- **{pgroup['name']}** : {', '.join(parts)}")

    # 4. Listes et consommables (compétences, capacités, sorts, cyphers, équipement...)
    for lst in view["lists"]:
        items = lst["items"]
        formatted_items = []
        for elem in items:
            if isinstance(elem, dict):
                name_part = elem.get("name", str(elem))
                qty = elem.get("quantity")
                formatted_items.append(f"{name_part} (x{qty})" if qty and qty > 1 else str(name_part))
            else:
                formatted_items.append(str(elem))
        limit_suffix = f" ({len(items)}/{lst['max_items']})" if lst.get("max_items") is not None else ""
        st.markdown(f"**{lst['label']}{limit_suffix} :** {', '.join(formatted_items)}")

    # 5. Informations additionnelles hors schéma
    other_info = view["other"]
    if other_info:
        with st.expander("Autres informations"):
            for k, v in other_info.items():
                label = k.replace("_", " ").capitalize()
                if isinstance(v, (str, int, float)):
                    st.write(f"**{label} :** {v}")
                elif isinstance(v, list):
                    st.write(f"**{label} :** {', '.join(map(str, v))}")
                elif isinstance(v, dict):
                    st.write(f"**{label} :**")
                    st.json(v)


st.title("🎲 RPG Oracle")
st.caption("Votre assistant de jeu de rôle intelligent (Multi-Agents)")

# Découverte des System Packs disponibles dans systems/
available_packs = list_available_packs()
pack_options = [p["id"] for p in available_packs]
pack_labels = {p["id"]: f"{p['name']} ({p['id']})" for p in available_packs}

# Initialisation de l'agent dans la session
if "agent" not in st.session_state:
    default_pack_id = os.getenv("SYSTEM_PACK")
    if not default_pack_id and pack_options:
        default_pack_id = pack_options[0]
    st.session_state.agent = RPGAgent(pack_id=default_pack_id)

# Sidebar pour les options
with st.sidebar:
    st.header("Options")

    current_pack = getattr(getattr(st.session_state.agent, "gse", None), "pack", None)
    current_pack_id = current_pack.manifest.id if (current_pack and getattr(current_pack, "manifest", None)) else None

    if pack_options:
        default_idx = pack_options.index(current_pack_id) if current_pack_id in pack_options else 0
        selected_pack_id = st.selectbox(
            "Système de règles",
            options=pack_options,
            index=default_idx,
            format_func=lambda pid: pack_labels.get(pid, pid),
        )
        if selected_pack_id != current_pack_id:
            st.session_state.agent.set_pack(selected_pack_id)
            st.rerun()

    if st.button("Réinitialiser la conversation"):
        st.session_state.agent.clear_history()
        st.session_state.pop("game_loaded", None)
        st.rerun()

    active_pack = getattr(getattr(st.session_state.agent, "gse", None), "pack", None)
    if active_pack and getattr(active_pack, "manifest", None):
        st.info(f"Système : {active_pack.manifest.name} (`{active_pack.manifest.id}`)")
    else:
        st.warning("Système : Aucun pack chargé (`SYSTEM_PACK` manquant)")

    st.info(f"Modèle : {config.LLM_MODEL or config.OLLAMA_MODEL}")
    st.info(f"Interface : http://{config.SERVER_ADDRESS}:{config.SERVER_PORT}")

    if st.session_state.agent.character_data:
        st.header("👤 Personnage")
        display_character_info(st.session_state.agent.character_data, pack=active_pack)
        with st.expander("JSON brut"):
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
                    display_character_info(char_data, pack=active_pack)

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
                display_character_info(char_data, pack=active_pack)
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
                            st.session_state.game_loaded = True
                            st.session_state.show_logs = True
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
    st.success("✅ La création de votre personnage est terminée !")
    st.info("Vous pouvez maintenant lancer l'aventure. Le Maître du Jeu va extraire le scénario et préparer l'introduction.")
    col_adv, col_cont = st.columns(2)
    with col_adv:
        if st.button("🚀 Lancer l'aventure"):
            with st.spinner("Génération du scénario et introduction... (cela peut prendre environ 30-40 secondes)"):
                try:
                    intro = st.session_state.agent.start_adventure()
                    if "Erreur" in intro:
                        st.error(intro)
                    else:
                        st.session_state.show_logs = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Une erreur critique est survenue : {e}")
    with col_cont:
        if st.button("✏️ Poursuivre / Modifier la création"):
            st.session_state.agent.game_state = "CREATION"
            if os.path.exists("Memory/character.json"):
                with open("Memory/character.json", "r", encoding="utf-8") as f:
                    st.session_state.agent.character_data = json.load(f)
                st.session_state.agent.gse.reload()
            if isinstance(st.session_state.agent.character_data, dict):
                st.session_state.agent.character_data.pop("status", None)
            st.rerun()

if st.session_state.get("show_logs"):
    if st.session_state.agent.setup_logs:
        with st.expander("📊 Détails du lancement", expanded=True):
            for log_msg in st.session_state.agent.setup_logs:
                st.code(log_msg)
    if st.button("C'est parti !"):
        st.session_state.show_logs = False
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
