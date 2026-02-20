import streamlit as st
import config
import os
import chromadb
import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from orchestrateur import Orchestrateur
import memory_manager

st.set_page_config(page_title="RPG Oracle - Multi-Agent", layout="wide")

st.title("🧙‍♂️ RPG Oracle - Système Multi-Agent")

# --- Configuration & Initialization ---
@st.cache_resource
def get_vectorstores():
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )

    if not os.path.exists(config.CHROMA_PATH) or not os.listdir(config.CHROMA_PATH):
        st.warning("⚠️ La base de données est vide. Veuillez lancer l'indexation (./run.sh).")
        return None, None

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collections = [c.name for c in client.list_collections()]

        codex_db = None
        if config.COLLECTION_CODEX in collections:
            codex_db = Chroma(
                client=client,
                collection_name=config.COLLECTION_CODEX,
                embedding_function=embeddings
            )

        intrigue_db = None
        if config.COLLECTION_INTRIGUE in collections:
            intrigue_db = Chroma(
                client=client,
                collection_name=config.COLLECTION_INTRIGUE,
                embedding_function=embeddings
            )

        return codex_db, intrigue_db
    except Exception as e:
        st.error(f"Erreur lors du chargement des bases : {e}")
        return None, None

codex_db, intrigue_db = get_vectorstores()

@st.cache_resource
def get_orchestrateur(_codex, _intrigue):
    return Orchestrateur(_codex, _intrigue)

if codex_db and intrigue_db:
    orchestrateur = get_orchestrateur(codex_db, intrigue_db)
else:
    orchestrateur = None

# --- Sidebar ---
with st.sidebar:
    st.header("📜 État du Jeu")
    memory = memory_manager.load_memory()
    if memory:
        st.subheader("👤 Personnage")
        st.json(memory.get("personnage", {}))
        st.subheader("🌍 Monde")
        st.write(f"**Lieu :** {memory.get('monde', {}).get('lieu_actuel')}")
        st.write("**Événements :**")
        for ev in memory.get('monde', {}).get('evenements_marquants', [])[-5:]:
            st.write(f"- {ev}")

    st.markdown("---")
    if st.button("🔄 Réinitialiser la Mémoire"):
        # Reset memory to default
        default_mem = {
          "personnage": {
            "nom": "Aventurier",
            "stats": {"force": 10, "agilite": 10, "intelligence": 10, "pv": 20, "pv_max": 20},
            "inventaire": ["Épée rouillée", "Gourde d'eau"],
            "xp": 0, "niveau": 1
          },
          "monde": {
            "lieu_actuel": "Auberge du Dragon Vert",
            "factions": {}, "evenements_marquants": [], "secrets_decouverts": []
          },
          "historique": []
        }
        memory_manager.save_memory(default_mem)
        st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "reflection" in message:
            with st.expander("💭 Réflexion des Agents"):
                # On affiche les réflexions de manière structurée
                for agent, content in message["reflection"].items():
                    st.subheader(agent)
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.write(content)

# Fonction pour exécuter un tour de jeu
def run_game_turn(user_query):
    with st.chat_message("assistant"):
        reflection_placeholder = st.empty()
        response_placeholder = st.empty()

        reflections = {}
        full_response = ""

        with st.status("Les agents réfléchissent...", expanded=True) as status:
            for step in orchestrateur.run(user_query):
                for node_name, output in step.items():
                    if node_name == "consult_regles":
                        st.write("⚖️ L'Agent Règles vérifie le Codex...")
                        reflections["Règles"] = output["regles_info"]
                    elif node_name == "consult_monde":
                        st.write("🌍 L'Agent Monde consulte l'Intrigue...")
                        reflections["Monde"] = output["world_info"]
                    elif node_name == "narrate":
                        st.write("🎙️ Le MJ Narrateur prépare sa réponse...")
                        full_response = output["narration"]
                    elif node_name == "update_memory":
                        st.write("🧠 L'Agent Mémoire met à jour l'état...")
                        reflections["Mémoire (Updates)"] = output["updates"]

            status.update(label="Réflexion terminée !", state="complete", expanded=False)

        response_placeholder.markdown(full_response)

        with st.expander("💭 Détails de la réflexion", expanded=False):
            if not reflections:
                st.info("Aucune réflexion technique disponible pour ce tour.")
            for agent, content in reflections.items():
                st.markdown(f"**{agent}**")
                if isinstance(content, dict):
                    st.json(content)
                else:
                    st.write(content)
                st.markdown("---")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "reflection": reflections
    })

# --- Introduction Automatique ---
if not st.session_state.messages and orchestrateur:
    welcome_query = "Le jeu commence. Présente-toi brièvement comme le MJ et décris la scène initiale pour plonger le joueur dans l'aventure selon le lieu actuel et l'intrigue."
    run_game_turn(welcome_query)
    st.rerun()

if prompt := st.chat_input("Que faites-vous ?"):
    if not orchestrateur:
        st.error("L'orchestrateur n'est pas prêt. Vérifiez les bases de données.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        run_game_turn(prompt)
        st.rerun()
