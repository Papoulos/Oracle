import streamlit as st
import config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="RPG Oracle - Test Agent", layout="wide")

st.title("🧙‍♂️ RPG Oracle - Agent de Test")

# --- Configuration & Initialization ---
@st.cache_resource
def get_vectorstores():
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )

    try:
        codex_db = Chroma(
            persist_directory=config.CHROMA_PATH,
            collection_name=config.COLLECTION_CODEX,
            embedding_function=embeddings
        )
        intrigue_db = Chroma(
            persist_directory=config.CHROMA_PATH,
            collection_name=config.COLLECTION_INTRIGUE,
            embedding_function=embeddings
        )
        return codex_db, intrigue_db
    except Exception as e:
        st.error(f"Erreur lors du chargement des bases : {e}")
        return None, None

codex_db, intrigue_db = get_vectorstores()

# --- Sidebar ---
with st.sidebar:
    st.header("Statut des Bases")
    if codex_db:
        st.success("Codex: Chargé")
    else:
        st.error("Codex: Non trouvé")

    if intrigue_db:
        st.success("Intrigue: Chargé")
    else:
        st.error("Intrigue: Non trouvé")

    st.info(f"Modèle: {config.OLLAMA_MODEL}")
    st.markdown("---")
    st.markdown("""
    ### Instructions
    1. Indexez vos documents avec `python indexer.py codex` et `python indexer.py intrigue`.
    2. Posez vos questions à l'agent ci-contre.
    """)

# --- RAG Logic ---
def get_response(query):
    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.7
    )

    # Simple strategy: retrieve from both and combine context
    context_docs = []
    if codex_db:
        context_docs.extend(codex_db.similarity_search(query, k=3))
    if intrigue_db:
        context_docs.extend(intrigue_db.similarity_search(query, k=3))

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = ChatPromptTemplate.from_template("""
    Tu es un assistant de jeu de rôle omniscient. Ton rôle est d'aider le maître du jeu ou les joueurs
    en répondant à des questions basées UNIQUEMENT sur le CODEX (règles) et l'INTRIGUE (scénario) fournis ci-dessous.

    Si la réponse n'est pas dans le contexte, dis-le poliment.

    CONTEXTE:
    {context}

    QUESTION:
    {question}

    REPONSE:
    """)

    chain = (
        {"context": lambda x: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream(query)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez une question sur les règles ou l'intrigue..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in get_response(prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
