import os
import argparse
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import config

def get_embeddings():
    if config.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.EMBEDDING_BASE_URL
        )
    else: # openai / llama-cpp
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.EMBEDDING_BASE_URL,
            api_key="sk-no-key-required"
        )

def index_directory(source_dir, collection_name, client, embeddings):
    print(f"Indexation des PDFs de {source_dir} dans la collection '{collection_name}'...")

    if not os.path.exists(source_dir):
        print(f"Avertissement : Le répertoire {source_dir} n'existe pas.")
        return

    documents = []
    for file in os.listdir(source_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(source_dir, file))
            documents.extend(loader.load())

    if not documents:
        print(f"Aucun fichier PDF trouvé dans {source_dir}.")
        return

    print(f"Chargement de {len(documents)} pages depuis {source_dir}.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Découpé en {len(chunks)} morceaux.")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name
    )
    print(f"Indexation réussie dans '{collection_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Indexer les documents pour le RPG Oracle.")
    parser.add_argument("--clear", action="store_true", help="Vider la base de données avant l'indexation.")
    args = parser.parse_args()

    if args.clear:
        if os.path.exists(config.CHROMA_PATH):
            print(f"Suppression de la base de données existante à {config.CHROMA_PATH}...")
            shutil.rmtree(config.CHROMA_PATH)
        else:
            print("Aucune base de données à supprimer.")

    embeddings = get_embeddings()

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    # Indexation du Core
    index_directory(config.CORE_DATA_PATH, config.CORE_COLLECTION_NAME, client, embeddings)

    # Indexation du Scénario
    index_directory(config.SCENARIO_DATA_PATH, config.SCENARIO_COLLECTION_NAME, client, embeddings)

if __name__ == "__main__":
    main()
