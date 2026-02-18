import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import config

def index_pdfs(source_dir, collection_name):
    print(f"Indexation des PDFs de {source_dir} dans la collection '{collection_name}'...")

    # Vérifie si le répertoire existe
    if not os.path.exists(source_dir):
        print(f"Erreur : Le répertoire {source_dir} n'existe pas.")
        return

    # Chargement des PDFs
    documents = []
    for file in os.listdir(source_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(source_dir, file))
            documents.extend(loader.load())

    if not documents:
        print("Aucun fichier PDF trouvé.")
        return

    print(f"Chargement de {len(documents)} pages.")

    # Découpage du texte
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Découpé en {len(chunks)} morceaux.")

    # Initialisation des Embeddings
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )

    # Création/Mise à jour du Vector Store
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PATH,
        collection_name=collection_name
    )

    print(f"Indexation réussie dans '{collection_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexe des fichiers PDF dans ChromaDB.")
    parser.add_argument("type", choices=["codex", "intrigue"], help="Type de base à indexer (codex ou intrigue)")

    args = parser.parse_args()

    if args.type == "codex":
        index_pdfs(os.path.join(config.DATA_PATH, "codex"), config.COLLECTION_CODEX)
    else:
        index_pdfs(os.path.join(config.DATA_PATH, "intrigue"), config.COLLECTION_INTRIGUE)
