import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb
import config

def index_pdfs():
    source_dir = config.DATA_PATH
    collection_name = config.COLLECTION_NAME

    print(f"Indexation des PDFs de {source_dir} dans la collection '{collection_name}'...")

    if not os.path.exists(source_dir):
        print(f"Erreur : Le répertoire {source_dir} n'existe pas.")
        return

    documents = []
    for file in os.listdir(source_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(source_dir, file))
            documents.extend(loader.load())

    if not documents:
        print("Aucun fichier PDF trouvé dans le dossier data/.")
        return

    print(f"Chargement de {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Découpé en {len(chunks)} morceaux.")

    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name
    )

    print(f"Indexation réussie dans '{collection_name}'.")

if __name__ == "__main__":
    index_pdfs()
