import os
import argparse
import shutil
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import config
from scenario_agents import ManualGeneratorAgent

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

def index_scenes(scenes_path, collection_name, client, embeddings):
    """
    Charge Memory/scenes.json, construit des Documents LangChain par scène
    et les indexe dans la collection scenario_collection.
    """
    if not os.path.exists(scenes_path):
        print(f"Avertissement : Le fichier de scènes {scenes_path} n'existe pas.")
        return

    try:
        with open(scenes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des scènes : {e}")
        return

    scenes = data.get("scenes", [])
    if not scenes:
        print("Aucune scène à indexer dans le fichier de scènes.")
        return

    documents = []
    for scene in scenes:
        # Construction d'un page_content concaténé et lisible
        pnjs_str = ", ".join(scene.get("pnjs", []))
        elements_str = ", ".join(scene.get("elements_a_preserver", []))

        reactions_str = ""
        for reaction in scene.get("reactions_anticipees", []):
            act = reaction.get("action_probable", "")
            cons = reaction.get("consequence", "")
            reactions_str += f"- Action : {act} -> Conséquence : {cons}\n"

        content_parts = [
            f"Titre : {scene.get('titre', '')}",
            f"Lieu : {scene.get('lieu', '')}",
            f"PNJs présents : {pnjs_str}",
            f"Esprit de la scène : {scene.get('esprit_de_la_scene', '')}",
            f"Éléments à préserver : {elements_str}",
            f"Objectif de la scène : {scene.get('objectif_atteint_si', '')}",
            f"Réactions anticipées :\n{reactions_str}"
        ]
        page_content = "\n".join(content_parts)

        metadata = {
            "type": "scene",
            "scene_id": scene.get("id")
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"Indexation de {len(documents)} scènes dans la collection '{collection_name}'...")
    db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    db.add_documents(documents)
    print("✓ Indexation des scènes réussie.")

def main():
    parser = argparse.ArgumentParser(description="Indexer les documents pour le RPG Oracle.")
    parser.add_argument("--clear", action="store_true", help="Vider la base de données avant l'indexation.")
    parser.add_argument("--core", action="store_true", help="Indexer uniquement les fichiers de règles.")
    parser.add_argument("--scenario", action="store_true", help="Indexer uniquement les fichiers de scénario.")
    parser.add_argument("--pj", action="store_true", help="Générer le manuel de création de personnage.")
    parser.add_argument("--reset", action="store_true", help="Supprimer toutes les données (ChromaDB + Memory) et recommencer.")
    args = parser.parse_args()

    # Si aucun argument n'est fourni (à part --clear qui est géré séparément pour compatibilité),
    # on indexe tout et on génère le manuel.
    index_all = not (args.core or args.scenario or args.pj or args.reset)

    if args.reset:
        print("Réinitialisation complète demandée...")
        if os.path.exists(config.CHROMA_PATH):
            print(f"Suppression de la base de données à {config.CHROMA_PATH}...")
            shutil.rmtree(config.CHROMA_PATH)
        if os.path.exists("Memory"):
            print("Suppression du dossier Memory...")
            shutil.rmtree("Memory")
        os.makedirs("Memory", exist_ok=True)

    if args.clear and not args.reset:
        if os.path.exists(config.CHROMA_PATH):
            print(f"Suppression de la base de données existante à {config.CHROMA_PATH}...")
            shutil.rmtree(config.CHROMA_PATH)
        else:
            print("Aucune base de données à supprimer.")

    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    # Création des répertoires si nécessaire
    os.makedirs(config.CORE_DATA_PATH, exist_ok=True)
    os.makedirs(config.SCENARIO_DATA_PATH, exist_ok=True)

    # Indexation du Core
    if index_all or args.core or args.reset:
        index_directory(config.CORE_DATA_PATH, config.CORE_COLLECTION_NAME, client, embeddings)

    # Indexation du Scénario
    if index_all or args.scenario or args.reset:
        index_directory(config.SCENARIO_DATA_PATH, config.SCENARIO_COLLECTION_NAME, client, embeddings)

    # Génération du manuel PJ
    if index_all or args.pj or args.reset:
        print("Génération du manuel de création de personnage...")
        core_store = Chroma(
            client=client,
            collection_name=config.CORE_COLLECTION_NAME,
            embedding_function=embeddings
        )
        generator = ManualGeneratorAgent(core_store)
        generator.generate()

if __name__ == "__main__":
    main()
