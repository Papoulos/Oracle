import os
import argparse
import shutil
import json
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import config
from scenario_agents import ManualGeneratorAgent, GameplayRulesAgent

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
    print(f"Indexing PDFs from {source_dir} into collection '{collection_name}'...")

    if not os.path.exists(source_dir):
        print(f"Warning: Directory {source_dir} does not exist.")
        return

    documents = []
    for file in os.listdir(source_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(source_dir, file))
            documents.extend(loader.load())

    if not documents:
        print(f"No PDF file found in {source_dir}.")
        return

    print(f"Loaded {len(documents)} pages from {source_dir}.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name
    )
    print(f"Successfully indexed in '{collection_name}'.")

def index_scenes(scenes_path, collection_name, client, embeddings):
    """
    Loads Memory/scenes.json, builds LangChain Documents per scene
    and indexes them in the scenario_collection.
    """
    if not os.path.exists(scenes_path):
        print(f"Warning: Scenes file {scenes_path} does not exist.")
        return

    try:
        with open(scenes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading scenes: {e}")
        return

    scenes = data.get("scenes", [])
    if not scenes:
        print("No scene to index in the scenes file.")
        return

    documents = []
    for scene in scenes:
        # Construct clean concatenated page content
        pnjs_str = ", ".join(scene.get("pnjs", []))
        elements_str = ", ".join(scene.get("elements_a_preserver", []))

        reactions_str = ""
        for reaction in scene.get("reactions_anticipees", []):
            act = reaction.get("action_probable", "")
            cons = reaction.get("consequence", "")
            reactions_str += f"- Action : {act} -> Consequence : {cons}\n"

        content_parts = [
            f"Titre : {scene.get('titre', '')}",
            f"Lieu : {scene.get('lieu', '')}",
            f"PNJs presents : {pnjs_str}",
            f"Esprit de la scene : {scene.get('esprit_de_la_scene', '')}",
            f"Elements a preserver : {elements_str}",
            f"Objectif de la scene : {scene.get('objectif_atteint_si', '')}",
            f"Reactions anticipees :\n{reactions_str}"
        ]
        page_content = "\n".join(content_parts)

        metadata = {
            "type": "scene",
            "scene_id": scene.get("id")
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"Indexing {len(documents)} scenes into collection '{collection_name}'...")
    db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    db.add_documents(documents)
    print("✓ Scene indexing successful.")

def main():
    parser = argparse.ArgumentParser(description="Index documents for the Oracle RPG.")
    parser.add_argument("--clear", action="store_true", help="Clear DB before indexing.")
    parser.add_argument("--core", action="store_true", help="Index only rules files.")
    parser.add_argument("--scenario", action="store_true", help="Index only scenario files.")
    parser.add_argument("--pj", action="store_true", help="Generate character creation manual.")
    parser.add_argument("--reset", action="store_true", help="Wipe all data (ChromaDB + Memory) and start over.")
    parser.add_argument("--log", action="store_true",
                        help="Activate detailed logging (sent prompts and raw LLM responses) in indexer_debug.log")
    args = parser.parse_args()

    # If no specific mode argument is provided, we index everything and generate the manual.
    index_all = not (args.core or args.scenario or args.pj or args.reset)

    if args.reset:
        print("Complete reset requested...")
        if os.path.exists(config.CHROMA_PATH):
            print(f"Deleting DB at {config.CHROMA_PATH}...")
            shutil.rmtree(config.CHROMA_PATH)
        if os.path.exists("Memory"):
            print("Deleting Memory folder...")
            shutil.rmtree("Memory")
        os.makedirs("Memory", exist_ok=True)

    if args.clear and not args.reset:
        if os.path.exists(config.CHROMA_PATH):
            print(f"Deleting existing DB at {config.CHROMA_PATH}...")
            shutil.rmtree(config.CHROMA_PATH)
        else:
            print("No database to delete.")

    if args.log:
        # Clear existing handlers to allow basicConfig reinitialization during tests
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename="indexer_debug.log",
            filemode="w",
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            encoding="utf-8"
        )
        logging.root.setLevel(logging.DEBUG)
        print("Detailed logging activated -> indexer_debug.log")
    verbose = args.log

    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    # Create directories if needed
    os.makedirs(config.CORE_DATA_PATH, exist_ok=True)
    os.makedirs(config.SCENARIO_DATA_PATH, exist_ok=True)

    # Core indexing
    if index_all or args.core or args.reset:
        index_directory(config.CORE_DATA_PATH, config.CORE_COLLECTION_NAME, client, embeddings)

    # Scenario indexing
    if index_all or args.scenario or args.reset:
        index_directory(config.SCENARIO_DATA_PATH, config.SCENARIO_COLLECTION_NAME, client, embeddings)

    # Character creation manual generation
    if index_all or args.pj or args.reset:
        print("Generating character creation manual...")
        core_store = Chroma(
            client=client,
            collection_name=config.CORE_COLLECTION_NAME,
            embedding_function=embeddings
        )
        generator = ManualGeneratorAgent(core_store, verbose=verbose)
        generator.generate()

        print("Generating recovery rules and action catalog...")
        gameplay_agent = GameplayRulesAgent(core_store, verbose=verbose)
        gameplay_agent.generate_recovery_rules()
        gameplay_agent.generate_action_catalog()

if __name__ == "__main__":
    main()
