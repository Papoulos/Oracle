import os
import argparse
import shutil
import json
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import config
import re
from pack_extractor import PackExtractorAgent
from systems.promote import get_draft_dir

def get_embeddings():
    # Use base_utils get_embeddings
    from base_utils import get_embeddings as utils_get_embeddings
    return utils_get_embeddings()

def index_directory(source_dir, collection_name, client, embeddings, index_json=False):
    msg = f"Indexing PDFs and JSONs from {source_dir}" if index_json else f"Indexing PDFs from {source_dir}"
    print(f"{msg} into collection '{collection_name}'...")

    if not os.path.exists(source_dir):
        print(f"Warning: Directory {source_dir} does not exist.")
        return

    documents = []
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".json") and index_json:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Dump JSON as string to treat it as text
                    text_content = json.dumps(data, indent=2, ensure_ascii=False)

                    # Provide metadata
                    metadata = {"source": file_path}
                    documents.append(Document(page_content=text_content, metadata=metadata))
            except Exception as e:
                print(f"Error loading JSON file {file_path}: {e}")

    if not documents:
        msg = "No PDF or JSON file found" if index_json else "No PDF file found"
        print(f"{msg} in {source_dir}.")
        return

    print(f"Loaded {len(documents)} documents from {source_dir}.")

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
    parser.add_argument("--scenario", action="store_true", help="Index only scenario files.")
    parser.add_argument("--reset", action="store_true", help="Wipe all data (ChromaDB + Memory) and start over.")
    parser.add_argument("--log", action="store_true",
                        help="Activate detailed logging (sent prompts and raw LLM responses) in indexer_debug.log")
    parser.add_argument("--pack", action="store_true", help="Generate draft pack for a system.")
    parser.add_argument("--pack-id", type=str, help="The id of the pack to draft (alphanumeric and underscore only).")
    parser.add_argument("--pdf", type=str, action="append", help="Specific PDF file path(s) to process for the pack.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing draft pack.")
    parser.add_argument("--language", type=str, help="ISO 639-1 language code to use (overrides detection).")
    args = parser.parse_args()

    # If no specific mode argument is provided, we index scenario files.
    index_all = not (args.scenario or args.reset or args.pack)

    if args.reset:
        print("Complete reset requested...")
        chroma_path = config.CHROMA_PATH if config.CHROMA_PATH else "./chroma_db"
        if os.path.exists(chroma_path):
            print(f"Deleting DB at {chroma_path}...")
            shutil.rmtree(chroma_path)
        if os.path.exists("Memory"):
            print("Deleting Memory folder...")
            shutil.rmtree("Memory")
        os.makedirs("Memory", exist_ok=True)

    if args.clear and not args.reset:
        chroma_path = config.CHROMA_PATH if config.CHROMA_PATH else "./chroma_db"
        if os.path.exists(chroma_path):
            print(f"Deleting existing DB at {chroma_path}...")
            shutil.rmtree(chroma_path)
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

    if args.pack:
        if not args.pack_id:
             print("Error: --pack-id is required when using --pack")
             return

        if not re.fullmatch(r"[a-z0-9_]+", args.pack_id):
             print(f"Error: Invalid pack id '{args.pack_id}'. Must be lowercase alphanumeric and underscore.")
             return

        if not args.pdf:
             print("Error: At least one --pdf <path> is required when using --pack")
             return

        pdf_files = args.pdf
        missing_pdfs = [p for p in pdf_files if not os.path.isfile(p)]
        if missing_pdfs:
             print(f"Error: PDF file(s) not found: {', '.join(missing_pdfs)}")
             return

        draft_dir = get_draft_dir(args.pack_id)
        if os.path.exists(draft_dir):
             if not args.force:
                  print(f"Error: Draft directory {draft_dir} already exists. Use --force to overwrite.")
                  return
             else:
                  print(f"Removing existing draft directory: {draft_dir}")
                  shutil.rmtree(draft_dir)

        print(f"Generating draft pack '{args.pack_id}' from {len(pdf_files)} PDF(s)...")
        agent = PackExtractorAgent(verbose=verbose)
        try:
             agent.extract(args.pack_id, pdf_files, override_language=args.language)
        except Exception as e:
             print(f"Error generating pack: {e}")
        return

    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=config.CHROMA_PATH if config.CHROMA_PATH else "./chroma_db")

    # Create scenario directory if needed
    scenario_data_path = config.SCENARIO_DATA_PATH if config.SCENARIO_DATA_PATH else "./data/scenario"
    scenario_coll_name = config.SCENARIO_COLLECTION_NAME if config.SCENARIO_COLLECTION_NAME else "default_scenario"

    os.makedirs(scenario_data_path, exist_ok=True)

    # Scenario indexing
    if index_all or args.scenario or args.reset:
        index_directory(scenario_data_path, scenario_coll_name, client, embeddings, index_json=True)

if __name__ == "__main__":
    main()
