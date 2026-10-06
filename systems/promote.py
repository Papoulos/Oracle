import os
import sys
import argparse
import json
import shutil
import re
import yaml
from .validate import validate_pack
from .provenance import ProvenanceData, REVIEW_CONFIDENCE_THRESHOLD

SYSTEMS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_draft_dir(pack_id: str) -> str:
    return os.path.join(SYSTEMS_DIR, "draft", pack_id)

def get_target_dir(pack_id: str) -> str:
    return os.path.join(SYSTEMS_DIR, pack_id)

def init_review(pack_id: str) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", pack_id):
        print(f"Error: Invalid pack id '{pack_id}'. Must be lowercase alphanumeric and underscore.")
        sys.exit(1)

    draft_dir = get_draft_dir(pack_id)
    if not os.path.isdir(draft_dir):
        print(f"Error: Draft directory {draft_dir} does not exist.")
        sys.exit(1)

    prov_path = os.path.join(draft_dir, "provenance.json")
    if not os.path.isfile(prov_path):
        print(f"Error: Missing provenance.json in {draft_dir}.")
        sys.exit(1)

    try:
        with open(prov_path, "r", encoding="utf-8") as f:
            prov_data = ProvenanceData.model_validate_json(f.read())
    except Exception as e:
        print(f"Error reading provenance.json: {e}")
        sys.exit(1)

    review_data = {
        "reviewed_by": None,
        "validated": {},
        "notes": {}
    }

    for entry in prov_data.entries:
        if entry.needs_review or entry.confidence < REVIEW_CONFIDENCE_THRESHOLD:
            key = f"{entry.file}:{entry.path}"
            review_data["validated"][key] = False

    review_path = os.path.join(draft_dir, "review.json")
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2)

    print(f"Initialized review.json with {len(review_data['validated'])} entries to review.")

def promote_pack(pack_id: str, force: bool) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", pack_id):
        print(f"Error: Invalid pack id '{pack_id}'. Must be lowercase alphanumeric and underscore.")
        sys.exit(1)

    draft_dir = get_draft_dir(pack_id)
    target_dir = get_target_dir(pack_id)

    if not os.path.isdir(draft_dir):
        print(f"Error: Draft directory {draft_dir} does not exist.")
        sys.exit(1)

    # 0. Check manifest.id == pack_id
    manifest_path = os.path.join(draft_dir, "manifest.yaml")
    if not os.path.isfile(manifest_path):
        print(f"Error: Missing manifest.yaml in {draft_dir}.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = yaml.safe_load(f)
        if manifest_data.get("id") != pack_id:
            print(f"Promotion refused: manifest.id '{manifest_data.get('id')}' does not match pack_id '{pack_id}'.")
            sys.exit(1)

    # 1. Check status in provenance.json
    prov_path = os.path.join(draft_dir, "provenance.json")
    if not os.path.isfile(prov_path):
        print(f"Error: Missing provenance.json in {draft_dir}.")
        sys.exit(1)

    with open(prov_path, "r", encoding="utf-8") as f:
         prov_data = ProvenanceData.model_validate_json(f.read())

    if prov_data.status != "complete":
        print(f"Promotion refused: provenance status is '{prov_data.status}', not 'complete'.")
        sys.exit(1)

    # 2. Check validate_pack (this will ignore provenance.json and review.json)
    issues = validate_pack(draft_dir)
    error_issues = [i for i in issues if i.severity == "error"]
    if error_issues:
        print("Promotion refused: validation errors found in draft files:")
        for issue in error_issues:
            print(f" - {issue.file}:{issue.path} - {issue.message}")
        sys.exit(1)

    # 3. Check review.json for required validations
    required_review_keys = set()
    for entry in prov_data.entries:
         if entry.needs_review or entry.confidence < REVIEW_CONFIDENCE_THRESHOLD:
             required_review_keys.add(f"{entry.file}:{entry.path}")

    if required_review_keys:
        review_path = os.path.join(draft_dir, "review.json")
        if not os.path.isfile(review_path):
             print("Promotion refused: review.json is missing, but there are fields requiring review.")
             sys.exit(1)

        with open(review_path, "r", encoding="utf-8") as f:
             review_data = json.load(f)

        validated = review_data.get("validated", {})
        for key in required_review_keys:
             if not validated.get(key, False):
                  print(f"Promotion refused: Entry '{key}' requires review but is not validated to true in review.json.")
                  sys.exit(1)

    # 4. Move files
    bak_dir = None
    if os.path.exists(target_dir):
        if not force:
             print(f"Promotion refused: Target directory {target_dir} already exists. Use --force to overwrite.")
             sys.exit(1)
        else:
             bak_dir = target_dir + ".bak"
             if os.path.exists(bak_dir):
                 shutil.rmtree(bak_dir)
             shutil.move(target_dir, bak_dir)

    try:
        shutil.move(draft_dir, target_dir)
        debug_dir = os.path.join(target_dir, "debug")
        if os.path.isdir(debug_dir):
            shutil.rmtree(debug_dir)
        if bak_dir:
            shutil.rmtree(bak_dir)
        print(f"Successfully promoted draft to {target_dir}!")
    except Exception as e:
        if bak_dir:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(bak_dir, target_dir)
        print(f"Promotion failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Promote a draft system pack.")
    parser.add_argument("id", help="The pack ID to promote")
    parser.add_argument("--init-review", action="store_true", help="Initialize review.json in the draft folder")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing pack")

    args = parser.parse_args()

    if args.init_review:
        init_review(args.id)
    else:
        promote_pack(args.id, args.force)

if __name__ == "__main__":
    main()
