import os
import sys
import argparse
import json
import shutil
from .validate import validate_pack

def get_draft_dir(pack_id: str) -> str:
    return os.path.join("systems", "draft", pack_id)

def get_target_dir(pack_id: str) -> str:
    return os.path.join("systems", pack_id)

def init_review(pack_id: str) -> None:
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
            prov_data = json.load(f)
    except Exception as e:
        print(f"Error reading provenance.json: {e}")
        sys.exit(1)

    review_data = {
        "reviewed_by": None,
        "validated": {},
        "notes": {}
    }

    entries = prov_data.get("entries", [])
    for entry in entries:
        needs_review = entry.get("needs_review", False)
        confidence = entry.get("confidence", 0)

        # We flag entries for review if needs_review=True OR confidence < 70
        if needs_review or confidence < 70:
            key = f"{entry.get('file', '')}:{entry.get('path', '')}"
            review_data["validated"][key] = False

    review_path = os.path.join(draft_dir, "review.json")
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2)

    print(f"Initialized review.json with {len(review_data['validated'])} entries to review.")

def promote_pack(pack_id: str, force: bool) -> None:
    draft_dir = get_draft_dir(pack_id)
    target_dir = get_target_dir(pack_id)

    if not os.path.isdir(draft_dir):
        print(f"Error: Draft directory {draft_dir} does not exist.")
        sys.exit(1)

    # 1. Check status in provenance.json
    prov_path = os.path.join(draft_dir, "provenance.json")
    if not os.path.isfile(prov_path):
        print(f"Error: Missing provenance.json in {draft_dir}.")
        sys.exit(1)

    with open(prov_path, "r", encoding="utf-8") as f:
         prov_data = json.load(f)

    if prov_data.get("status") != "complete":
        print(f"Promotion refused: provenance status is '{prov_data.get('status')}', not 'complete'.")
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
    entries = prov_data.get("entries", [])
    required_review_keys = set()
    for entry in entries:
         if entry.get("needs_review", False) or entry.get("confidence", 0) < 70:
             required_review_keys.add(f"{entry.get('file', '')}:{entry.get('path', '')}")

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
    if os.path.exists(target_dir):
        if not force:
             print(f"Promotion refused: Target directory {target_dir} already exists. Use --force to overwrite.")
             sys.exit(1)
        else:
             shutil.rmtree(target_dir)

    shutil.move(draft_dir, target_dir)
    print(f"Successfully promoted draft to {target_dir}!")

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
