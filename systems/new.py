import os
import sys
import json
import yaml
import argparse

import re
from mechanics.models import FAMILIES, Manifest, ResourcesConfig


def generate_new_pack(pack_id: str, family_name: str) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", pack_id):
        print(f"Erreur : pack_id invalide '{pack_id}'. Il doit respecter l'expression régulière ^[a-z0-9_]+$", file=sys.stderr)
        sys.exit(1)

    if family_name not in FAMILIES:
        available = ", ".join(FAMILIES.keys())
        print(f"Erreur : Famille '{family_name}' inconnue. Familles disponibles : {available}", file=sys.stderr)
        sys.exit(1)

    pack_dir = os.path.join("systems", pack_id)
    if os.path.exists(pack_dir):
        print(f"Erreur : Le dossier '{pack_dir}' existe déjà. Refus de l'écraser.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(pack_dir)

    # Generate Manifest
    manifest = Manifest.template(pack_id=pack_id, family=family_name)
    manifest_path = os.path.join(pack_dir, "manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest.model_dump(), f, allow_unicode=True, sort_keys=False)

    # Generate ResolutionConfig
    resolution_model_class = FAMILIES[family_name]
    resolution = resolution_model_class.template()
    resolution_path = os.path.join(pack_dir, "resolution.json")
    with open(resolution_path, "w", encoding="utf-8") as f:
        json.dump(resolution.model_dump(exclude_none=True), f, indent=2, ensure_ascii=False)

    # Generate ResourcesConfig
    resources = ResourcesConfig.template()
    resources_path = os.path.join(pack_dir, "resources.json")
    with open(resources_path, "w", encoding="utf-8") as f:
        json.dump(resources.model_dump(exclude_none=True), f, indent=2, ensure_ascii=False)

    print(f"Le pack '{pack_id}' de la famille '{family_name}' a été généré avec succès dans {pack_dir}.")
    print("Veuillez remplir les valeurs 'TODO' dans les fichiers.")


def main():
    parser = argparse.ArgumentParser(description="Générer un squelette de System Pack.")
    parser.add_argument("id", help="Identifiant du nouveau pack (nom du dossier)")
    parser.add_argument("--family", required=True, help="Famille de résolution")

    args = parser.parse_args()

    generate_new_pack(args.id, args.family)


if __name__ == "__main__":
    main()
