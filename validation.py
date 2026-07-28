"""
Fonction de validation/réparation partagée, appelée à la fois en sortie du
pipeline d'extraction (ScenarioExtractorAgent) et sur un JSON déposé
directement dans data/scenario/.

Règles :
- lieu_rattache_id invalide sur une scène -> mis à None, avertissement (pas de fallback inventé).
- pnj_presents orphelins -> filtrés, avertissement.
- scenes_incluses orphelines (id de scène inexistant) -> filtrées, avertissement.
- cohérence bidirectionnelle acte<->scène : si scene.acte_rattache_id existe mais
  que l'acte ne liste pas la scène dans scenes_incluses -> réparation automatique
  (ajout de l'id à la liste), avertissement. Si acte_rattache_id ne correspond à
  aucun acte connu -> ERREUR bloquante (pas de réparation possible sans deviner).
- destination_scene_id invalide dans sorties_logiques -> filtré, avertissement.
- metadata.scene_initiale manquant -> repli sur la première scène du premier acte, avertissement.
- horloges_globales[].seuil manquant -> valeur par défaut (6), avertissement.
- noeuds_sceniques[].condition_resolution manquant -> ERREUR bloquante (contenu
  sémantique que la validation ne peut pas inventer sans fausser le scénario).
"""

DEFAULT_SEUIL = 6


def _check_duplicates(items, id_key, label, warnings):
    seen = set()
    for item in items:
        iid = item.get(id_key)
        if iid in seen:
            warnings.append(f"{label} : id dupliqué '{iid}' détecté - seule la dernière occurrence sera retenue en aval.")
        seen.add(iid)


def validate_scenario_structure(data: dict) -> tuple[dict, list[str], list[str]]:
    """
    Retourne (data_corrige, avertissements, erreurs_bloquantes).
    Les erreurs bloquantes signalent un besoin de régénération/correction manuelle,
    pas quelque chose que la validation peut réparer silencieusement.
    """
    warnings = []
    errors = []

    # --- Doublons d'ID ---
    _check_duplicates(data.get("entites", {}).get("pnj", []), "id", "PNJ", warnings)
    _check_duplicates(data.get("entites", {}).get("lieux", []), "id", "Lieu", warnings)
    _check_duplicates(data.get("noeuds_sceniques", []), "id_scene", "Scène", warnings)
    _check_duplicates(data.get("macro_structure", []), "id_acte", "Acte", warnings)

    pnj_ids = {p["id"] for p in data.get("entites", {}).get("pnj", []) if "id" in p}
    lieu_ids = {l["id"] for l in data.get("entites", {}).get("lieux", []) if "id" in l}
    scene_ids = {s["id_scene"] for s in data.get("noeuds_sceniques", []) if "id_scene" in s}
    acte_ids = {a["id_acte"] for a in data.get("macro_structure", []) if "id_acte" in a}

    # --- PNJ : localisation_habituelle ---
    for p in data.get("entites", {}).get("pnj", []):
        loc = p.get("localisation_habituelle")
        if loc not in lieu_ids:
            if loc is not None:
                warnings.append(
                    f"PNJ {p.get('id')} : localisation_habituelle '{loc}' inconnue -> mise à null."
                )
            p["localisation_habituelle"] = None

    # --- Nœuds scéniques ---
    for s in data.get("noeuds_sceniques", []):
        sid = s["id_scene"]

        if s.get("lieu_rattache_id") not in lieu_ids:
            if s.get("lieu_rattache_id") is not None:
                warnings.append(
                    f"Scène {sid} : lieu_rattache_id '{s['lieu_rattache_id']}' inconnu -> mis à null."
                )
            s["lieu_rattache_id"] = None

        valid_pnjs = [p for p in s.get("pnj_presents", []) if p in pnj_ids]
        removed = set(s.get("pnj_presents", [])) - set(valid_pnjs)
        for p in removed:
            warnings.append(f"Scène {sid} : PNJ orphelin '{p}' retiré de pnj_presents.")
        s["pnj_presents"] = valid_pnjs

        valid_sorties = []
        for sortie in s.get("sorties_logiques", []):
            if sortie.get("destination_scene_id") in scene_ids:
                valid_sorties.append(sortie)
            else:
                warnings.append(
                    f"Scène {sid} : sortie_logique vers scène inconnue "
                    f"'{sortie.get('destination_scene_id')}' retirée."
                )
        s["sorties_logiques"] = valid_sorties

        if not s.get("condition_resolution"):
            errors.append(
                f"Scène {sid} : condition_resolution manquant - "
                f"nécessite une régénération de cette scène, non réparable automatiquement."
            )

        acte = s.get("acte_rattache_id")
        if acte not in acte_ids:
            errors.append(
                f"Scène {sid} : acte_rattache_id '{acte}' ne correspond à aucun acte "
                f"connu - nécessite une correction manuelle."
            )

    # --- Cohérence bidirectionnelle acte <-> scenes_incluses ---
    for a in data.get("macro_structure", []):
        declared = set(a.get("scenes_incluses", []))
        cleaned = [sid for sid in a.get("scenes_incluses", []) if sid in scene_ids]
        for sid in declared - set(cleaned):
            warnings.append(
                f"Acte {a['id_acte']} : scene_incluse orpheline '{sid}' retirée."
            )
        a["scenes_incluses"] = cleaned

    for s in data.get("noeuds_sceniques", []):
        acte = s.get("acte_rattache_id")
        if acte in acte_ids:
            acte_obj = next(a for a in data["macro_structure"] if a["id_acte"] == acte)
            if s["id_scene"] not in acte_obj["scenes_incluses"]:
                acte_obj["scenes_incluses"].append(s["id_scene"])
                warnings.append(
                    f"Incohérence réparée : scène {s['id_scene']} ajoutée à "
                    f"scenes_incluses de {acte} (déclarée par la scène mais absente de l'acte)."
                )

    # --- Métadonnées ---
    if not data.get("metadata", {}).get("scene_initiale"):
        if "metadata" not in data:
            data["metadata"] = {}
        if data.get("macro_structure") and data["macro_structure"][0].get("scenes_incluses"):
            fallback = data["macro_structure"][0]["scenes_incluses"][0]
            data["metadata"]["scene_initiale"] = fallback
            warnings.append(f"metadata.scene_initiale manquant -> repli sur '{fallback}'.")
        else:
            errors.append("metadata.scene_initiale manquant et aucun repli possible (macro_structure vide).")

    # --- Horloges ---
    for h in data.get("horloges_globales", []):
        if "seuil" not in h or h["seuil"] is None:
            h["seuil"] = DEFAULT_SEUIL
            warnings.append(f"Horloge '{h['nom']}' : seuil manquant -> valeur par défaut {DEFAULT_SEUIL}.")

    return data, warnings, errors


def _get_nested(data: dict, path: str):
    """Résout un chemin en points ('ressources.points_de_vie') dans un dict imbriqué."""
    value = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_character_sheet(character_data: dict, schema: dict) -> tuple[bool, list[str]]:
    """
    Retourne (est_complet, champs_manquants).

    `schema` est généré par ruleset (voir ManualGeneratorAgent étendu ci-dessous),
    avec ce format :
    {
      "champs_requis": [
        {"chemin": "nom", "type": "string"},
        {"chemin": "caracteristiques", "type": "object", "sous_champs": ["Force", "Dexterite", ...]},
        {"chemin": "ressources.points_de_vie", "type": "object", "sous_champs": ["actuels", "max"]},
        {"chemin": "equipement", "type": "list", "non_vide": true}
      ]
    }
    """
    if not character_data:
        return False, ["character_data absent"]

    if not schema or not schema.get("champs_requis"):
        return False, ["schema de validation absent ou vide - la création ne peut pas être confirmée automatiquement, vérifier Memory/character_schema.json"]

    missing = []

    for field in schema.get("champs_requis", []):
        chemin = field["chemin"]
        ftype = field.get("type", "string")
        value = _get_nested(character_data, chemin)

        if ftype == "string":
            if not value or not isinstance(value, str):
                missing.append(chemin)

        elif ftype == "number":
            if not isinstance(value, (int, float)):
                missing.append(chemin)

        elif ftype == "object":
            sous_champs = field.get("sous_champs", [])
            if not isinstance(value, dict):
                missing.append(chemin)
            else:
                absents = [
                    sc for sc in sous_champs
                    if not isinstance(value.get(sc), (int, float, str)) or value.get(sc) in (None, "")
                ]
                if absents:
                    missing.append(f"{chemin} ({', '.join(absents)})")

        elif ftype == "list":
            if not isinstance(value, list):
                missing.append(chemin)
            elif field.get("non_vide", False) and len(value) == 0:
                missing.append(chemin)

    return (len(missing) == 0), missing
