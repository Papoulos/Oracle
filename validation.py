"""
Shared validation and correction functions, called both at the end of the
extraction pipeline (ScenarioExtractorAgent) and on any JSON file dropped
directly in data/scenario/.

Rules:
- Invalid location_id on a scene -> set to None, warning (no invented fallback).
- Orphan present_npcs -> filtered, warning.
- Orphan included_scenes (non-existent scene ID) -> filtered, warning.
- Bidirectional act <-> scene coherence: if scene.act_id exists but the act does not list the scene in included_scenes -> automatic repair (add ID to list), warning. If act_id does not correspond to any known act -> blocking ERROR (no repair possible without guessing).
- Invalid destination_scene_id in logical_exits -> filtered, warning.
- Missing metadata.starting_scene -> fallback to the first scene of the first act, warning.
- Missing global_clocks[].threshold -> default value (6), warning.
- Missing scene_nodes[].resolution_condition -> blocking ERROR (semantic content that validation cannot invent without distorting the scenario).
"""

DEFAULT_THRESHOLD = 6


def _check_duplicates(items, id_key, label, warnings):
    seen = set()
    for item in items:
        iid = item.get(id_key)
        if iid in seen:
            warnings.append(f"{label}: duplicate id '{iid}' detected - only the last occurrence will be retained.")
        seen.add(iid)


def validate_scenario_structure(data: dict) -> tuple[dict, list[str], list[str]]:
    """
    Returns (corrected_data, warnings, blocking_errors).
    Blocking errors signal a need for regeneration or manual correction,
    not something validation can repair silently.
    """
    warnings = []
    errors = []

    # --- ID Duplicates ---
    _check_duplicates(data.get("entities", {}).get("npcs", []), "id", "NPC", warnings)
    _check_duplicates(data.get("entities", {}).get("locations", []), "id", "Location", warnings)
    _check_duplicates(data.get("scene_nodes", []), "scene_id", "Scene", warnings)
    _check_duplicates(data.get("acts", []), "act_id", "Act", warnings)

    npc_ids = {p["id"] for p in data.get("entities", {}).get("npcs", []) if "id" in p}
    location_ids = {l["id"] for l in data.get("entities", {}).get("locations", []) if "id" in l}
    scene_ids = {s["scene_id"] for s in data.get("scene_nodes", []) if "scene_id" in s}
    act_ids = {a["act_id"] for a in data.get("acts", []) if "act_id" in a}

    # --- NPC: usual_location ---
    for p in data.get("entities", {}).get("npcs", []):
        loc = p.get("usual_location")
        if loc not in location_ids:
            if loc is not None:
                warnings.append(
                    f"NPC {p.get('id')}: usual_location '{loc}' unknown -> set to null."
                )
            p["usual_location"] = None

    # --- Scene Nodes ---
    for s in data.get("scene_nodes", []):
        sid = s["scene_id"]

        if s.get("location_id") not in location_ids:
            if s.get("location_id") is not None:
                warnings.append(
                    f"Scene {sid}: location_id '{s['location_id']}' unknown -> set to null."
                )
            s["location_id"] = None

        valid_npcs = [p for p in s.get("present_npcs", []) if p in npc_ids]
        removed = set(s.get("present_npcs", [])) - set(valid_npcs)
        for p in removed:
            warnings.append(f"Scene {sid}: orphan NPC '{p}' removed from present_npcs.")
        s["present_npcs"] = valid_npcs

        valid_exits = []
        for logic_exit in s.get("logical_exits", []):
            if logic_exit.get("destination_scene_id") in scene_ids:
                valid_exits.append(logic_exit)
            else:
                warnings.append(
                    f"Scene {sid}: logical_exit to unknown scene "
                    f"'{logic_exit.get('destination_scene_id')}' removed."
                )
        s["logical_exits"] = valid_exits

        if not s.get("resolution_condition"):
            errors.append(
                f"Scene {sid}: resolution_condition missing - "
                f"requires regeneration of this scene, cannot be automatically repaired."
            )

        act = s.get("act_id")
        if act not in act_ids:
            errors.append(
                f"Scene {sid}: act_id '{act}' does not correspond to any known "
                f"act - manual correction required."
            )

    # --- Bidirectional coherence: act <-> included_scenes ---
    for a in data.get("acts", []):
        declared = set(a.get("included_scenes", []))
        cleaned = [sid for sid in a.get("included_scenes", []) if sid in scene_ids]
        for sid in declared - set(cleaned):
            warnings.append(
                f"Act {a['act_id']}: orphan included_scene '{sid}' removed."
            )
        a["included_scenes"] = cleaned

    for s in data.get("scene_nodes", []):
        act = s.get("act_id")
        if act in act_ids:
            act_obj = next(a for a in data["acts"] if a["act_id"] == act)
            if s["scene_id"] not in act_obj["included_scenes"]:
                act_obj["included_scenes"].append(s["scene_id"])
                warnings.append(
                    f"Coherence repaired: scene {s['scene_id']} added to "
                    f"included_scenes of {act} (declared by scene but absent in act)."
                )

    # --- Metadata ---
    if not data.get("metadata", {}).get("starting_scene"):
        if "metadata" not in data:
            data["metadata"] = {}
        if data.get("acts") and data["acts"][0].get("included_scenes"):
            fallback = data["acts"][0]["included_scenes"][0]
            data["metadata"]["starting_scene"] = fallback
            warnings.append(f"metadata.starting_scene missing -> fallback to '{fallback}'.")
        else:
            errors.append("metadata.starting_scene missing and no fallback possible (acts is empty).")

    # --- Clocks ---
    for h in data.get("global_clocks", []):
        if "threshold" not in h or h["threshold"] is None:
            h["threshold"] = DEFAULT_THRESHOLD
            warnings.append(f"Clock '{h['name']}': threshold missing -> fallback to default {DEFAULT_THRESHOLD}.")

    return data, warnings, errors


def _get_nested(data: dict, path: str):
    """Resolves a dot-separated path ('resources.hit_points') in a nested dict."""
    value = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_character_sheet(character_data: dict, schema: dict) -> tuple[bool, list[str]]:
    """
    Returns (is_complete, missing_fields).

    `schema` is generated by the ruleset (see ManualGeneratorAgent),
    with this format:
    {
      "required_fields": [
        {"path": "name", "type": "string"},
        {"path": "statistics", "type": "object", "sub_fields": ["Strength", "Dexterity", ...]},
        {"path": "resources.hit_points", "type": "object", "sub_fields": ["current", "max"]},
        {"path": "equipment", "type": "list", "non_empty": true}
      ]
    }
    """
    if not character_data:
        return False, ["character_data missing"]

    if not schema or not schema.get("required_fields"):
        return False, ["validation schema missing or empty - creation cannot be confirmed automatically, check Memory/character_schema.json"]

    missing = []

    for field in schema.get("required_fields", []):
        chemin = field["path"]
        ftype = field.get("type", "string")
        value = _get_nested(character_data, chemin)

        if ftype == "string":
            if not value or not isinstance(value, str):
                missing.append(chemin)

        elif ftype == "number":
            if not isinstance(value, (int, float)):
                missing.append(chemin)

        elif ftype == "object":
            sub_fields = field.get("sub_fields", [])
            if not isinstance(value, dict):
                missing.append(chemin)
            else:
                absents = [
                    sc for sc in sub_fields
                    if not isinstance(value.get(sc), (int, float, str)) or value.get(sc) in (None, "")
                ]
                if absents:
                    missing.append(f"{chemin} ({', '.join(absents)})")

        elif ftype == "list":
            if not isinstance(value, list):
                missing.append(chemin)
            elif field.get("non_empty", False) and len(value) == 0:
                missing.append(chemin)

    return (len(missing) == 0), missing
