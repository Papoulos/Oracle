import copy
from typing import Any

from mechanics.paths import get_by_path, set_by_path

def build_task_analysis_prompt(pack, sheet: dict, user_input: str, chronicle_text: str, npcs_json: str, state_summary: str, mechanical_context: str, scale_text: str) -> str:
    """
    Builds the prompt for the Orchestrator's Task Analysis phase.
    """
    prompt = f"""Analyze the player's action: "{user_input}"

Adventure history (Chronicle):
{chronicle_text}

Present NPCs and their secrets (if relevant):
{npcs_json}

MECHANICAL STATE: {state_summary}
{mechanical_context}

"""
    prompt += "AVAILABLE ACTIONS:\n"
    if pack.actions and pack.actions.actions:
        for action in pack.actions.actions:
            prompt += f"- {action.id} ({action.name}): {action.description}\n"
            if action.guidance:
                prompt += f"  Guidance: {action.guidance}\n"

    prompt += "\n"
    prompt += "DIFFICULTY SCALE:\n"
    prompt += scale_text
    prompt += "\n\n"

    # Identify all pools required by at least one action
    required_pools = set()
    if pack.actions and pack.actions.actions:
        for action in pack.actions.actions:
            for p in action.pools:
                required_pools.add(p)

    if required_pools and pack.resources and pack.resources.pools:
        prompt += "CHARACTER POOLS (Current / Max):\n"
        for pool_id in required_pools:
            pool_def = next((p for p in pack.resources.pools if p.id == pool_id), None)
            if pool_def:
                try:
                    current_val = get_by_path(sheet, pool_def.current_path)
                except KeyError:
                    current_val = "?"
                current_str = str(current_val) if isinstance(current_val, int) and not isinstance(current_val, bool) else "?"

                max_str = "?"
                if pool_def.max_value is not None:
                    max_str = str(pool_def.max_value)
                elif pool_def.max_path:
                    try:
                        max_val = get_by_path(sheet, pool_def.max_path)
                        max_str = str(max_val) if isinstance(max_val, int) and not isinstance(max_val, bool) else "?"
                    except KeyError:
                        max_str = "?"

                prompt += f"- {pool_id} ({pool_def.name}): {current_str}/{max_str}\n"
        prompt += "\n"

    prompt += """According to the rules and the character state, determines if a roll is required.
If a roll is required, select the appropriate action_id, pool, difficulty, and determine effort and ease/hinder adjustments.

Reply in JSON format:
{
    "need_roll": boolean,
    "action_id": "action_id_or_null",
    "pool": "pool_id_or_null",
    "difficulty": integer_or_null,
    "ease": {"skill": integer, "asset": integer},
    "hinder": integer,
    "effort_levels": integer,
    "reason": "short explanation"
}
"""
    return prompt

def _parse_int_val(v: Any) -> int:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            val = float(v)
            if val.is_integer():
                return int(val)
        except ValueError:
            pass
    return 0

def sanitize_task_decision(raw: Any, pack) -> dict | None:
    """
    Sanitizes and validates the task decision JSON.
    Returns None if the decision is invalid or need_roll is not strictly True.
    """
    if not isinstance(raw, dict):
        return None

    if raw.get("need_roll") is not True:
        return None

    if not pack.actions or not pack.actions.actions:
        return None

    action_id = raw.get("action_id")
    action_def = None
    if action_id:
        action_def = next((a for a in pack.actions.actions if a.id == action_id), None)

    if not action_def:
        # Fallback to first action
        action_def = pack.actions.actions[0]
        action_id = action_def.id

    pool_id = raw.get("pool")
    if pool_id and pool_id not in action_def.pools:
        # Invalid pool
        return None

    diff = raw.get("difficulty")
    if not isinstance(diff, int) or isinstance(diff, bool):
        return None

    min_diff = 0
    max_diff = 10
    if pack.actions.difficulty_scale:
        min_diff = min(d.level for d in pack.actions.difficulty_scale)
        max_diff = max(d.level for d in pack.actions.difficulty_scale)

    if not (min_diff <= diff <= max_diff):
        return None

    ease = {}
    raw_ease = raw.get("ease", {})
    if isinstance(raw_ease, dict):
        effort_key = action_def.effort.adjustment_key if action_def.effort else None
        allowed_caps = pack.resolution.adjustment_caps.keys() if pack.resolution and pack.resolution.adjustment_caps else []
        allowed_keys = [k for k in allowed_caps if k != effort_key]

        for k, v in raw_ease.items():
            if k in allowed_keys:
                parsed_v = _parse_int_val(v)
                if parsed_v > 0:
                    ease[k] = parsed_v

    hinder = _parse_int_val(raw.get("hinder"))
    if hinder < 0: hinder = 0

    effort_levels = _parse_int_val(raw.get("effort_levels"))
    if effort_levels < 0: effort_levels = 0

    reason = str(raw.get("reason", ""))

    return {
        "need_roll": True,
        "action_id": action_id,
        "pool": pool_id,
        "difficulty": diff,
        "ease": ease,
        "hinder": hinder,
        "effort_levels": effort_levels,
        "reason": reason
    }

def format_task_roll_line(state_changes: dict, pack) -> str:
    """
    Formats the task result into a readable narrative line.
    """
    task_res = state_changes.get("task")
    if not task_res:
        return ""

    diff_info = task_res.get("difficulty", {})
    init_diff = diff_info.get("initial")
    final_diff = diff_info.get("final")

    # Resolve names from pack if available
    init_name = ""
    final_name = ""

    if pack.actions and pack.actions.difficulty_scale:
        init_level = next((d for d in pack.actions.difficulty_scale if d.level == init_diff), None)
        if init_level:
            init_name = init_level.name
        final_level = next((d for d in pack.actions.difficulty_scale if d.level == final_diff), None)
        if final_level:
            final_name = final_level.name

    # Or use provided names if present (fallback)
    if not init_name and diff_info.get("name"):
        init_name = diff_info.get("name") # Usually final name is stored in state_changes

    init_name_str = f" ({init_name})" if init_name else ""
    final_name_str = f" ({final_name})" if final_name else ""
    if diff_info.get("name") and not final_name:
        final_name_str = f" ({diff_info.get('name')})"

    pool_id = task_res.get("pool")
    pool_name = pool_id
    if pool_id and pack.resources and pack.resources.pools:
        pool_def = next((p for p in pack.resources.pools if p.id == pool_id), None)
        if pool_def:
            pool_name = pool_def.name

    effort_res = state_changes.get("effort")
    effort_str = ""
    if effort_res and not effort_res.get("waived") and effort_res.get("cost", 0) > 0:
        cost = effort_res.get("cost")
        levels = effort_res.get("levels")
        before = effort_res.get("before")
        after = effort_res.get("after")
        effort_str = f" Effort : {levels} niveau(x), coût {cost}, {pool_name} {before}→{after}."

    target = task_res.get("target")
    roll = task_res.get("roll")
    outcome = task_res.get("outcome")
    margin = task_res.get("margin")

    if outcome == "SUCCESS":
        outcome_str = "Succès"
    elif outcome == "FAILURE":
        outcome_str = "Échec"
    else:
        outcome_str = "Impossible"

    margin_str = ""
    if margin is not None:
        margin_str = f"+{margin}" if margin >= 0 else f"{margin}"

    base_str = f"Tâche — {pool_name} : "

    if task_res.get("impossible"):
        res = f"{base_str}cible {target} inatteignable : aucun dé, échec. Effort non dépensé."
    elif task_res.get("auto_success"):
        res = f"{base_str}difficulté {init_diff}{init_name_str} → {final_diff}{final_name_str} : réussite automatique, aucun dé."
    else:
        res = f"{base_str}difficulté {init_diff}{init_name_str} → {final_diff}{final_name_str}, cible {target}, d20 = {roll} → {outcome_str} (marge {margin_str})."

    res += effort_str

    warnings = state_changes.get("warnings", [])
    if warnings:
        warn_str = " ; ".join(warnings)
        res += f" Attention : {warn_str}."

    return res

def _clean_empty_dicts(d: dict):
    """Recursively removes empty dictionaries, in-place if possible."""
    keys_to_del = []
    for k, v in d.items():
        if isinstance(v, dict):
            _clean_empty_dicts(v)
            if not v:
                keys_to_del.append(k)
    for k in keys_to_del:
        del d[k]

def _get_by_path_strict(data: dict, path: str):
    keys = path.split('.')
    val = data
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            raise KeyError(k)
        val = val[k]
    return val

def enforce_engine_owned_fields(sheet: dict, snapshot: dict, pack) -> tuple[dict, list[str]]:
    """
    Enforces engine-owned fields on a modified sheet, using the snapshot as ground truth.
    Returns a deep-copied corrected sheet and a list of restored paths.
    """
    new_sheet = copy.deepcopy(sheet)
    restored_paths = []

    engine_paths = ["recovery_step", "pending_allocation"]

    if pack.resources:
        if pack.resources.pools:
            for pool in pack.resources.pools:
                engine_paths.append(pool.current_path)
                if pool.max_path:
                    engine_paths.append(pool.max_path)
        if pack.resources.pool_groups:
            for group in pack.resources.pool_groups:
                engine_paths.append(group.path)

    for path in engine_paths:
        try:
            snap_val = _get_by_path_strict(snapshot, path)
            snap_exists = True
        except KeyError:
            snap_exists = False

        try:
            sheet_val = _get_by_path_strict(new_sheet, path)
            sheet_exists = True
        except KeyError:
            sheet_exists = False

        if snap_exists:
            # We want strict dict equality. For dicts, check order/keys/values.
            if not sheet_exists or sheet_val != snap_val:
                set_by_path(new_sheet, path, copy.deepcopy(snap_val))
                restored_paths.append(path)
        else:
            if sheet_exists:
                # Need to delete the path from new_sheet
                parts = path.split('.')
                curr = new_sheet
                can_delete = True
                for part in parts[:-1]:
                    if part in curr:
                        curr = curr[part]
                    else:
                        can_delete = False
                        break
                if can_delete and parts[-1] in curr:
                    del curr[parts[-1]]
                    restored_paths.append(path)

    _clean_empty_dicts(new_sheet)
    return new_sheet, restored_paths
