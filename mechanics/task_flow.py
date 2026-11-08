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

                prompt += f"- {pool_id} ({pool_def.name}) : {current_str}/{max_str}\n"
        prompt += "\n"

    dmg_cfg = getattr(pack.resources, "damage", None) if pack.resources else None
    if dmg_cfg:
        armor_val = get_by_path(sheet, dmg_cfg.armor_path) if dmg_cfg.armor_path else None
        track_val = get_by_path(sheet, dmg_cfg.damage_track_path) if dmg_cfg.damage_track_path else None
        default_state = dmg_cfg.damage_track_states[0] if dmg_cfg.damage_track_states else "normal"
        if armor_val is not None or track_val is not None:
            prompt += f"COMBAT STATUS: Armor={armor_val if armor_val is not None else 0}, Damage Track={track_val or default_state}\n\n"

    family = getattr(pack.resolution, "family", "StepTargetD20") if getattr(pack, "resolution", None) else "StepTargetD20"
    if isinstance(sheet, dict) and (sheet.get("statistics") or sheet.get("modifiers")):
        stats_dict = sheet.get("statistics") or {}
        mods_dict = sheet.get("modifiers") or {}
        stat_parts = []
        for k, v in stats_dict.items():
            if k in mods_dict:
                stat_parts.append(f"{k}: {v} (mod {mods_dict[k]:+d})" if isinstance(mods_dict[k], int) else f"{k}: {v}")
            else:
                stat_parts.append(f"{k}: {v}")
        if stat_parts:
            prompt += f"CHARACTER STATISTICS: {', '.join(stat_parts)}\n\n"

    consumables_cfg = getattr(pack.resources, "consumables", None) if pack.resources else None
    if consumables_cfg and isinstance(sheet, dict):
        cons_lines = []
        for cdef in consumables_cfg:
            items = get_by_path(sheet, cdef.path)
            if isinstance(items, list):
                limit_val = cdef.max_items if cdef.max_items is not None else (get_by_path(sheet, cdef.max_items_path) if cdef.max_items_path else None)
                limit_str = f"/{limit_val}" if isinstance(limit_val, int) else ""
                items_str = ", ".join(i.get("name", str(i)) if isinstance(i, dict) else str(i) for i in items) if items else "aucun"
                cons_lines.append(f"- {cdef.id} ({cdef.name}, {len(items)}{limit_str}) : {items_str}")
        if cons_lines:
            prompt += "CONSUMABLES / ITEMS:\n" + "\n".join(cons_lines) + "\n\n"

    prog_cfg = getattr(pack.resources, "progression", None) if pack.resources else None
    if prog_cfg and isinstance(sheet, dict):
        xp_val = get_by_path(sheet, prog_cfg.xp_path)
        lvl_val = get_by_path(sheet, prog_cfg.level_path) if prog_cfg.level_path else None
        taken_val = get_by_path(sheet, prog_cfg.advancements_path) if prog_cfg.advancements_path else None
        prog_parts = [f"XP={xp_val if isinstance(xp_val, int) else 0}"]
        if lvl_val is not None:
            prog_parts.append(f"{prog_cfg.level_path}={lvl_val}")
        if isinstance(taken_val, list):
            prog_parts.append(f"{prog_cfg.advancements_path}={taken_val}")
        if prog_cfg.advancements:
            adv_ids = [f"{a.id} ({a.cost} XP)" for a in prog_cfg.advancements]
            prog_parts.append(f"advancements=[{', '.join(adv_ids)}]")
        prompt += "PROGRESSION: " + ", ".join(prog_parts) + "\n\n"

    prompt += """According to the rules and the character state, determines if a roll is required.
If a roll is required, select the appropriate action_id, pool (if applicable), difficulty, and modifiers/adjustments.
For attack actions (role="attack"), you may also specify "damage_effort_levels", "base_damage", "target_npc", "npc_health_max", and "npc_armor".
For defense actions (role="defend"), you may also specify "incoming_damage", "damage_pool", and "ignore_armor".
If the player uses a consumable/cypher/potion, specify "consume_item". If they find/receive one, specify "add_item".
If XP is awarded or spent (or an advancement/level-up is requested), specify "xp_gain", "spend_xp", "buy_advancement", or "level_up".

Reply in JSON format:
{
    "need_roll": boolean,
    "action_id": "action_id_or_null",
    "pool": "pool_id_or_null",
    "difficulty": integer_or_null,
    "stat": "stat_name_or_null_optional",
    "modifier": integer_optional,
    "roll_bonus": integer_optional,
    "advantage": "none_or_advantage_or_disadvantage_optional",
    "pool_size": integer_optional,
    "ease": {"skill": integer, "asset": integer},
    "hinder": integer,
    "effort_levels": integer,
    "damage_effort_levels": integer_optional,
    "base_damage": integer_or_null_optional,
    "target_npc": "npc_id_or_null_optional",
    "npc_health_max": integer_or_null_optional,
    "npc_armor": integer_optional,
    "incoming_damage": integer_or_null_optional,
    "damage_pool": "pool_id_or_null_optional",
    "ignore_armor": boolean_optional,
    "consume_item": {"list": "list_id_optional", "item": "item_name", "heal_pool": "pool_id_optional", "heal_amount": integer_optional, "heal_expr": "formula_optional"},
    "add_item": {"list": "list_id_optional", "item": "item_name_or_object"},
    "xp_gain": integer_optional,
    "spend_xp": {"amount": integer_optional, "reason": "string_optional"},
    "buy_advancement": {"id": "advancement_id", "target_path": "path_optional", "allocation": {"pool": integer}, "value": "string_optional"},
    "level_up": boolean_optional,
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

    family = getattr(pack.resolution, "family", "StepTargetD20") if getattr(pack, "resolution", None) else "StepTargetD20"

    action_id = raw.get("action_id")
    action_def = None
    if action_id:
        action_def = next((a for a in pack.actions.actions if a.id == action_id), None)

    if not action_def:
        # Fallback to first action
        action_def = pack.actions.actions[0]
        action_id = action_def.id

    pool_id = raw.get("pool")
    if not pool_id:
        pool_id = None

    if action_def.pools:
        if pool_id is None:
            return None
        if pool_id not in action_def.pools:
            return None
    else:
        pool_id = None

    diff = raw.get("difficulty")
    if diff is None and family == "D20VsTarget" and "dc" in raw:
        diff = raw.get("dc")

    if family == "StepTargetD20":
        if not isinstance(diff, int) or isinstance(diff, bool):
            return None
        min_diff = 0
        max_diff = 10
        if pack.actions.difficulty_scale:
            min_diff = min(d.level for d in pack.actions.difficulty_scale)
            max_diff = max(d.level for d in pack.actions.difficulty_scale)
        if not (min_diff <= diff <= max_diff):
            return None
    elif family == "D20VsTarget":
        if not isinstance(diff, int) or isinstance(diff, bool) or diff < 0:
            return None
    elif family == "PbtA2d6":
        if diff is None:
            diff = 0
        elif not isinstance(diff, int) or isinstance(diff, bool):
            return None
    elif family == "DicePoolSuccess":
        if diff is None:
            diff = 1
        elif not isinstance(diff, int) or isinstance(diff, bool) or diff < 1:
            return None
    else:
        return None

    ease = {}
    raw_ease = raw.get("ease", {})
    if isinstance(raw_ease, dict):
        effort_key = action_def.effort.adjustment_key if action_def.effort else None
        allowed_caps = pack.resolution.adjustment_caps.keys() if pack.resolution and getattr(pack.resolution, "adjustment_caps", None) else []
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

    out = {
        "need_roll": True,
        "action_id": action_id,
        "pool": pool_id,
        "difficulty": diff,
        "ease": ease,
        "hinder": hinder,
        "effort_levels": effort_levels,
        "reason": reason
    }

    if isinstance(raw.get("stat"), str) and raw["stat"].strip():
        out["stat"] = raw["stat"].strip()

    if "modifier" in raw and raw.get("modifier") is not None:
        out["modifier"] = _parse_int_val(raw.get("modifier"))
    elif "bonus" in raw and raw.get("bonus") is not None:
        out["modifier"] = _parse_int_val(raw.get("bonus"))

    if "roll_bonus" in raw and raw.get("roll_bonus") is not None:
        out["roll_bonus"] = _parse_int_val(raw.get("roll_bonus"))

    if raw.get("advantage") in ("advantage", "disadvantage"):
        out["advantage"] = raw["advantage"]

    if "pool_size" in raw and raw.get("pool_size") is not None:
        ps = _parse_int_val(raw.get("pool_size"))
        if ps > 0:
            out["pool_size"] = ps

    if "damage_effort_levels" in raw:
        dmg_eff = _parse_int_val(raw.get("damage_effort_levels"))
        if dmg_eff > 0:
            out["damage_effort_levels"] = dmg_eff

    if "base_damage" in raw and raw.get("base_damage") is not None:
        base_dmg = _parse_int_val(raw.get("base_damage"))
        if base_dmg > 0:
            out["base_damage"] = base_dmg

    if isinstance(raw.get("target_npc"), str) and raw["target_npc"].strip():
        out["target_npc"] = raw["target_npc"].strip()

    if "npc_health_max" in raw and raw.get("npc_health_max") is not None:
        npc_hp = _parse_int_val(raw.get("npc_health_max"))
        if npc_hp > 0:
            out["npc_health_max"] = npc_hp

    if "npc_armor" in raw and raw.get("npc_armor") is not None:
        npc_arm = _parse_int_val(raw.get("npc_armor"))
        if npc_arm > 0:
            out["npc_armor"] = npc_arm

    if "incoming_damage" in raw and raw.get("incoming_damage") is not None:
        inc_dmg = _parse_int_val(raw.get("incoming_damage"))
        if inc_dmg > 0:
            out["incoming_damage"] = inc_dmg

    if isinstance(raw.get("damage_pool"), str) and raw["damage_pool"].strip():
        out["damage_pool"] = raw["damage_pool"].strip()

    if raw.get("ignore_armor") is True:
        out["ignore_armor"] = True

    return out


def sanitize_turn_mechanics(raw: Any, pack) -> dict:
    """
    Extracts optional non-roll turn mechanics (consumable use/addition, XP gain/spend,
    advancement purchase, level-up) from the Orchestrator's analysis JSON.
    Works regardless of whether need_roll is True or False.
    """
    if not isinstance(raw, dict) or not pack or not getattr(pack, "resources", None):
        return {}

    out: dict[str, Any] = {}

    raw_consume = raw.get("consume_item")
    if isinstance(raw_consume, dict) and isinstance(raw_consume.get("item"), str) and raw_consume["item"].strip():
        c_dict: dict[str, Any] = {"item": raw_consume["item"].strip()}
        lst = raw_consume.get("list") or raw_consume.get("list_id")
        if isinstance(lst, str) and lst.strip():
            c_dict["list"] = lst.strip()
        if isinstance(raw_consume.get("heal_pool"), str) and raw_consume["heal_pool"].strip():
            c_dict["heal_pool"] = raw_consume["heal_pool"].strip()
        if raw_consume.get("heal_amount") is not None:
            ha = _parse_int_val(raw_consume.get("heal_amount"))
            if ha > 0:
                c_dict["heal_amount"] = ha
        if isinstance(raw_consume.get("heal_expr"), str) and raw_consume["heal_expr"].strip():
            c_dict["heal_expr"] = raw_consume["heal_expr"].strip()
        out["consume_item"] = c_dict

    raw_add = raw.get("add_item")
    if isinstance(raw_add, dict) and raw_add.get("item"):
        item_val = raw_add["item"]
        if (isinstance(item_val, str) and item_val.strip()) or isinstance(item_val, dict):
            a_dict: dict[str, Any] = {"item": item_val.strip() if isinstance(item_val, str) else item_val}
            lst = raw_add.get("list") or raw_add.get("list_id")
            if isinstance(lst, str) and lst.strip():
                a_dict["list"] = lst.strip()
            out["add_item"] = a_dict

    if raw.get("xp_gain") is not None:
        xp_g = _parse_int_val(raw.get("xp_gain"))
        if xp_g > 0:
            out["xp_gain"] = xp_g

    raw_spend = raw.get("spend_xp")
    if isinstance(raw_spend, dict):
        amt = _parse_int_val(raw_spend.get("amount"))
        reason = raw_spend.get("reason")
        reason_str = reason.strip() if isinstance(reason, str) and reason.strip() else None
        if amt > 0 or reason_str:
            out["spend_xp"] = {"amount": amt, "reason": reason_str}

    raw_adv = raw.get("buy_advancement")
    if isinstance(raw_adv, dict) and isinstance(raw_adv.get("id"), str) and raw_adv["id"].strip():
        adv_dict: dict[str, Any] = {"id": raw_adv["id"].strip()}
        tp = raw_adv.get("target_path") or raw_adv.get("pool")
        if isinstance(tp, str) and tp.strip():
            adv_dict["target_path"] = tp.strip()
        raw_alloc = raw_adv.get("allocation") or raw_adv.get("allocations")
        if isinstance(raw_alloc, dict):
            alloc = {}
            for k, v in raw_alloc.items():
                iv = _parse_int_val(v)
                if iv > 0:
                    alloc[str(k)] = iv
            if alloc:
                adv_dict["allocation"] = alloc
        val = raw_adv.get("value")
        if val is None:
            val = raw_adv.get("skill") if raw_adv.get("skill") is not None else raw_adv.get("ability")
        if val is not None:
            adv_dict["value"] = val
        out["buy_advancement"] = adv_dict

    if raw.get("level_up") is True:
        out["level_up"] = True

    return out

def format_task_roll_line(state_changes: dict, pack) -> str:
    """
    Formats the task result into a readable narrative line.
    """
    task_res = state_changes.get("task")
    if not task_res:
        return ""

    family = getattr(pack.resolution, "family", "StepTargetD20") if getattr(pack, "resolution", None) else "StepTargetD20"

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

    if not init_name and diff_info.get("name"):
        init_name = diff_info.get("name")

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

    target = task_res.get("target")
    impossible = bool(task_res.get("impossible", False))
    rolled_list = task_res.get("rolled", [])

    outcome = task_res.get("outcome")
    if outcome == "CRITICAL_SUCCESS":
        outcome_str = "Succès critique"
    elif outcome == "SUCCESS":
        outcome_str = "Succès"
    elif outcome == "PARTIAL_SUCCESS":
        outcome_str = "Succès partiel"
    elif outcome == "CRITICAL_FAILURE":
        outcome_str = "Échec critique"
    elif outcome == "FAILURE":
        outcome_str = "Échec"
    else:
        outcome_str = "Impossible"

    margin = task_res.get("margin")
    margin_str = ""
    if margin is not None:
        margin_str = f" (marge +{margin})" if margin >= 0 else f" (marge {margin})"

    action_id = task_res.get("action", "task")
    action_def = None
    if pack.actions and pack.actions.actions:
        action_def = next((a for a in pack.actions.actions if a.id == action_id), None)

    action_label = action_def.name if (action_def and action_def.name) else "Tâche"
    action_role = getattr(action_def, "role", "general") if action_def else "general"

    if family == "StepTargetD20":
        base_str = f"{action_label} — {pool_name} : "
        if impossible:
            res = f"{base_str}difficulté {init_diff}{init_name_str} → {final_diff}{final_name_str}, cible {target} inatteignable : aucun dé, échec."
        elif not rolled_list:
            res = f"{base_str}difficulté {init_diff}{init_name_str} → {final_diff}{final_name_str} : réussite automatique, aucun dé."
        else:
            roll = rolled_list[0]
            res = f"{base_str}difficulté {init_diff}{init_name_str} → {final_diff}{final_name_str}, cible {target}, d20 = {roll} → {outcome_str}{margin_str}."
    elif family == "D20VsTarget":
        stat_label = task_res.get("stat") or pool_name
        prefix = f"{action_label} ({stat_label})" if stat_label else action_label
        mod_val = task_res.get("modifier", 0)
        mod_str = f" + {mod_val}" if mod_val >= 0 else f" - {abs(mod_val)}"
        adv = task_res.get("advantage")
        roll_display = f"{rolled_list} ({adv})" if adv else (str(rolled_list[0]) if rolled_list else "?")
        total_val = task_res.get("total", 0)
        res = f"{prefix} — DD {target}{init_name_str} : d20 = {roll_display}{mod_str} = {total_val} → {outcome_str}{margin_str}."
    elif family == "PbtA2d6":
        stat_label = task_res.get("stat") or pool_name
        prefix = f"{action_label} ({stat_label})" if stat_label else action_label
        mod_val = task_res.get("modifier", 0)
        mod_str = f" + {mod_val}" if mod_val >= 0 else f" - {abs(mod_val)}"
        total_val = task_res.get("total", 0)
        tier_lbl = task_res.get("tier_label")
        tier_str = f" ({tier_lbl})" if tier_lbl else ""
        res = f"{prefix} : 2d6 = {rolled_list}{mod_str} = {total_val} → {outcome_str}{tier_str}."
    else:  # DicePoolSuccess
        total_val = task_res.get("total", 0)
        res = f"{action_label} — seuil {target} succès : dés = {rolled_list} → {total_val} succès → {outcome_str}{margin_str}."

    # Effort block
    effort_res = state_changes.get("effort")
    if effort_res:
        waived = bool(effort_res.get("waived", False))
        refunded = bool(effort_res.get("refunded", False))
        cost = effort_res.get("cost", 0)
        levels = effort_res.get("levels", 0)

        if waived:
            res += " Effort non dépensé."
        elif refunded:
            res += f" Effort : {levels} niveau(x), coût 0 (remboursé sur 20 naturel)."
        elif cost > 0:
            res += f" Effort : {levels} niveau(x), coût {cost}"
            pool_changes = state_changes.get(pool_id)
            if pool_changes and "avant" in pool_changes and "apres" in pool_changes:
                res += f", {pool_name} {pool_changes['avant']}→{pool_changes['apres']}."
            else:
                res += "."

    # Special roll narrative tag (for combat actions or when refunded)
    special_roll = task_res.get("special_roll")
    if special_roll and action_role != "general":
        special_labels = getattr(action_def, "special_roll_labels", {}) if action_def else {}
        label = special_labels.get(special_roll, special_roll)
        res += f" {label} !"

    # Attack damage block
    atk_dmg = state_changes.get("attack_damage")
    if atk_dmg:
        total_dmg = atk_dmg.get("total", 0)
        npc_dmg = state_changes.get("npc_damage")
        if npc_dmg:
            npc_id = npc_dmg.get("npc_id", "PNJ")
            avant = npc_dmg.get("avant", 0)
            apres = npc_dmg.get("apres", 0)
            defeated_str = ", vaincu" if npc_dmg.get("defeated") else ""
            res += f" Dégâts infligés : {total_dmg} ({npc_id} {avant}→{apres} PV{defeated_str})."
        else:
            res += f" Dégâts infligés : {total_dmg}."

    # Defense damage taken block
    dmg_taken = state_changes.get("damage_taken")
    if dmg_taken:
        eff_dmg = dmg_taken.get("effective", 0)
        raw_dmg = dmg_taken.get("raw", 0)
        arm_dmg = dmg_taken.get("armor", 0)
        track_info = dmg_taken.get("damage_track", {})
        track_str = ""
        if track_info and track_info.get("avant") != track_info.get("apres"):
            track_str = f", état {track_info.get('avant')}→{track_info.get('apres')}"
        res += f" Dégâts subis : {eff_dmg} (brut {raw_dmg}, armure -{arm_dmg}{track_str})."

    warnings = task_res.get("warnings", [])
    if warnings:
        warn_str = " ; ".join(warnings)
        res += f" Attention : {warn_str}."

    return res

def _get_by_path_strict(data: dict, path: str):
    keys = path.split('.')
    val = data
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            raise KeyError(k)
        val = val[k]
    return val

def enforce_engine_owned_fields(sheet: dict, snapshot: dict, pack, extra_paths: list[str] | None = None) -> tuple[dict, list[str]]:
    """
    Enforces engine-owned fields on a modified sheet, using the snapshot as ground truth.
    Returns a deep-copied corrected sheet and a list of restored paths.
    """
    new_sheet = copy.deepcopy(sheet)
    restored_paths = []

    engine_paths = ["recovery_step", "pending_allocation", "npc_health"]

    if pack.resources:
        if pack.resources.pools:
            for pool in pack.resources.pools:
                engine_paths.append(pool.current_path)
                if pool.max_path:
                    engine_paths.append(pool.max_path)
        if pack.resources.pool_groups:
            for group in pack.resources.pool_groups:
                engine_paths.append(group.path)
        if getattr(pack.resources, "damage", None) and pack.resources.damage.damage_track_path:
            engine_paths.append(pack.resources.damage.damage_track_path)
        if getattr(pack.resources, "consumables", None):
            for cdef in pack.resources.consumables:
                if getattr(cdef, "engine_owned", False) and cdef.path not in engine_paths:
                    engine_paths.append(cdef.path)
        if getattr(pack.resources, "progression", None):
            prog = pack.resources.progression
            if prog.xp_path and prog.xp_path not in engine_paths:
                engine_paths.append(prog.xp_path)
            if prog.level_path and prog.level_path not in engine_paths:
                engine_paths.append(prog.level_path)
            if prog.advancements_path and prog.advancements_path not in engine_paths:
                engine_paths.append(prog.advancements_path)

    if extra_paths:
        for ep in extra_paths:
            if ep and ep not in engine_paths:
                engine_paths.append(ep)

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
                    if isinstance(curr, dict) and part in curr:
                        curr = curr[part]
                    else:
                        can_delete = False
                        break
                if can_delete and isinstance(curr, dict) and parts[-1] in curr:
                    del curr[parts[-1]]
                    restored_paths.append(path)

    return new_sheet, restored_paths

def _pool_values(pack, sheet: dict, pool_id: str) -> tuple[Any, Any, Any]:
    pool_def = None
    if pack.resources and pack.resources.pools:
        pool_def = next((p for p in pack.resources.pools if p.id == pool_id), None)

    if not pool_def:
        return None, "?", "?"

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

    return pool_def, current_str, max_str


def build_allocation_prompt(pack, sheet: dict, pending: dict, user_input: str) -> str:
    points = pending.get("points", "?")
    if not isinstance(points, int):
        points = "?"

    among = pending.get("among", [])
    if not isinstance(among, list):
        among = []

    prompt = f"Points to allocate: {points}\n\n"
    prompt += "AVAILABLE POOLS (Current / Max):\n"

    if among and pack.resources and pack.resources.pools:
        for pool_id in among:
            pool_def, current_str, max_str = _pool_values(pack, sheet, pool_id)
            if pool_def:
                pool_line = f"- {pool_id} ({pool_def.name}) : {current_str}/{max_str}"

                if current_str != "?" and max_str != "?":
                    try:
                        c = int(current_str)
                        m = int(max_str)
                        missing = m - c
                        if missing < 0:
                            missing = 0
                        pool_line += f" (manque {missing})"
                    except ValueError:
                        pass

                prompt += f"{pool_line}\n"
            else:
                prompt += f"- {pool_id} (?) : ?/?\n"
    elif among:
        for pool_id in among:
            prompt += f"- {pool_id} (?) : ?/?\n"

    prompt += f"\nPlayer message: \"{user_input}\"\n\n"

    prompt += """Instructions:
You must determine how the player wants to allocate their points based ONLY on their message.
Do NOT invent an allocation if the player does not explicitly specify one.
If the message does not specify an allocation, set "understood" to false.
If the player explicitly declines or cancels the allocation, set "decline" to true.

Reply strictly in JSON format:
{
    "understood": boolean,
    "decline": boolean,
    "allocation": {"pool_id": integer, ...}
}
"""
    return prompt

def sanitize_allocation(raw: Any, pending: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None

    points = pending.get("points")
    if not isinstance(points, int) or points < 0:
        return None

    among = pending.get("among")
    if not isinstance(among, list) or not among:
        return None
    if not all(isinstance(a, str) for a in among):
        return None

    understood = raw.get("understood")
    if understood is not True:
        return None

    decline = raw.get("decline")
    if decline is True:
        return {"action": "decline", "allocation": {}}

    allocation_raw = raw.get("allocation")
    if not isinstance(allocation_raw, dict):
        return None

    allocation = {}
    total_allocated = 0

    for k, v in allocation_raw.items():
        if k not in among:
            continue

        if isinstance(v, bool):
            return None

        # Custom explicit parsing since _parse_int_val defaults to 0 on failure. We want strict integers/convertible floats
        parsed_v = None
        if isinstance(v, int):
            parsed_v = v
        elif isinstance(v, float) and v.is_integer():
            parsed_v = int(v)
        elif isinstance(v, str):
            try:
                f_val = float(v)
                if f_val.is_integer():
                    parsed_v = int(f_val)
            except ValueError:
                pass

        if parsed_v is None or parsed_v < 0:
             return None

        if parsed_v > 0:
            allocation[k] = parsed_v
            total_allocated += parsed_v

    if total_allocated == 0:
        return None

    if total_allocated > points:
        return None

    return {"action": "allocate", "allocation": allocation}

def format_pending_allocation_note(pending: dict, sheet: dict, pack) -> str:
    """
    Returns a formatted, localized (French) string explaining the pending allocation to the player.
    """
    if not isinstance(pending, dict):
        return ""

    points = pending.get("points")
    among = pending.get("among")

    if not isinstance(points, int) or not isinstance(among, list) or not among:
        return ""

    formatted_items = []
    for pool_id in among:
        pool_def, current_str, max_str = _pool_values(pack, sheet, pool_id)
        if pool_def:
            formatted_items.append(f"{pool_id} ({pool_def.name}) {current_str}/{max_str}")
        else:
            formatted_items.append(f"{pool_id}")

    liste = ", ".join(formatted_items)

    return f"\n⏳ RÉPARTITION EN ATTENTE : {points} point(s) de récupération à répartir entre : {liste}. Demande au joueur comment les répartir ; ne décide pas à sa place."
