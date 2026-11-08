import json
import os
import logging
import unicodedata
import re
import random
from dataclasses import dataclass, field
from typing import Optional
from systems.loader import SystemPack
from mechanics.models import ResolutionRequest, Outcome
from mechanics.resolution import resolve
from mechanics.paths import get_by_path, set_by_path
from mechanics.expr import evaluate_int, ExprError

logger = logging.getLogger(__name__)

@dataclass
class ActionResult:
    success: bool
    message: str                    # technical message for the Orchestrator
    state_changes: dict = field(default_factory=dict)  # what has changed
    blocked_reason: Optional[str] = None  # if success=False


@dataclass
class DetectedAction:
    rule_id: str
    kind: str
    target: Optional[str] = None
    key: Optional[str] = None
    amount: int = 1
    trigger: Optional[str] = None
    sequence: Optional[str] = None


class GameStateEngine:
    """
    Central Game State Engine — Pure Python, zero LLM.
    Validates and applies all mechanical game state changes.
    """

    def __init__(self, character_path: str = "Memory/character.json", pack: Optional[SystemPack] = None, rng: Optional[random.Random] = None):
        self.character_path = character_path
        self.pack = pack
        self.rng = rng if rng is not None else random.Random()
        if self.pack is None:
            logger.warning("mode expérimental, aucun system pack chargé")
        else:
            logger.info(f"System pack loaded: {self.pack.manifest.id} (v{self.pack.manifest.version})")
            if not self.pack.triggers:
                logger.warning(f"No triggers.json found in pack {self.pack.manifest.id}. Automatic action detection disabled.")
        self.state = self._load()

    # ── Load / Save ──────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.character_path):
            try:
                with open(self.character_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save(self):
        os.makedirs("Memory", exist_ok=True)
        with open(self.character_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4, ensure_ascii=False)

    def reload(self):
        """Reload from disk (after external update)."""
        self.state = self._load()

    # ── Accessors ────────────────────────────────────────────

    def _get_health_pool_def(self):
        if self.pack and getattr(self.pack, "resources", None) and self.pack.resources.pools:
            return next((p for p in self.pack.resources.pools if p.kind == "health"), None)
        return None

    def get_hp(self) -> tuple[int, int]:
        """Returns (current_hp, max_hp)."""
        hp_pool = self._get_health_pool_def()
        if hp_pool:
            cur = get_by_path(self.state, hp_pool.current_path)
            max_val = hp_pool.max_value if hp_pool.max_value is not None else (get_by_path(self.state, hp_pool.max_path) if hp_pool.max_path else 0)
            return (cur if isinstance(cur, int) else 0), (max_val if isinstance(max_val, int) else 0)
        resources = self.state.get("resources", {})
        pv = resources.get("hit_points", {})
        return pv.get("current", 0), pv.get("max", 0)

    def get_state_summary(self) -> str:
        """Readable summary of the state for the Orchestrator."""
        if not self.state:
            return "No state available."

        hp_cur, hp_max = self.get_hp()
        lines = [
            f"Level: {self.state.get('level', '?')}",
            f"XP: {self.state.get('xp', 0)}/{self.state.get('next_level_xp', self.state.get('xp_prochain_niveau', '?'))}",
        ]

        if not self.pack:
            lines.insert(0, f"HP: {hp_cur}/{hp_max}")

        if self.pack:
            # Dynamically build summary from pack resources
            for pool in self.pack.resources.pools:
                cur = get_by_path(self.state, pool.current_path)
                if cur is None:
                    logger.warning(f"Resource path '{pool.current_path}' not found in state for pool '{pool.id}'. Ignored.")
                    continue

                max_val = None
                if pool.max_value is not None:
                    max_val = pool.max_value
                elif pool.max_path is not None:
                    max_val = get_by_path(self.state, pool.max_path)

                if max_val is not None:
                    lines.append(f"{pool.name}: {cur}/{max_val}")
                else:
                    lines.append(f"{pool.name}: {cur}")

            for group in self.pack.resources.pool_groups:
                group_dict = get_by_path(self.state, group.path)
                if isinstance(group_dict, dict):
                    keys = list(group_dict.keys())

                    def natural_sort_key(s):
                        import re
                        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

                    sorted_keys = sorted(keys, key=natural_sort_key)

                    for key in sorted_keys:
                        values = group_dict[key]
                        if isinstance(values, dict):
                            lines.append(f"{group.name} {key}: {values.get('current', 0)}/{values.get('max', 0)}")

            dmg_cfg = getattr(self.pack.resources, "damage", None)
            if dmg_cfg:
                if dmg_cfg.armor_path:
                    armor_val = get_by_path(self.state, dmg_cfg.armor_path)
                    if isinstance(armor_val, int) and not isinstance(armor_val, bool):
                        lines.append(f"Armor: {armor_val}")
                if dmg_cfg.damage_track_path:
                    track_val = get_by_path(self.state, dmg_cfg.damage_track_path)
                    if isinstance(track_val, str) and track_val:
                        lines.append(f"Damage Track: {track_val}")

            npc_health = self.state.get("npc_health")
            if isinstance(npc_health, dict) and npc_health:
                npc_parts = []
                for npc_id, info in npc_health.items():
                    if isinstance(info, dict):
                        npc_parts.append(f"{npc_id} {info.get('current', 0)}/{info.get('max', 0)}")
                if npc_parts:
                    lines.append(f"NPC Health: {', '.join(npc_parts)}")

        return " | ".join(lines)

    # ── Modifications Application ────────────────────────

    def _compute_damage_track(self) -> Optional[str]:
        """
        Computes the damage track state configured in the pack's damage settings
        based on how many cascade pools are at or below min_value.
        """
        if not self.pack or not getattr(self.pack, "resources", None) or not getattr(self.pack.resources, "damage", None):
            return None
        dmg_cfg = self.pack.resources.damage
        if not dmg_cfg.damage_track_states:
            return None
        zero_count = 0
        present_pools = 0
        for pid in dmg_cfg.cascade_pools:
            pool_def = next((p for p in self.pack.resources.pools if p.id == pid), None)
            if not pool_def:
                continue
            cur = get_by_path(self.state, pool_def.current_path)
            if isinstance(cur, int) and not isinstance(cur, bool):
                present_pools += 1
                if cur <= pool_def.min_value:
                    zero_count += 1
        if present_pools == 0:
            return None
        states = dmg_cfg.damage_track_states
        idx = min(zero_count, len(states) - 1)
        return states[idx]

    def advance_damage_track(self, steps: int = 1) -> ActionResult:
        """
        Directly shifts the character's damage track by `steps` (positive to worsen,
        negative to recover), as configured in the pack's damage_track_states.
        """
        if not self.pack or not getattr(self.pack, "resources", None) or not getattr(self.pack.resources, "damage", None):
            return ActionResult(success=False, message="No damage track configured in pack.", blocked_reason="no_damage_track")
        dmg_cfg = self.pack.resources.damage
        states = dmg_cfg.damage_track_states
        if not states or not dmg_cfg.damage_track_path:
            return ActionResult(success=False, message="No damage track states configured in pack.", blocked_reason="no_damage_track")

        old_track = get_by_path(self.state, dmg_cfg.damage_track_path)
        old_idx = states.index(old_track) if old_track in states else 0
        new_idx = max(0, min(len(states) - 1, old_idx + steps))
        new_track = states[new_idx]
        set_by_path(self.state, dmg_cfg.damage_track_path, new_track)
        self.save()
        return ActionResult(
            success=True,
            message=f"Damage track shifted: {old_track or states[0]} -> {new_track}.",
            state_changes={dmg_cfg.damage_track_path: {"avant": old_track or states[0], "apres": new_track}},
        )

    def update_damage_track(self) -> Optional[dict]:
        """
        Updates the damage track in state if configured and returns {"avant": old, "apres": new}
        if it changed, or None.
        """
        if not self.pack or not getattr(self.pack, "resources", None) or not getattr(self.pack.resources, "damage", None):
            return None
        dmg_cfg = self.pack.resources.damage
        new_track = self._compute_damage_track()
        if new_track is None:
            return None
        old_track = get_by_path(self.state, dmg_cfg.damage_track_path)
        if old_track != new_track:
            set_by_path(self.state, dmg_cfg.damage_track_path, new_track)
            return {"avant": old_track, "apres": new_track}
        return None

    def apply_damage(self, amount: int, pool_id: Optional[str] = None, ignore_armor: bool = False) -> ActionResult:
        """Applies damage to the character (supports pack pool cascade + armor or legacy hit_points)."""
        if type(amount) is bool or not isinstance(amount, int) or amount < 0:
            return ActionResult(success=False, message="damage amount must be a non-negative integer", blocked_reason="invalid_damage")

        if (
            self.pack
            and getattr(self.pack, "resources", None)
            and getattr(self.pack.resources, "damage", None)
            and getattr(self.pack.resources.damage, "cascade_pools", None)
        ):
            dmg_cfg = self.pack.resources.damage
            armor_val = 0
            if not ignore_armor and dmg_cfg.armor_path:
                raw_armor = get_by_path(self.state, dmg_cfg.armor_path)
                if isinstance(raw_armor, int) and not isinstance(raw_armor, bool) and raw_armor > 0:
                    armor_val = raw_armor

            effective = max(0, amount - armor_val)
            cascade_order = list(dmg_cfg.cascade_pools)
            if pool_id and pool_id in cascade_order:
                cascade_order.remove(pool_id)
                cascade_order.insert(0, pool_id)

            remaining = effective
            pools_affected = {}
            for pid in cascade_order:
                if remaining <= 0:
                    break
                pool_def = next((p for p in self.pack.resources.pools if p.id == pid), None)
                if not pool_def:
                    continue
                cur = get_by_path(self.state, pool_def.current_path)
                if not isinstance(cur, int) or isinstance(cur, bool):
                    continue
                avail = max(0, cur - pool_def.min_value)
                loss = min(avail, remaining)
                if loss > 0:
                    new_val = cur - loss
                    set_by_path(self.state, pool_def.current_path, new_val)
                    pools_affected[pid] = {"avant": cur, "apres": new_val, "loss": loss}
                    remaining -= loss

            default_track = dmg_cfg.damage_track_states[0] if dmg_cfg.damage_track_states else None
            old_track = (get_by_path(self.state, dmg_cfg.damage_track_path) if dmg_cfg.damage_track_path else None) or default_track
            new_track = self._compute_damage_track() or old_track
            if dmg_cfg.damage_track_path and new_track is not None:
                set_by_path(self.state, dmg_cfg.damage_track_path, new_track)
            self.save()

            dmg_summary = {
                "raw": amount,
                "armor": armor_val,
                "effective": effective,
                "unabsorbed": remaining,
                "pools": pools_affected,
                "damage_track": {"avant": old_track, "apres": new_track},
            }
            state_changes = {"damage": dmg_summary}
            for pid, pchange in pools_affected.items():
                state_changes[pid] = {"avant": pchange["avant"], "apres": pchange["apres"]}
            if old_track != new_track:
                state_changes[dmg_cfg.damage_track_path] = {"avant": old_track, "apres": new_track}

            pool_parts = [f"{pid} {c['avant']}->{c['apres']}" for pid, c in pools_affected.items()]
            pools_str = ", ".join(pool_parts) if pool_parts else "none"
            msg = f"Damage: {amount} raw - {armor_val} armor = {effective} effective ({pools_str}). Track: {old_track}->{new_track}."
            return ActionResult(success=True, message=msg, state_changes=state_changes)

        hp_pool = self._get_health_pool_def()
        hp_cur, hp_max = self.get_hp()
        min_hp = hp_pool.min_value if hp_pool else 0
        new_hp = max(min_hp, hp_cur - amount)
        pool_key = hp_pool.id if hp_pool else "hit_points"
        if hp_pool:
            set_by_path(self.state, hp_pool.current_path, new_hp)
        else:
            self.state.setdefault("resources", {}).setdefault("hit_points", {})["current"] = new_hp
        self.save()
        status = "inconscient" if new_hp == 0 else f"{new_hp}/{hp_max} HP remaining"
        return ActionResult(
            success=True,
            message=f"-{amount} HP. {status}",
            state_changes={pool_key: {"avant": hp_cur, "apres": new_hp}}
        )

    def apply_npc_damage(
        self,
        npc_id: str,
        damage: int,
        npc_level: int = 1,
        npc_health_max: Optional[int] = None,
        npc_armor: int = 0,
    ) -> dict:
        """
        Applies damage to a target NPC tracked in self.state['npc_health'][npc_id].
        Defaults NPC max health to 3 * npc_level when not previously tracked or specified.
        """
        npc_table = self.state.setdefault("npc_health", {})
        entry = npc_table.get(npc_id)
        if not isinstance(entry, dict):
            max_hp = npc_health_max if (isinstance(npc_health_max, int) and npc_health_max > 0) else max(1, npc_level * 3)
            armor_val = npc_armor if (isinstance(npc_armor, int) and npc_armor >= 0) else 0
            entry = {
                "current": max_hp,
                "max": max_hp,
                "level": npc_level,
                "armor": armor_val,
            }
            npc_table[npc_id] = entry
        else:
            if isinstance(npc_health_max, int) and npc_health_max > 0 and "max" not in entry:
                entry["max"] = npc_health_max
            if isinstance(npc_armor, int) and npc_armor > 0 and not entry.get("armor"):
                entry["armor"] = npc_armor

        armor_used = entry.get("armor", 0)
        if not isinstance(armor_used, int) or armor_used < 0:
            armor_used = 0
        effective = max(0, damage - armor_used)
        before = entry.get("current", entry.get("max", 0))
        after = max(0, before - effective)
        entry["current"] = after
        self.save()
        return {
            "npc_id": npc_id,
            "raw_damage": damage,
            "armor": armor_used,
            "effective_damage": effective,
            "avant": before,
            "apres": after,
            "max": entry.get("max", before),
            "defeated": after == 0,
        }

    def apply_healing(self, amount: int) -> ActionResult:
        """Applies healing to the character."""
        hp_pool = self._get_health_pool_def()
        hp_cur, hp_max = self.get_hp()
        new_hp = min(hp_max, hp_cur + amount) if hp_max > 0 else (hp_cur + amount)
        pool_key = hp_pool.id if hp_pool else "hit_points"
        if hp_pool:
            set_by_path(self.state, hp_pool.current_path, new_hp)
        else:
            self.state.setdefault("resources", {}).setdefault("hit_points", {})["current"] = new_hp
        self.save()
        return ActionResult(
            success=True,
            message=f"+{amount} HP. {new_hp}/{hp_max} HP.",
            state_changes={pool_key: {"avant": hp_cur, "apres": new_hp}}
        )

    def consume_pool(self, target_id: str, key: Optional[str] = None, amount: int = 1) -> ActionResult:
        """Consumes from a pool or pool_group configured in the system pack."""
        if not self.pack:
            return ActionResult(success=False, message="consume_pool requires a system pack.", blocked_reason="no_pack")

        pool = next((p for p in self.pack.resources.pools if p.id == target_id), None)
        group = next((g for g in self.pack.resources.pool_groups if g.id == target_id), None)

        if not pool and not group:
            return ActionResult(success=False, message=f"Target '{target_id}' not found in resources.", blocked_reason="unknown_target")

        if pool:
            current = get_by_path(self.state, pool.current_path)
            if current is None or current < amount:
                return ActionResult(success=False, message=f"Not enough {pool.name}.", blocked_reason="insufficient_resources")

            new_val = current - amount
            set_by_path(self.state, pool.current_path, new_val)
            self.save()
            return ActionResult(
                success=True,
                message=f"Consumed {amount} from {pool.name} ({new_val} remaining).",
                state_changes={pool.id: {"avant": current, "apres": new_val}}
            )

        if group:
            if not key:
                return ActionResult(success=False, message=f"Missing key for pool_group '{group.name}' (e.g. Please specify the level).", blocked_reason="missing_key")

            group_dict = get_by_path(self.state, group.path)
            if not group_dict or key not in group_dict:
                return ActionResult(success=False, message=f"Pool '{key}' not found in '{group.name}'.", blocked_reason="unknown_pool")

            current = group_dict[key].get("current")
            if current is None or current < amount:
                return ActionResult(success=False, message=f"Not enough in {group.name} [{key}].", blocked_reason="insufficient_resources")

            new_val = current - amount
            group_dict[key]["current"] = new_val
            # On resauvegarde le dictionnaire complet bien que ce soit par référence
            set_by_path(self.state, group.path, group_dict)
            self.save()
            return ActionResult(
                success=True,
                message=f"Consumed {amount} from {group.name} [{key}] ({new_val} remaining).",
                state_changes={group.id: {key: {"avant": current, "apres": new_val}}}
            )

    def add_xp(self, amount: int, auto_level_up: bool = False) -> ActionResult:
        """Adds XP and detects or applies level/rank progression configured in the pack."""
        if type(amount) is bool or not isinstance(amount, int) or amount < 0:
            return ActionResult(success=False, message="XP amount must be a non-negative integer.", blocked_reason="invalid_xp")

        prog = getattr(self.pack.resources, "progression", None) if (self.pack and getattr(self.pack, "resources", None)) else None
        xp_path = prog.xp_path if prog else "xp"
        level_path = prog.level_path if prog else "level"

        raw_xp = get_by_path(self.state, xp_path)
        current_xp = raw_xp if (isinstance(raw_xp, int) and not isinstance(raw_xp, bool)) else 0
        new_xp = current_xp + amount
        set_by_path(self.state, xp_path, new_xp)

        raw_lvl = get_by_path(self.state, level_path) if level_path else None
        current_lvl = raw_lvl if (isinstance(raw_lvl, int) and not isinstance(raw_lvl, bool)) else 1

        next_threshold = None
        if "next_level_xp" in self.state or "xp_prochain_niveau" in self.state:
            next_threshold = self.state.get("next_level_xp", self.state.get("xp_prochain_niveau"))
        elif prog:
            if prog.max_level is None or current_lvl < prog.max_level:
                if prog.thresholds and str(current_lvl + 1) in prog.thresholds:
                    next_threshold = prog.thresholds[str(current_lvl + 1)]
                elif prog.threshold_expr:
                    try:
                        next_threshold = evaluate_int(prog.threshold_expr, {"character": self.state}, self.rng)
                    except ExprError:
                        next_threshold = None

        level_up_ready = bool(isinstance(next_threshold, int) and new_xp >= next_threshold)
        advancement_ready = False
        if prog and prog.advancements:
            min_cost = min((a.cost for a in prog.advancements if a.cost > 0), default=0)
            if min_cost > 0 and new_xp >= min_cost:
                advancement_ready = True

        state_changes = {
            "xp": {"avant": current_xp, "apres": new_xp},
            "level_up": level_up_ready,
        }
        if advancement_ready:
            state_changes["advancement_available"] = True

        denom = next_threshold if next_threshold is not None else 999999
        msg = f"+{amount} XP ({new_xp}/{denom})"
        if level_up_ready:
            msg += " — LEVEL UP available !"
        elif advancement_ready:
            msg += " — Advancement available !"

        if auto_level_up and level_up_ready and prog:
            lvl_res = self.level_up()
            if lvl_res.success:
                state_changes["rank_up"] = lvl_res.state_changes.get("rank_up")
                msg += f" {lvl_res.message}"
                return ActionResult(success=True, message=msg, state_changes=state_changes)

        self.save()
        return ActionResult(success=True, message=msg, state_changes=state_changes)

    def spend_xp(self, amount: int = 0, reason: Optional[str] = None) -> ActionResult:
        """Spends XP directly (e.g. for an immediate reroll, refusing an intrusion, or narrative benefit)."""
        prog = getattr(self.pack.resources, "progression", None) if (self.pack and getattr(self.pack, "resources", None)) else None
        xp_path = prog.xp_path if prog else "xp"

        cost = amount
        if (cost is None or cost == 0) and reason and prog and prog.xp_spends and reason in prog.xp_spends:
            cost = prog.xp_spends[reason]

        if type(cost) is bool or not isinstance(cost, int) or cost <= 0:
            return ActionResult(success=False, message="XP spend amount must be a positive integer.", blocked_reason="invalid_xp_cost")

        raw_xp = get_by_path(self.state, xp_path)
        current_xp = raw_xp if (isinstance(raw_xp, int) and not isinstance(raw_xp, bool)) else 0
        if current_xp < cost:
            return ActionResult(
                success=False,
                message=f"Insufficient XP ({current_xp} available, {cost} required).",
                blocked_reason="insufficient_xp",
            )

        new_xp = current_xp - cost
        set_by_path(self.state, xp_path, new_xp)
        self.save()
        reason_str = f" ({reason})" if reason else ""
        return ActionResult(
            success=True,
            message=f"Spent {cost} XP{reason_str} ({new_xp} XP remaining).",
            state_changes={"xp": {"avant": current_xp, "apres": new_xp, "spent": cost, "reason": reason}},
        )

    def _apply_rank_up_effects(self, prog) -> dict:
        """Increments the rank/level/tier and applies configured on_rank_up formulas."""
        changes = {}
        if prog.level_path:
            raw_lvl = get_by_path(self.state, prog.level_path)
            old_lvl = raw_lvl if (isinstance(raw_lvl, int) and not isinstance(raw_lvl, bool)) else 1
            new_lvl = old_lvl + 1
            set_by_path(self.state, prog.level_path, new_lvl)
            changes[prog.level_path] = {"avant": old_lvl, "apres": new_lvl}

        for effect in getattr(prog, "on_rank_up", []):
            val = evaluate_int(effect.expr, {"character": self.state}, self.rng)
            old_target = get_by_path(self.state, effect.target_path)
            if effect.mode == "set_formula":
                new_target = val
                set_by_path(self.state, effect.target_path, new_target)
                changes[effect.target_path] = {"avant": old_target, "apres": new_target}
            elif effect.mode == "add_formula":
                base_val = old_target if (isinstance(old_target, int) and not isinstance(old_target, bool)) else 0
                new_target = base_val + val
                set_by_path(self.state, effect.target_path, new_target)
                changes[effect.target_path] = {"avant": base_val, "apres": new_target}
                if effect.sync_path:
                    old_sync = get_by_path(self.state, effect.sync_path)
                    base_sync = old_sync if (isinstance(old_sync, int) and not isinstance(old_sync, bool)) else 0
                    set_by_path(self.state, effect.sync_path, base_sync + val)
                    changes[effect.sync_path] = {"avant": base_sync, "apres": base_sync + val}
        return changes

    def level_up(self) -> ActionResult:
        """Advances the character to the next level/rank if thresholds (when configured) are met."""
        if not self.pack or not getattr(self.pack, "resources", None) or not getattr(self.pack.resources, "progression", None):
            return ActionResult(success=False, message="No progression configured in pack.", blocked_reason="no_progression")

        prog = self.pack.resources.progression
        if not prog.level_path:
            return ActionResult(success=False, message="This system does not use discrete levels/ranks.", blocked_reason="no_levels")

        raw_lvl = get_by_path(self.state, prog.level_path)
        current_lvl = raw_lvl if (isinstance(raw_lvl, int) and not isinstance(raw_lvl, bool)) else 1
        if prog.max_level is not None and current_lvl >= prog.max_level:
            return ActionResult(success=False, message=f"Maximum level/rank ({prog.max_level}) already reached.", blocked_reason="max_level_reached")

        raw_xp = get_by_path(self.state, prog.xp_path)
        current_xp = raw_xp if (isinstance(raw_xp, int) and not isinstance(raw_xp, bool)) else 0

        required_xp = None
        if prog.thresholds and str(current_lvl + 1) in prog.thresholds:
            required_xp = prog.thresholds[str(current_lvl + 1)]
        elif prog.threshold_expr:
            try:
                required_xp = evaluate_int(prog.threshold_expr, {"character": self.state}, self.rng)
            except ExprError as e:
                return ActionResult(success=False, message=f"Threshold formula error: {e}", blocked_reason="formula_error")

        if required_xp is not None and current_xp < required_xp:
            return ActionResult(
                success=False,
                message=f"Insufficient XP for level {current_lvl + 1} ({current_xp}/{required_xp}).",
                blocked_reason="insufficient_xp",
            )

        state_changes = {}
        if prog.reset_xp_on_level_up and required_xp is not None:
            new_xp = max(0, current_xp - required_xp)
            set_by_path(self.state, prog.xp_path, new_xp)
            state_changes[prog.xp_path] = {"avant": current_xp, "apres": new_xp}

        try:
            rank_changes = self._apply_rank_up_effects(prog)
        except ExprError as e:
            return ActionResult(success=False, message=f"Rank-up formula error: {e}", blocked_reason="formula_error")

        state_changes["rank_up"] = rank_changes
        self.synchronize_and_recalculate()
        self.save()
        new_lvl = get_by_path(self.state, prog.level_path)
        return ActionResult(
            success=True,
            message=f"Rank/Level increased: {current_lvl} -> {new_lvl}.",
            state_changes=state_changes,
        )

    def apply_advancement(
        self,
        advancement_id: str,
        target_path: Optional[str] = None,
        allocation: Optional[dict[str, int]] = None,
        value: object = None,
        *,
        allocations: Optional[dict[str, int]] = None,
        pool: Optional[str] = None,
        skill: Optional[str] = None,
        ability: Optional[str] = None,
    ) -> ActionResult:
        """
        Purchases and applies an advancement from the pack's progression catalog.
        Automatically advances rank/tier if advancements_per_rank is reached.
        """
        if allocation is None and allocations is not None:
            allocation = allocations
        if target_path is None and pool is not None:
            target_path = pool
        if value is None:
            value = skill if skill is not None else ability
        if not self.pack or not getattr(self.pack, "resources", None) or not getattr(self.pack.resources, "progression", None):
            return ActionResult(success=False, message="No progression configured in pack.", blocked_reason="no_progression")

        prog = self.pack.resources.progression
        adv = next((a for a in prog.advancements if a.id == advancement_id), None)
        if not adv:
            valid_ids = [a.id for a in prog.advancements]
            return ActionResult(
                success=False,
                message=f"Unknown advancement '{advancement_id}'. Valid options: {', '.join(valid_ids)}",
                blocked_reason="unknown_advancement",
            )

        taken_list = get_by_path(self.state, prog.advancements_path)
        if not isinstance(taken_list, list):
            taken_list = []
        if not adv.repeatable_per_rank and advancement_id in taken_list:
            return ActionResult(
                success=False,
                message=f"Advancement '{advancement_id}' has already been purchased in the current rank/tier.",
                blocked_reason="advancement_already_taken",
            )

        try:
            cost = evaluate_int(adv.cost_expr, {"character": self.state}, self.rng) if adv.cost_expr else adv.cost
        except ExprError as e:
            return ActionResult(success=False, message=f"Advancement cost formula error: {e}", blocked_reason="formula_error")

        raw_xp = get_by_path(self.state, prog.xp_path)
        current_xp = raw_xp if (isinstance(raw_xp, int) and not isinstance(raw_xp, bool)) else 0
        if current_xp < cost:
            return ActionResult(
                success=False,
                message=f"Insufficient XP for '{adv.name}' ({current_xp}/{cost}).",
                blocked_reason="insufficient_xp",
            )

        effect_changes = {}
        if adv.kind == "add_stat":
            resolved_path = target_path or adv.target_path
            if adv.allowed_paths:
                if resolved_path and resolved_path not in adv.allowed_paths:
                    # Allow short key (e.g. "might" -> "edge.might")
                    candidate = next((p for p in adv.allowed_paths if p.endswith(f".{resolved_path}")), None)
                    if candidate:
                        resolved_path = candidate
                if not resolved_path or resolved_path not in adv.allowed_paths:
                    return ActionResult(
                        success=False,
                        message=f"target_path for '{advancement_id}' must be one of {adv.allowed_paths}",
                        blocked_reason="invalid_advancement_target",
                    )
            if not resolved_path:
                return ActionResult(success=False, message=f"Missing target_path for '{advancement_id}'.", blocked_reason="invalid_advancement_target")

            cur_val = get_by_path(self.state, resolved_path)
            base_val = cur_val if (isinstance(cur_val, int) and not isinstance(cur_val, bool)) else 0
            new_val = base_val + adv.amount
            set_by_path(self.state, resolved_path, new_val)
            effect_changes[resolved_path] = {"avant": base_val, "apres": new_val}

            sync_p = adv.sync_paths.get(resolved_path)
            if sync_p:
                cur_sync = get_by_path(self.state, sync_p)
                base_sync = cur_sync if (isinstance(cur_sync, int) and not isinstance(cur_sync, bool)) else 0
                set_by_path(self.state, sync_p, base_sync + adv.amount)
                effect_changes[sync_p] = {"avant": base_sync, "apres": base_sync + adv.amount}

        elif adv.kind == "distribute_points":
            if not isinstance(allocation, dict) or not allocation:
                return ActionResult(success=False, message=f"allocation dict required for '{advancement_id}'.", blocked_reason="invalid_advancement_allocation")

            normalized_alloc = {}
            for k, v in allocation.items():
                if type(v) is bool or not isinstance(v, int) or v < 0:
                    return ActionResult(success=False, message=f"Allocation values must be non-negative integers.", blocked_reason="invalid_advancement_allocation")
                if v == 0:
                    continue
                resolved_k = k
                if adv.allowed_paths and resolved_k not in adv.allowed_paths:
                    candidate = next((p for p in adv.allowed_paths if f".{k}." in p or p.endswith(f".{k}")), None)
                    if candidate:
                        resolved_k = candidate
                if adv.allowed_paths and resolved_k not in adv.allowed_paths:
                    return ActionResult(
                        success=False,
                        message=f"Path '{k}' not allowed for '{advancement_id}'. Allowed: {adv.allowed_paths}",
                        blocked_reason="invalid_advancement_allocation",
                    )
                normalized_alloc[resolved_k] = normalized_alloc.get(resolved_k, 0) + v

            total_pts = sum(normalized_alloc.values())
            if total_pts != adv.amount:
                return ActionResult(
                    success=False,
                    message=f"Total allocated points ({total_pts}) must equal {adv.amount}.",
                    blocked_reason="invalid_advancement_allocation",
                )

            for path, pts in normalized_alloc.items():
                cur_val = get_by_path(self.state, path)
                base_val = cur_val if (isinstance(cur_val, int) and not isinstance(cur_val, bool)) else 0
                set_by_path(self.state, path, base_val + pts)
                effect_changes[path] = {"avant": base_val, "apres": base_val + pts}
                sync_p = adv.sync_paths.get(path)
                if sync_p:
                    cur_sync = get_by_path(self.state, sync_p)
                    base_sync = cur_sync if (isinstance(cur_sync, int) and not isinstance(cur_sync, bool)) else 0
                    set_by_path(self.state, sync_p, base_sync + pts)
                    effect_changes[sync_p] = {"avant": base_sync, "apres": base_sync + pts}

        elif adv.kind == "append_list":
            resolved_path = target_path or adv.target_path
            if not resolved_path or not value:
                return ActionResult(success=False, message=f"value is required for '{advancement_id}'.", blocked_reason="invalid_advancement_value")
            lst = get_by_path(self.state, resolved_path)
            if not isinstance(lst, list):
                lst = []
            lst = list(lst)
            lst.append(value)
            set_by_path(self.state, resolved_path, lst)
            effect_changes[resolved_path] = {"added": value}

        # Deduct XP and record advancement
        new_xp = current_xp - cost
        set_by_path(self.state, prog.xp_path, new_xp)
        new_taken = list(taken_list) + [advancement_id]

        state_changes = {
            "advancement": {
                "id": advancement_id,
                "name": adv.name,
                "cost": cost,
                "changes": effect_changes,
            },
            prog.xp_path: {"avant": current_xp, "apres": new_xp},
        }

        msg = f"Advancement '{adv.name}' purchased (-{cost} XP, {new_xp} remaining)."

        if prog.advancements_per_rank and len(new_taken) >= prog.advancements_per_rank:
            raw_lvl = get_by_path(self.state, prog.level_path) if prog.level_path else 1
            cur_lvl = raw_lvl if (isinstance(raw_lvl, int) and not isinstance(raw_lvl, bool)) else 1
            if prog.max_level is None or cur_lvl < prog.max_level:
                rank_changes = self._apply_rank_up_effects(prog)
                set_by_path(self.state, prog.advancements_path, [])
                state_changes["rank_up"] = rank_changes
                new_lvl = get_by_path(self.state, prog.level_path)
                msg += f" All {prog.advancements_per_rank} advancements completed: Rank/Tier increased {cur_lvl} -> {new_lvl}!"
            else:
                set_by_path(self.state, prog.advancements_path, new_taken)
        else:
            set_by_path(self.state, prog.advancements_path, new_taken)

        self.synchronize_and_recalculate()
        self.save()
        return ActionResult(success=True, message=msg, state_changes=state_changes)

    # ── Consumables Management ───────────────────────────────

    def _find_consumable_def(self, target_id: Optional[str] = None):
        if not self.pack or not getattr(self.pack, "resources", None) or not getattr(self.pack.resources, "consumables", None):
            return None
        consumables = self.pack.resources.consumables
        if not consumables:
            return None
        if target_id:
            return next((c for c in consumables if c.id == target_id or c.path == target_id), None)
        return consumables[0]

    def add_consumable(
        self,
        item: object,
        target_id: Optional[str] = None,
        *,
        list_id: Optional[str] = None,
    ) -> ActionResult:
        """Adds an item to a declared consumable list, enforcing max_items / max_items_path if configured."""
        eff_target_id = target_id or list_id
        cdef = self._find_consumable_def(eff_target_id)
        if not cdef:
            return ActionResult(success=False, message=f"Consumable list '{eff_target_id}' not configured in pack.", blocked_reason="unknown_consumable_list")

        items = get_by_path(self.state, cdef.path)
        if not isinstance(items, list):
            items = []
        else:
            items = list(items)

        max_limit = cdef.max_items
        if max_limit is None and cdef.max_items_path:
            raw_max = get_by_path(self.state, cdef.max_items_path)
            if isinstance(raw_max, int) and not isinstance(raw_max, bool) and raw_max > 0:
                max_limit = raw_max

        if max_limit is not None and len(items) >= max_limit:
            return ActionResult(
                success=False,
                message=f"Cannot add item to '{cdef.name}': limit of {max_limit} reached.",
                blocked_reason="consumable_limit_reached",
            )

        items.append(item)
        set_by_path(self.state, cdef.path, items)
        self.save()
        label = item.get("name", str(item)) if isinstance(item, dict) else str(item)
        return ActionResult(
            success=True,
            message=f"Added '{label}' to {cdef.name} ({len(items)}/{max_limit if max_limit is not None else '∞'}).",
            state_changes={"consumable_added": {"list": cdef.id, "item": item, "count": len(items)}},
        )

    def use_consumable(
        self,
        item_name: str,
        target_id: Optional[str] = None,
        *,
        list_id: Optional[str] = None,
        heal_pool: Optional[str] = None,
        heal_amount: Optional[int] = None,
        heal_expr: Optional[str] = None,
    ) -> ActionResult:
        """
        Consumes an item by name from a declared consumable list (decrementing quantity if stacked
        or removing it from the list) and optionally applies a healing/pool restoration effect.
        """
        eff_target_id = target_id or list_id
        if not self.pack or not getattr(self.pack, "resources", None) or not getattr(self.pack.resources, "consumables", None):
            return ActionResult(success=False, message="No consumables configured in pack.", blocked_reason="no_consumables")

        if not isinstance(item_name, str) or not item_name.strip():
            return ActionResult(success=False, message="item_name must be a non-empty string.", blocked_reason="invalid_item")

        query = item_name.strip().lower()
        candidates = [self._find_consumable_def(eff_target_id)] if eff_target_id else list(self.pack.resources.consumables)
        candidates = [c for c in candidates if c is not None]
        if not candidates and eff_target_id:
            candidates = list(self.pack.resources.consumables)

        matched_cdef = None
        matched_idx = None
        matched_list = None

        for cdef in candidates:
            items = get_by_path(self.state, cdef.path)
            if not isinstance(items, list):
                continue
            # 1. Exact case-insensitive match
            for idx, entry in enumerate(items):
                entry_str = entry.get("name", "") if isinstance(entry, dict) else str(entry)
                if entry_str.strip().lower() == query:
                    matched_cdef, matched_idx, matched_list = cdef, idx, list(items)
                    break
            if matched_cdef:
                break
            # 2. Substring case-insensitive match
            for idx, entry in enumerate(items):
                entry_str = entry.get("name", "") if isinstance(entry, dict) else str(entry)
                if query in entry_str.strip().lower():
                    matched_cdef, matched_idx, matched_list = cdef, idx, list(items)
                    break
            if matched_cdef:
                break

        if not matched_cdef or matched_idx is None:
            return ActionResult(
                success=False,
                message=f"Consumable '{item_name}' not found in character inventory.",
                blocked_reason="item_not_found",
            )

        entry = matched_list[matched_idx]
        consumed_label = entry.get("name", str(entry)) if isinstance(entry, dict) else str(entry)

        # Handle stacked items (dict quantity/charges or string "(xN)")
        if isinstance(entry, dict) and isinstance(entry.get("quantity"), int) and entry["quantity"] > 1:
            updated = dict(entry)
            updated["quantity"] = entry["quantity"] - 1
            matched_list[matched_idx] = updated
        elif isinstance(entry, str):
            m = re.search(r"\(\s*x(\d+)\s*\)", entry, re.IGNORECASE)
            if m and int(m.group(1)) > 1:
                new_qty = int(m.group(1)) - 1
                matched_list[matched_idx] = entry[:m.start()] + f"(x{new_qty})" + entry[m.end():]
            else:
                matched_list.pop(matched_idx)
        else:
            matched_list.pop(matched_idx)

        set_by_path(self.state, matched_cdef.path, matched_list)

        state_changes = {
            "consumable_used": {
                "list": matched_cdef.id,
                "item": consumed_label,
                "remaining_count": len(matched_list),
            }
        }
        msg = f"Used consumable '{consumed_label}' from {matched_cdef.name}."

        # Optional healing / pool restoration
        gain = None
        if isinstance(heal_amount, int) and not isinstance(heal_amount, bool) and heal_amount > 0:
            gain = heal_amount
        elif isinstance(heal_expr, str) and heal_expr.strip():
            try:
                gain = max(0, evaluate_int(heal_expr, {"character": self.state}, self.rng))
            except ExprError as e:
                return ActionResult(success=False, message=f"Consumable heal_expr error: {e}", blocked_reason="formula_error")

        if gain and gain > 0:
            target_pool = None
            if heal_pool:
                target_pool = next((p for p in self.pack.resources.pools if p.id == heal_pool), None)
            if not target_pool:
                target_pool = self._get_health_pool_def()
            if target_pool:
                cur_val = get_by_path(self.state, target_pool.current_path)
                max_val = target_pool.max_value if target_pool.max_value is not None else get_by_path(self.state, target_pool.max_path)
                base_cur = cur_val if (isinstance(cur_val, int) and not isinstance(cur_val, bool)) else 0
                new_cur = min(max_val, base_cur + gain) if isinstance(max_val, int) else (base_cur + gain)
                set_by_path(self.state, target_pool.current_path, new_cur)
                state_changes[target_pool.id] = {"avant": base_cur, "apres": new_cur}
                if self.pack.resources.damage and get_by_path(self.state, self.pack.resources.damage.damage_track_path) is not None:
                    track_change = self.update_damage_track()
                    if track_change:
                        state_changes["damage_track"] = track_change
                msg += f" Restored +{new_cur - base_cur} to {target_pool.name} ({base_cur}->{new_cur})."

        self.save()
        return ActionResult(success=True, message=msg, state_changes=state_changes)

    def next_rest(self, sequence_id: str) -> ActionResult:
        """
        Advances and applies the next rest in a defined recovery sequence.
        """
        if not self.pack or not getattr(self.pack.resources, "recovery_sequences", []):
            return ActionResult(success=False, message="No recovery sequences defined in this pack.")

        if "pending_allocation" in self.state:
            return ActionResult(success=False, message="Cannot execute sequence while there is a pending recovery allocation.")

        seq = next((s for s in self.pack.resources.recovery_sequences if s.id == sequence_id), None)
        if not seq:
            valid_seqs = [s.id for s in self.pack.resources.recovery_sequences]
            return ActionResult(success=False, message=f"Unknown recovery sequence '{sequence_id}'. Valid sequences are: {', '.join(valid_seqs)}.")

        # Read index
        current_index = get_by_path(self.state, seq.state_path)
        if current_index is None:
            current_index = 0
        elif not isinstance(current_index, int) or isinstance(current_index, bool) or current_index < 0:
            logger.warning(f"Invalid state for sequence index at '{seq.state_path}'. Defaulting to 0.")
            current_index = 0
        elif current_index >= len(seq.steps):
            logger.warning(f"Sequence index at '{seq.state_path}' out of bounds. Wrapping around.")
            current_index = current_index % len(seq.steps)

        trigger = seq.steps[current_index]
        next_index = (current_index + 1) % len(seq.steps)

        # We need to execute the rest without saving, append our sequence changes, then save
        # To do this safely without rewriting the whole rest(), we can temporarily patch save()
        # But a cleaner way is just to call a modified internal _apply_rest

        return self._inner_rest(trigger, sequence_info={"id": sequence_id, "step": trigger, "index": current_index, "next_index": next_index, "path": seq.state_path})

    def rest(self, trigger_id: str = "long") -> ActionResult:
        return self._inner_rest(trigger_id)

    def _inner_rest(self, trigger_id: str = "long", sequence_info: dict = None) -> ActionResult:
        """
        Applies a rest/recovery action.

        Formula Evaluation Context (ctx):
        When using the "formula" recovery mode via System Packs, the formula evaluates
        against a specific context containing:
          - character: The full character sheet state (self.state)
          - pool: Information about the pool being recovered, containing:
              - current: The current value of the pool.
              - max: The maximum value of the pool.
              - min: The minimum value of the pool (configuration 'min_value').
        """
        if self.pack:
            if "pending_allocation" in self.state:
                return ActionResult(success=False, message="Cannot rest while there is a pending recovery allocation.")

            # Validate trigger_id
            triggers = self.pack.resources.recovery_triggers
            trigger_found = any(t.id == trigger_id for t in triggers)

            if not trigger_found:
                valid_triggers = [t.id for t in triggers]
                return ActionResult(success=False, message=f"Unknown recovery trigger '{trigger_id}'. Valid triggers are: {', '.join(valid_triggers)}.")

            restored = []
            state_changes = {}

            def apply_rule(current_val, max_val, min_val, rule):
                if rule.mode == "full":
                    gain = max_val - current_val
                    return min(max_val, current_val + gain), min(max_val - current_val, gain)
                elif rule.mode == "fixed":
                    gain = rule.amount
                    return min(max_val, current_val + gain), min(max_val - current_val, gain)
                elif rule.mode == "percent":
                    gain = max(1, max_val * rule.amount // 100)
                    return min(max_val, current_val + gain), min(max_val - current_val, gain)
                elif rule.mode == "formula":
                    ctx = {
                        "character": self.state,
                        "pool": {"current": current_val, "max": max_val, "min": min_val}
                    }
                    gain = max(0, evaluate_int(rule.amount_expr, ctx, self.rng))
                    return min(max_val, current_val + gain), min(max_val - current_val, gain)
                return current_val, 0

            pending_changes = []

            # Gather pools
            for pool in self.pack.resources.pools:
                rule = next((r for r in pool.recovery if r.trigger == trigger_id), None)
                if rule:
                    current = get_by_path(self.state, pool.current_path)

                    max_val = None
                    if pool.max_value is not None:
                        max_val = pool.max_value
                    elif pool.max_path is not None:
                        max_val = get_by_path(self.state, pool.max_path)

                    if current is not None:
                        if max_val is None:
                            logger.warning(f"Unresolvable max value for pool '{pool.id}'. Ignored during rest.")
                            continue
                        try:
                            new_val, gain = apply_rule(current, max_val, pool.min_value, rule)
                            pending_changes.append({
                                "type": "pool",
                                "pool": pool,
                                "current": current,
                                "new_val": new_val,
                                "gain": gain
                            })
                        except ExprError as e:
                            return ActionResult(success=False, message=f"Recovery formula failed for '{pool.id}': {e}")

            # Gather pool_groups
            for group in self.pack.resources.pool_groups:
                rule = next((r for r in group.recovery if r.trigger == trigger_id), None)
                if rule:
                    group_dict = get_by_path(self.state, group.path)
                    if isinstance(group_dict, dict):
                        for key, sub_res in group_dict.items():
                            if isinstance(sub_res, dict) and "current" in sub_res and "max" in sub_res:
                                current = sub_res["current"]
                                max_val = sub_res["max"]
                                min_val = 0 # Sub-pools default to 0 min_value
                                try:
                                    new_val, gain = apply_rule(current, max_val, min_val, rule)
                                    pending_changes.append({
                                        "type": "group",
                                        "group": group,
                                        "key": key,
                                        "current": current,
                                        "new_val": new_val,
                                        "gain": gain,
                                        "dict_ref": sub_res
                                    })
                                except ExprError as e:
                                    return ActionResult(success=False, message=f"Recovery formula failed for '{group.id}': {e}")

            pooled_changes = []
            pending_alloc = None

            for pooled_rule in getattr(self.pack.resources, "pooled_recoveries", []):
                if trigger_id in pooled_rule.triggers:
                    try:
                        points = max(0, evaluate_int(pooled_rule.points_expr, {"character": self.state}, self.rng))

                        if pooled_rule.allocation == "auto_in_order":
                            projected_values = {}
                            max_values = {}

                            for pool_id in pooled_rule.among:
                                pool = next((p for p in self.pack.resources.pools if p.id == pool_id), None)
                                if pool:
                                    max_val = pool.max_value if pool.max_value is not None else get_by_path(self.state, pool.max_path)
                                    curr_val = get_by_path(self.state, pool.current_path)
                                    if max_val is not None and curr_val is not None:
                                        projected_values[pool.id] = curr_val
                                        max_values[pool.id] = max_val

                            for change in pending_changes:
                                if change["type"] == "pool":
                                    pid = change["pool"].id
                                    if pid in projected_values:
                                        projected_values[pid] = change["new_val"]
                                elif change["type"] == "group_pool":
                                    pid = change["pool"].id
                                    if pid in projected_values:
                                        projected_values[pid] = change["new_val"]

                            allocated = {}
                            unallocated = points

                            for pool_id in pooled_rule.among:
                                if pool_id not in projected_values:
                                    continue

                                missing = max_values[pool_id] - projected_values[pool_id]
                                if missing > 0 and unallocated > 0:
                                    gain = min(missing, unallocated)
                                    avant = projected_values[pool_id]
                                    projected_values[pool_id] += gain
                                    unallocated -= gain

                                    allocated[pool_id] = {
                                        "avant": avant,
                                        "apres": projected_values[pool_id],
                                        "gain": gain
                                    }

                                    pool = next(p for p in self.pack.resources.pools if p.id == pool_id)
                                    pooled_changes.append({
                                        "type": "pool",
                                        "pool": pool,
                                        "current": avant,
                                        "new_val": projected_values[pool_id],
                                        "gain": gain
                                    })

                            state_changes.setdefault("pooled", {})[pooled_rule.id] = {
                                "allocated": allocated,
                                "wasted": {
                                    "total": unallocated,
                                    "unallocated": unallocated,
                                    "capped": {}
                                }
                            }

                        elif pooled_rule.allocation == "player_choice":
                            pending_alloc = {
                                "rule_id": pooled_rule.id,
                                "trigger": trigger_id,
                                "points": points,
                                "among": pooled_rule.among
                            }
                    except ExprError as e:
                        return ActionResult(success=False, message=f"Recovery formula failed for pooled rule '{pooled_rule.id}': {e}")

            if pending_alloc:
                self.state["pending_allocation"] = pending_alloc
                state_changes["pending_allocation"] = pending_alloc

            # Apply changes
            for change in pending_changes + pooled_changes:
                if change["type"] == "pool":
                    pool = change["pool"]
                    if change["current"] != change["new_val"]:
                        set_by_path(self.state, pool.current_path, change["new_val"])
                        if pool.name not in restored:
                            restored.append(pool.name)
                        state_changes[pool.id] = {"avant": change["current"], "apres": change["new_val"], "gain": change["gain"]}

                        if pool.kind == "health" and "pv" in self.state:
                            self.state["pv"] = change["new_val"]
                elif change["type"] == "group":
                    group = change["group"]
                    if change["current"] != change["new_val"]:
                        change["dict_ref"]["current"] = change["new_val"]
                        if group.name not in restored:
                            restored.append(group.name)
                        if group.id not in state_changes:
                            state_changes[group.id] = {}
                        state_changes[group.id][change["key"]] = {"avant": change["current"], "apres": change["new_val"], "gain": change["gain"]}

            if sequence_info:
                set_by_path(self.state, sequence_info["path"], sequence_info["next_index"])
                state_changes["sequence"] = {
                    "id": sequence_info["id"],
                    "step": sequence_info["step"],
                    "next_index": sequence_info["next_index"]
                }

            dmg_cfg = getattr(self.pack.resources, "damage", None)
            if dmg_cfg and get_by_path(self.state, dmg_cfg.damage_track_path) is not None:
                track_change = self.update_damage_track()
                if track_change:
                    state_changes["damage_track"] = track_change

            self.synchronize_and_recalculate()
            self.save()

            final_changes = {"rest": trigger_id, "restored": {}}
            for key, val in state_changes.items():
                if key in ("pending_allocation", "pooled", "sequence", "restored", "damage_track"):
                    final_changes[key] = val
                else:
                    final_changes["restored"][key] = val

            msg = f"Rest '{trigger_id}' completed. Restored: {', '.join(restored) if restored else 'nothing'}."
            if sequence_info:
                seq = next((s for s in getattr(self.pack.resources, 'recovery_sequences', []) if s.id == sequence_info['id']), None)
                msg += f" (step {sequence_info['index'] + 1}/{len(seq.steps) if seq else '?'})"

            return ActionResult(
                success=True,
                message=msg,
                state_changes=final_changes
            )

        return ActionResult(success=False, message=f"Unknown recovery tier '{trigger_id}'.", blocked_reason="no_pack")

    def allocate_recovery(self, allocation: dict[str, int]) -> ActionResult:
        """
        Distributes a pending pooled recovery among chosen pools.
        """
        pending = self.state.get("pending_allocation")
        if not pending:
            return ActionResult(success=False, message="No pending allocation to resolve.")

        among = set(pending["among"])
        points = pending["points"]

        # Validation
        for k, v in allocation.items():
            if k not in among:
                return ActionResult(success=False, message=f"Key '{k}' is not in the allowed targets ({', '.join(among)}).")
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return ActionResult(success=False, message=f"Value for '{k}' must be a non-negative integer.")

        total_requested = sum(allocation.values())
        if total_requested > points:
            return ActionResult(success=False, message=f"Total allocated points ({total_requested}) exceed available points ({points}).")

        unallocated = points - total_requested

        # Calculate actual gains and wastes
        allocated = {}
        capped = {}

        for k, req in allocation.items():
            if req == 0:
                continue

            pool = next((p for p in self.pack.resources.pools if p.id == k), None)
            if not pool:
                continue

            curr = get_by_path(self.state, pool.current_path)
            max_val = pool.max_value if pool.max_value is not None else get_by_path(self.state, pool.max_path)

            if curr is None or max_val is None:
                continue

            missing = max(0, max_val - curr)
            gain = min(req, missing)
            cap = req - gain

            if gain > 0:
                allocated[k] = {
                    "avant": curr,
                    "apres": curr + gain,
                    "gain": gain
                }
                set_by_path(self.state, pool.current_path, curr + gain)

            if cap > 0:
                capped[k] = cap

        wasted_total = unallocated + sum(capped.values())
        wasted = {
            "total": wasted_total,
            "unallocated": unallocated,
            "capped": capped
        }

        state_changes = {
            "allocated": allocated,
            "wasted": wasted,
            "rule_id": pending["rule_id"]
        }

        del self.state["pending_allocation"]
        if self.pack and getattr(self.pack, "resources", None):
            dmg_cfg = getattr(self.pack.resources, "damage", None)
            if dmg_cfg and get_by_path(self.state, dmg_cfg.damage_track_path) is not None:
                track_change = self.update_damage_track()
                if track_change:
                    state_changes["damage_track"] = track_change
        self.synchronize_and_recalculate()
        self.save()

        return ActionResult(
            success=True,
            message="Recovery allocation applied successfully.",
            state_changes=state_changes
        )

    def _resolve_stat_modifier(self, stat: Optional[str], explicit_modifier: int = 0) -> int:
        if explicit_modifier != 0 or not stat or not isinstance(self.state, dict):
            return explicit_modifier
        mods = self.state.get("modifiers")
        if isinstance(mods, dict):
            for k, v in mods.items():
                if k.lower() == stat.lower() and isinstance(v, int) and not isinstance(v, bool):
                    return v
        stats = self.state.get("statistics")
        if isinstance(stats, dict):
            for k, v in stats.items():
                if k.lower() == stat.lower() and isinstance(v, int) and not isinstance(v, bool):
                    return v if -5 <= v <= 10 else (v - 10) // 2
        return 0

    def execute_task(
        self,
        action_id: str,
        pool_id: Optional[str] = None,
        difficulty: Optional[int] = None,
        step_adjustments: Optional[dict[str, int]] = None,
        effort_levels: int = 0,
        impaired: Optional[bool] = None,
        damage_effort_levels: int = 0,
        base_damage: Optional[int] = None,
        target_npc: Optional[str] = None,
        npc_health_max: Optional[int] = None,
        npc_armor: int = 0,
        incoming_damage: Optional[int] = None,
        damage_pool: Optional[str] = None,
        ignore_armor: bool = False,
        refund_on_20: Optional[bool] = None,
        modifier: int = 0,
        roll_bonus: int = 0,
        advantage: str = "none",
        pool_size: Optional[int] = None,
        stat: Optional[str] = None,
    ) -> ActionResult:
        """
        Executes a task action (e.g. attack, defend, climb, ability_check) dynamically from the system pack.
        Supports StepTargetD20, D20VsTarget, PbtA2d6, and DicePoolSuccess resolution families.
        """
        if not self.pack:
            return ActionResult(success=False, message="No system pack loaded.", blocked_reason="no_pack")
        if not getattr(self.pack, "actions", None) or not self.pack.actions.actions:
            return ActionResult(success=False, message="System pack has no actions configured.", blocked_reason="no_actions")

        family = self.pack.resolution.family
        if family not in ("StepTargetD20", "D20VsTarget", "PbtA2d6", "DicePoolSuccess"):
            return ActionResult(success=False, message=f"Unsupported resolution family '{family}'.", blocked_reason="unsupported_family")

        action = next((a for a in self.pack.actions.actions if a.id == action_id and a.kind == "task"), None)
        if not action:
            valid_actions = [a.id for a in self.pack.actions.actions if a.kind == "task"]
            return ActionResult(success=False, message=f"Unknown task action '{action_id}'. Valid actions: {', '.join(valid_actions)}", blocked_reason="unknown_action")

        # Damage track check (configured blocking_states block tasks; impaired_state auto-applies unless overridden)
        dmg_cfg = getattr(self.pack.resources, "damage", None) if getattr(self.pack, "resources", None) else None
        current_track = None
        if dmg_cfg and dmg_cfg.damage_track_path:
            current_track = get_by_path(self.state, dmg_cfg.damage_track_path)
            if current_track and current_track in dmg_cfg.blocking_states:
                return ActionResult(
                    success=False,
                    message=f"Character is {current_track} and cannot perform tasks.",
                    blocked_reason=current_track,
                )

        is_impaired = (
            bool(impaired)
            if impaired is not None
            else bool(dmg_cfg and dmg_cfg.impaired_state and current_track == dmg_cfg.impaired_state)
        )

        # Validation of pool_id
        if action.pools:
            if pool_id is None:
                return ActionResult(success=False, message=f"pool_id is required for action '{action_id}'", blocked_reason="invalid_pool")
            if pool_id not in action.pools:
                return ActionResult(success=False, message=f"pool '{pool_id}' is not allowed for action '{action_id}'", blocked_reason="invalid_pool")
        else:
            if pool_id is not None:
                return ActionResult(success=False, message=f"action '{action_id}' does not use a pool", blocked_reason="invalid_pool")
            if effort_levels > 0 or damage_effort_levels > 0:
                return ActionResult(success=False, message="effort requires a pool", blocked_reason="invalid_pool")

        pool_def = None
        if pool_id:
            pool_def = next((p for p in self.pack.resources.pools if p.id == pool_id), None)
            if not pool_def:
                return ActionResult(success=False, message=f"Pool '{pool_id}' not found in resources.", blocked_reason="invalid_pool")

        # Family-specific difficulty & parameter validation
        if family == "StepTargetD20":
            if type(difficulty) is bool or not isinstance(difficulty, int):
                return ActionResult(success=False, message="difficulty must be an integer", blocked_reason="invalid_difficulty")
            if not (self.pack.resolution.min_difficulty <= difficulty <= self.pack.resolution.max_difficulty):
                return ActionResult(success=False, message=f"difficulty {difficulty} out of bounds [{self.pack.resolution.min_difficulty}, {self.pack.resolution.max_difficulty}]", blocked_reason="invalid_difficulty")
        elif family == "D20VsTarget":
            if type(difficulty) is bool or not isinstance(difficulty, int) or difficulty < 0:
                return ActionResult(success=False, message="difficulty (DC/AC) must be a non-negative integer", blocked_reason="invalid_difficulty")
            if advantage not in ("none", "advantage", "disadvantage"):
                return ActionResult(success=False, message=f"invalid advantage mode '{advantage}'", blocked_reason="invalid_advantage")
        elif family == "PbtA2d6":
            if difficulty is None:
                difficulty = 0
            elif type(difficulty) is bool or not isinstance(difficulty, int):
                return ActionResult(success=False, message="difficulty must be an integer", blocked_reason="invalid_difficulty")
        elif family == "DicePoolSuccess":
            if difficulty is None:
                difficulty = 1
            elif type(difficulty) is bool or not isinstance(difficulty, int) or difficulty < 1:
                return ActionResult(success=False, message="difficulty (required successes) must be a positive integer", blocked_reason="invalid_difficulty")
            eff_pool_size = pool_size if pool_size is not None else getattr(self.pack.resolution, "default_pool_size", None)
            if type(eff_pool_size) is bool or not isinstance(eff_pool_size, int) or eff_pool_size < 1:
                return ActionResult(success=False, message="pool_size must be a positive integer", blocked_reason="invalid_pool_size")

        # Validation of effort
        if type(effort_levels) is bool or not isinstance(effort_levels, int) or effort_levels < 0:
            return ActionResult(success=False, message="effort_levels must be a non-negative integer", blocked_reason="invalid_effort")
        if type(damage_effort_levels) is bool or not isinstance(damage_effort_levels, int) or damage_effort_levels < 0:
            return ActionResult(success=False, message="damage_effort_levels must be a non-negative integer", blocked_reason="invalid_effort")

        total_effort_levels = effort_levels + damage_effort_levels
        adj = dict(step_adjustments or {})
        effort_cost = 0
        edge = 0

        if action.effort:
            if action.effort.adjustment_key in adj and effort_levels > 0:
                return ActionResult(success=False, message=f"effort_levels conflict: '{action.effort.adjustment_key}' already in step_adjustments", blocked_reason="effort_conflict")

            max_levels = None
            if action.effort.max_levels_path:
                max_levels = get_by_path(self.state, action.effort.max_levels_path)

            if max_levels is not None and total_effort_levels > max_levels:
                return ActionResult(success=False, message=f"effort_levels ({total_effort_levels}) exceeds max_levels ({max_levels})", blocked_reason="invalid_effort")

            if action.effort.edge_path_template and pool_id:
                edge_path = action.effort.edge_path_template.format(pool=pool_id)
                e_val = get_by_path(self.state, edge_path)
                if isinstance(e_val, int) and type(e_val) is not bool and e_val >= 0:
                    edge = e_val

            ctx = {
                "character": self.state,
                "effort": {"levels": total_effort_levels, "edge": edge, "impaired": is_impaired}
            }
            if total_effort_levels > 0:
                try:
                    raw_cost = evaluate_int(action.effort.cost_expr, ctx, self.rng)
                    effort_cost = max(0, raw_cost)
                except ExprError as e:
                    return ActionResult(success=False, message=f"formula_error: {e}", blocked_reason="formula_error")

            if effort_levels > 0:
                adj[action.effort.adjustment_key] = -effort_levels

        current_val = 0
        if pool_def:
            val = get_by_path(self.state, pool_def.current_path)
            if isinstance(val, int) and type(val) is not bool:
                current_val = val

            if current_val - effort_cost < pool_def.min_value:
                return ActionResult(success=False, message=f"Insufficient resources in '{pool_id}' to pay effort cost of {effort_cost}", blocked_reason="insufficient_resources")

        eff_modifier = self._resolve_stat_modifier(stat, modifier)

        if family == "StepTargetD20":
            req = ResolutionRequest(
                difficulty=difficulty,
                step_adjustments=adj
            )
        elif family == "D20VsTarget":
            req = ResolutionRequest(
                difficulty=difficulty,
                modifier=eff_modifier,
                roll_bonus=roll_bonus,
                advantage=advantage,
            )
        elif family == "PbtA2d6":
            req = ResolutionRequest(
                difficulty=difficulty or 0,
                modifier=eff_modifier,
                roll_bonus=roll_bonus,
            )
        else:  # DicePoolSuccess
            req = ResolutionRequest(
                difficulty=difficulty,
                pool_size=pool_size if pool_size is not None else getattr(self.pack.resolution, "default_pool_size", None),
            )

        result = resolve(self.pack.resolution, req, self.rng)
        if result.rolled:
            if family == "D20VsTarget" and advantage == "advantage":
                nat_roll = max(result.rolled)
            elif family == "D20VsTarget" and advantage == "disadvantage":
                nat_roll = min(result.rolled)
            else:
                nat_roll = result.rolled[0]
        else:
            nat_roll = None

        # Special roll detection (configured in action.special_rolls; suppressed when impaired if in impaired_suppressed_special_rolls)
        special_roll = None
        if nat_roll is not None and getattr(action, "special_rolls", None):
            suppressed = getattr(action, "impaired_suppressed_special_rolls", []) or []
            if not (is_impaired and str(nat_roll) in suppressed):
                special_roll = action.special_rolls.get(str(nat_roll))

        # Build output message and task_dict
        pool_name = pool_def.name if pool_def else (stat if stat else "no pool")
        initial_name = next((s.name for s in self.pack.actions.difficulty_scale if s.level == difficulty), None) if self.pack.actions.difficulty_scale else None
        final_name = next((s.name for s in self.pack.actions.difficulty_scale if s.level == result.difficulty_final), None) if (self.pack.actions.difficulty_scale and result.difficulty_final is not None) else initial_name
        eff_target = result.target if result.target is not None else difficulty

        if family == "StepTargetD20":
            diff_str = f"difficulty {difficulty}"
            if initial_name:
                diff_str += f" ({initial_name})"
            if result.difficulty_initial != result.difficulty_final:
                diff_str += f" reduced to {result.difficulty_final}"
                if final_name:
                    diff_str += f" ({final_name})"

            if result.impossible:
                phrase = f"Task '{action_id}' with {pool_name}: {diff_str}, target {eff_target} is unreachable: no die rolled, FAILURE."
            elif result.difficulty_final == self.pack.resolution.auto_success_at:
                phrase = f"Task '{action_id}' with {pool_name}: {diff_str}: automatic success, no die rolled."
            else:
                margin_str = f"+{result.margin}" if result.margin is not None and result.margin >= 0 else str(result.margin)
                rolled_str = str(result.rolled[0]) if result.rolled else "none"
                phrase = f"Task '{action_id}' with {pool_name}: {diff_str}, target {eff_target}, rolled {rolled_str}, {result.outcome.name} (margin {margin_str})."
        elif family == "D20VsTarget":
            margin_str = f"+{result.margin}" if result.margin is not None and result.margin >= 0 else str(result.margin)
            total_mod = eff_modifier + roll_bonus
            mod_str = f"+{total_mod}" if total_mod >= 0 else str(total_mod)
            adv_str = f" ({advantage})" if advantage != "none" else ""
            phrase = f"Task '{action_id}' ({pool_name}): DC {eff_target}, rolled {result.rolled}{adv_str} {mod_str} = {result.total}, {result.outcome.name} (margin {margin_str})."
        elif family == "PbtA2d6":
            total_mod = eff_modifier + roll_bonus
            mod_str = f"+{total_mod}" if total_mod >= 0 else str(total_mod)
            tier_str = f" [{result.tier_label}]" if result.tier_label else ""
            phrase = f"Task '{action_id}' ({pool_name}): rolled {result.rolled} {mod_str} = {result.total}, {result.outcome.name}{tier_str}."
        else:  # DicePoolSuccess
            margin_str = f"+{result.margin}" if result.margin is not None and result.margin >= 0 else str(result.margin)
            phrase = f"Task '{action_id}': rolled {result.rolled} -> {result.total} successes vs {eff_target}, {result.outcome.name} (margin {margin_str})."

        effort_waived = False
        effort_refunded = False
        should_refund_20 = refund_on_20 if refund_on_20 is not None else bool(action.effort and action.effort.refund_on_20)

        if total_effort_levels > 0 and result.impossible:
            effort_waived = True
            effort_cost = 0
        elif total_effort_levels > 0 and nat_roll == 20 and should_refund_20:
            effort_refunded = True
            effort_cost = 0

        if total_effort_levels > 0 and pool_def:
            if effort_waived:
                phrase += " Effort not spent: the task is impossible."
            elif effort_refunded:
                phrase += f" Effort: {total_effort_levels} level(s), cost 0 (refunded on natural 20, {current_val} remaining)."
            else:
                rem = current_val - effort_cost
                phrase += f" Effort: {total_effort_levels} level(s), cost {effort_cost} ({rem} remaining)."

        if special_roll:
            phrase += f" Special roll ({nat_roll}): {special_roll}."

        if result.warnings:
            phrase += f" Warnings: {', '.join(result.warnings)}."

        task_dict = {
            "action": action_id,
            "pool": pool_id,
            "difficulty": {
                "initial": result.difficulty_initial if result.difficulty_initial is not None else difficulty,
                "final": result.difficulty_final if result.difficulty_final is not None else difficulty,
                "name": final_name
            },
            "target": eff_target,
            "rolled": result.rolled,
            "total": result.total,
            "outcome": result.outcome.name,
            "margin": result.margin,
            "impossible": result.impossible,
            "warnings": list(result.warnings)
        }
        if family in ("D20VsTarget", "PbtA2d6"):
            task_dict["modifier"] = eff_modifier + roll_bonus
            if stat:
                task_dict["stat"] = stat
            if family == "D20VsTarget" and advantage != "none":
                task_dict["advantage"] = advantage
            if family == "PbtA2d6":
                task_dict["tier_id"] = result.tier_id
                task_dict["tier_label"] = result.tier_label
        if special_roll:
            task_dict["special_roll"] = special_roll

        state_changes = {"task": task_dict}

        if total_effort_levels > 0 and pool_def:
            effort_dict = {
                "levels": total_effort_levels,
                "edge": edge,
                "cost": effort_cost,
                "waived": effort_waived,
            }
            if damage_effort_levels > 0:
                effort_dict["ease_levels"] = effort_levels
                effort_dict["damage_levels"] = damage_effort_levels
            if effort_refunded:
                effort_dict["refunded"] = True
            state_changes["effort"] = effort_dict

        state_dirty = False
        # Apply effort deduction first
        if effort_cost > 0 and pool_def:
            new_val = current_val - effort_cost
            set_by_path(self.state, pool_def.current_path, new_val)
            if pool_def.kind == "health" and "pv" in self.state:
                self.state["pv"] = new_val

            state_changes[pool_id] = {
                "avant": current_val,
                "apres": new_val
            }
            if dmg_cfg and get_by_path(self.state, dmg_cfg.damage_track_path) is not None:
                track_change = self.update_damage_track()
                if track_change:
                    state_changes[dmg_cfg.damage_track_path] = track_change
            state_dirty = True

        # Combat consequences: Attack
        action_role = getattr(action, "role", "general")
        is_attack_hit = (
            result.outcome in (Outcome.SUCCESS, Outcome.CRITICAL_SUCCESS, Outcome.PARTIAL_SUCCESS)
            and not result.impossible
        )
        is_defend_fail = (
            result.outcome in (Outcome.FAILURE, Outcome.CRITICAL_FAILURE)
            or result.impossible
        )

        if action_role == "attack" and is_attack_hit:
            base_dmg = base_damage if (isinstance(base_damage, int) and not isinstance(base_damage, bool) and base_damage >= 0) else getattr(action, "default_damage", 0)
            dmg_per_lvl = action.effort.damage_per_level if action.effort else 0
            effort_dmg = damage_effort_levels * dmg_per_lvl
            roll_bonuses = getattr(action, "roll_damage_bonuses", {}) or {}
            dmg_roll_bonus = roll_bonuses.get(str(nat_roll), 0) if nat_roll is not None else 0
            max_imp_bonus = getattr(action, "impaired_max_roll_bonus", None)
            if is_impaired and max_imp_bonus is not None and dmg_roll_bonus > max_imp_bonus:
                dmg_roll_bonus = max_imp_bonus
            total_dmg = base_dmg + effort_dmg + dmg_roll_bonus
            if total_dmg > 0 or base_damage is not None or getattr(action, "default_damage", 0) > 0:
                state_changes["attack_damage"] = {
                    "base": base_dmg,
                    "effort_bonus": effort_dmg,
                    "roll_bonus": dmg_roll_bonus,
                    "total": total_dmg,
                }
                phrase += f" Attack damage: {total_dmg} (base {base_dmg}, effort +{effort_dmg}, roll +{dmg_roll_bonus})."
                if target_npc:
                    npc_lvl = difficulty if (isinstance(difficulty, int) and difficulty > 0) else 1
                    npc_res = self.apply_npc_damage(
                        target_npc,
                        total_dmg,
                        npc_level=npc_lvl,
                        npc_health_max=npc_health_max,
                        npc_armor=npc_armor,
                    )
                    state_changes["npc_damage"] = npc_res
                    phrase += f" NPC '{target_npc}' health: {npc_res['avant']}->{npc_res['apres']}"
                    if npc_res["defeated"]:
                        phrase += " (DEFEATED)."
                    else:
                        phrase += "."
                    state_dirty = False  # apply_npc_damage already saved

        # Combat consequences: Defend
        elif action_role == "defend" and is_defend_fail:
            inc_dmg = incoming_damage if (isinstance(incoming_damage, int) and not isinstance(incoming_damage, bool) and incoming_damage >= 0) else (difficulty if family == "StepTargetD20" and isinstance(difficulty, int) else 0)
            if inc_dmg > 0:
                default_pool = dmg_cfg.cascade_pools[0] if (dmg_cfg and dmg_cfg.cascade_pools) else None
                dmg_res = self.apply_damage(inc_dmg, pool_id=damage_pool or default_pool, ignore_armor=ignore_armor)
                if dmg_res.success:
                    if "damage" in dmg_res.state_changes:
                        dmg_info = dmg_res.state_changes["damage"]
                        state_changes["damage_taken"] = dmg_info
                        for pid, pchange in dmg_info.get("pools", {}).items():
                            if pid in state_changes and "avant" in state_changes[pid]:
                                state_changes[pid]["apres"] = pchange["apres"]
                            else:
                                state_changes[pid] = {"avant": pchange["avant"], "apres": pchange["apres"]}
                        if dmg_cfg and dmg_cfg.damage_track_path in dmg_res.state_changes:
                            state_changes[dmg_cfg.damage_track_path] = dmg_res.state_changes[dmg_cfg.damage_track_path]
                    else:
                        state_changes["damage_taken"] = {"raw": inc_dmg, "armor": 0, "effective": inc_dmg}
                        for k, v in dmg_res.state_changes.items():
                            state_changes[k] = v
                    phrase += f" Defense failed: {dmg_res.message}"
                    state_dirty = False  # apply_damage already saved

        if state_dirty:
            self.save()

        return ActionResult(
            success=True,
            message=phrase,
            state_changes=state_changes
        )

    def synchronize_and_recalculate(self):
        """
        Ensures sheet consistency by clamping all pools and pool_groups declared
        in the active SystemPack to their configured bounds [min_value, max] and
        normalizing progression and damage track fields.
        """
        if not self.state:
            return

        if self.pack is not None and getattr(self.pack, "resources", None):
            res_cfg = self.pack.resources
            for pool in getattr(res_cfg, "pools", []):
                cur = get_by_path(self.state, pool.current_path)
                max_val = pool.max_value if pool.max_value is not None else (
                    get_by_path(self.state, pool.max_path) if pool.max_path else None
                )
                if isinstance(cur, int) and not isinstance(cur, bool):
                    clamped = max(pool.min_value, cur)
                    if isinstance(max_val, int) and not isinstance(max_val, bool) and max_val > 0 and clamped > max_val:
                        clamped = max_val
                    if clamped != cur:
                        set_by_path(self.state, pool.current_path, clamped)
                elif cur is None and isinstance(max_val, int) and not isinstance(max_val, bool) and max_val > 0:
                    set_by_path(self.state, pool.current_path, max_val)

            for group in getattr(res_cfg, "pool_groups", []):
                group_dict = get_by_path(self.state, group.path)
                if isinstance(group_dict, dict):
                    for sub_obj in group_dict.values():
                        if isinstance(sub_obj, dict):
                            s_cur = sub_obj.get("current")
                            s_max = sub_obj.get("max")
                            if isinstance(s_cur, int) and not isinstance(s_cur, bool):
                                clamped = max(0, s_cur)
                                if isinstance(s_max, int) and not isinstance(s_max, bool) and s_max > 0 and clamped > s_max:
                                    clamped = s_max
                                sub_obj["current"] = clamped
                            elif s_cur is None and isinstance(s_max, int) and not isinstance(s_max, bool) and s_max > 0:
                                sub_obj["current"] = s_max

            prog = getattr(res_cfg, "progression", None)
            if prog:
                for p_path in (prog.level_path, prog.xp_path):
                    if p_path:
                        raw_v = get_by_path(self.state, p_path)
                        if raw_v is not None and not isinstance(raw_v, bool):
                            try:
                                set_by_path(self.state, p_path, int(raw_v))
                            except (ValueError, TypeError):
                                pass

            if getattr(res_cfg, "damage", None) and getattr(res_cfg.damage, "damage_track_path", None):
                self.update_damage_track()
        else:
            for k in ("level", "xp"):
                if k in self.state and not isinstance(self.state[k], bool):
                    try:
                        self.state[k] = int(self.state[k])
                    except (ValueError, TypeError):
                        pass

        self.save()

    # ── Info Extraction ──────────────────────────────────────

    def describe_difficulty_scale(self) -> str:
        """
        Returns a formatted string describing the difficulty scale or tiers for the current pack.
        Returns an empty string if no pack or no difficulty information is configured.
        """
        if not self.pack or not getattr(self.pack, "resolution", None):
            return ""

        family = self.pack.resolution.family
        scale = getattr(self.pack.actions, "difficulty_scale", None) if getattr(self.pack, "actions", None) else None

        if family == "StepTargetD20":
            if not scale:
                return ""
            multiplier = getattr(self.pack.resolution, "multiplier", 1)
            sorted_scale = sorted(scale, key=lambda x: x.level)
            lines = []
            for level_info in sorted_scale:
                target = level_info.level * multiplier
                line = f"Niveau {level_info.level} ({level_info.name}, cible {target})"
                if level_info.guidance:
                    line += f" : {level_info.guidance}"
                lines.append(line)
            return "\n".join(lines)

        if family == "D20VsTarget":
            if not scale:
                return ""
            sorted_scale = sorted(scale, key=lambda x: x.level)
            lines = []
            for level_info in sorted_scale:
                line = f"DD {level_info.level} ({level_info.name})"
                if level_info.guidance:
                    line += f" : {level_info.guidance}"
                lines.append(line)
            return "\n".join(lines)

        if family == "DicePoolSuccess":
            if not scale:
                return ""
            sorted_scale = sorted(scale, key=lambda x: x.level)
            lines = []
            for level_info in sorted_scale:
                line = f"{level_info.level} succès ({level_info.name})"
                if level_info.guidance:
                    line += f" : {level_info.guidance}"
                lines.append(line)
            return "\n".join(lines)

        if family == "PbtA2d6":
            tiers = getattr(self.pack.resolution, "tiers", [])
            if not tiers:
                return ""
            lines = []
            for t in tiers:
                lines.append(f"{t.id} ({t.range[0]}-{t.range[1]}) : {t.label}")
            return "\n".join(lines)

        return ""

    # ── Automatic Detection from Text ─────────────────

    def detect_action(self, user_input: str) -> Optional[DetectedAction]:
        """
        Detects an action based on triggers and returns a DetectedAction dataclass
        containing the rule id, extracted keys, and target.
        """
        def normalize(t):
            t = t.lower()
            return ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')

        text = normalize(user_input)

        if not self.pack or not self.pack.triggers:
            return None

        lang = self.pack.manifest.language
        for rule in self.pack.triggers.rules:
            keywords = rule.keywords.get(lang, [])
            if not keywords:
                continue

            for kw in keywords:
                norm_kw = normalize(kw)
                if re.search(r'\b' + re.escape(norm_kw) + r'\b', text):
                    extracted_key = None
                    if rule.kind == "consume" and rule.key_regex and lang in rule.key_regex:
                        match = re.search(rule.key_regex[lang], text, re.IGNORECASE)
                        if match and match.groups():
                            extracted_key = match.group(1)

                    final_key = extracted_key or rule.key_default
                    if final_key is not None and rule.key_template is not None:
                        final_key = rule.key_template.format(key=final_key)

                    return DetectedAction(
                        rule_id=rule.id,
                        kind=rule.kind,
                        target=getattr(rule, "target", None),
                        key=final_key,
                        amount=getattr(rule, "amount", 1),
                        trigger=getattr(rule, "trigger", None),
                        sequence=getattr(rule, "sequence", None)
                    )
        return None

    def detect_action_type(self, user_input: str) -> Optional[str]:
        """
        Detects action type from player's text input using the loaded SystemPack.
        Returns action key or None if no mechanical action is detected.
        """
        if self.pack:
            action = self.detect_action(user_input)
            if action:
                if action.kind == "recover":
                    if action.sequence:
                        return f"rest_seq:{action.sequence}"
                    return f"rest:{action.trigger}"
                return action.rule_id
        return None
