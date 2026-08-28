import json
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ActionResult:
    success: bool
    message: str                    # technical message for the Orchestrator
    state_changes: dict = field(default_factory=dict)  # what has changed
    blocked_reason: Optional[str] = None  # if success=False


class GameStateEngine:
    """
    Central Game State Engine — Pure Python, zero LLM.
    Validates and applies all mechanical game state changes.
    """

    RESOURCE_KEYWORDS = {
        "sort": "spells_per_day",
        "magie": "spells_per_day",
        "spell": "spells_per_day",
        "rage": "points_de_rage",
        "inspiration": "inspiration",
    }

    def __init__(self, character_path: str = "Memory/character.json"):
        self.character_path = character_path
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

    def get_hp(self) -> tuple[int, int]:
        """Returns (current_hp, max_hp)."""
        resources = self.state.get("resources", {})
        pv = resources.get("hit_points", {})
        return pv.get("current", 0), pv.get("max", 0)

    def get_resource(self, resource_key: str, level: Optional[int] = None) -> tuple[int, int]:
        """
        Returns (current, max) for a resource.
        If level is provided, searches in resource_key.level_N (for spells).
        """
        resources = self.state.get("resources", {})
        resource = resources.get(resource_key, {})
        if level is not None:
            slot = resource.get(f"level_{level}", {})
            return slot.get("current", 0), slot.get("max", 0)
        return resource.get("current", 0), resource.get("max", 0)

    def get_state_summary(self) -> str:
        """Readable summary of the state for the Orchestrator."""
        if not self.state:
            return "No state available."

        hp_cur, hp_max = self.get_hp()
        lines = [
            f"HP: {hp_cur}/{hp_max}",
            f"Level: {self.state.get('level', '?')}",
            f"XP: {self.state.get('xp', 0)}/{self.state.get('next_level_xp', self.state.get('xp_prochain_niveau', '?'))}",
        ]

        resources = self.state.get("resources", {})
        spells = resources.get("spells_per_day", {})
        if isinstance(spells, dict):
            for slot, values in spells.items():
                if isinstance(values, dict):
                    lines.append(f"Spells {slot}: {values.get('current', 0)}/{values.get('max', 0)}")

        return " | ".join(lines)

    # ── Validation ────────────────────────────────────────────

    def can_use_spell(self, spell_level: int) -> ActionResult:
        """Verifies if the character can cast a spell of the given level."""
        current, max_val = self.get_resource("spells_per_day", level=spell_level)
        if max_val == 0:
            return ActionResult(
                success=False,
                message=f"The character has no spell slots of level {spell_level}.",
                blocked_reason="no_spell_slot"
            )
        if current <= 0:
            return ActionResult(
                success=False,
                message=f"No spell slots of level {spell_level} remaining. Rest required.",
                blocked_reason="no_spell_slot_remaining"
            )
        return ActionResult(success=True, message=f"Spell slot level {spell_level} available ({current}/{max_val}).")

    def can_use_resource(self, resource_key: str) -> ActionResult:
        """Verifies if a generic resource is available."""
        current, max_val = self.get_resource(resource_key)
        if max_val == 0:
            return ActionResult(
                success=False,
                message=f"Resource '{resource_key}' unavailable for this character.",
                blocked_reason="resource_unavailable"
            )
        if current <= 0:
            return ActionResult(
                success=False,
                message=f"Resource '{resource_key}' depleted. Rest required.",
                blocked_reason="resource_depleted"
            )
        return ActionResult(success=True, message=f"Resource '{resource_key}' available ({current}/{max_val}).")

    # ── Modifications Application ────────────────────────

    def apply_damage(self, amount: int) -> ActionResult:
        """Applies damage to the character."""
        hp_cur, hp_max = self.get_hp()
        new_hp = max(0, hp_cur - amount)
        self.state.setdefault("resources", {}).setdefault("hit_points", {})["current"] = new_hp
        if "pv" in self.state:
            self.state["pv"] = new_hp
        self.save()
        status = "inconscient" if new_hp == 0 else f"{new_hp}/{hp_max} HP remaining"
        return ActionResult(
            success=True,
            message=f"-{amount} HP. {status}",
            state_changes={"hit_points": {"avant": hp_cur, "apres": new_hp}}
        )

    def apply_healing(self, amount: int) -> ActionResult:
        """Applies healing to the character."""
        hp_cur, hp_max = self.get_hp()
        new_hp = min(hp_max, hp_cur + amount)
        self.state.setdefault("resources", {}).setdefault("hit_points", {})["current"] = new_hp
        if "pv" in self.state:
            self.state["pv"] = new_hp
        self.save()
        return ActionResult(
            success=True,
            message=f"+{amount} HP. {new_hp}/{hp_max} HP.",
            state_changes={"hit_points": {"avant": hp_cur, "apres": new_hp}}
        )

    def consume_spell_slot(self, spell_level: int) -> ActionResult:
        """Consumes a spell slot."""
        check = self.can_use_spell(spell_level)
        if not check.success:
            return check
        current, max_val = self.get_resource("spells_per_day", level=spell_level)
        slot_key = f"level_{spell_level}"
        self.state.setdefault("resources", {}).setdefault("spells_per_day", {}).setdefault(slot_key, {})["current"] = current - 1
        self.save()
        return ActionResult(
            success=True,
            message=f"Spell slot level {spell_level} consumed ({current - 1}/{max_val} remaining).",
            state_changes={"spells_per_day": {slot_key: {"avant": current, "apres": current - 1}}}
        )

    def consume_resource(self, resource_key: str, amount: int = 1) -> ActionResult:
        """Consumes a generic resource."""
        check = self.can_use_resource(resource_key)
        if not check.success:
            return check
        current, max_val = self.get_resource(resource_key)
        new_val = max(0, current - amount)
        self.state.setdefault("resources", {})[resource_key]["current"] = new_val
        self.save()
        return ActionResult(
            success=True,
            message=f"'{resource_key}' consumed ({new_val}/{max_val} remaining).",
            state_changes={resource_key: {"avant": current, "apres": new_val}}
        )

    def add_xp(self, amount: int) -> ActionResult:
        """Adds XP and detects level up."""
        current_xp = self.state.get("xp", 0)
        next_level_xp = self.state.get("next_level_xp", self.state.get("xp_prochain_niveau", 999999))
        new_xp = current_xp + amount
        self.state["xp"] = new_xp
        level_up = new_xp >= next_level_xp
        self.save()
        msg = f"+{amount} XP ({new_xp}/{next_level_xp})"
        if level_up:
            msg += " — LEVEL UP available !"
        return ActionResult(success=True, message=msg, state_changes={"xp": {"avant": current_xp, "apres": new_xp}, "level_up": level_up})

    def _load_recovery_rules(self) -> dict:
        path = "Memory/recovery_rules.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"recovery_tiers": []}

    def rest(self, palier_id: str = "long") -> ActionResult:
        recovery_rules = self._load_recovery_rules()
        palier = next((p for p in recovery_rules.get("recovery_tiers", []) if p["id"] == palier_id), None)

        # Fallback to legacy resting logic if palier is not found but it is a standard legacy type (long or short)
        if not palier:
            if palier_id in ("long", "short"):
                return self._legacy_rest(palier_id)
            return ActionResult(success=False, message=f"Unknown recovery tier '{palier_id}'.")

        resources = self.state.get("resources", {})
        restored = []

        for effect in palier.get("effects", []):
            path = effect.get("resource", "")
            parts = path.split(".", 1)
            key = parts[1] if len(parts) > 1 and parts[0] == "resources" else path
            res = resources.get(key, {})

            # Handle nested sub-dicts
            if isinstance(res, dict) and "max" not in res:
                is_sub_dict = any(isinstance(v, dict) and "max" in v for v in res.values())
                if is_sub_dict:
                    for sub_key, sub_res in res.items():
                        if isinstance(sub_res, dict) and "max" in sub_res:
                            action = effect.get("action")
                            val_key = "current"
                            if action == "restore_full":
                                sub_res[val_key] = sub_res["max"]
                            elif action == "restore_percentage":
                                gain = max(1, sub_res["max"] * effect.get("value", 100) // 100)
                                sub_res[val_key] = min(sub_res["max"], sub_res.get(val_key, 0) + gain)
                            elif action == "restore_fixed_value":
                                gain = effect.get("value", 0)
                                sub_res[val_key] = min(sub_res["max"], sub_res.get(val_key, 0) + gain)
                    restored.append(key)
                    continue

            if not isinstance(res, dict) or "max" not in res:
                continue

            val_key = "current"
            action = effect.get("action")
            if action == "restore_full":
                res[val_key] = res["max"]
            elif action == "restore_percentage":
                gain = max(1, res["max"] * effect.get("value", 100) // 100)
                res[val_key] = min(res["max"], res.get(val_key, 0) + gain)
            elif action == "restore_fixed_value":
                gain = effect.get("value", 0)
                res[val_key] = min(res["max"], res.get(val_key, 0) + gain)
            else:
                continue
            restored.append(key)

        if "hit_points" in resources:
            self.state["pv"] = resources.get("hit_points", {}).get("current", self.state.get("pv", 0))

        self.state["resources"] = resources
        self.save()
        return ActionResult(
            success=True,
            message=f"Rest '{palier['name']}' completed. Restored: {', '.join(restored) if restored else 'nothing'}.",
            state_changes={"rest": palier_id, "restored": restored}
        )

    def _legacy_rest(self, rest_type: str = "long") -> ActionResult:
        resources = self.state.get("resources", {})
        restored = []

        if rest_type == "long":
            # Max HP
            pv = resources.get("hit_points", {})
            if pv:
                pv["current"] = pv.get("max", pv.get("current", 0))
                if "pv" in self.state:
                    self.state["pv"] = pv["current"]
                restored.append("HP")

            # Spells
            spells = resources.get("spells_per_day", {})
            for slot, values in spells.items():
                if isinstance(values, dict) and "max" in values:
                    values["current"] = values["max"]
            if spells:
                restored.append("spells")

            # Other resources with max
            for key, values in resources.items():
                if key not in ("hit_points", "spells_per_day") and isinstance(values, dict) and "max" in values:
                    values["current"] = values["max"]
                    restored.append(key)

        elif rest_type == "short":
            # Only HP
            pv = resources.get("hit_points", {})
            if pv:
                heal = max(1, pv.get("max", 1) // 4)
                pv["current"] = min(pv.get("max", 0), pv.get("current", 0) + heal)
                if "pv" in self.state:
                    self.state["pv"] = pv["current"]
                restored.append(f"HP +{heal}")

        self.state["resources"] = resources
        self.save()
        return ActionResult(
            success=True,
            message=f"Rest {rest_type} completed. Restored: {', '.join(restored) if restored else 'nothing'}.",
            state_changes={"rest": rest_type, "restored": restored}
        )

    def synchronize_and_recalculate(self):
        """
        Recalculates derived values (HP, AC, etc.) based on stats, race and class.
        Ensures sheet consistency.
        """
        if not self.state:
            return

        stats = self.state.get("statistics", {})

        resources = self.state.setdefault("resources", {})
        pv = resources.setdefault("hit_points", {})

        hp_max = pv.get("max", 0)
        hp_cur = pv.get("current", 0)

        if hp_max == 0 and "pv" in self.state and isinstance(self.state["pv"], int):
            hp_max = self.state["pv"]
            pv["max"] = hp_max

        if hp_cur > hp_max and hp_max > 0:
            pv["current"] = hp_max

        if hp_max > 0:
            self.state["pv"] = pv["current"]

        if "level" in self.state:
            try: self.state["level"] = int(self.state["level"])
            except: pass
        if "xp" in self.state:
            try: self.state["xp"] = int(self.state["xp"])
            except: pass

        self.save()

    # ── Automatic Detection from Text ─────────────────

    def detect_action_type(self, user_input: str) -> Optional[str]:
        """
        Detects action type from player's text input.
        Returns action key or None if no mechanical action is detected.
        """
        text = user_input.lower()

        recovery_rules = self._load_recovery_rules()
        for palier in recovery_rules.get("recovery_tiers", []):
            if any(kw.lower() in text for kw in palier.get("text_triggers", [])):
                return f"rest:{palier['id']}"

        if any(w in text for w in ["lance", "utilise", "sort", "magie", "spell", "incantation"]):
            return "spell"
        if any(w in text for w in ["rage", "berserk", "fureur"]):
            return "rage"
        if any(w in text for w in ["repos", "dors", "camp", "bivouac", "long rest", "short rest", "pause"]):
            return "rest"
        return None

    def apply_orchestrator_decision(self, decision: dict) -> ActionResult:
        """
        Applies a mechanical decision from the Orchestrator.
        Expected format: {"action": "damage"|"heal"|"xp", "amount": int}
        """
        action = decision.get("action")
        amount = decision.get("amount", 0)

        if action == "damage":
            return self.apply_damage(amount)
        elif action == "heal":
            return self.apply_healing(amount)
        elif action == "xp":
            return self.add_xp(amount)

        return ActionResult(success=False, message=f"Unknown mechanical action: {action}")
