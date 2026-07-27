import json
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ActionResult:
    success: bool
    message: str                    # message technique pour l'Orchestrateur
    state_changes: dict = field(default_factory=dict)  # ce qui a changé
    blocked_reason: Optional[str] = None  # si success=False


class GameStateEngine:
    """
    Moteur d'état central — Python pur, zéro LLM.
    Valide et applique toutes les modifications mécaniques du jeu.
    """

    RESOURCE_KEYWORDS = {
        "sort": "sorts_par_jour",
        "magie": "sorts_par_jour",
        "spell": "sorts_par_jour",
        "rage": "points_de_rage",
        "inspiration": "inspiration",
    }

    def __init__(self, character_path: str = "Memory/character.json"):
        self.character_path = character_path
        self.state = self._load()

    # ── Chargement / Persistence ──────────────────────────────

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
        """Recharger depuis le disque (après une mise à jour externe)."""
        self.state = self._load()

    # ── Accesseurs ────────────────────────────────────────────

    def get_hp(self) -> tuple[int, int]:
        """Retourne (hp_actuels, hp_max)."""
        ressources = self.state.get("ressources", {})
        pv = ressources.get("points_de_vie", {})
        return pv.get("actuels", 0), pv.get("max", 0)

    def get_resource(self, resource_key: str, level: Optional[int] = None) -> tuple[int, int]:
        """
        Retourne (restants, max) pour une ressource.
        Si level est fourni, cherche dans resource_key.niveau_N (pour les sorts).
        """
        ressources = self.state.get("ressources", {})
        resource = ressources.get(resource_key, {})
        if level is not None:
            slot = resource.get(f"niveau_{level}", {})
            return slot.get("restants", 0), slot.get("max", 0)
        return resource.get("restants", 0), resource.get("max", 0)

    def get_state_summary(self) -> str:
        """Résumé lisible de l'état pour l'Orchestrateur."""
        if not self.state:
            return "Aucun état disponible."

        hp_cur, hp_max = self.get_hp()
        lines = [
            f"PV : {hp_cur}/{hp_max}",
            f"Niveau : {self.state.get('niveau', '?')}",
            f"XP : {self.state.get('xp', 0)}/{self.state.get('xp_prochain_niveau', '?')}",
        ]

        ressources = self.state.get("ressources", {})
        sorts = ressources.get("sorts_par_jour", {})
        if isinstance(sorts, dict):
            for slot, values in sorts.items():
                if isinstance(values, dict):
                    lines.append(f"Sorts {slot} : {values.get('restants', 0)}/{values.get('max', 0)}")

        return " | ".join(lines)

    # ── Validation ────────────────────────────────────────────

    def can_use_spell(self, spell_level: int) -> ActionResult:
        """Vérifie si le personnage peut lancer un sort du niveau donné."""
        restants, max_val = self.get_resource("sorts_par_jour", level=spell_level)
        if max_val == 0:
            return ActionResult(
                success=False,
                message=f"Le personnage n'a pas d'emplacements de sort de niveau {spell_level}.",
                blocked_reason="no_spell_slot"
            )
        if restants <= 0:
            return ActionResult(
                success=False,
                message=f"Plus d'emplacements de sort de niveau {spell_level} disponibles. Repos requis.",
                blocked_reason="no_spell_slot_remaining"
            )
        return ActionResult(success=True, message=f"Sort de niveau {spell_level} disponible ({restants}/{max_val}).")

    def can_use_resource(self, resource_key: str) -> ActionResult:
        """Vérifie si une ressource générique est disponible."""
        restants, max_val = self.get_resource(resource_key)
        if max_val == 0:
            return ActionResult(
                success=False,
                message=f"Ressource '{resource_key}' non disponible pour ce personnage.",
                blocked_reason="resource_unavailable"
            )
        if restants <= 0:
            return ActionResult(
                success=False,
                message=f"Ressource '{resource_key}' épuisée. Repos requis.",
                blocked_reason="resource_depleted"
            )
        return ActionResult(success=True, message=f"Ressource '{resource_key}' disponible ({restants}/{max_val}).")

    # ── Application des modifications ────────────────────────

    def apply_damage(self, amount: int) -> ActionResult:
        """Applique des dégâts au personnage."""
        hp_cur, hp_max = self.get_hp()
        new_hp = max(0, hp_cur - amount)
        self.state.setdefault("ressources", {}).setdefault("points_de_vie", {})["actuels"] = new_hp
        # Sync with legacy pv field if it exists
        if "pv" in self.state:
            self.state["pv"] = new_hp
        self.save()
        status = "inconscient" if new_hp == 0 else f"{new_hp}/{hp_max} PV restants"
        return ActionResult(
            success=True,
            message=f"-{amount} PV. {status}",
            state_changes={"points_de_vie": {"avant": hp_cur, "apres": new_hp}}
        )

    def apply_healing(self, amount: int) -> ActionResult:
        """Applique des soins au personnage."""
        hp_cur, hp_max = self.get_hp()
        new_hp = min(hp_max, hp_cur + amount)
        self.state.setdefault("ressources", {}).setdefault("points_de_vie", {})["actuels"] = new_hp
        # Sync with legacy pv field if it exists
        if "pv" in self.state:
            self.state["pv"] = new_hp
        self.save()
        return ActionResult(
            success=True,
            message=f"+{amount} PV. {new_hp}/{hp_max} PV.",
            state_changes={"points_de_vie": {"avant": hp_cur, "apres": new_hp}}
        )

    def consume_spell_slot(self, spell_level: int) -> ActionResult:
        """Consomme un emplacement de sort."""
        check = self.can_use_spell(spell_level)
        if not check.success:
            return check
        restants, max_val = self.get_resource("sorts_par_jour", level=spell_level)
        slot_key = f"niveau_{spell_level}"
        self.state.setdefault("ressources", {}).setdefault("sorts_par_jour", {}).setdefault(slot_key, {})["restants"] = restants - 1
        self.save()
        return ActionResult(
            success=True,
            message=f"Emplacement sort niveau {spell_level} consommé ({restants - 1}/{max_val} restants).",
            state_changes={"sorts_par_jour": {slot_key: {"avant": restants, "apres": restants - 1}}}
        )

    def consume_resource(self, resource_key: str, amount: int = 1) -> ActionResult:
        """Consomme une ressource générique."""
        check = self.can_use_resource(resource_key)
        if not check.success:
            return check
        restants, max_val = self.get_resource(resource_key)
        new_val = max(0, restants - amount)
        self.state.setdefault("ressources", {})[resource_key]["restants"] = new_val
        self.save()
        return ActionResult(
            success=True,
            message=f"'{resource_key}' consommé ({new_val}/{max_val} restants).",
            state_changes={resource_key: {"avant": restants, "apres": new_val}}
        )

    def add_xp(self, amount: int) -> ActionResult:
        """Ajoute de l'XP et détecte une montée de niveau."""
        current_xp = self.state.get("xp", 0)
        next_level_xp = self.state.get("xp_prochain_niveau", 999999)
        new_xp = current_xp + amount
        self.state["xp"] = new_xp
        level_up = new_xp >= next_level_xp
        self.save()
        msg = f"+{amount} XP ({new_xp}/{next_level_xp})"
        if level_up:
            msg += " — MONTÉE DE NIVEAU disponible !"
        return ActionResult(success=True, message=msg, state_changes={"xp": {"avant": current_xp, "apres": new_xp}, "level_up": level_up})

    def _load_recovery_rules(self) -> dict:
        path = "Memory/recovery_rules.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"paliers_repos": []}

    def rest(self, palier_id: str = "long") -> ActionResult:
        recovery_rules = self._load_recovery_rules()
        palier = next((p for p in recovery_rules.get("paliers_repos", []) if p["id"] == palier_id), None)

        # Fallback to legacy resting logic if palier is not found but it is a standard legacy type (long or short)
        if not palier:
            if palier_id in ("long", "short"):
                return self._legacy_rest(palier_id)
            return ActionResult(success=False, message=f"Palier de récupération '{palier_id}' inconnu.")

        ressources = self.state.get("ressources", {})
        restored = []

        for effet in palier.get("effets", []):
            chemin = effet.get("ressource", "")
            parts = chemin.split(".", 1)
            cle = parts[1] if len(parts) > 1 and parts[0] == "ressources" else chemin
            res = ressources.get(cle, {})

            # Handle nested sub-dicts
            if isinstance(res, dict) and "max" not in res:
                is_sub_dict = any(isinstance(v, dict) and "max" in v for v in res.values())
                if is_sub_dict:
                    for sub_key, sub_res in res.items():
                        if isinstance(sub_res, dict) and "max" in sub_res:
                            action = effet.get("action")
                            val_key = "actuels" if sub_key == "points_de_vie" else "restants"
                            if action == "restaurer_complet":
                                sub_res[val_key] = sub_res["max"]
                            elif action == "restaurer_pourcentage":
                                gain = max(1, sub_res["max"] * effet.get("valeur", 100) // 100)
                                sub_res[val_key] = min(sub_res["max"], sub_res.get(val_key, 0) + gain)
                            elif action == "restaurer_valeur_fixe":
                                gain = effet.get("valeur", 0)
                                sub_res[val_key] = min(sub_res["max"], sub_res.get(val_key, 0) + gain)
                    restored.append(cle)
                    continue

            if not isinstance(res, dict) or "max" not in res:
                continue

            val_key = "actuels" if cle == "points_de_vie" else "restants"
            action = effet.get("action")
            if action == "restaurer_complet":
                res[val_key] = res["max"]
            elif action == "restaurer_pourcentage":
                gain = max(1, res["max"] * effet.get("valeur", 100) // 100)
                res[val_key] = min(res["max"], res.get(val_key, 0) + gain)
            elif action == "restaurer_valeur_fixe":
                gain = effet.get("valeur", 0)
                res[val_key] = min(res["max"], res.get(val_key, 0) + gain)
            else:
                continue
            restored.append(cle)

        if "points_de_vie" in ressources:
            self.state["pv"] = ressources.get("points_de_vie", {}).get("actuels", self.state.get("pv", 0))

        self.state["ressources"] = ressources
        self.save()
        return ActionResult(
            success=True,
            message=f"Repos '{palier['nom']}' effectué. Restauré : {', '.join(restored) if restored else 'rien'}.",
            state_changes={"rest": palier_id, "restored": restored}
        )

    def _legacy_rest(self, rest_type: str = "long") -> ActionResult:
        ressources = self.state.get("ressources", {})
        restored = []

        if rest_type == "long":
            # PV max
            pv = ressources.get("points_de_vie", {})
            if pv:
                pv["actuels"] = pv.get("max", pv.get("actuels", 0))
                if "pv" in self.state:
                    self.state["pv"] = pv["actuels"]
                restored.append("PV")

            # Sorts
            sorts = ressources.get("sorts_par_jour", {})
            for slot, values in sorts.items():
                if isinstance(values, dict) and "max" in values:
                    values["restants"] = values["max"]
            if sorts:
                restored.append("sorts")

            # Autres ressources avec max
            for key, values in ressources.items():
                if key not in ("points_de_vie", "sorts_par_jour") and isinstance(values, dict) and "max" in values:
                    values["restants"] = values["max"]
                    restored.append(key)

        elif rest_type == "short":
            # Uniquement PV
            pv = ressources.get("points_de_vie", {})
            if pv:
                heal = max(1, pv.get("max", 1) // 4)
                pv["actuels"] = min(pv.get("max", 0), pv.get("actuels", 0) + heal)
                if "pv" in self.state:
                    self.state["pv"] = pv["actuels"]
                restored.append(f"PV +{heal}")

        self.state["ressources"] = ressources
        self.save()
        return ActionResult(
            success=True,
            message=f"Repos {rest_type} effectué. Restauré : {', '.join(restored) if restored else 'rien'}.",
            state_changes={"rest": rest_type, "restored": restored}
        )

    def synchronize_and_recalculate(self):
        """
        Recalcule les valeurs dérivées (PV, CA, etc.) basées sur les stats, race et classe.
        Assure la cohérence de la fiche.
        """
        if not self.state:
            return

        # 1. Calcul des modificateurs de caractéristiques (Règle standard D&D/OSRIC : (stat-10)//2)
        # Note: Dans certains vieux systèmes (BX), c'est différent.
        # Mais on va proposer une logique de base qui pourra être affinée.
        stats = self.state.get("statistiques", {})

        # 2. Mise à jour des PV si nécessaire
        # Si PV max est absent ou incohérent, on tente un calcul de base (ex: 8 + mod CON)
        # Dans un vrai système RAG, on laisserait l'LLM extraire le dé de vie,
        # mais ici on s'assure au moins que les PV actuels ne dépassent pas le max.
        ressources = self.state.setdefault("ressources", {})
        pv = ressources.setdefault("points_de_vie", {})

        hp_max = pv.get("max", 0)
        hp_cur = pv.get("actuels", 0)

        if hp_max == 0 and "pv" in self.state and isinstance(self.state["pv"], int):
            hp_max = self.state["pv"]
            pv["max"] = hp_max

        if hp_cur > hp_max and hp_max > 0:
            pv["actuels"] = hp_max

        # 3. Synchronisation legacy
        if hp_max > 0:
            self.state["pv"] = pv["actuels"]

        # 4. S'assurer que le niveau et l'XP sont des entiers
        if "niveau" in self.state:
            try: self.state["niveau"] = int(self.state["niveau"])
            except: pass
        if "xp" in self.state:
            try: self.state["xp"] = int(self.state["xp"])
            except: pass

        self.save()

    # ── Détection automatique depuis le texte ─────────────────

    def detect_action_type(self, user_input: str) -> Optional[str]:
        """
        Détecte le type d'action mécanique depuis le texte du joueur.
        Retourne une clé d'action ou None si pas d'action mécanique détectée.
        """
        text = user_input.lower()

        recovery_rules = self._load_recovery_rules()
        for palier in recovery_rules.get("paliers_repos", []):
            if any(kw.lower() in text for kw in palier.get("declencheurs_texte", [])):
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
        Applique une décision mécanique venant de l'Orchestrateur.
        Format attendu : {"action": "damage"|"heal"|"xp", "amount": int}
        """
        action = decision.get("action")
        amount = decision.get("amount", 0)

        if action == "damage":
            return self.apply_damage(amount)
        elif action == "heal":
            return self.apply_healing(amount)
        elif action == "xp":
            return self.add_xp(amount)

        return ActionResult(success=False, message=f"Action mécanique inconnue : {action}")
