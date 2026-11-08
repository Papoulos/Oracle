import copy
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from mechanics.paths import get_by_path

class FieldSpec(BaseModel):
    path: str
    type: Literal["string", "number", "object", "list"]
    label: Optional[str] = None
    sub_fields: list[str] = Field(default_factory=list)
    non_empty: bool = False
    min_items: int = 0

class GlossaryEntry(BaseModel):
    path: str
    label: Optional[str] = None
    description: str

class CharacterSchemaConfig(BaseModel):
    required_fields: list[FieldSpec]
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    defaults: dict = Field(default_factory=dict)

class CreationOption(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    description: str = ""
    effects: dict[str, Any] = Field(default_factory=dict)

class CreationManualStep(BaseModel):
    step: int
    name: str
    description: str

class CreationManualConfig(BaseModel):
    steps: list[CreationManualStep] = Field(default_factory=list)
    general_rules: str = ""
    options: dict[str, list[CreationOption]] = Field(default_factory=dict)

def schema_as_legacy_dict(cfg: CharacterSchemaConfig) -> dict:
    """
    Returns a dictionary equivalent to the legacy character_schema format
    (containing only required_fields) usable directly by validate_character_sheet.
    """
    return {
        "required_fields": [f.model_dump() for f in cfg.required_fields]
    }

def _deep_merge_defaults(target: dict, defaults: dict) -> None:
    """Recursively merges defaults into target without overwriting existing non-None values."""
    for k, default_val in defaults.items():
        if k not in target or target[k] is None:
            target[k] = copy.deepcopy(default_val)
        elif isinstance(default_val, dict) and isinstance(target[k], dict):
            _deep_merge_defaults(target[k], default_val)

def apply_schema_defaults(sheet: dict, cfg: CharacterSchemaConfig) -> dict:
    """
    Returns a copy of `sheet` with `cfg.defaults` recursively applied
    for any missing or None fields.
    """
    result = copy.deepcopy(sheet) if isinstance(sheet, dict) else {}
    if cfg and cfg.defaults:
        _deep_merge_defaults(result, cfg.defaults)
    return result


def _humanize_path(path: str) -> str:
    last = path.split(".")[-1]
    return last.replace("_", " ").capitalize()


def build_character_view(character_data: dict, pack: Any = None) -> dict:
    """
    Builds a pack-driven, UI-agnostic view model of a character sheet so the UI
    (e.g. Streamlit in app.py) can render any System Pack without hardcoded field names.
    """
    if not isinstance(character_data, dict) or not character_data:
        return {
            "name": "Aventurier Inconnu",
            "scalars": [],
            "pools": [],
            "pool_groups": [],
            "stat_groups": [],
            "lists": [],
            "other": {},
        }

    # Unwrap single-key root wrapper if present
    data = character_data
    if len(data) == 1 and isinstance(next(iter(data.values())), dict):
        only_key = next(iter(data.keys()))
        if only_key.lower() in ("personnage", "character", "pj"):
            data = data[only_key]

    consumed_top_keys: set[str] = {"name", "nom", "status", "image", "portrait", "pending_allocation", "recovery_step"}
    consumed_paths: set[str] = {"name", "nom", "status", "image", "portrait", "pending_allocation", "recovery_step"}

    def mark_consumed(path: str) -> None:
        consumed_paths.add(path)
        consumed_top_keys.add(path.split(".")[0])

    # Name resolution
    char_name = data.get("name") or data.get("nom") or "Aventurier Inconnu"

    scalars: list[dict[str, Any]] = []
    pools_view: list[dict[str, Any]] = []
    pool_groups_view: list[dict[str, Any]] = []
    stat_groups_view: list[dict[str, Any]] = []
    lists_view: list[dict[str, Any]] = []

    if pack is not None:
        schema: Optional[CharacterSchemaConfig] = getattr(pack, "character_schema", None)
        res_cfg = getattr(pack, "resources", None)

        # Build label map from schema required_fields and glossary
        label_map: dict[str, str] = {}
        if schema:
            for g in schema.glossary:
                if g.label:
                    label_map[g.path] = g.label
            for f in schema.required_fields:
                if f.label:
                    label_map[f.path] = f.label

        def get_label(path: str) -> str:
            if path in label_map:
                return label_map[path]
            return _humanize_path(path)

        # Identify pool parent paths so we don't duplicate them as stat_groups
        pool_parent_paths: set[str] = set()
        if res_cfg:
            for pool in getattr(res_cfg, "pools", []):
                mark_consumed(pool.current_path)
                if "." in pool.current_path:
                    pool_parent_paths.add(pool.current_path.rsplit(".", 1)[0])
                if pool.max_path:
                    mark_consumed(pool.max_path)
                    if "." in pool.max_path:
                        pool_parent_paths.add(pool.max_path.rsplit(".", 1)[0])
                cur_val = get_by_path(data, pool.current_path)
                max_val = pool.max_value if pool.max_value is not None else (get_by_path(data, pool.max_path) if pool.max_path else None)
                if cur_val is not None or max_val is not None:
                    pools_view.append({
                        "id": pool.id,
                        "name": pool.name,
                        "kind": pool.kind,
                        "current": cur_val,
                        "max": max_val,
                    })

            for group in getattr(res_cfg, "pool_groups", []):
                mark_consumed(group.path)
                pool_parent_paths.add(group.path)
                g_dict = get_by_path(data, group.path)
                if isinstance(g_dict, dict) and g_dict:
                    entries = []
                    for k, v in g_dict.items():
                        if isinstance(v, dict):
                            entries.append({
                                "key": k,
                                "current": v.get("current", v.get("restants", 0)),
                                "max": v.get("max", v.get("total", 0)),
                            })
                    if entries:
                        pool_groups_view.append({
                            "id": group.id,
                            "name": group.name,
                            "path": group.path,
                            "entries": entries,
                        })

        # Candidate paths from schema & resources
        scalar_paths: list[str] = []
        object_specs: list[tuple[str, list[str]]] = []
        list_paths: list[str] = []

        if schema:
            for f in schema.required_fields:
                if f.path in ("name", "nom") or f.path in pool_parent_paths:
                    continue
                if f.type in ("string", "number"):
                    if f.path not in scalar_paths:
                        scalar_paths.append(f.path)
                elif f.type == "object":
                    object_specs.append((f.path, list(f.sub_fields)))
                elif f.type == "list":
                    if f.path not in list_paths:
                        list_paths.append(f.path)

            for g in schema.glossary:
                if g.path in ("name", "nom") or g.path in pool_parent_paths or g.path in consumed_paths:
                    continue
                if any(g.path.startswith(p + ".") for p in pool_parent_paths):
                    continue
                if any(g.path == obj_p or g.path.startswith(obj_p + ".") for obj_p, _ in object_specs):
                    continue
                val = get_by_path(data, g.path)
                if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                    if g.path not in scalar_paths:
                        scalar_paths.append(g.path)
                elif isinstance(val, list):
                    if g.path not in list_paths:
                        list_paths.append(g.path)
                elif isinstance(val, dict) and g.path != "resources":
                    object_specs.append((g.path, list(val.keys())))

        if res_cfg:
            prog = getattr(res_cfg, "progression", None)
            if prog:
                for p_path in (prog.level_path, prog.xp_path):
                    if p_path and p_path not in scalar_paths:
                        scalar_paths.append(p_path)
                if prog.advancements_path and prog.advancements_path not in list_paths:
                    list_paths.append(prog.advancements_path)

            dmg = getattr(res_cfg, "damage", None)
            if dmg:
                for d_path in (dmg.armor_path, dmg.damage_track_path):
                    if d_path and d_path not in scalar_paths:
                        scalar_paths.append(d_path)

        # Populate scalars
        for sp in scalar_paths:
            if sp in consumed_paths:
                continue
            val = get_by_path(data, sp)
            mark_consumed(sp)
            if val is not None and val != "":
                scalars.append({
                    "path": sp,
                    "label": get_label(sp),
                    "value": val,
                })

        # Populate stat_groups (objects)
        for obj_path, sub_fields in object_specs:
            if obj_path in consumed_paths or obj_path in pool_parent_paths:
                continue
            obj_val = get_by_path(data, obj_path)
            mark_consumed(obj_path)
            if isinstance(obj_val, dict) and obj_val:
                ordered_keys = list(sub_fields) + [k for k in obj_val.keys() if k not in sub_fields]
                items = []
                for k in ordered_keys:
                    if k in obj_val and obj_val[k] is not None:
                        sub_path = f"{obj_path}.{k}"
                        items.append({
                            "key": k,
                            "label": get_label(sub_path),
                            "value": obj_val[k],
                        })
                if items:
                    stat_groups_view.append({
                        "path": obj_path,
                        "label": get_label(obj_path),
                        "items": items,
                    })

        # Consumables & Lists
        consumable_by_path: dict[str, Any] = {}
        if res_cfg and getattr(res_cfg, "consumables", None):
            for cdef in res_cfg.consumables:
                consumable_by_path[cdef.path] = cdef
                if cdef.path not in list_paths:
                    list_paths.append(cdef.path)
                if cdef.max_items_path:
                    mark_consumed(cdef.max_items_path)

        for lp in list_paths:
            if lp in consumed_paths:
                continue
            lst_val = get_by_path(data, lp)
            mark_consumed(lp)
            cdef = consumable_by_path.get(lp)
            max_limit = None
            if cdef:
                max_limit = cdef.max_items
                if max_limit is None and cdef.max_items_path:
                    raw_max = get_by_path(data, cdef.max_items_path)
                    if isinstance(raw_max, int) and not isinstance(raw_max, bool):
                        max_limit = raw_max
            if isinstance(lst_val, list) and lst_val:
                label = cdef.name if cdef else get_label(lp)
                lists_view.append({
                    "path": lp,
                    "label": label,
                    "items": lst_val,
                    "max_items": max_limit,
                })

    else:
        # Generic fallback when no pack is loaded: inspect types dynamically
        for k, v in data.items():
            if k in consumed_top_keys or v is None or v == "":
                continue
            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                consumed_top_keys.add(k)
                scalars.append({"path": k, "label": _humanize_path(k), "value": v})
            elif isinstance(v, list) and v:
                consumed_top_keys.add(k)
                lists_view.append({"path": k, "label": _humanize_path(k), "items": v, "max_items": None})
            elif isinstance(v, dict) and v:
                if k.lower() in ("resources", "ressources"):
                    consumed_top_keys.add(k)
                    for rk, rv in v.items():
                        if isinstance(rv, dict):
                            cur = rv.get("current", rv.get("restants"))
                            mx = rv.get("max", rv.get("total"))
                            if cur is not None or mx is not None:
                                pools_view.append({
                                    "id": rk,
                                    "name": _humanize_path(rk),
                                    "kind": "resource",
                                    "current": cur,
                                    "max": mx,
                                })
                            else:
                                entries = []
                                for sk, sv in rv.items():
                                    if isinstance(sv, dict):
                                        entries.append({
                                            "key": sk,
                                            "current": sv.get("current", sv.get("restants", 0)),
                                            "max": sv.get("max", sv.get("total", 0)),
                                        })
                                if entries:
                                    pool_groups_view.append({
                                        "id": rk,
                                        "name": _humanize_path(rk),
                                        "path": f"{k}.{rk}",
                                        "entries": entries,
                                    })
                elif all(isinstance(sv, (int, float, str)) for sv in v.values()):
                    consumed_top_keys.add(k)
                    stat_groups_view.append({
                        "path": k,
                        "label": _humanize_path(k),
                        "items": [{"key": sk, "label": _humanize_path(sk), "value": sv} for sk, sv in v.items()],
                    })

    other_info = {
        k: v for k, v in data.items()
        if k not in consumed_top_keys and v is not None and v != "" and v != [] and v != {}
    }

    return {
        "name": char_name,
        "scalars": scalars,
        "pools": pools_view,
        "pool_groups": pool_groups_view,
        "stat_groups": stat_groups_view,
        "lists": lists_view,
        "other": other_info,
    }
