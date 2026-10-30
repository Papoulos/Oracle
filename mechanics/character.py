import copy
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field

class FieldSpec(BaseModel):
    path: str
    type: Literal["string", "number", "object", "list"]
    sub_fields: list[str] = Field(default_factory=list)
    non_empty: bool = False

class GlossaryEntry(BaseModel):
    path: str
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
