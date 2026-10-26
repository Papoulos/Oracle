from typing import Literal, Optional
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

class CreationManualStep(BaseModel):
    step: int
    name: str
    description: str

class CreationManualConfig(BaseModel):
    steps: list[CreationManualStep] = Field(default_factory=list)
    general_rules: str = ""

def schema_as_legacy_dict(cfg: CharacterSchemaConfig) -> dict:
    """
    Returns a dictionary equivalent to the legacy character_schema format
    (containing only required_fields) usable directly by validate_character_sheet.
    """
    return {
        "required_fields": [f.model_dump() for f in cfg.required_fields]
    }
