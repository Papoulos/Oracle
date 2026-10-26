import os
import json
import yaml
from dataclasses import dataclass
from typing import Optional

from pydantic import TypeAdapter
from mechanics.models import Manifest, ResolutionConfig, ResourcesConfig, TriggersConfig
from mechanics.actions import ActionsConfig
from mechanics.character import CharacterSchemaConfig, CreationManualConfig
from systems.validate import validate_pack


class PackNotFoundError(Exception):
    pass


class PackInvalidError(Exception):
    pass


@dataclass
class SystemPack:
    manifest: Manifest
    resolution: ResolutionConfig
    resources: ResourcesConfig
    triggers: Optional[TriggersConfig]
    actions: Optional[ActionsConfig] = None
    character_schema: Optional[CharacterSchemaConfig] = None
    creation_manual: Optional[CreationManualConfig] = None


def load_pack(pack_id: str, base_dir: str = None) -> SystemPack:
    """
    Loads a System Pack from the given base directory.
    Validates it first, raises PackInvalidError if validation fails.
    Raises PackNotFoundError if pack directory doesn't exist.
    """
    import re
    if not re.fullmatch(r"[a-z0-9_]+", pack_id):
        raise ValueError(f"Invalid pack_id '{pack_id}'. Must match ^[a-z0-9_]+$")

    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "systems")

    pack_dir = os.path.join(base_dir, pack_id)
    if not os.path.exists(pack_dir) or not os.path.isdir(pack_dir):
        raise PackNotFoundError(f"Pack '{pack_id}' not found at {pack_dir}")

    # Validate first
    issues = validate_pack(pack_dir)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        error_msgs = "\n".join([f"{i.file}:{i.path} - {i.message}" for i in errors])
        raise PackInvalidError(f"Pack '{pack_id}' is invalid:\n{error_msgs}")

    # Load Manifest
    with open(os.path.join(pack_dir, "manifest.yaml"), "r", encoding="utf-8") as f:
        manifest_data = yaml.safe_load(f)
    manifest = Manifest.model_validate(manifest_data)

    # Load Resolution
    with open(os.path.join(pack_dir, "resolution.json"), "r", encoding="utf-8") as f:
        resolution_data = json.load(f)
    config_adapter = TypeAdapter(ResolutionConfig)
    resolution = config_adapter.validate_python(resolution_data)

    # Load Resources
    with open(os.path.join(pack_dir, "resources.json"), "r", encoding="utf-8") as f:
        resources_data = json.load(f)
    resources = ResourcesConfig.model_validate(resources_data)

    # Load Triggers (optional)
    triggers = None
    triggers_path = os.path.join(pack_dir, "triggers.json")
    if os.path.exists(triggers_path):
        with open(triggers_path, "r", encoding="utf-8") as f:
            triggers_data = json.load(f)
        triggers = TriggersConfig.model_validate(triggers_data)

    # Load Actions (optional)
    actions = None
    actions_path = os.path.join(pack_dir, "actions.json")
    if os.path.exists(actions_path):
        with open(actions_path, "r", encoding="utf-8") as f:
            actions_data = json.load(f)
        actions = ActionsConfig.model_validate(actions_data)

    # Load Character Schema (optional)
    character_schema = None
    character_schema_path = os.path.join(pack_dir, "character_schema.json")
    if os.path.exists(character_schema_path):
        with open(character_schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        character_schema = CharacterSchemaConfig.model_validate(schema_data)

    # Load Creation Manual (optional)
    creation_manual = None
    creation_manual_path = os.path.join(pack_dir, "creation_manual.json")
    if os.path.exists(creation_manual_path):
        with open(creation_manual_path, "r", encoding="utf-8") as f:
            manual_data = json.load(f)
        creation_manual = CreationManualConfig.model_validate(manual_data)

    return SystemPack(
        manifest=manifest,
        resolution=resolution,
        resources=resources,
        triggers=triggers,
        actions=actions,
        character_schema=character_schema,
        creation_manual=creation_manual
    )
