import json
import pytest
from mechanics.character import CharacterSchemaConfig, schema_as_legacy_dict
from validation import validate_character_sheet
from systems.validate import validate_pack
from game_state_engine import GameStateEngine
from systems.loader import load_pack

def setup_pack(tmp_path, pack_id="test_pack", schema_data=None, manual_data=None, pools=None):
    pack_dir = tmp_path / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest
    (pack_dir / "manifest.yaml").write_text("""id: test_pack
name: Test Pack
version: 1.0.0
language: fr
family: StepTargetD20
source_pdfs: []
""")

    # Write resolution
    (pack_dir / "resolution.json").write_text(json.dumps({
        "family": "StepTargetD20",
        "multiplier": 3,
        "min_difficulty": 0,
        "max_difficulty": 10,
        "auto_success_at": 0
    }))

    if pools is None:
        pools = []

    # Write resources
    (pack_dir / "resources.json").write_text(json.dumps({
        "recovery_triggers": [],
        "pools": pools,
        "pool_groups": []
    }))

    # Write optional files
    if schema_data is not None:
        (pack_dir / "character_schema.json").write_text(json.dumps(schema_data))
    if manual_data is not None:
        (pack_dir / "creation_manual.json").write_text(json.dumps(manual_data))

    return str(pack_dir)


def test_schema_as_legacy_dict():
    schema_json = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "resources.hit_points", "type": "object", "sub_fields": ["current", "max"]}
        ]
    }
    cfg = CharacterSchemaConfig.model_validate(schema_json)
    legacy = schema_as_legacy_dict(cfg)

    # Valid character
    char_valid = {
        "name": "Hero",
        "resources": {
            "hit_points": {"current": 10, "max": 10}
        }
    }
    is_complete, missing = validate_character_sheet(char_valid, legacy)
    assert is_complete
    assert not missing

    # Invalid character
    char_invalid = {
        "name": "Hero"
    }
    is_complete, missing = validate_character_sheet(char_invalid, legacy)
    assert not is_complete
    assert missing == ["resources.hit_points"]


def test_validate_pack_character_files_valid(tmp_path):
    schema_data = {
        "required_fields": [{"path": "name", "type": "string"}]
    }
    manual_data = {
        "steps": [{"step": 1, "name": "Step 1", "description": "Do stuff"}]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data, manual_data=manual_data)
    issues = validate_pack(pack_dir)
    assert not any(i.severity == "error" for i in issues)


def test_validate_pack_schema_invalid_path(tmp_path):
    schema_data = {
        "required_fields": [{"path": "invalid..path", "type": "string"}]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data)
    issues = validate_pack(pack_dir)
    assert any(i.severity == "error" and i.file == "character_schema.json" and "Invalid path format" in i.message for i in issues)


def test_validate_pack_schema_duplicate_path(tmp_path):
    schema_data = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "name", "type": "string"}
        ]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data)
    issues = validate_pack(pack_dir)
    assert any(i.severity == "error" and i.file == "character_schema.json" and "Duplicate path" in i.message for i in issues)


def test_validate_pack_schema_pool_coverage(tmp_path):
    pools = [{
        "id": "might",
        "name": "Might",
        "kind": "health",
        "current_path": "resources.might.current",
        "max_path": "resources.might.max",
        "recovery": []
    }]
    schema_data = {
        "required_fields": [{"path": "name", "type": "string"}]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data, pools=pools)
    issues = validate_pack(pack_dir)
    warnings = [i for i in issues if i.severity == "warning" and i.file == "character_schema.json" and "not covered" in i.message]
    assert len(warnings) == 1
    assert "pool 'might' (resources.might.current) is not covered" in warnings[0].message

    # Now cover it
    schema_data_covered = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "resources.might", "type": "object", "sub_fields": ["current", "max"]}
        ]
    }
    pack_dir_covered = setup_pack(tmp_path, schema_data=schema_data_covered, pools=pools)
    issues_covered = validate_pack(pack_dir_covered)
    assert not any(i.severity == "warning" and "not covered" in i.message for i in issues_covered)


def test_validate_pack_manual_steps(tmp_path):
    # Empty steps should warn
    manual_empty = {"steps": []}
    pack_dir = setup_pack(tmp_path, manual_data=manual_empty)
    issues = validate_pack(pack_dir)
    assert any(i.severity == "warning" and i.file == "creation_manual.json" and "no steps" in i.message for i in issues)

    # Non-consecutive steps should error
    manual_non_consecutive = {
        "steps": [
            {"step": 1, "name": "Step 1", "description": "a"},
            {"step": 3, "name": "Step 3", "description": "b"}
        ]
    }
    pack_dir2 = setup_pack(tmp_path, pack_id="test2", manual_data=manual_non_consecutive)
    issues2 = validate_pack(pack_dir2)
    assert any(i.severity == "error" and i.file == "creation_manual.json" and "Expected step 2, got 3" in i.message for i in issues2)


def test_synchronize_and_recalculate_hit_points_logic(tmp_path):
    # Pack with hp kind health but no hit_points id
    pools1 = [{
        "id": "hp",
        "name": "HP",
        "kind": "health",
        "current_path": "resources.hp.current",
        "max_path": "resources.hp.max",
        "recovery": []
    }]
    pack_dir1 = setup_pack(tmp_path, pack_id="pack_health", pools=pools1)
    pack1 = load_pack("pack_health", base_dir=str(tmp_path))

    engine1 = GameStateEngine(pack=pack1)
    engine1.state = {"pv": 10}
    engine1.synchronize_and_recalculate()
    assert "resources" not in engine1.state or "hit_points" not in engine1.state.get("resources", {})

    # Pack with hit_points id
    pools2 = [{
        "id": "hit_points",
        "name": "Hit Points",
        "kind": "health",
        "current_path": "resources.hit_points.current",
        "max_path": "resources.hit_points.max",
        "recovery": []
    }]
    pack_dir2 = setup_pack(tmp_path, pack_id="pack_hit_points", pools=pools2)
    pack2 = load_pack("pack_hit_points", base_dir=str(tmp_path))

    engine2 = GameStateEngine(pack=pack2)
    engine2.state = {"pv": 10}
    engine2.synchronize_and_recalculate()
    assert "hit_points" in engine2.state["resources"]
    assert engine2.state["resources"]["hit_points"]["max"] == 10

    # No pack (fallback logic)
    engine3 = GameStateEngine(pack=None)
    engine3.state = {"pv": 10}
    engine3.synchronize_and_recalculate()
    assert "hit_points" in engine3.state["resources"]


def test_synchronize_hit_points_regression_139(tmp_path):
    """
    Test pour la régression #139 : la normalisation des points de vie (hit_points) doit
    se déclencher UNIQUEMENT si le pack gère explicitement "resources.hit_points" dans ses
    chemins de pools ou pool_groups, ou s'il n'y a pas de pack.
    """
    from systems.loader import load_pack
    from pathlib import Path

    base_sys_dir = Path(__file__).parent.parent / "systems"

    # 1. Pack dnd5e_srd : transfert simple de pv vers resources.hit_points
    pack_dnd = load_pack("dnd5e_srd", base_dir=str(base_sys_dir))
    engine_dnd = GameStateEngine(pack=pack_dnd)
    engine_dnd.state = {"pv": 12, "resources": {}}
    engine_dnd.synchronize_and_recalculate()
    assert "hit_points" in engine_dnd.state["resources"]
    assert engine_dnd.state["resources"]["hit_points"]["max"] == 12

    # 2. Pack dnd5e_srd : current > max est ramené au max, pv mis à jour
    engine_dnd.state = {"pv": 10, "resources": {"hit_points": {"current": 30, "max": 20}}}
    engine_dnd.synchronize_and_recalculate()
    assert engine_dnd.state["resources"]["hit_points"]["current"] == 20
    assert engine_dnd.state["pv"] == 20

    # 3. Pack cypher_system : ne gère pas hit_points, aucune clé ajoutée
    pack_cypher = load_pack("cypher_system", base_dir=str(base_sys_dir))
    engine_cypher = GameStateEngine(pack=pack_cypher)
    engine_cypher.state = {"pv": 15, "resources": {}}
    engine_cypher.synchronize_and_recalculate()
    assert "hit_points" not in engine_cypher.state["resources"]

    # 4. Pack de test : current_path = "resources.hp.current"
    pools_hp = [{
        "id": "hp",
        "name": "HP",
        "kind": "health",
        "current_path": "resources.hp.current",
        "max_path": "resources.hp.max",
        "recovery": []
    }]
    setup_pack(tmp_path, pack_id="pack_hp_only", pools=pools_hp)
    pack_hp = load_pack("pack_hp_only", base_dir=str(tmp_path))
    engine_hp = GameStateEngine(pack=pack_hp)
    engine_hp.state = {"pv": 10, "resources": {}}
    engine_hp.synchronize_and_recalculate()
    assert "hit_points" not in engine_hp.state.get("resources", {})

    # 5. Pack de test : id "vie", current_path = "resources.hit_points.current" (déclenche normalisation)
    pools_vie = [{
        "id": "vie",
        "name": "Vie",
        "kind": "health",
        "current_path": "resources.hit_points.current",
        "max_path": "resources.hit_points.max",
        "recovery": []
    }]
    setup_pack(tmp_path, pack_id="pack_vie", pools=pools_vie)
    pack_vie = load_pack("pack_vie", base_dir=str(tmp_path))
    engine_vie = GameStateEngine(pack=pack_vie)
    engine_vie.state = {"pv": 10, "resources": {}}
    engine_vie.synchronize_and_recalculate()
    assert "hit_points" in engine_vie.state["resources"]
    assert engine_vie.state["resources"]["hit_points"]["max"] == 10

    # 6. Pack de test : current_path = "resources.hit_points_extra.current" (rejeté, préfixe exact)
    pools_extra = [{
        "id": "extra",
        "name": "Extra",
        "kind": "health",
        "current_path": "resources.hit_points_extra.current",
        "max_path": "resources.hit_points_extra.max",
        "recovery": []
    }]
    setup_pack(tmp_path, pack_id="pack_extra", pools=pools_extra)
    pack_extra = load_pack("pack_extra", base_dir=str(tmp_path))
    engine_extra = GameStateEngine(pack=pack_extra)
    engine_extra.state = {"pv": 10, "resources": {}}
    engine_extra.synchronize_and_recalculate()
    assert "hit_points" not in engine_extra.state.get("resources", {})

    # 7. Sans pack : comportement inchangé
    engine_no_pack = GameStateEngine(pack=None)
    engine_no_pack.state = {"pv": 10, "resources": {}}
    engine_no_pack.synchronize_and_recalculate()
    assert "hit_points" in engine_no_pack.state["resources"]
    assert engine_no_pack.state["resources"]["hit_points"]["max"] == 10
