import os
import pytest
from systems.validate import validate_pack

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def test_validate_valid_pbta():
    pack_path = os.path.join(FIXTURES_DIR, "pbta_pack")
    issues = validate_pack(pack_path)

    assert len(issues) == 0, f"Expected no issues, got: {issues}"

def test_validate_invalid_pbta_gaps():
    pack_path = os.path.join(FIXTURES_DIR, "invalid_pbta_pack")
    issues = validate_pack(pack_path)

    assert len(issues) > 0
    assert any("Gap detected between 5 and 7" in i.message for i in issues)

def test_validate_dnd5e_srd():
    pack_path = os.path.join(os.path.dirname(__file__), "..", "systems", "dnd5e_srd")
    issues = validate_pack(pack_path)

    assert len(issues) == 0, f"Expected no issues for dnd5e_srd reference pack, got: {issues}"

def test_validate_step_target():
    pack_path = os.path.join(FIXTURES_DIR, "step_target_pack")
    issues = validate_pack(pack_path)

    assert len(issues) == 0

def test_validate_dice_pool():
    pack_path = os.path.join(FIXTURES_DIR, "dice_pool_pack")
    issues = validate_pack(pack_path)

    assert len(issues) == 0
