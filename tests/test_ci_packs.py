import os
import pytest
from systems.validate import validate_pack
from mechanics.models import FAMILIES

SYSTEMS_DIR = os.path.join(os.path.dirname(__file__), "..", "systems")

def get_system_packs():
    packs = []
    if os.path.exists(SYSTEMS_DIR):
        for entry in os.listdir(SYSTEMS_DIR):
            full_path = os.path.join(SYSTEMS_DIR, entry)
            # Skip non-directories, __pycache__, draft
            if os.path.isdir(full_path) and entry not in ("__pycache__", "draft"):
                packs.append(entry)
    return packs

@pytest.mark.parametrize("pack_id", get_system_packs())
def test_all_system_packs_valid(pack_id):
    pack_path = os.path.join(SYSTEMS_DIR, pack_id)
    issues = validate_pack(pack_path)

    # Assert no errors
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 0, f"Pack {pack_id} has errors: {errors}"

    # Check for warnings based on whether it is a skeleton pack
    # A skeleton pack contains 'skeleton' or 'template' in its name
    is_skeleton = "skeleton" in pack_id.lower() or "template" in pack_id.lower()

    warnings = [i for i in issues if i.severity == "warning"]

    if not is_skeleton:
        assert len(warnings) == 0, f"Pack {pack_id} has TODO warnings, which is not allowed in CI for non-skeleton packs: {warnings}"
