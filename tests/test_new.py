import os
import shutil
import tempfile
import pytest
import subprocess
import json

from mechanics.models import FAMILIES
from systems.validate import validate_pack


@pytest.fixture(scope="module")
def temp_systems_dir():
    # Use a temporary directory for generating packs
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("family", FAMILIES.keys())
def test_new_pack_generation(family, temp_systems_dir, monkeypatch):
    # Mock SYSTEMS_DIR generation to go into the temp dir
    # we will just call the script using subprocess and CWD or modify the script?
    # Better: run via subprocess with an overridden path?
    # Actually `systems.new` hardcodes `os.path.join("systems", pack_id)`.
    # Let's mock `os.path.join` or run it from a patched CWD.

    pack_id = f"test_pack_{family.lower()}"

    # We patch the CWD so that 'systems/' is created inside the temp dir
    monkeypatch.chdir(temp_systems_dir)

    # Add project root to PYTHONPATH so we can resolve systems.new
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # Run the generation script
    result = subprocess.run(["python3", "-m", "systems.new", pack_id, "--family", family], capture_output=True, text=True, env=env)
    assert result.returncode == 0

    pack_path = os.path.join("systems", pack_id)
    assert os.path.exists(pack_path)

    # Validate the generated pack
    issues = validate_pack(pack_path)

    # Should have no errors
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 0

    # Should have at least one TODO warning
    warnings = [i for i in issues if i.severity == "warning" and "TODO" in i.message]
    assert len(warnings) > 0

    # Verify replacing TODOs makes it strict-valid
    # Replace all "TODO" with "Done" in all 3 files
    for filename in ["manifest.yaml", "resolution.json", "resources.json"]:
        filepath = os.path.join(pack_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # for resources.json, we have `{"TODO": "TODO"}`, let's replace it with valid key/value
        content = content.replace("TODO", "Done")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    # Validate again
    strict_issues = validate_pack(pack_path)
    assert len(strict_issues) == 0, f"Expected 0 issues after fixing TODOs, got {strict_issues}"


def test_new_pack_refuses_overwrite(temp_systems_dir, monkeypatch):
    monkeypatch.chdir(temp_systems_dir)

    pack_id = "test_overwrite"
    family = "D20VsTarget"

    # Add project root to PYTHONPATH so we can resolve systems.new
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # First generation
    result1 = subprocess.run(["python3", "-m", "systems.new", pack_id, "--family", family], capture_output=True, text=True, env=env)
    assert result1.returncode == 0

    # Second generation should fail
    result2 = subprocess.run(["python3", "-m", "systems.new", pack_id, "--family", family], capture_output=True, text=True, env=env)
    assert result2.returncode == 1
    assert "existe déjà" in result2.stderr

def test_new_pack_invalid_id(temp_systems_dir, monkeypatch):
    monkeypatch.chdir(temp_systems_dir)
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(["python3", "-m", "systems.new", "Invalid-Name!", "--family", "D20VsTarget"], capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert "invalide" in result.stderr

    result2 = subprocess.run(["python3", "-m", "systems.new", "abc\n", "--family", "D20VsTarget"], capture_output=True, text=True, env=env)
    assert result2.returncode == 1
    assert "invalide" in result2.stderr

    result3 = subprocess.run(["python3", "-m", "systems.new", "../x", "--family", "D20VsTarget"], capture_output=True, text=True, env=env)
    assert result3.returncode == 1
    assert "invalide" in result3.stderr
