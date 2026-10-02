import os
import pytest
from systems.loader import load_pack, PackNotFoundError

def test_load_pack_base_dir(monkeypatch, tmp_path):
    # Change to temp dir
    monkeypatch.chdir(tmp_path)

    # Should find dnd5e_srd because base_dir is computed via __file__
    pack = load_pack("dnd5e_srd")
    assert pack.manifest.id == "dnd5e_srd"
    assert pack.resolution.family == "D20VsTarget"

def test_load_pack_not_found():
    with pytest.raises(PackNotFoundError):
        load_pack("does_not_exist")
