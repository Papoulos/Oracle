import pytest
from pydantic import ValidationError

from mechanics.models import TriggerRule
from game_state_engine import GameStateEngine, DetectedAction
from systems.validate import validate_pack

class DummyResources:
    def __init__(self, sequences=None):
        from mechanics.models import RecoverySequence
        self.recovery_sequences = sequences or []

class DummyPack:
    def __init__(self, triggers=None, resources=None):
        self.triggers = triggers
        self.resources = resources

# 1. TriggerRule model tests
def test_trigger_rule_recover():
    # Valid: only trigger
    TriggerRule(id="1", kind="recover", trigger="rest", keywords={"fr": ["repos"]})

    # Valid: only sequence
    TriggerRule(id="2", kind="recover", sequence="rest_seq", keywords={"fr": ["repos"]})

    # Invalid: both
    with pytest.raises(ValueError, match="recover rule requires exactly one of 'trigger' or 'sequence'"):
        TriggerRule(id="3", kind="recover", trigger="rest", sequence="rest_seq", keywords={"fr": ["repos"]})

    # Invalid: neither
    with pytest.raises(ValueError, match="recover rule requires exactly one of 'trigger' or 'sequence'"):
        TriggerRule(id="4", kind="recover", keywords={"fr": ["repos"]})

def test_trigger_rule_consume():
    # Valid: no sequence
    TriggerRule(id="1", kind="consume", target="pool", keywords={"fr": ["use"]})

    # Valid: with trigger (trigger is ignored but allowed for backwards compatibility)
    TriggerRule(id="2", kind="consume", target="pool", trigger="something", keywords={"fr": ["use"]})

    # Invalid: no target
    with pytest.raises(ValueError, match="target must be provided when kind is 'consume'"):
        TriggerRule(id="3", kind="consume", keywords={"fr": ["use"]})

    # Invalid: sequence
    with pytest.raises(ValueError, match="sequence cannot be provided when kind is 'consume'"):
        TriggerRule(id="4", kind="consume", target="pool", sequence="seq", keywords={"fr": ["use"]})

# 2. validate_pack tests for sequence
def test_validate_pack_sequence(tmp_path, monkeypatch):
    import json
    import os

    pack_dir = tmp_path / "test_pack"
    pack_dir.mkdir()

    # manifest.yaml
    with open(pack_dir / "manifest.yaml", "w", encoding="utf-8") as f:
        f.write("id: test_pack\nname: Test Pack\nversion: 1.0\nlanguage: fr\nfamily: D20VsTarget\nsource_pdfs: []")

    # resolution.json
    with open(pack_dir / "resolution.json", "w", encoding="utf-8") as f:
        json.dump({"family": "D20VsTarget", "advantage_enabled": False, "advantage_dice": 2, "critical_success_on": 20, "critical_failure_on": 1}, f)

    # resources.json (empty sequences)
    with open(pack_dir / "resources.json", "w", encoding="utf-8") as f:
        json.dump({
            "recovery_triggers": [],
            "pools": [],
            "pool_groups": [],
            "pooled_recoveries": [],
            "recovery_sequences": []
        }, f)

    # triggers.json (sequence not in resources)
    with open(pack_dir / "triggers.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "rules": [
                {
                    "id": "rest_request",
                    "kind": "recover",
                    "sequence": "cypher_rests",
                    "keywords": {"fr": ["je me repose"]}
                }
            ]
        }, f)

    issues = validate_pack(str(pack_dir))
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) >= 1
    seq_error = next((e for e in errors if "sequence 'cypher_rests' not found in resources.json recovery_sequences" in e.message), None)
    assert seq_error is not None
    assert seq_error.path == "/rules/0/sequence"

# 3. game_state_engine tests
def test_engine_detect_action_type():
    from systems.loader import load_pack
    from pathlib import Path
    systems_dir = str(Path(__file__).parent.parent / "systems")
    cypher_pack = load_pack("cypher_system", base_dir=systems_dir)
    engine = GameStateEngine(pack=cypher_pack)

    # Detect action manually
    action = engine.detect_action("je me repose")
    assert action is not None
    assert action.kind == "recover"
    assert action.sequence == "cypher_rests"
    assert action.trigger is None

    # Detect action type
    action_type = engine.detect_action_type("je me repose")
    assert action_type == "rest_seq:cypher_rests"

    # Different case and padding
    action_type = engine.detect_action_type("Je me repose un peu")
    assert action_type == "rest_seq:cypher_rests"

    # Unrelated text
    assert engine.detect_action_type("je récupère l'épée") is None

def test_engine_detect_action_type_trigger():
    from systems.loader import load_pack
    from pathlib import Path
    systems_dir = str(Path(__file__).parent.parent / "systems")
    dnd5e_pack = load_pack("dnd5e_srd", base_dir=systems_dir)
    engine = GameStateEngine(pack=dnd5e_pack)

    # dnd5e_srd has 'long_rest' trigger
    action_type = engine.detect_action_type("je prends un repos long")
    assert action_type == "rest:long_rest"
