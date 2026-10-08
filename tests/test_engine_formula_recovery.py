import pytest
import random
import os
import shutil
import json
from game_state_engine import GameStateEngine
from systems.loader import load_pack
from mechanics.models import Manifest, ResolutionConfig, ResourcesConfig, PoolDef, RecoveryRule, TriggersConfig
from mechanics.models import D20VsTargetConfig

@pytest.fixture
def test_dir(tmp_path):
    d = tmp_path / "formula_pack"
    d.mkdir()

    # Manifest
    manifest = {
        "id": "formula_pack",
        "name": "Test Pack",
        "version": "1.0.0",
        "language": "en",
        "family": "D20VsTarget",
        "source_pdfs": []
    }
    with open(d / "manifest.yaml", "w") as f:
        import yaml
        yaml.dump(manifest, f)

    # Resolution
    res = {
        "family": "D20VsTarget",
        "advantage_enabled": True,
        "advantage_dice": 2,
        "critical_success_on": 20,
        "critical_failure_on": 1
    }
    with open(d / "resolution.json", "w") as f:
        json.dump(res, f)

    # Resources
    resources = {
        "recovery_triggers": [
            {"id": "long_rest", "name": "Long Rest"},
            {"id": "short_rest", "name": "Short Rest"}
        ],
        "pools": [
            {
                "id": "hit_dice",
                "name": "Hit Dice",
                "kind": "counter",
                "min_value": 0,
                "current_path": "resources.hit_dice.current",
                "max_path": "resources.hit_dice.max",
                "recovery": [
                    {"trigger": "long_rest", "mode": "formula", "amount_expr": "max(1, floor(pool.max / 2))"}
                ]
            },
            {
                "id": "stamina",
                "name": "Stamina",
                "kind": "counter",
                "min_value": 0,
                "current_path": "resources.stamina.current",
                "max_path": "resources.stamina.max",
                "recovery": [
                    {"trigger": "short_rest", "mode": "formula", "amount_expr": "d(1, 6) + character.tier"}
                ]
            },
            {
                "id": "health_pool",
                "name": "Health Pool",
                "kind": "health",
                "min_value": 0,
                "current_path": "resources.hit_points.current",
                "max_path": "resources.hit_points.max",
                "recovery": [
                    {"trigger": "short_rest", "mode": "formula", "amount_expr": "10"}
                ]
            }
        ],
        "pool_groups": []
    }
    with open(d / "resources.json", "w") as f:
        json.dump(resources, f)

    with open(d / "triggers.json", "w") as f:
        json.dump({"version": 1, "rules": []}, f)

    return d

@pytest.fixture
def engine(test_dir, tmp_path):
    pack = load_pack("formula_pack", base_dir=str(test_dir.parent))
    char_path = tmp_path / "character.json"
    state = {
        "tier": 2,
        "pv": 10,
        "resources": {
            "hit_dice": {"current": 2, "max": 7},
            "stamina": {"current": 0, "max": 20},
            "hit_points": {"current": 10, "max": 50}
        }
    }
    with open(char_path, "w") as f:
        json.dump(state, f)

    rng = random.Random(42)
    return GameStateEngine(character_path=str(char_path), pack=pack, rng=rng)


def test_rest_formula_logic(engine):
    # hit_dice max 7, current 2 -> floor(7/2) = 3 -> 5
    res = engine.rest("long_rest")
    assert res.success
    assert engine.state["resources"]["hit_dice"]["current"] == 5

    # check state_changes gain
    assert res.state_changes["restored"]["hit_dice"]["gain"] == 3

    # capped at max
    engine.state["resources"]["hit_dice"]["current"] = 6
    res = engine.rest("long_rest")
    assert res.success
    assert engine.state["resources"]["hit_dice"]["current"] == 7
    assert res.state_changes["restored"]["hit_dice"]["gain"] == 1

    # max 1, current 0 -> 1
    engine.state["resources"]["hit_dice"]["current"] = 0
    engine.state["resources"]["hit_dice"]["max"] = 1
    res = engine.rest("long_rest")
    assert res.success
    assert engine.state["resources"]["hit_dice"]["current"] == 1
    assert res.state_changes["restored"]["hit_dice"]["gain"] == 1

def test_stamina_rng_formula(engine):
    # stamina logic: d(1, 6) + character.tier
    # deterministic seed 42 rolls a 6 on d6
    # 6 + 2 (tier) = 8
    res = engine.rest("short_rest")
    assert res.success
    assert engine.state["resources"]["stamina"]["current"] == 8
    assert res.state_changes["restored"]["stamina"]["gain"] == 8

    # negative tier capped at 0 minimum gain
    engine.state["tier"] = -10
    engine.state["resources"]["stamina"]["current"] = 0
    # RNG advances. Next d6 from seed 42. Let's just say it rolls x, x - 10 <= 0 -> gain 0
    res = engine.rest("short_rest")
    assert res.success
    assert engine.state["resources"]["stamina"]["current"] == 0
    # Because there is no change, there is no key added to restored
    assert "stamina" not in res.state_changes["restored"]

def test_atomicity_error(engine):
    # Remove tier to force ExprError on stamina short_rest
    del engine.state["tier"]

    # Current values
    stamina_cur = engine.state["resources"]["stamina"]["current"]
    hp_cur = engine.state["resources"]["hit_points"]["current"]

    res = engine.rest("short_rest")
    assert not res.success
    assert "Recovery formula failed for 'stamina'" in res.message

    # Ensure neither stamina NOR health_pool changed
    assert engine.state["resources"]["stamina"]["current"] == stamina_cur
    assert engine.state["resources"]["hit_points"]["current"] == hp_cur

def test_pv_mirroring(engine):
    assert engine.state["pv"] == 10
    res = engine.rest("short_rest")
    assert res.success
    assert engine.state["resources"]["hit_points"]["current"] == 20
    assert engine.state["pv"] == 20

from systems.validate import validate_pack

def test_validate_pack_formula(test_dir):
    # Invalid formula rules
    resources_path = test_dir / "resources.json"
    with open(resources_path, "r") as f:
        res = json.load(f)

    res["pools"][0]["recovery"][0]["amount_expr"] = "max(1, floor(pool.max / 2)) + unknown_var"
    res["pools"][1]["recovery"][0]["mode"] = "formula"
    del res["pools"][1]["recovery"][0]["amount_expr"]

    with open(resources_path, "w") as f:
        json.dump(res, f)

    issues = validate_pack(str(test_dir))
    errors = [i for i in issues if i.severity == "error"]
    assert any("amount_expr must be provided when mode is 'formula'" in e.message for e in errors)
    assert any("Allowed: variables starting with 'character.' or one of pool.current, pool.max, pool.min." in e.message for e in errors)
    assert any("unknown_var" in e.message for e in errors)
