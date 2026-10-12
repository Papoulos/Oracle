import os
import random
from systems.validate import validate_pack
from game_state_engine import GameStateEngine
from systems.loader import load_pack
from mechanics.models import ResolutionRequest, ResolutionResult, ResolutionConfig, Outcome
from mechanics.resolution import resolve
from pydantic import TypeAdapter

def test_cypher_system_valid():
    pack_path = os.path.join(os.path.dirname(__file__), "..", "systems", "cypher_system")
    issues = validate_pack(pack_path)
    assert len(issues) == 0, f"Expected no issues (strict), got: {issues}"

def test_cypher_system_rest(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEM_PACK", "cypher_system")

    char_file = tmp_path / "char.json"
    char_file.write_text("{}") # dummy init

    pack = load_pack("cypher_system", os.path.join(os.path.dirname(__file__), "..", "systems"))
    engine = GameStateEngine(str(char_file), pack=pack)

    engine.state = {
        "tier": 2,
        "recovery_step": 0, # The state_path is "recovery_step", not "resources.cypher_rests.recovery_step"
        "resources": {
            "might": {"current": 1, "max": 12},
            "speed": {"current": 4, "max": 10},
            "intellect": {"current": 10, "max": 10}
        }
    }

    engine.synchronize_and_recalculate()
    engine.rng = random.Random(11)

    allocations = ["might", "speed", "might", "speed"]
    # expected points based on "d(1, 6) + character.tier (2)"
    # random.Random(11).randint(1, 6) gives 4, 5, 4, 4
    # so points = 6, 7, 6, 6
    expected_points = [6, 7, 6, 6]
    expected_gains = [6, 6, 5, 0] # actual gain due to capping: might:1+6=7, speed:4+7=11->cap10(gain 6), might:7+6=13->cap12(gain 5), speed:10+6=16->cap10(gain 0)
    expected_wasted = [0, 1, 1, 6]

    for i in range(4):
        engine.next_rest("cypher_rests")
        points = engine.state.get("pending_allocation", {}).get("points", 0)
        assert points == expected_points[i]

        alloc_res = engine.allocate_recovery({allocations[i]: points})

        if alloc_res.state_changes:
            gained = 0
            if alloc_res.state_changes.get("allocated", {}).get(allocations[i]):
                gained = alloc_res.state_changes["allocated"][allocations[i]]["gain"]
            assert gained == expected_gains[i]
            assert alloc_res.state_changes["wasted"]["total"] == expected_wasted[i]

    assert engine.state["resources"]["might"]["current"] == 12
    assert engine.state["resources"]["speed"]["current"] == 10
    assert engine.state["resources"]["intellect"]["current"] == 10
    assert engine.state["recovery_step"] == 0

def test_cypher_system_resolution():
    import json
    pack_path = os.path.join(os.path.dirname(__file__), "..", "systems", "cypher_system", "resolution.json")
    with open(pack_path, "r") as f:
        config_data = json.load(f)

    adapter = TypeAdapter(ResolutionConfig)
    config = adapter.validate_python(config_data)

    req1 = ResolutionRequest(
        difficulty=4,
        modifier=0,
        step_adjustments={"skill": -1, "effort": -1}
    )
    rng = random.Random(42)
    res1 = resolve(config, req1, rng)
    # diff 4, adj -2 -> final diff 2 -> target 6 (diff * 3)
    assert res1.target == 6

    req2 = ResolutionRequest(
        difficulty=2,
        step_adjustments={"skill": -2}
    )
    res2 = resolve(config, req2, rng)
    # diff 2, adj -2 -> final diff 0 -> automatic success
    assert res2.outcome == Outcome.SUCCESS
    assert res2.rolled == []

    req3 = ResolutionRequest(
        difficulty=7,
        step_adjustments={}
    )
    res3 = resolve(config, req3, rng)
    # diff 7, adj 0 -> final diff 7 -> target 21
    assert res3.target == 21
