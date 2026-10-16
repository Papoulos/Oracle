import pytest
import os
import shutil
import json
from systems.validate import validate_pack

@pytest.fixture
def base_pack(tmp_path):
    pack_dir = tmp_path / "cypher_system"
    shutil.copytree("systems/cypher_system", pack_dir)
    return str(pack_dir)

def test_validate_actions_intact_cypher(base_pack):
    issues = validate_pack(base_pack)
    assert not any(i.severity == "error" for i in issues)

def test_validate_actions_no_actions_json(base_pack):
    os.remove(os.path.join(base_pack, "actions.json"))
    issues = validate_pack(base_pack)
    # actions.json is optional, should not report issues for it missing
    assert not any(i.file == "actions.json" and i.severity == "error" for i in issues)

def modify_actions(pack_path, modifier_fn):
    actions_path = os.path.join(pack_path, "actions.json")
    with open(actions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    modifier_fn(data)
    with open(actions_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def test_validate_actions_duplicate_action_id(base_pack):
    def mod(d):
        d["actions"].append(d["actions"][0].copy())
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and i.file == "actions.json" and "Duplicate action id" in i.message), None)
    assert issue is not None
    assert "/actions/" in issue.path

def test_validate_actions_inexistent_pool(base_pack):
    def mod(d):
        d["actions"][0]["pools"] = ["invalid_pool"]
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and "not found in resources.json" in i.message), None)
    assert issue is not None

def test_validate_actions_duplicate_difficulty_level(base_pack):
    def mod(d):
        d["difficulty_scale"].append(d["difficulty_scale"][0].copy())
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and "Duplicate difficulty level" in i.message), None)
    assert issue is not None

def test_validate_actions_level_out_of_bounds(base_pack):
    def mod(d):
        d["difficulty_scale"].append({"level": 11, "name": "Impossible"})
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and "out of resolution bounds" in i.message), None)
    assert issue is not None

def test_validate_actions_incomplete_scale(base_pack):
    def mod(d):
        d["difficulty_scale"].pop()
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "warning" and "does not cover all levels" in i.message), None)
    assert issue is not None

def test_validate_actions_invalid_variable(base_pack):
    def mod(d):
        # Find an action with effort
        for a in d["actions"]:
            if a.get("effort"):
                a["effort"]["cost_expr"] = "levels.foo"
                break
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and "Invalid variable" in i.message), None)
    assert issue is not None

def test_validate_actions_invalid_formula(base_pack):
    def mod(d):
        for a in d["actions"]:
            if a.get("effort"):
                a["effort"]["cost_expr"] = "3 +"
                break
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and i.path.endswith("cost_expr") and "3 +" not in i.message), None)
    # The message should come from pydantic ValidationError or ExprError
    assert any(i.severity == "error" and i.path.endswith("cost_expr") for i in issues)

def test_validate_actions_invalid_path(base_pack):
    def mod(d):
        for a in d["actions"]:
            if a.get("effort"):
                a["effort"]["max_levels_path"] = "a/b"
                break
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and "Invalid pointer path syntax" in i.message), None)
    assert issue is not None

def test_validate_actions_adjustment_key_absent(base_pack):
    def mod(d):
        for a in d["actions"]:
            if a.get("effort"):
                a["effort"]["adjustment_key"] = "missing_key"
                break
    modify_actions(base_pack, mod)
    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "warning" and "not found in resolution.adjustment_caps" in i.message), None)
    assert issue is not None

def test_validate_actions_family_d20vstarget(base_pack):
    # Change resolution.json to D20VsTarget
    res_path = os.path.join(base_pack, "resolution.json")
    with open(res_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    res["family"] = "D20VsTarget"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f)

    issues = validate_pack(base_pack)
    issue = next((i for i in issues if i.severity == "error" and i.file == "actions.json" and "actions require StepTargetD20 resolution" in i.message), None)
    assert issue is not None
