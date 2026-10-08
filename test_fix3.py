import re

# Need to monkeypatch SYSTEMS_DIR inside systems.promote where get_draft_dir resides

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

code = code.replace("monkeypatch.setattr(\"systems.promote.SYSTEMS_DIR\", \"tests/fixtures\")", "monkeypatch.setattr(\"systems.promote.SYSTEMS_DIR\", \"tests/fixtures\")\\n    monkeypatch.setattr(\"game_state_engine.get_draft_dir\", lambda _: \"tests/fixtures/\" + os.environ[\"SYSTEM_PACK\"])")


with open('tests/test_pooled_recoveries.py', 'w') as f:
    f.write(code)

with open('tests/fixtures/cypher_recovery_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "tiers": [],
  "modifiers": {
    "attribute": 0,
    "skill": 0,
    "situational": 0,
    "gear": 0
  }
}""")

with open('tests/fixtures/cypher_recovery_auto_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "tiers": [],
  "modifiers": {
    "attribute": 0,
    "skill": 0,
    "situational": 0,
    "gear": 0
  }
}""")

print("Fixes 3 applied")
