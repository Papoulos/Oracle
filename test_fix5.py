import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Fix mock for systems dir - getting rid of getattr for game_state_engine.SYSTEMS_DIR
# game_state_engine doesn't define SYSTEMS_DIR. it imports get_draft_dir from systems.promote.
# so just setting SYSTEM_PACK and SYSTEMS_DIR env variables is enough.

code = code.replace("monkeypatch.setattr(\"game_state_engine.SYSTEMS_DIR\", \"tests/fixtures\")", "")


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
  },
  "advantages": {
    "boons": 0,
    "banes": 0
  },
  "step_modifiers": {}
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
  },
  "advantages": {
    "boons": 0,
    "banes": 0
  },
  "step_modifiers": {}
}""")

print("Fixes 5 applied")
