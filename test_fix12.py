import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Pass base_dir to load_pack!
code = code.replace("load_pack(pack_id)", "load_pack(pack_id, base_dir='tests/fixtures')")

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
  "step_modifiers": {
    "trainings": 0,
    "assets": 0,
    "efforts": 0,
    "inability": 0
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
  },
  "advantages": {
    "boons": 0,
    "banes": 0
  },
  "step_modifiers": {
    "trainings": 0,
    "assets": 0,
    "efforts": 0,
    "inability": 0
  }
}""")


print("Fixes 12 applied")
