import re
import json

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Fix mock for systems dir
code = code.replace("monkeypatch.setattr(\"systems.promote.SYSTEMS_DIR\", \"tests/fixtures\")", "monkeypatch.setenv(\"SYSTEMS_DIR\", \"tests/fixtures\")\\n    monkeypatch.setattr(\"systems.promote.SYSTEMS_DIR\", \"tests/fixtures\")")

with open('tests/test_pooled_recoveries.py', 'w') as f:
    f.write(code)

with open('tests/fixtures/cypher_recovery_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "tiers": [],
  "modifiers": {
    "attribute": 0,
    "skill": 0
  }
}""")

with open('tests/fixtures/cypher_recovery_auto_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "tiers": [],
  "modifiers": {
    "attribute": 0,
    "skill": 0
  }
}""")

print("Fixes 2 applied")
