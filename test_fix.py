import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Fix monkeypatch
code = code.replace("monkeypatch.setattr(\"game_state_engine.SYSTEMS_DIR\", \"tests/fixtures\")", "monkeypatch.setattr(\"systems.promote.SYSTEMS_DIR\", \"tests/fixtures\")")

with open('tests/test_pooled_recoveries.py', 'w') as f:
    f.write(code)

with open('tests/fixtures/cypher_recovery_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "modifiers": {
    "attribute": 0,
    "skill": 0
  }
}""")

with open('tests/fixtures/cypher_recovery_auto_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "modifiers": {
    "attribute": 0,
    "skill": 0
  }
}""")

print("Fixes applied")
