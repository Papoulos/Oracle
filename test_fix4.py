import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Fix mock for systems dir - getting rid of get_draft_dir since we don't mock it that way
# we only need os.environ["SYSTEM_PACK"] and monkeypatch SYSTEMS_DIR inside game_state_engine

code = code.replace("monkeypatch.setattr(\"game_state_engine.get_draft_dir\", lambda _: \"tests/fixtures/\" + os.environ[\"SYSTEM_PACK\"])", "")
code = code.replace("monkeypatch.setattr(\"systems.promote.SYSTEMS_DIR\", \"tests/fixtures\")", "monkeypatch.setattr(\"game_state_engine.SYSTEMS_DIR\", \"tests/fixtures\")")


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
  }
}""")

print("Fixes 4 applied")
