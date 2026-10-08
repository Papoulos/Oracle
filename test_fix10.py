import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Ah SystemPack loading in loader.py expects load_system_pack() not SystemPack() constructor probably.
# Let's check loader.py

replace = """
from systems.loader import load_system_pack
def get_engine(character_file, pack_id):
    pack = load_system_pack(pack_id)
    return GameStateEngine(character_file, pack=pack)
"""

code = code.replace("from systems.loader import SystemPack\n\ndef get_engine(character_file, pack_id):\n    pack = SystemPack(f\"tests/fixtures/{pack_id}\")\n    return GameStateEngine(character_file, pack=pack)", "from systems.loader import load_system_pack\n\ndef get_engine(character_file, pack_id):\n    pack = load_system_pack(pack_id)\n    return GameStateEngine(character_file, pack=pack)")


with open('tests/test_pooled_recoveries.py', 'w') as f:
    f.write(code)

with open('tests/fixtures/cypher_recovery_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "modifiers": {}
}""")

with open('tests/fixtures/cypher_recovery_auto_pack/resolution.json', 'w') as f:
    f.write("""{
  "family": "StepTargetD20",
  "modifiers": {}
}""")

print("Fixes 10 applied")
