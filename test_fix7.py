import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Ah! GameStateEngine doesn't load the pack itself via SYSTEM_PACK, it's passed in!
# Let's fix the tests to load the pack and pass it to the engine.

replace = """
from systems.system_pack import SystemPack

def get_engine(character_file, pack_id):
    pack = SystemPack(f"tests/fixtures/{pack_id}")
    return GameStateEngine(character_file, pack=pack)
"""

code = code.replace("from systems.validate import validate_pack", "from systems.validate import validate_pack\nfrom mechanics.system_pack import SystemPack\n\ndef get_engine(character_file, pack_id):\n    pack = SystemPack(f\"tests/fixtures/{pack_id}\")\n    return GameStateEngine(character_file, pack=pack)\n")

code = code.replace("engine = GameStateEngine(character_file)", "engine = get_engine(character_file, os.environ['SYSTEM_PACK'])")


with open('tests/test_pooled_recoveries.py', 'w') as f:
    f.write(code)

print("Fixes 7 applied")
