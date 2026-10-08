import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

code = code.replace("from systems.system_pack import SystemPack", "from systems.loader import SystemPack")

with open('tests/test_pooled_recoveries.py', 'w') as f:
    f.write(code)

print("Fixes 9 applied")
