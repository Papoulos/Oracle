import re

with open('game_state_engine.py', 'r') as f:
    code = f.read()

# Fix allocate_recovery missing can be negative
search = "            missing = max_val - curr\n            gain = min(req, missing)"
replace = "            missing = max(0, max_val - curr)\n            gain = min(req, missing)"

code = code.replace(search, replace)
with open('game_state_engine.py', 'w') as f:
    f.write(code)

print("Gain fixed")
