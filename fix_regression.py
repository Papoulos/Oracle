import re

with open('game_state_engine.py', 'r') as f:
    code = f.read()

# Fix regression: the classic pool modifications must go into `restored` sub-dictionary in `state_changes`
# The agent changed it to:
# final_changes = {"rest": trigger_id, "restored": state_changes.pop("restored", {})}
# final_changes.update(state_changes)
# Which flattens state_changes.
# We should keep `state_changes` contents inside `restored` key, EXCEPT for pending_allocation, pooled, sequence

search = """            final_changes = {"rest": trigger_id, "restored": state_changes.pop("restored", {})}
            final_changes.update(state_changes)"""

replace = """            final_changes = {"rest": trigger_id, "restored": {}}
            for key, val in state_changes.items():
                if key in ("pending_allocation", "pooled", "sequence", "restored"):
                    final_changes[key] = val
                else:
                    final_changes["restored"][key] = val"""

code = code.replace(search, replace)

with open('game_state_engine.py', 'w') as f:
    f.write(code)


# Revert tampered tests
with open('tests/test_engine_formula_recovery.py', 'r') as f:
    code = f.read()
code = code.replace("res.state_changes[\"hit_dice\"]", "res.state_changes[\"restored\"][\"hit_dice\"]")
code = code.replace("res.state_changes[\"stamina\"]", "res.state_changes[\"restored\"][\"stamina\"]")
with open('tests/test_engine_formula_recovery.py', 'w') as f:
    f.write(code)

print("Regression fixed")
