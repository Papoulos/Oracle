import re

with open('game_state_engine.py', 'r') as f:
    code = f.read()

# Fix sequence length calculation logic in msg formatting to be robust against missing sequences
# and find the correct one

search = """            msg = f"Rest '{trigger_id}' completed. Restored: {', '.join(restored) if restored else 'nothing'}."
            if sequence_info:
                msg += f" (step {sequence_info['index'] + 1}/{len(self.pack.resources.recovery_sequences[0].steps) if self.pack and getattr(self.pack.resources, 'recovery_sequences', []) else '?'})\""""

replace = """            msg = f"Rest '{trigger_id}' completed. Restored: {', '.join(restored) if restored else 'nothing'}."
            if sequence_info:
                seq = next((s for s in getattr(self.pack.resources, 'recovery_sequences', []) if s.id == sequence_info['id']), None)
                msg += f" (step {sequence_info['index'] + 1}/{len(seq.steps) if seq else '?'})\""""

code = code.replace(search, replace)

with open('game_state_engine.py', 'w') as f:
    f.write(code)
print("Fixed sequence msg formatting in game_state_engine.py")
