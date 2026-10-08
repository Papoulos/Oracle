import re

with open('game_state_engine.py', 'r') as f:
    code = f.read()

add_next_rest = """
    def next_rest(self, sequence_id: str) -> ActionResult:
        \"\"\"
        Advances and applies the next rest in a defined recovery sequence.
        \"\"\"
        if not self.pack or not getattr(self.pack.resources, "recovery_sequences", []):
            return ActionResult(success=False, message="No recovery sequences defined in this pack.")

        if "pending_allocation" in self.state:
            return ActionResult(success=False, message="Cannot execute sequence while there is a pending recovery allocation.")

        seq = next((s for s in self.pack.resources.recovery_sequences if s.id == sequence_id), None)
        if not seq:
            valid_seqs = [s.id for s in self.pack.resources.recovery_sequences]
            return ActionResult(success=False, message=f"Unknown recovery sequence '{sequence_id}'. Valid sequences are: {', '.join(valid_seqs)}.")

        # Read index
        current_index = get_by_path(self.state, seq.state_path)
        if current_index is None:
            current_index = 0
        elif not isinstance(current_index, int) or isinstance(current_index, bool) or current_index < 0:
            logger.warning(f"Invalid state for sequence index at '{seq.state_path}'. Defaulting to 0.")
            current_index = 0
        elif current_index >= len(seq.steps):
            logger.warning(f"Sequence index at '{seq.state_path}' out of bounds. Wrapping around.")
            current_index = current_index % len(seq.steps)

        trigger = seq.steps[current_index]
        next_index = (current_index + 1) % len(seq.steps)

        # We need to execute the rest without saving, append our sequence changes, then save
        # To do this safely without rewriting the whole rest(), we can temporarily patch save()
        # But a cleaner way is just to call a modified internal _apply_rest

        return self._inner_rest(trigger, sequence_info={"id": sequence_id, "step": trigger, "index": current_index, "next_index": next_index, "path": seq.state_path})

"""

code = code.replace("    def rest(self, trigger_id: str = \"long\") -> ActionResult:", add_next_rest + "    def rest(self, trigger_id: str = \"long\") -> ActionResult:\n        return self._inner_rest(trigger_id)\n\n    def _inner_rest(self, trigger_id: str = \"long\", sequence_info: dict = None) -> ActionResult:")

with open('game_state_engine.py', 'w') as f:
    f.write(code)
print("Added next_rest method structure")
