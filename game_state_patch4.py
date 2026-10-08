import re

with open('game_state_engine.py', 'r') as f:
    code = f.read()

# Make _inner_rest handle save properly
search = """            self.synchronize_and_recalculate()
            self.save()
            return ActionResult(
                success=True,
                message=f"Rest '{trigger_id}' completed. Restored: {', '.join(restored) if restored else 'nothing'}.",
                state_changes={"rest": trigger_id, "restored": state_changes}
            )"""

replace = """            if sequence_info:
                set_by_path(self.state, sequence_info["path"], sequence_info["next_index"])
                state_changes["sequence"] = {
                    "id": sequence_info["id"],
                    "step": sequence_info["step"],
                    "next_index": sequence_info["next_index"]
                }

            self.synchronize_and_recalculate()
            self.save()

            final_changes = {"rest": trigger_id, "restored": state_changes.pop("restored", {})}
            final_changes.update(state_changes)

            msg = f"Rest '{trigger_id}' completed. Restored: {', '.join(restored) if restored else 'nothing'}."
            if sequence_info:
                msg += f" (step {sequence_info['index'] + 1}/{len(self.pack.resources.recovery_sequences[0].steps) if self.pack and getattr(self.pack.resources, 'recovery_sequences', []) else '?'})"

            return ActionResult(
                success=True,
                message=msg,
                state_changes=final_changes
            )"""

code = code.replace(search, replace)

with open('game_state_engine.py', 'w') as f:
    f.write(code)
print("Updated inner_rest to handle sequence_info")
