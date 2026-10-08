import re

with open('game_state_engine.py', 'r') as f:
    code = f.read()

# Add allocate_recovery method
add_method_code = """
    def allocate_recovery(self, allocation: dict[str, int]) -> ActionResult:
        \"\"\"
        Distributes a pending pooled recovery among chosen pools.
        \"\"\"
        pending = self.state.get("pending_allocation")
        if not pending:
            return ActionResult(success=False, message="No pending allocation to resolve.")

        among = set(pending["among"])
        points = pending["points"]

        # Validation
        for k, v in allocation.items():
            if k not in among:
                return ActionResult(success=False, message=f"Key '{k}' is not in the allowed targets ({', '.join(among)}).")
            if not isinstance(v, int) or v < 0:
                return ActionResult(success=False, message=f"Value for '{k}' must be a non-negative integer.")

        total_requested = sum(allocation.values())
        if total_requested > points:
            return ActionResult(success=False, message=f"Total allocated points ({total_requested}) exceed available points ({points}).")

        unallocated = points - total_requested

        # Calculate actual gains and wastes
        allocated = {}
        capped = {}

        for k, req in allocation.items():
            if req == 0:
                continue

            pool = next((p for p in self.pack.resources.pools if p.id == k), None)
            if not pool:
                continue

            curr = get_by_path(self.state, pool.current_path)
            max_val = pool.max_value if pool.max_value is not None else get_by_path(self.state, pool.max_path)

            if curr is None or max_val is None:
                continue

            missing = max_val - curr
            gain = min(req, missing)
            cap = req - gain

            if gain > 0:
                allocated[k] = {
                    "avant": curr,
                    "apres": curr + gain,
                    "gain": gain
                }
                set_by_path(self.state, pool.current_path, curr + gain)

            if cap > 0:
                capped[k] = cap

        wasted_total = unallocated + sum(capped.values())
        wasted = {
            "total": wasted_total,
            "unallocated": unallocated,
            "capped": capped
        }

        state_changes = {
            "allocated": allocated,
            "wasted": wasted,
            "rule_id": pending["rule_id"]
        }

        del self.state["pending_allocation"]
        self.synchronize_and_recalculate()
        self.save()

        return ActionResult(
            success=True,
            message="Recovery allocation applied successfully.",
            state_changes=state_changes
        )
"""

code = code.replace("    def _legacy_rest(self, rest_type: str = \"long\") -> ActionResult:", add_method_code + "\n    def _legacy_rest(self, rest_type: str = \"long\") -> ActionResult:")


with open('game_state_engine.py', 'w') as f:
    f.write(code)
print("Added allocate_recovery")
