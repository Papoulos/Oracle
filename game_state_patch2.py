import re

with open('game_state_engine.py', 'r') as f:
    code = f.read()

# Add pending check in rest
search = """        if self.pack:
            # Check if recovery_rules.json exists and log warning"""
replace = """        if self.pack:
            if "pending_allocation" in self.state:
                return ActionResult(success=False, message="Cannot rest while there is a pending recovery allocation.")

            # Check if recovery_rules.json exists and log warning"""
code = code.replace(search, replace)

# Add pooled_recoveries handling
search2 = """            # Apply changes
            for change in pending_changes:"""
replace2 = """            pooled_changes = []
            pending_alloc = None

            for pooled_rule in getattr(self.pack.resources, "pooled_recoveries", []):
                if trigger_id in pooled_rule.triggers:
                    try:
                        points = max(0, evaluate_int(pooled_rule.points_expr, {"character": self.state}, self.rng))

                        if pooled_rule.allocation == "auto_in_order":
                            projected_values = {}
                            max_values = {}

                            for pool_id in pooled_rule.among:
                                pool = next((p for p in self.pack.resources.pools if p.id == pool_id), None)
                                if pool:
                                    max_val = pool.max_value if pool.max_value is not None else get_by_path(self.state, pool.max_path)
                                    curr_val = get_by_path(self.state, pool.current_path)
                                    if max_val is not None and curr_val is not None:
                                        projected_values[pool.id] = curr_val
                                        max_values[pool.id] = max_val

                            for change in pending_changes:
                                if change["type"] == "pool":
                                    pid = change["pool"].id
                                    if pid in projected_values:
                                        projected_values[pid] = change["new_val"]
                                elif change["type"] == "group_pool":
                                    pid = change["pool"].id
                                    if pid in projected_values:
                                        projected_values[pid] = change["new_val"]

                            allocated = {}
                            unallocated = points

                            for pool_id in pooled_rule.among:
                                if pool_id not in projected_values:
                                    continue

                                missing = max_values[pool_id] - projected_values[pool_id]
                                if missing > 0 and unallocated > 0:
                                    gain = min(missing, unallocated)
                                    avant = projected_values[pool_id]
                                    projected_values[pool_id] += gain
                                    unallocated -= gain

                                    allocated[pool_id] = {
                                        "avant": avant,
                                        "apres": projected_values[pool_id],
                                        "gain": gain
                                    }

                                    pool = next(p for p in self.pack.resources.pools if p.id == pool_id)
                                    pooled_changes.append({
                                        "type": "pool",
                                        "pool": pool,
                                        "current": avant,
                                        "new_val": projected_values[pool_id],
                                        "gain": gain
                                    })

                            state_changes.setdefault("pooled", {})[pooled_rule.id] = {
                                "allocated": allocated,
                                "wasted": {
                                    "total": unallocated,
                                    "unallocated": unallocated,
                                    "capped": {}
                                }
                            }

                        elif pooled_rule.allocation == "player_choice":
                            pending_alloc = {
                                "rule_id": pooled_rule.id,
                                "trigger": trigger_id,
                                "points": points,
                                "among": pooled_rule.among
                            }
                    except ExprError as e:
                        return ActionResult(success=False, message=f"Recovery formula failed for pooled rule '{pooled_rule.id}': {e}")

            if pending_alloc:
                self.state["pending_allocation"] = pending_alloc
                state_changes["pending_allocation"] = pending_alloc

            # Apply changes
            for change in pending_changes + pooled_changes:"""
code = code.replace(search2, replace2)


with open('game_state_engine.py', 'w') as f:
    f.write(code)
print("Added pooled_recoveries handling")
