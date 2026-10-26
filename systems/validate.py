import os
import json
import yaml
import sys
import argparse
from typing import Literal, Any
import re
from pydantic import BaseModel, ValidationError, TypeAdapter

from mechanics.models import Manifest, ResolutionConfig, ResourcesConfig, PbtA2d6Config, StepTargetD20Config, TriggersConfig
from mechanics.actions import ActionsConfig
from mechanics.character import CharacterSchemaConfig, CreationManualConfig
from mechanics.expr import compile_expr, referenced_variables, ExprError


class Issue(BaseModel):
    severity: Literal["error", "warning"]
    file: str
    path: str
    message: str


def check_for_todos(data: Any, file_name: str, path: str, issues: list[Issue]):
    """Recursively checks for the literal string 'TODO' in the data."""
    if isinstance(data, str):
        if data == "TODO":
            issues.append(Issue(severity="warning", file=file_name, path=path, message="Field contains 'TODO'"))
    elif isinstance(data, dict):
        for key, value in data.items():
            check_for_todos(value, file_name, f"{path}.{key}" if path else key, issues)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            check_for_todos(item, file_name, f"{path}[{i}]", issues)


def validate_pack(pack_path: str) -> list[Issue]:
    issues: list[Issue] = []

    if not os.path.isdir(pack_path):
        issues.append(Issue(severity="error", file="N/A", path=pack_path, message=f"Pack path is not a directory: {pack_path}"))
        return issues

    manifest_path = os.path.join(pack_path, "manifest.yaml")
    resolution_path = os.path.join(pack_path, "resolution.json")
    resources_path = os.path.join(pack_path, "resources.json")

    # 1. Manifest
    if not os.path.isfile(manifest_path):
        issues.append(Issue(severity="error", file="manifest.yaml", path=manifest_path, message="Missing manifest.yaml"))
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = yaml.safe_load(f)

            manifest = Manifest.model_validate(manifest_data)
            check_for_todos(manifest_data, "manifest.yaml", "", issues)
        except Exception as e:
            issues.append(Issue(severity="error", file="manifest.yaml", path=manifest_path, message=str(e)))

    # 2. Resolution Config
    config_adapter = TypeAdapter(ResolutionConfig)
    if not os.path.isfile(resolution_path):
        issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message="Missing resolution.json"))
    else:
        try:
            with open(resolution_path, "r", encoding="utf-8") as f:
                resolution_data = json.load(f)

            resolution = config_adapter.validate_python(resolution_data)
            check_for_todos(resolution_data, "resolution.json", "", issues)

            # Additional logic checks
            if isinstance(resolution, PbtA2d6Config):
                # Check for gaps and overlaps in tiers
                tiers = resolution.tiers
                # We expect tiers to cover -inf to inf without overlaps
                # First, ensure they are sorted by min

                # To properly sort, we treat None as -inf for min, and inf for max
                def get_min(t): return t.min if t.min is not None else float('-inf')
                def get_max(t): return t.max if t.max is not None else float('inf')

                sorted_tiers = sorted(tiers, key=get_min)

                if not sorted_tiers:
                    issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message="PbtA2d6Config must have at least one tier"))
                else:
                    if get_min(sorted_tiers[0]) != float('-inf'):
                        issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message="Tiers do not cover negative infinity (missing unbounded lower tier)"))

                    if get_max(sorted_tiers[-1]) != float('inf'):
                        issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message="Tiers do not cover positive infinity (missing unbounded upper tier)"))

                    for i in range(len(sorted_tiers) - 1):
                        current_max = get_max(sorted_tiers[i])
                        next_min = get_min(sorted_tiers[i+1])

                        if current_max == float('inf'):
                            issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message="Overlap detected: intermediate tier has infinite max"))
                            break

                        if current_max + 1 < next_min:
                            issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message=f"Gap detected between {current_max} and {next_min}"))
                        elif current_max >= next_min:
                            issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message=f"Overlap detected between {current_max} and {next_min}"))

        except ValidationError as e:
             issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message=str(e)))
        except json.JSONDecodeError as e:
            issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message=f"Invalid JSON: {e}"))
        except Exception as e:
             issues.append(Issue(severity="error", file="resolution.json", path=resolution_path, message=str(e)))

    # 3. Resources Config
    resources = None
    if not os.path.isfile(resources_path):
        issues.append(Issue(severity="error", file="resources.json", path=resources_path, message="Missing resources.json"))
    else:
        try:
            with open(resources_path, "r", encoding="utf-8") as f:
                resources_data = json.load(f)

            # First pass: Validate formulas even if resources validation fails
            for i, pool in enumerate(resources_data.get("pools", [])):
                for j, rule in enumerate(pool.get("recovery", [])):
                    if rule.get("mode") == "formula" and rule.get("amount_expr"):
                        path = f"/pools/{i}/recovery/{j}/amount_expr"
                        try:
                            compile_expr(rule["amount_expr"])
                            vars_ref = referenced_variables(rule["amount_expr"])
                            for var in vars_ref:
                                if not var.startswith("character.") and var not in ("pool.current", "pool.max", "pool.min"):
                                    issues.append(Issue(
                                        severity="error",
                                        file="resources.json",
                                        path=path,
                                        message=f"Invalid variable '{var}' in recovery rule (pool '{pool.get('id', 'unknown')}', trigger '{rule.get('trigger', 'unknown')}'). Allowed: variables starting with 'character.' or one of pool.current, pool.max, pool.min."
                                    ))
                        except ExprError as e:
                            issues.append(Issue(severity="error", file="resources.json", path=path, message=str(e)))

            for i, group in enumerate(resources_data.get("pool_groups", [])):
                for j, rule in enumerate(group.get("recovery", [])):
                    if rule.get("mode") == "formula" and rule.get("amount_expr"):
                        path = f"/pool_groups/{i}/recovery/{j}/amount_expr"
                        try:
                            compile_expr(rule["amount_expr"])
                            vars_ref = referenced_variables(rule["amount_expr"])
                            for var in vars_ref:
                                if not var.startswith("character.") and var not in ("pool.current", "pool.max", "pool.min"):
                                    issues.append(Issue(
                                        severity="error",
                                        file="resources.json",
                                        path=path,
                                        message=f"Invalid variable '{var}' in recovery rule (pool_group '{group.get('id', 'unknown')}', trigger '{rule.get('trigger', 'unknown')}'). Allowed: variables starting with 'character.' or one of pool.current, pool.max, pool.min."
                                    ))
                        except ExprError as e:
                            issues.append(Issue(severity="error", file="resources.json", path=path, message=str(e)))

            for i, rule in enumerate(resources_data.get("pooled_recoveries", [])):
                if rule.get("points_expr"):
                    path = f"/pooled_recoveries/{i}/points_expr"
                    try:
                        compile_expr(rule["points_expr"])
                        vars_ref = referenced_variables(rule["points_expr"])
                        for var in vars_ref:
                            if not var.startswith("character."):
                                issues.append(Issue(
                                    severity="error",
                                    file="resources.json",
                                    path=path,
                                    message=f"Invalid variable '{var}' in pooled recovery rule '{rule.get('id', 'unknown')}'. Allowed: variables starting with 'character.'."
                                ))
                    except ExprError as e:
                        issues.append(Issue(severity="error", file="resources.json", path=path, message=str(e)))

            resources = ResourcesConfig.model_validate(resources_data)

            # Cross-validation for resources
            recovery_triggers = {t.id for t in resources.recovery_triggers}
            pools = {p.id: p for p in resources.pools}

            # Check unique IDs in pooled_recoveries and recovery_sequences
            pooled_ids = set()
            for i, rule in enumerate(resources.pooled_recoveries):
                if rule.id in pooled_ids:
                    issues.append(Issue(severity="error", file="resources.json", path=f"/pooled_recoveries/{i}", message=f"Duplicate pooled_recovery id: {rule.id}"))
                pooled_ids.add(rule.id)

                # Check triggers exist
                for trigger in rule.triggers:
                    if trigger not in recovery_triggers:
                        issues.append(Issue(severity="error", file="resources.json", path=f"/pooled_recoveries/{i}", message=f"Trigger '{trigger}' not found in recovery_triggers"))

                # Check among targets valid pools
                for target in rule.among:
                    if target not in pools:
                        issues.append(Issue(severity="error", file="resources.json", path=f"/pooled_recoveries/{i}", message=f"Target pool '{target}' not found in pools"))
                    elif pools[target].kind not in ("health", "counter"):
                        issues.append(Issue(severity="error", file="resources.json", path=f"/pooled_recoveries/{i}", message=f"Target pool '{target}' must be of kind 'health' or 'counter'"))

            sequence_ids = set()
            for i, seq in enumerate(resources.recovery_sequences):
                if seq.id in sequence_ids:
                    issues.append(Issue(severity="error", file="resources.json", path=f"/recovery_sequences/{i}", message=f"Duplicate recovery_sequence id: {seq.id}"))
                sequence_ids.add(seq.id)

                # Check steps exist in triggers
                for step in seq.steps:
                    if step not in recovery_triggers:
                        issues.append(Issue(severity="error", file="resources.json", path=f"/recovery_sequences/{i}", message=f"Step '{step}' not found in recovery_triggers"))

                # Check state_path syntax (pointer path syntax without resources. prefix)
                if not re.match(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$', seq.state_path):
                    issues.append(Issue(severity="error", file="resources.json", path=f"/recovery_sequences/{i}", message=f"Invalid state_path syntax: {seq.state_path}"))
                if seq.state_path.startswith("resources."):
                    issues.append(Issue(severity="error", file="resources.json", path=f"/recovery_sequences/{i}", message=f"state_path should not start with 'resources.': {seq.state_path}"))

            # Warn if a recovery_trigger has no rules
            used_triggers = set()
            for pool in resources.pools:
                for rule in pool.recovery:
                    used_triggers.add(rule.trigger)
            for group in resources.pool_groups:
                for rule in group.recovery:
                    used_triggers.add(rule.trigger)
            for rule in resources.pooled_recoveries:
                for trigger in rule.triggers:
                    used_triggers.add(trigger)

            for trigger_data in resources_data.get("recovery_triggers", []):
                trigger = trigger_data.get("id")
                if trigger not in used_triggers and not trigger_data.get("informational"):
                    issues.append(Issue(severity="warning", file="resources.json", path=f"/recovery_triggers", message=f"Recovery trigger '{trigger}' is not used by any pool, pool_group, or pooled_recovery rule"))
            check_for_todos(resources_data, "resources.json", "", issues)

        except ValidationError as e:
            issues.append(Issue(severity="error", file="resources.json", path=resources_path, message=str(e)))
        except json.JSONDecodeError as e:
            issues.append(Issue(severity="error", file="resources.json", path=resources_path, message=f"Invalid JSON: {e}"))
        except Exception as e:
            issues.append(Issue(severity="error", file="resources.json", path=resources_path, message=str(e)))

    # 4. Triggers Config (Optional)
    triggers_path = os.path.join(pack_path, "triggers.json")
    if os.path.isfile(triggers_path):
        try:
            with open(triggers_path, "r", encoding="utf-8") as f:
                triggers_data = json.load(f)

            triggers = TriggersConfig.model_validate(triggers_data)

            # Additional checks for triggers
            if resources:
                pool_ids = {p.id for p in resources.pools}
                group_ids = {g.id for g in resources.pool_groups}
                all_targets = pool_ids.union(group_ids)

                recovery_triggers = {t.id for t in resources.recovery_triggers}

                rule_ids = set()
                for i, rule in enumerate(triggers.rules):
                    # check unique id
                    if rule.id in rule_ids:
                        issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}", message=f"Duplicate rule id: {rule.id}"))
                    rule_ids.add(rule.id)

                    # check target
                    if rule.kind == "consume" and rule.target not in all_targets:
                        issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}", message=f"Target '{rule.target}' not found in resources.json pools or pool_groups"))

                    # check trigger
                    if rule.kind == "recover" and rule.trigger is not None:
                         if rule.trigger not in recovery_triggers:
                             issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}", message=f"Trigger '{rule.trigger}' not found in resources.json recovery_triggers"))

                    # check sequence
                    if rule.kind == "recover" and getattr(rule, "sequence", None) is not None:
                         valid_sequences = {seq.id for seq in resources.recovery_sequences} if resources and hasattr(resources, "recovery_sequences") else set()
                         if rule.sequence not in valid_sequences:
                             issues.append(Issue(severity="error", file="triggers.json", path=f"/rules/{i}/sequence", message=f"sequence '{rule.sequence}' not found in resources.json recovery_sequences"))

                    # check regex compilable
                    if rule.key_regex:
                        for lang, regex_str in rule.key_regex.items():
                            try:
                                re.compile(regex_str)
                            except re.error as e:
                                issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}.key_regex.{lang}", message=f"Invalid regex: {e}"))

                    # check keywords not empty
                    for lang, words in rule.keywords.items():
                        if not words:
                            issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}.keywords.{lang}", message="Keywords list cannot be empty"))

                    # check if manifest language is supported in keywords
                    if 'manifest' in locals() and manifest:
                        if manifest.language not in rule.keywords:
                            issues.append(Issue(severity="warning", file="triggers.json", path=f"rules.{rule.id}", message=f"Rule does not have keywords for manifest language '{manifest.language}'"))


        except ValidationError as e:
            issues.append(Issue(severity="error", file="triggers.json", path=triggers_path, message=str(e)))
        except json.JSONDecodeError as e:
            issues.append(Issue(severity="error", file="triggers.json", path=triggers_path, message=f"Invalid JSON: {e}"))
        except Exception as e:
            issues.append(Issue(severity="error", file="triggers.json", path=triggers_path, message=str(e)))

    # 5. Actions Config (Optional)
    actions_path = os.path.join(pack_path, "actions.json")
    if os.path.isfile(actions_path):
        try:
            with open(actions_path, "r", encoding="utf-8") as f:
                actions_data = json.load(f)

            if not os.path.isfile(resolution_path):
                issues.append(Issue(severity="error", file="actions.json", path=actions_path, message="actions require StepTargetD20 resolution (missing resolution.json)"))
            elif not isinstance(resolution, StepTargetD20Config):
                issues.append(Issue(severity="error", file="actions.json", path=actions_path, message="actions require StepTargetD20 resolution"))

            actions = ActionsConfig.model_validate(actions_data)

            # Check action IDs are unique
            action_ids = set()
            for action in actions.actions:
                if action.id in action_ids:
                    issues.append(Issue(severity="error", file="actions.json", path=f"/actions/{action.id}", message=f"Duplicate action id: {action.id}"))
                action_ids.add(action.id)

                # Check pools exist
                if resources:
                    valid_pools = {p.id for p in resources.pools}
                    for pool_id in action.pools:
                        if pool_id not in valid_pools:
                            issues.append(Issue(severity="error", file="actions.json", path=f"/actions/{action.id}/pools", message=f"Pool '{pool_id}' not found in resources.json pools"))

                # Check effort config
                if action.effort:
                    # adjustment_key must exist in resolution.adjustment_caps
                    if isinstance(resolution, StepTargetD20Config) and action.effort.adjustment_key not in resolution.adjustment_caps:
                        issues.append(Issue(severity="warning", file="actions.json", path=f"/actions/{action.id}/effort", message=f"adjustment_key '{action.effort.adjustment_key}' not found in resolution.adjustment_caps"))

                    # effort expression vars whitelist
                    try:
                        compile_expr(action.effort.cost_expr)
                        vars_ref = referenced_variables(action.effort.cost_expr)
                        allowed_effort_vars = {"effort.levels", "effort.edge", "effort.impaired"}
                        for var in vars_ref:
                            if not var.startswith("character.") and var not in allowed_effort_vars:
                                issues.append(Issue(severity="error", file="actions.json", path=f"/actions/{action.id}/effort/cost_expr", message=f"Invalid variable '{var}'. Allowed: 'character.*', 'effort.levels', 'effort.edge', 'effort.impaired'."))
                    except ExprError as e:
                        issues.append(Issue(severity="error", file="actions.json", path=f"/actions/{action.id}/effort/cost_expr", message=str(e)))

                    # path validation
                    ptr_regex = r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$'
                    if action.effort.max_levels_path and not re.match(ptr_regex, action.effort.max_levels_path):
                        issues.append(Issue(severity="error", file="actions.json", path=f"/actions/{action.id}/effort/max_levels_path", message=f"Invalid pointer path syntax: {action.effort.max_levels_path}"))

                    if action.effort.edge_path_template:
                        test_str = action.effort.edge_path_template.replace("{pool}", "test_pool")
                        if not re.match(ptr_regex, test_str):
                            issues.append(Issue(severity="error", file="actions.json", path=f"/actions/{action.id}/effort/edge_path_template", message=f"Invalid pointer path syntax: {action.effort.edge_path_template}"))

            # Check difficulty_scale
            levels = set()
            for scale in actions.difficulty_scale:
                if scale.level in levels:
                    issues.append(Issue(severity="error", file="actions.json", path=f"/difficulty_scale", message=f"Duplicate difficulty level: {scale.level}"))
                levels.add(scale.level)
                if isinstance(resolution, StepTargetD20Config):
                    if not (resolution.min_difficulty <= scale.level <= resolution.max_difficulty):
                        issues.append(Issue(severity="error", file="actions.json", path=f"/difficulty_scale", message=f"Difficulty level {scale.level} is out of resolution bounds ({resolution.min_difficulty} to {resolution.max_difficulty})"))

            if actions.difficulty_scale and isinstance(resolution, StepTargetD20Config):
                expected_levels = set(range(resolution.min_difficulty, resolution.max_difficulty + 1))
                if not expected_levels.issubset(levels):
                    issues.append(Issue(severity="warning", file="actions.json", path="/difficulty_scale", message="difficulty_scale does not cover all levels from min_difficulty to max_difficulty"))


        except ValidationError as e:
            for err in e.errors():
                loc = err.get("loc", [])
                path = "/" + "/".join(str(x) for x in loc) if loc else actions_path
                msg = err.get("msg", "")
                if msg.startswith("Value error, "):
                    msg = msg[13:]
                msg = msg.replace("\n", " ").replace("\r", " ")
                issues.append(Issue(severity="error", file="actions.json", path=path, message=msg))
        except json.JSONDecodeError as e:
            issues.append(Issue(severity="error", file="actions.json", path=actions_path, message=f"Invalid JSON: {e}"))
        except Exception as e:
            issues.append(Issue(severity="error", file="actions.json", path=actions_path, message=str(e)))

    # 6. Character Schema Config (Optional)
    schema_path = os.path.join(pack_path, "character_schema.json")
    if os.path.isfile(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)

            schema = CharacterSchemaConfig.model_validate(schema_data)

            # Check paths format and duplicates
            ptr_regex = r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$'
            seen_paths = set()
            for idx, field in enumerate(schema.required_fields):
                if not re.match(ptr_regex, field.path):
                    issues.append(Issue(severity="error", file="character_schema.json", path=f"/required_fields/{idx}/path", message=f"Invalid path format: {field.path}"))
                if field.path in seen_paths:
                    issues.append(Issue(severity="error", file="character_schema.json", path=f"/required_fields/{idx}/path", message=f"Duplicate path: {field.path}"))
                seen_paths.add(field.path)

            # Check pool coverage
            if resources:
                covered_paths = set(seen_paths)
                for pool in resources.pools:
                    is_covered = False
                    for p in covered_paths:
                        if p == pool.current_path:
                            is_covered = True
                            break
                        if pool.max_path and p == pool.max_path:
                            is_covered = True
                            break
                        # Parent path coverage
                        if "." in pool.current_path and p == pool.current_path.rsplit(".", 1)[0]:
                            is_covered = True
                            break
                        # Prefix coverage (e.g. p is 'resources.might' and pool.current_path is 'resources.might.current')
                        if pool.current_path.startswith(p + "."):
                            is_covered = True
                            break
                    if not is_covered:
                        issues.append(Issue(severity="warning", file="character_schema.json", path="/required_fields", message=f"pool '{pool.id}' ({pool.current_path}) is not covered by any required field"))

        except ValidationError as e:
            for err in e.errors():
                loc = err.get("loc", [])
                path = "/" + "/".join(str(x) for x in loc) if loc else schema_path
                msg = err.get("msg", "")
                if msg.startswith("Value error, "):
                    msg = msg[13:]
                msg = msg.replace("\n", " ").replace("\r", " ")
                issues.append(Issue(severity="error", file="character_schema.json", path=path, message=msg))
        except json.JSONDecodeError as e:
            issues.append(Issue(severity="error", file="character_schema.json", path=schema_path, message=f"Invalid JSON: {e}"))
        except Exception as e:
            issues.append(Issue(severity="error", file="character_schema.json", path=schema_path, message=str(e)))

    # 7. Creation Manual Config (Optional)
    manual_path = os.path.join(pack_path, "creation_manual.json")
    if os.path.isfile(manual_path):
        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                manual_data = json.load(f)

            manual = CreationManualConfig.model_validate(manual_data)

            if not manual.steps:
                issues.append(Issue(severity="warning", file="creation_manual.json", path="/steps", message="creation manual has no steps"))
            else:
                expected_step = 1
                for idx, step in enumerate(manual.steps):
                    if step.step != expected_step:
                        issues.append(Issue(severity="error", file="creation_manual.json", path=f"/steps/{idx}/step", message=f"Expected step {expected_step}, got {step.step}"))
                    expected_step += 1

        except ValidationError as e:
            for err in e.errors():
                loc = err.get("loc", [])
                path = "/" + "/".join(str(x) for x in loc) if loc else manual_path
                msg = err.get("msg", "")
                if msg.startswith("Value error, "):
                    msg = msg[13:]
                msg = msg.replace("\n", " ").replace("\r", " ")
                issues.append(Issue(severity="error", file="creation_manual.json", path=path, message=msg))
        except json.JSONDecodeError as e:
            issues.append(Issue(severity="error", file="creation_manual.json", path=manual_path, message=f"Invalid JSON: {e}"))
        except Exception as e:
            issues.append(Issue(severity="error", file="creation_manual.json", path=manual_path, message=str(e)))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate a system pack.")
    parser.add_argument("path", help="Path to the system pack directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")

    args = parser.parse_args()

    pack_path = args.path

    if not os.path.exists(pack_path):
        if args.json:
            print(json.dumps([{"severity": "error", "file": "N/A", "path": pack_path, "message": "Directory not found"}]))
        else:
            print(f"[ERROR] N/A:{pack_path} Directory not found", file=sys.stderr)
        sys.exit(2)

    issues = validate_pack(pack_path)

    has_errors = any(i.severity == "error" for i in issues)
    has_warnings = any(i.severity == "warning" for i in issues)

    if args.json:
        print(json.dumps([i.model_dump() for i in issues], indent=2))
    else:
        for issue in issues:
            prefix = "[ERROR]" if issue.severity == "error" else "[WARNING]"
            out = sys.stderr if issue.severity == "error" else sys.stdout
            print(f"{prefix} {issue.file}:{issue.path} {issue.message}", file=out)

    if has_errors:
        sys.exit(1)
    if args.strict and has_warnings:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
