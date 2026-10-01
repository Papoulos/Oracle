import os
import json
import yaml
import sys
import argparse
from typing import Literal, Any
import re
from pydantic import BaseModel, ValidationError, TypeAdapter

from mechanics.models import Manifest, ResolutionConfig, ResourcesConfig, PbtA2d6Config, TriggersConfig


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

            resources = ResourcesConfig.model_validate(resources_data)
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
                for rule in triggers.rules:
                    # check unique id
                    if rule.id in rule_ids:
                        issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}", message=f"Duplicate rule id: {rule.id}"))
                    rule_ids.add(rule.id)

                    # check target
                    if rule.kind == "consume" and rule.target not in all_targets:
                        issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}", message=f"Target '{rule.target}' not found in resources.json pools or pool_groups"))

                    # check trigger
                    if rule.kind == "recover" and rule.trigger not in recovery_triggers:
                         issues.append(Issue(severity="error", file="triggers.json", path=f"rules.{rule.id}", message=f"Trigger '{rule.trigger}' not found in resources.json recovery_triggers"))

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
