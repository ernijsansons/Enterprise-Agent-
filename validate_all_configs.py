#!/usr/bin/env python3
"""Validate all YAML configuration files."""

import sys
import yaml
from pathlib import Path


def validate_yaml_file(file_path):
    """Validate a single YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            return f"WARNING: Empty config file: {file_path}"

        return f"PASS: {file_path} - {len(config)} top-level keys"
    except yaml.YAMLError as e:
        return f"FAIL: {file_path} - YAML syntax error: {e}"
    except Exception as e:
        return f"FAIL: {file_path} - Error: {e}"


def main():
    """Validate all YAML configuration files."""
    config_dir = Path("configs")
    yaml_files = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))

    # Also check CI/CD workflows
    github_dir = Path(".github")
    if github_dir.exists():
        yaml_files.extend(list(github_dir.rglob("*.yaml")) + list(github_dir.rglob("*.yml")))

    print(f"Validating {len(yaml_files)} YAML files...\n")

    results = []
    failed = 0

    for yaml_file in yaml_files:
        result = validate_yaml_file(yaml_file)
        results.append(result)
        if result.startswith("FAIL"):
            failed += 1
        print(result)

    print(f"\nSummary: {len(yaml_files) - failed} passed, {failed} failed")

    if failed > 0:
        print("\nFailed files need attention!")
        sys.exit(1)
    else:
        print("\nAll YAML files are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()