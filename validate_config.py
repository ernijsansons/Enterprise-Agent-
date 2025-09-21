#!/usr/bin/env python3
"""Configuration validation script for Enterprise Agent v3.4."""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Handle yaml import with fallback
try:
    import yaml
except ImportError:
    print("Installing required dependency: PyYAML...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "--quiet"])
        import yaml
    except Exception as e:
        print(f"ERROR: Failed to install PyYAML: {e}")
        print("Please install manually: pip install pyyaml")
        sys.exit(1)


class ConfigValidator:
    """Validates Enterprise Agent configuration files."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_range(self, value: Any, min_val: float, max_val: float, name: str) -> bool:
        """Validate that a value is within a specified range."""
        if not isinstance(value, (int, float)):
            self.errors.append(f"{name}: Expected numeric value, got {type(value).__name__}")
            return False

        if not (min_val <= value <= max_val):
            self.errors.append(f"{name}: Value {value} not in range [{min_val}, {max_val}]")
            return False

        return True

    def validate_enum(self, value: Any, allowed_values: List[str], name: str) -> bool:
        """Validate that a value is in a list of allowed values."""
        if value not in allowed_values:
            self.errors.append(f"{name}: Value '{value}' not in allowed values {allowed_values}")
            return False
        return True

    def validate_path(self, value: str, name: str, must_exist: bool = False) -> bool:
        """Validate that a path is valid."""
        if not isinstance(value, str):
            self.errors.append(f"{name}: Expected string path, got {type(value).__name__}")
            return False

        try:
            path = Path(value)
            if must_exist and not path.exists():
                self.errors.append(f"{name}: Path '{value}' does not exist")
                return False
        except Exception as e:
            self.errors.append(f"{name}: Invalid path '{value}': {e}")
            return False

        return True

    def validate_model_config(self, config: Dict[str, Any], prefix: str = "") -> None:
        """Validate model configuration section."""
        if "timeout" in config:
            self.validate_range(config["timeout"], 1, 3600, f"{prefix}timeout")

        if "retry" in config:
            self.validate_range(config["retry"], 0, 10, f"{prefix}retry")

        if "temperature" in config:
            self.validate_range(config["temperature"], 0.0, 2.0, f"{prefix}temperature")

    def validate_caching_config(self, config: Dict[str, Any]) -> None:
        """Validate caching configuration section."""
        if "default_ttl" in config:
            self.validate_range(config["default_ttl"], 60, 86400, "caching.default_ttl")

        if "max_size" in config:
            self.validate_range(config["max_size"], 100, 100000, "caching.max_size")

        if "cleanup_interval" in config:
            self.validate_range(config["cleanup_interval"], 10, 3600, "caching.cleanup_interval")

        if "quality_threshold" in config:
            self.validate_range(config["quality_threshold"], 0.0, 1.0, "caching.quality_threshold")

        if "high_quality_ttl_multiplier" in config:
            self.validate_range(config["high_quality_ttl_multiplier"], 1.0, 10.0, "caching.high_quality_ttl_multiplier")

        if "low_quality_ttl_multiplier" in config:
            self.validate_range(config["low_quality_ttl_multiplier"], 0.1, 1.0, "caching.low_quality_ttl_multiplier")

        if "compression_threshold" in config:
            self.validate_range(config["compression_threshold"], 512, 10485760, "caching.compression_threshold")

        if "eviction_policy" in config:
            self.validate_enum(config["eviction_policy"], ["lru", "lfu", "ttl"], "caching.eviction_policy")

        if "persistence_path" in config:
            self.validate_path(config["persistence_path"], "caching.persistence_path")

    def validate_metrics_config(self, config: Dict[str, Any]) -> None:
        """Validate metrics configuration section."""
        if "buffer_size" in config:
            self.validate_range(config["buffer_size"], 100, 100000, "metrics.buffer_size")

        if "flush_interval" in config:
            self.validate_range(config["flush_interval"], 1.0, 3600.0, "metrics.flush_interval")

        if "export_path" in config:
            self.validate_path(config["export_path"], "metrics.export_path")

        if "retention_hours" in config:
            self.validate_range(config["retention_hours"], 1, 8760, "metrics.retention_hours")  # Max 1 year

    def validate_reflection_config(self, config: Dict[str, Any]) -> None:
        """Validate reflection configuration section."""
        if "max_iterations" in config:
            self.validate_range(config["max_iterations"], 1, 20, "reflection.max_iterations")

        if "confidence_threshold" in config:
            self.validate_range(config["confidence_threshold"], 0.0, 1.0, "reflection.confidence_threshold")

        if "early_termination" in config:
            et_config = config["early_termination"]
            if "stagnation_threshold" in et_config:
                self.validate_range(et_config["stagnation_threshold"], 1, 10, "reflection.early_termination.stagnation_threshold")

            if "min_iterations" in et_config:
                self.validate_range(et_config["min_iterations"], 0, 10, "reflection.early_termination.min_iterations")

            if "progress_threshold" in et_config:
                self.validate_range(et_config["progress_threshold"], 0.0, 1.0, "reflection.early_termination.progress_threshold")

    def validate_config(self, config_path: str) -> bool:
        """Validate the complete configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            self.errors.append(f"Configuration file not found: {config_path}")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading configuration file: {e}")
            return False

        # Validate required top-level sections
        required_sections = ["default_model_config", "components", "enterprise_coding_agent"]
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")

        # Validate default model config
        if "default_model_config" in config:
            self.validate_model_config(config["default_model_config"], "default_model_config.")

        # Validate enterprise coding agent config
        if "enterprise_coding_agent" in config:
            agent_config = config["enterprise_coding_agent"]

            # Validate planning config
            if "planning" in agent_config:
                self.validate_model_config(agent_config["planning"], "planning.")

            # Validate coding config
            if "coding" in agent_config:
                self.validate_model_config(agent_config["coding"], "coding.")

            # Validate caching config
            if "caching" in agent_config:
                self.validate_caching_config(agent_config["caching"])

            # Validate metrics config
            if "observability" in agent_config and "metrics" in agent_config["observability"]:
                self.validate_metrics_config(agent_config["observability"]["metrics"])

            # Validate reflection config
            if "reflecting" in agent_config:
                self.validate_reflection_config(agent_config["reflecting"])

        return len(self.errors) == 0

    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print("FAIL: Configuration validation failed:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("WARN: Configuration warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("PASS: Configuration validation passed!")
        elif not self.errors:
            print("PASS: Configuration validation passed with warnings.")


def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python validate_config.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    validator = ConfigValidator()

    success = validator.validate_config(config_file)
    validator.print_results()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()