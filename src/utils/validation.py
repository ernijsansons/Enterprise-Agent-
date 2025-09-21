"""Input validation utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.exceptions import ValidationException


class Validator:
    """Base validator class."""

    def validate(self, value: Any) -> Any:
        """Validate and potentially transform the value."""
        raise NotImplementedError

    def __call__(self, value: Any) -> Any:
        """Allow validator to be called directly."""
        return self.validate(value)


class StringValidator(Validator):
    """Validate string inputs."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_chars: Optional[str] = None,
        strip_whitespace: bool = True,
        lowercase: bool = False,
        uppercase: bool = False,
    ):
        self.min_length = min_length
        self.max_length = max_length
        try:
            self.pattern = re.compile(pattern) if pattern else None
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern} - {e}")
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        self.strip_whitespace = strip_whitespace
        self.lowercase = lowercase
        self.uppercase = uppercase

    def validate(self, value: Any) -> str:
        """Validate string value."""
        if not isinstance(value, str):
            raise ValidationException(
                f"Expected string, got {type(value).__name__}",
                validation_type="type",
            )

        if self.strip_whitespace:
            value = value.strip()

        if self.lowercase:
            value = value.lower()
        elif self.uppercase:
            value = value.upper()

        if self.min_length and len(value) < self.min_length:
            raise ValidationException(
                f"String must be at least {self.min_length} characters long (provided: {len(value)} characters). "
                f"Please provide a longer input.",
                validation_type="length",
                provided_value=len(value),
                expected_range=f">={self.min_length}"
            )

        if self.max_length and len(value) > self.max_length:
            raise ValidationException(
                f"String must be at most {self.max_length} characters long (provided: {len(value)} characters). "
                f"Please shorten your input.",
                validation_type="length",
                provided_value=len(value),
                expected_range=f"<={self.max_length}"
            )

        if self.pattern and not self.pattern.match(value):
            raise ValidationException(
                f"String does not match the required format. Expected pattern: {self.pattern.pattern}. "
                f"Please check your input format and try again.",
                validation_type="pattern",
                provided_value=value[:50] + "..." if len(value) > 50 else value,
                expected_pattern=self.pattern.pattern
            )

        if self.allowed_chars:
            invalid_chars = set(value) - self.allowed_chars
            if invalid_chars:
                invalid_chars_display = "".join(sorted(invalid_chars)[:10])
                if len(invalid_chars) > 10:
                    invalid_chars_display += "..."
                raise ValidationException(
                    f"String contains {len(invalid_chars)} invalid character(s): '{invalid_chars_display}'. "
                    f"Only these characters are allowed: {self._format_allowed_chars()}",
                    validation_type="chars",
                    invalid_characters=list(invalid_chars),
                    allowed_characters=list(self.allowed_chars)
                )

        return value

    def _format_allowed_chars(self) -> str:
        """Format allowed characters for user-friendly display."""
        if not self.allowed_chars:
            return ""

        # Sort characters for consistent display
        chars = sorted(self.allowed_chars)

        # Group common character types
        if len(chars) > 20:
            # For large sets, show types instead of individual chars
            char_types = []
            if any(c.isalpha() for c in chars):
                char_types.append("letters")
            if any(c.isdigit() for c in chars):
                char_types.append("digits")
            if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in chars):
                char_types.append("special characters")

            return " and ".join(char_types) if char_types else "specified characters"
        else:
            # For small sets, show individual characters
            return "".join(chars)


class NumberValidator(Validator):
    """Validate numeric inputs."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_float: bool = True,
        allow_negative: bool = True,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.allow_float = allow_float
        self.allow_negative = allow_negative

    def validate(self, value: Any) -> Union[int, float]:
        """Validate numeric value."""
        if not isinstance(value, (int, float)):
            # Try to convert string to number
            if isinstance(value, str):
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    raise ValidationException(
                        f"Cannot convert '{value}' to a number. Please provide a valid numeric value (e.g., 42, 3.14, -10).",
                        validation_type="conversion",
                        provided_value=value,
                        expected_type="number"
                    )
            else:
                raise ValidationException(
                    f"Expected a number, but received {type(value).__name__}. "
                    f"Please provide a numeric value (integer or decimal).",
                    validation_type="type",
                    provided_type=type(value).__name__,
                    expected_type="number"
                )

        if not self.allow_float and isinstance(value, float) and not value.is_integer():
            raise ValidationException(
                f"Decimal values are not allowed. Please provide a whole number (integer). "
                f"Received: {value}",
                validation_type="type",
                provided_value=value,
                expected_type="integer"
            )

        if not self.allow_negative and value < 0:
            raise ValidationException(
                f"Negative values are not allowed. Please provide a positive number or zero. "
                f"Received: {value}",
                validation_type="range",
                provided_value=value,
                expected_range=">=0"
            )

        if self.min_value is not None and value < self.min_value:
            raise ValidationException(
                f"Value must be at least {self.min_value} (provided: {value}). "
                f"Please provide a larger number.",
                validation_type="range",
                provided_value=value,
                expected_range=f">={self.min_value}"
            )

        if self.max_value is not None and value > self.max_value:
            raise ValidationException(
                f"Value must be at most {self.max_value} (provided: {value}). "
                f"Please provide a smaller number.",
                validation_type="range",
                provided_value=value,
                expected_range=f"<={self.max_value}"
            )

        return int(value) if not self.allow_float else value


class ListValidator(Validator):
    """Validate list inputs."""

    def __init__(
        self,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        item_validator: Optional[Validator] = None,
        unique_items: bool = False,
    ):
        self.min_items = min_items
        self.max_items = max_items
        self.item_validator = item_validator
        self.unique_items = unique_items

    def validate(self, value: Any) -> List[Any]:
        """Validate list value."""
        if not isinstance(value, (list, tuple)):
            raise ValidationException(
                f"Expected list, got {type(value).__name__}",
                validation_type="type",
            )

        value = list(value)  # Convert tuple to list

        if self.min_items and len(value) < self.min_items:
            raise ValidationException(
                f"List too short (min {self.min_items} items)",
                validation_type="length",
            )

        if self.max_items and len(value) > self.max_items:
            raise ValidationException(
                f"List too long (max {self.max_items} items)",
                validation_type="length",
            )

        if self.unique_items and len(value) != len(set(map(str, value))):
            raise ValidationException(
                "List must contain unique items",
                validation_type="uniqueness",
            )

        if self.item_validator:
            validated_items = []
            for i, item in enumerate(value):
                try:
                    validated_items.append(self.item_validator.validate(item))
                except ValidationException as exc:
                    raise ValidationException(
                        f"Item {i} validation failed: {exc.message}",
                        validation_type="item",
                        failures=[{"index": i, "error": str(exc)}],
                    )
            value = validated_items

        return value


class DictValidator(Validator):
    """Validate dictionary inputs."""

    def __init__(
        self,
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        key_validators: Optional[Dict[str, Validator]] = None,
        allow_extra_keys: bool = True,
    ):
        self.required_keys = set(required_keys or [])
        self.optional_keys = set(optional_keys or [])
        self.key_validators = key_validators or {}
        self.allow_extra_keys = allow_extra_keys

    def validate(self, value: Any) -> Dict[str, Any]:
        """Validate dictionary value."""
        if not isinstance(value, dict):
            raise ValidationException(
                f"Expected dict, got {type(value).__name__}",
                validation_type="type",
            )

        # Check required keys
        missing_keys = self.required_keys - set(value.keys())
        if missing_keys:
            raise ValidationException(
                f"Missing required keys: {missing_keys}",
                validation_type="keys",
                failures=list(missing_keys),
            )

        # Check for extra keys
        if not self.allow_extra_keys:
            allowed_keys = self.required_keys | self.optional_keys
            extra_keys = set(value.keys()) - allowed_keys
            if extra_keys:
                raise ValidationException(
                    f"Extra keys not allowed: {extra_keys}",
                    validation_type="keys",
                    failures=list(extra_keys),
                )

        # Validate individual keys
        validated = {}
        for key, val in value.items():
            if key in self.key_validators:
                try:
                    validated[key] = self.key_validators[key].validate(val)
                except ValidationException as exc:
                    raise ValidationException(
                        f"Key '{key}' validation failed: {exc.message}",
                        validation_type="value",
                        failures=[{"key": key, "error": str(exc)}],
                    )
            else:
                validated[key] = val

        return validated


class PathValidator(Validator):
    """Validate file/directory paths."""

    def __init__(
        self,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_extensions: Optional[List[str]] = None,
        resolve: bool = True,
    ):
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir
        self.allowed_extensions = allowed_extensions
        self.resolve = resolve

    def validate(self, value: Any) -> Path:
        """Validate path value."""
        if not isinstance(value, (str, Path)):
            raise ValidationException(
                f"Expected path string, got {type(value).__name__}",
                validation_type="type",
            )

        path = Path(value)

        if self.resolve:
            try:
                path = path.resolve()
            except Exception as exc:
                raise ValidationException(
                    f"Cannot resolve path: {exc}",
                    validation_type="path",
                )

        if self.must_exist and not path.exists():
            raise ValidationException(
                f"Path does not exist: {path}",
                validation_type="existence",
            )

        if self.must_be_file and not path.is_file():
            raise ValidationException(
                f"Path is not a file: {path}",
                validation_type="type",
            )

        if self.must_be_dir and not path.is_dir():
            raise ValidationException(
                f"Path is not a directory: {path}",
                validation_type="type",
            )

        if self.allowed_extensions:
            if not any(str(path).endswith(ext) for ext in self.allowed_extensions):
                raise ValidationException(
                    f"Invalid file extension. Allowed: {self.allowed_extensions}",
                    validation_type="extension",
                )

        return path


class ModelNameValidator(Validator):
    """Validate model names."""

    VALID_MODELS = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o3"],
        "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "google": ["gemini-1.5-pro", "gemini-1.5-flash"],
    }

    def validate(self, value: Any) -> str:
        """Validate model name."""
        if not isinstance(value, str):
            raise ValidationException(
                f"Expected string model name, got {type(value).__name__}",
                validation_type="type",
            )

        # Check if it matches any known model pattern
        valid = False
        for provider, models in self.VALID_MODELS.items():
            for model in models:
                if model in value.lower():
                    valid = True
                    break
            if valid:
                break

        if not valid:
            raise ValidationException(
                f"Unknown model: {value}",
                validation_type="model",
            )

        return value


class DomainValidator(Validator):
    """Validate domain names."""

    VALID_DOMAINS = ["coding", "social_media", "content", "trading", "real_estate"]

    def validate(self, value: Any) -> str:
        """Validate domain name."""
        if not isinstance(value, str):
            raise ValidationException(
                f"Expected string domain name, got {type(value).__name__}",
                validation_type="type",
            )

        if value not in self.VALID_DOMAINS:
            raise ValidationException(
                f"Invalid domain '{value}'. Valid domains: {self.VALID_DOMAINS}",
                validation_type="domain",
            )

        return value


def validate_input(
    value: Any,
    validator: Union[Validator, Callable],
    default: Optional[Any] = None,
    raise_on_error: bool = True,
) -> Any:
    """Validate input with optional default fallback."""
    try:
        if isinstance(validator, Validator):
            return validator.validate(value)
        else:
            return validator(value)
    except ValidationException:
        if raise_on_error:
            raise
        if default is not None:
            return default
        return value


def create_validator(**kwargs) -> DictValidator:
    """Create a dictionary validator from keyword arguments."""
    key_validators = {}

    for key, validator_spec in kwargs.items():
        if isinstance(validator_spec, Validator):
            key_validators[key] = validator_spec
        elif isinstance(validator_spec, dict):
            # Create validator from specification
            vtype = validator_spec.get("type", "string")
            if vtype == "string":
                key_validators[key] = StringValidator(**validator_spec)
            elif vtype == "number":
                key_validators[key] = NumberValidator(**validator_spec)
            elif vtype == "list":
                key_validators[key] = ListValidator(**validator_spec)
            elif vtype == "dict":
                key_validators[key] = DictValidator(**validator_spec)
            elif vtype == "path":
                key_validators[key] = PathValidator(**validator_spec)

    return DictValidator(
        required_keys=list(kwargs.keys()),
        key_validators=key_validators,
    )


__all__ = [
    "Validator",
    "StringValidator",
    "NumberValidator",
    "ListValidator",
    "DictValidator",
    "PathValidator",
    "ModelNameValidator",
    "DomainValidator",
    "validate_input",
    "create_validator",
]
