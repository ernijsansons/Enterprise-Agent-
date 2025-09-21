"""Enterprise agent orchestrator coordinating multi-role workflows."""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.orchestration import build_graph
from src.orchestration.async_orchestrator import get_async_orchestrator
from src.providers.auth_manager import get_auth_manager
from src.providers.claude_code_provider import get_claude_code_provider
from src.roles import Coder, Planner, Reflector, Reviewer, Validator
from src.tools import invoke_codex_cli
from src.utils.cache import CacheConfig, ModelResponseCache, get_model_cache
from src.utils.concurrency import ExecutionManager

# Import structured error handling
from src.utils.errors import (
    EnterpriseAgentError,
    ErrorCode,
    ErrorSeverity,
    create_config_error,
    create_model_error,
    create_orchestration_error,
    get_error_handler,
    handle_error,
)
from src.utils.metrics import (
    MetricSeverity,
    get_metrics_collector,
    record_counter,
    record_event,
    record_gauge,
    timer,
)
from src.utils.notifications import notify_cli_failure
from src.utils.reflection_audit import (
    ReflectionDecision,
    ReflectionPhase,
    ValidationIssue,
    finish_reflection_session,
    get_reflection_auditor,
    log_reflection_step,
    start_reflection_session,
)
from src.utils.retry import retry_on_timeout
from src.utils.safety import scrub_pii
from src.utils.secrets import load_secrets
from src.utils.telemetry import record_metric
from src.utils.validation import DomainValidator, StringValidator

try:  # Optional dependencies; continue gracefully if missing
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore

try:
    from google.generativeai import GenerativeModel
except ImportError:  # pragma: no cover
    GenerativeModel = None  # type: ignore

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

try:
    from src.utils.costs import CostEstimator as ExternalCostEstimator  # type: ignore
except ImportError:  # pragma: no cover

    class ExternalCostEstimator:  # type: ignore[too-many-ancestors]
        """Lightweight fallback estimator when cost utilities are unavailable."""

        def __init__(self, *_: Any, **__: Any) -> None:
            self._tokens = 0
            self._total_cost = 0.0
            self._events: List[Dict[str, Any]] = []

        def track(self, tokens: int, role: str, operation: str, **extra: Any) -> None:
            tokens = max(tokens or 0, 0)
            self._tokens += tokens
            cost = extra.get("cost", tokens * 2e-6)
            self._total_cost += cost
            event = {
                "role": role,
                "operation": operation,
                "tokens": tokens,
                "cost": cost,
                **extra,
            }
            self._events.append(event)

        def track_estimated(
            self, tokens: int, role: str, operation: str, **extra: Any
        ) -> None:
            self.track(tokens, role, f"{operation}_estimated", **extra)

        def estimate(self, complexity: int, _domain: str) -> float:
            return max(complexity, 1) / 1000.0 * 0.05

        def summary(self) -> Dict[str, Any]:
            return {
                "tokens": self._tokens,
                "events": list(self._events),
                "total_cost": round(self._total_cost, 6),
            }


try:
    from src.memory import MemoryStore as ExternalMemoryStore  # type: ignore
except ImportError:  # pragma: no cover

    class ExternalMemoryStore:  # type: ignore[too-many-ancestors]
        """Simple in-memory store used when the real implementation is absent."""

        def __init__(self, *_: Any, **__: Any) -> None:
            self._data: Dict[str, Dict[str, Any]] = {}

        def store(self, scope: str, key: str, value: Any) -> None:
            self._data.setdefault(scope, {})[key] = value

        def retrieve(self, scope: str, key: str, default: Any = None) -> Any:
            return self._data.get(scope, {}).get(key, default)

        def prune(self) -> None:
            # Keep the fallback lightweight; no pruning heuristic required.
            if len(self._data) > 10:
                self._data.pop(next(iter(self._data)))


try:
    from src.governance import (  # type: ignore
        GovernanceChecker as ExternalGovernanceChecker,
    )
except ImportError:  # pragma: no cover

    class ExternalGovernanceChecker:  # type: ignore[too-many-ancestors]
        """Fallback governance checker that records metrics without enforcement."""

        def __init__(self, *_: Any, **__: Any) -> None:
            self.latest_metrics: Dict[str, Any] = {}

        def check(self, result: Dict[str, Any]) -> bool:
            self.latest_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "domain": result.get("domain"),
            }
            return True

        def hitl_check(self, *_: Any, **__: Any) -> bool:
            return True


logger = logging.getLogger(__name__)


MODEL_ALIASES: Dict[str, str] = {
    "openai_gpt_5": "gpt-3.5-turbo",  # Downgraded to basic model for backup
    "openai_gpt_5_codex": "gpt-3.5-turbo",  # Use basic model as fallback
    "claude_sonnet_4": "claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
    "claude_opus_4": "claude-3-opus-20240229",  # Claude Opus for complex tasks
    "gemini-2.5-pro": "gemini-1.5-pro-latest",
}


def _resolve_model_alias(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _detect_provider(model: str) -> str:
    resolved = _resolve_model_alias(model)
    if model.startswith("openai") or resolved.startswith(("gpt-", "o1", "o3")):
        return "openai"
    if model.startswith("claude") or resolved.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini") or resolved.startswith("gemini"):
        return "gemini"
    return "unknown"


class CostEstimator(ExternalCostEstimator):
    """Alias to allow type checking when the external implementation is present."""


class MemoryStore(ExternalMemoryStore):
    """Alias to allow type checking when the external implementation is present."""


class GovernanceChecker(ExternalGovernanceChecker):
    """Alias to allow type checking when the external implementation is present."""


class AgentState(dict):
    """Lightweight state container for LangGraph-compatible execution."""

    pass


class AgentOrchestrator:
    """Coordinate planner, coder, validator, reflector, reviewer, and governance roles."""

    def __init__(self, config_path: str = "configs/agent_config_v3.4.yaml") -> None:
        # Initialize error handler, metrics collector, and reflection auditor first
        self._error_handler = get_error_handler()
        self._metrics_collector = get_metrics_collector()
        self._reflection_auditor = get_reflection_auditor()

        try:
            self._state_lock = threading.Lock()
            self._model_lock = threading.Lock()
            self._executor = ExecutionManager(max_workers=4)
            self._domain_validator = DomainValidator()
            self._string_validator = StringValidator(
                max_length=100000, strip_whitespace=True
            )

            # Initialize Claude Code provider if enabled
            self._use_claude_code = (
                os.getenv("USE_CLAUDE_CODE", "false").lower() == "true"
            )
            self._claude_code_provider = None
            if self._use_claude_code:
                self._init_claude_code_provider()

            # Initialize async orchestrator for performance improvements
            self._async_enabled = os.getenv("ENABLE_ASYNC", "true").lower() == "true"
            self._async_orchestrator = None
            if self._async_enabled:
                try:
                    self._async_orchestrator = get_async_orchestrator(
                        self._build_async_config()
                    )
                    logger.info(
                        "Async orchestrator initialized for performance improvements"
                    )
                except Exception as e:
                    handle_error(
                        create_orchestration_error(
                            "Failed to initialize async orchestrator",
                            ErrorCode.ORCHESTRATION_INIT_FAILED,
                            context={"async_enabled": True, "error": str(e)},
                            cause=e,
                        )
                    )
                    logger.warning(f"Failed to initialize async orchestrator: {e}")
                    self._async_enabled = False

            # Load secrets with error handling
            try:
                self.secrets = load_secrets()
            except Exception as exc:
                handle_error(
                    EnterpriseAgentError(
                        ErrorCode.AUTH_SETUP_FAILED,
                        "Secret loading degraded; continuing with limited credentials",
                        context={"error": str(exc)},
                        severity=ErrorSeverity.LOW,
                        recovery_suggestions=[
                            "Check secret configuration",
                            "Verify environment variables",
                        ],
                        cause=exc,
                    )
                )
                self.secrets = {}

            # Load and validate configuration
            self._load_config(config_path)

            # Initialize components with error handling
            self._init_components()

            # Initialize clients, roles, and graph
            self._init_clients()
            self._init_roles()
            self._init_graph()

            logger.info("AgentOrchestrator initialized successfully")

        except Exception as e:
            handle_error(
                create_orchestration_error(
                    f"Failed to initialize AgentOrchestrator: {str(e)}",
                    ErrorCode.ORCHESTRATION_INIT_FAILED,
                    context={"config_path": config_path},
                    cause=e,
                )
            )
            raise

    def _load_config(self, config_path: str) -> None:
        """Load and validate configuration with structured error handling."""
        try:
            config_override = os.getenv("ENTERPRISE_AGENT_CONFIG")
            config_path = (
                Path(config_override) if config_override else Path(config_path)
            )

            if not config_path.is_absolute():
                candidate = Path(__file__).resolve().parents[1] / config_path
                config_path = (
                    candidate
                    if candidate.exists()
                    else (Path.cwd() / config_path).resolve()
                )

            self._project_root = Path(__file__).resolve().parents[1]

            if not config_path.exists():
                raise create_config_error(
                    f"Configuration file not found: {config_path}",
                    ErrorCode.CONFIG_FILE_NOT_FOUND,
                    context={"config_path": str(config_path)},
                    recovery_suggestions=[
                        "Check if the configuration file exists",
                        "Verify the ENTERPRISE_AGENT_CONFIG environment variable",
                        "Use the default config path: configs/agent_config_v3.4.yaml",
                    ],
                )

            with open(config_path, "r", encoding="utf-8") as handle:
                self.config = yaml.safe_load(handle) or {}

        except yaml.YAMLError as e:
            raise create_config_error(
                f"Failed to parse configuration file: {str(e)}",
                ErrorCode.CONFIG_PARSE_ERROR,
                context={"config_path": str(config_path), "yaml_error": str(e)},
                recovery_suggestions=[
                    "Check YAML syntax in configuration file",
                    "Validate configuration structure",
                    "Use a YAML validator tool",
                ],
                cause=e,
            )
        except Exception as e:
            raise create_config_error(
                f"Failed to load configuration: {str(e)}",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                context={"config_path": str(config_path)},
                cause=e,
            )

        # Validate required configuration sections
        if "enterprise_coding_agent" not in self.config:
            raise create_config_error(
                "Missing required 'enterprise_coding_agent' section in configuration",
                ErrorCode.CONFIG_MISSING_REQUIRED_FIELD,
                context={
                    "config_path": str(config_path),
                    "missing_section": "enterprise_coding_agent",
                },
                recovery_suggestions=[
                    "Add the 'enterprise_coding_agent' section to your configuration",
                    "Use the default configuration template",
                    "Check configuration documentation",
                ],
            )

    def _init_components(self) -> None:
        """Initialize core components with error handling."""
        try:
            self.agent_cfg = self.config.get("enterprise_coding_agent", {})
            self.orchestration_cfg = self.agent_cfg.get("orchestration", {})
            self.domain_packs = self._load_domain_packs()
            memory_cfg = self.agent_cfg.get("memory", {})
            optimizer_cfg = self.orchestration_cfg.get("runtime_optimizer", {})
            governance_cfg = self.agent_cfg.get("governance", {})

            # Initialize configurable caching
            self._init_cache_system()

            self.memory = MemoryStore(memory_cfg)
            self.cost_estimator = CostEstimator(optimizer_cfg)
            self.governance = GovernanceChecker(governance_cfg)

            self.openai_client = None
            self.anthropic_client = None
            self.gemini_client = None
            self._codex_available = (
                bool(shutil.which("codex"))
                and os.getenv("CODEX_CLI_ENABLED", "0") == "1"
            )

        except Exception as e:
            raise create_orchestration_error(
                f"Failed to initialize core components: {str(e)}",
                ErrorCode.ORCHESTRATION_INIT_FAILED,
                context={"component_init": "core_components"},
                cause=e,
            )

    def _init_cache_system(self) -> None:
        """Initialize configurable cache system."""
        try:
            cache_cfg = self.agent_cfg.get("caching", {})
            model_cache_cfg = cache_cfg.get("model_cache", {})

            # Create cache configuration from YAML config with environment overrides
            cache_config = CacheConfig.from_env()

            # Override with YAML configuration
            if cache_cfg:
                for key, value in cache_cfg.items():
                    if hasattr(cache_config, key) and key != "model_cache":
                        setattr(cache_config, key, value)

            # Initialize model cache with configuration
            model_cache_config = CacheConfig(
                enabled=model_cache_cfg.get("enabled", cache_config.enabled),
                default_ttl=model_cache_cfg.get(
                    "default_ttl", 900
                ),  # 15 minutes for models
                max_size=model_cache_cfg.get("max_size", 500),
                adaptive_ttl=cache_config.adaptive_ttl,
                quality_threshold=cache_config.quality_threshold,
                high_quality_ttl_multiplier=model_cache_cfg.get(
                    "high_confidence_ttl_multiplier", 2.0
                ),
                persistence_enabled=cache_config.persistence_enabled,
                persistence_path=cache_config.persistence_path,
                compression_enabled=cache_config.compression_enabled,
                compression_threshold=cache_config.compression_threshold,
                eviction_policy=cache_config.eviction_policy,
                metrics_enabled=cache_config.metrics_enabled,
            )

            self._model_cache = ModelResponseCache(config=model_cache_config)
            self._cache_config = cache_config

            # Store cache multipliers for Claude responses
            self._claude_ttl_multiplier = model_cache_cfg.get(
                "claude_ttl_multiplier", 1.5
            )

            logger.info(
                f"Initialized configurable cache system: enabled={cache_config.enabled}, "
                f"adaptive_ttl={cache_config.adaptive_ttl}, eviction_policy={cache_config.eviction_policy}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize cache system: {e}, using defaults")
            self._model_cache = get_model_cache()
            self._cache_config = CacheConfig()
            self._claude_ttl_multiplier = 1.5

    def _init_claude_code_provider(self) -> None:
        """Initialize Claude Code CLI provider with enhanced validation and error handling.

        This method handles the complete initialization of the Claude Code provider,
        including authentication validation, configuration setup, and error recovery.
        """
        try:
            # Enhanced authentication setup
            auth_manager = get_auth_manager()

            # Comprehensive setup validation
            validation = auth_manager.validate_setup()

            if not validation["ready"]:
                self._handle_validation_issues(validation)
                return

            # Ensure subscription mode
            if not self._ensure_subscription_mode(auth_manager):
                return

            # Initialize provider with enhanced configuration
            self._claude_code_provider = self._create_provider_instance()

            # Enhanced verification and session setup
            self._setup_provider_session(validation)

        except Exception as exc:
            logger.error(f"Failed to initialize Claude Code provider: {exc}")
            notify_cli_failure("provider_initialization", str(exc))
            self._use_claude_code = False
            self._claude_code_provider = None

    def _handle_validation_issues(self, validation: Dict[str, Any]) -> None:
        """Handle validation issues with detailed logging and recommendations.

        Args:
            validation: Validation results dictionary
        """
        logger.warning(f"Claude Code setup not ready: {validation['status']}")

        # Log specific issues and recommendations
        for issue in validation.get("issues", []):
            logger.warning(f"Setup issue: {issue}")

        for rec in validation.get("recommendations", []):
            logger.info(f"Recommendation: {rec}")

        if validation["status"] == "needs_installation":
            logger.error("Claude Code CLI not installed. Provider disabled.")
            self._use_claude_code = False
            self._claude_code_provider = None
        elif validation["status"] == "needs_login":
            logger.warning("Claude Code authentication required for optimal operation.")

    def _ensure_subscription_mode(self, auth_manager) -> bool:
        """Ensure subscription mode is properly configured.

        Args:
            auth_manager: Authentication manager instance

        Returns:
            True if subscription mode is ensured, False otherwise
        """
        try:
            if not auth_manager.ensure_subscription_mode():
                logger.warning("Failed to ensure subscription mode for Claude Code")
                return False
            return True
        except Exception as e:
            logger.error(f"Error ensuring subscription mode: {e}")
            return False

    def _create_provider_instance(self):
        """Create and configure the Claude Code provider instance.

        Returns:
            Configured Claude Code provider instance
        """
        provider_config = {
            "timeout": 60,
            "enable_fallback": True,
            "auto_mode": False,  # Require confirmations for safety
            "working_directory": os.getcwd(),
            "auto_remove_api_key": True,  # Automatically handle API key conflicts
        }
        return get_claude_code_provider(provider_config)

    def _setup_provider_session(self, validation: Dict[str, Any]) -> None:
        """Setup provider session and store validation information.

        Args:
            validation: Validation results dictionary
        """
        if validation.get("authenticated", False):
            logger.info(
                "Claude Code provider initialized successfully (subscription mode)"
            )
            # Initialize session for context retention
            self._session_id = (
                self._claude_code_provider.create_session()
                if self._claude_code_provider
                else None
            )
        else:
            logger.warning(
                "Claude Code provider initialized but not logged in. "
                "Run 'claude login' to use your Max subscription."
            )
            self._session_id = None

        # Store validation info for runtime monitoring
        self._claude_validation = validation

    # --------------------------------------------------------------------- setup
    def _emit_event(self, stage: str, **payload: Any) -> None:
        record_event(stage, **payload)

    def _enhance_prompt(self, prompt: str, role: str) -> str:
        enhancements = self.agent_cfg.get("prompt_enhancements", {})
        addition = enhancements.get(role.lower())
        if addition:
            return f"{addition}\n\n{prompt}"
        return prompt

    def _enhance_cli_params(self, params: List[str], role: str) -> List[str]:
        hints = self.agent_cfg.get("cli_enhancements", {})
        addition = hints.get(role.lower())
        if addition and addition not in params:
            return params + [addition]
        return params

    def _provide_quality_feedback(self, score: float, feedback: Dict[str, Any]) -> None:
        """Provide quality feedback to Claude Code provider for learning."""
        if self._use_claude_code and self._claude_code_provider:
            try:
                success = self._claude_code_provider.provide_quality_feedback(
                    score, feedback
                )
                if success:
                    logger.debug(
                        f"Provided quality feedback: score={score:.2f}, role={feedback.get('role')}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to provide quality feedback: {exc}")

    def _build_project_context(self) -> Dict[str, Any]:
        """Build project context for enhanced Claude reasoning."""
        context = {}

        # Add domain packs information
        if hasattr(self, "domain_packs") and self.domain_packs:
            context["available_domains"] = list(self.domain_packs.keys())

        # Add current configuration highlights
        context["tech_stack"] = ["Python", "YAML", "Enterprise Agent Framework"]

        # Add architectural patterns
        context["architecture"] = "Role-based multi-agent system with validation loops"

        # Add coding conventions
        context["conventions"] = [
            "Type hints required",
            "Comprehensive error handling",
            "Structured logging",
            "Test-driven development",
        ]

        # Add project-specific requirements
        context["requirements"] = [
            "Zero-cost Claude Code integration",
            "High-quality outputs through validation loops",
            "Structured reasoning patterns",
            "Session-based context retention",
        ]

        return context

    def _build_async_config(self) -> Dict[str, Any]:
        """Build configuration for async orchestrator."""
        return {
            "memory": self.agent_cfg.get("memory", {}),
            "costs": self.orchestration_cfg.get("runtime_optimizer", {}),
            "governance": self.agent_cfg.get("governance", {}),
            "use_claude_code": self._use_claude_code,
            "claude_code": {
                "timeout": 60,
                "enable_fallback": True,
                "working_directory": os.getcwd(),
            },
        }

    def _enrich_state_with_context(self, state: AgentState) -> AgentState:
        """Enrich state with project context for enhanced Claude reasoning."""
        if self._use_claude_code:
            state["project_context"] = self._build_project_context()
            # Add session tracking for context retention
            if hasattr(self, "_session_id") and self._session_id:
                state["session_id"] = self._session_id
        return state

    def _transfer_role_context(
        self, state: AgentState, from_role: str, to_role: str
    ) -> AgentState:
        """Transfer context between roles for enhanced coordination."""
        if not self._use_claude_code:
            return state

        # Build role transition context
        transition_context = {
            "from_role": from_role,
            "to_role": to_role,
            "previous_outputs": {},
        }

        # Capture relevant outputs from previous role
        if from_role == "planner":
            transition_context["previous_outputs"]["plan"] = state.get("plan", "")
            transition_context["previous_outputs"]["epics"] = state.get(
                "plan_epics", []
            )

        elif from_role == "coder":
            transition_context["previous_outputs"]["code"] = state.get("code", "")
            transition_context["previous_outputs"]["source"] = state.get(
                "code_source", ""
            )

        elif from_role == "validator":
            transition_context["previous_outputs"]["validation"] = state.get(
                "validation", {}
            )
            transition_context["previous_outputs"]["needs_reflect"] = state.get(
                "needs_reflect", False
            )

        elif from_role == "reflector":
            transition_context["previous_outputs"]["reflection"] = state.get(
                "reflection", {}
            )
            transition_context["previous_outputs"]["confidence"] = state.get(
                "confidence", 0.0
            )

        # Store in state for next role to access
        state[f"{to_role}_context"] = transition_context

        return state

    def _load_domain_packs(self) -> Dict[str, Dict[str, Any]]:
        packs: Dict[str, Dict[str, Any]] = {}
        base_dir = self._project_root / "configs" / "domains"
        if not base_dir.exists():  # pragma: no cover - defensive
            return packs
        for yaml_file in base_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as handle:
                    packs[yaml_file.stem] = yaml.safe_load(handle) or {}
            except yaml.YAMLError as exc:
                logger.warning(
                    "Failed to parse domain pack %s: %s", yaml_file.name, exc
                )
        return packs

    def _init_clients(self) -> None:
        openai_key = self.secrets.get("OPENAI_API_KEY")
        if openai and openai_key and openai_key != "STUBBED_FALLBACK":
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "OpenAI client initialisation failed; falling back to offline mode: %s",
                    exc,
                )
                self.openai_client = None

        anthropic_key = self.secrets.get("ANTHROPIC_API_KEY")
        if Anthropic and anthropic_key and anthropic_key != "STUBBED_FALLBACK":
            try:
                # Initialize Anthropic client with proper configuration
                # Handle different Anthropic SDK versions gracefully
                import inspect

                anthropic_params = {"api_key": anthropic_key}

                # Check if Anthropic constructor accepts http_client parameter
                if "http_client" in inspect.signature(Anthropic.__init__).parameters:
                    try:
                        import httpx

                        # Create HTTP client with proper timeout and no proxy issues
                        http_client = httpx.Client(
                            timeout=httpx.Timeout(30.0, connect=10.0),
                            limits=httpx.Limits(
                                max_keepalive_connections=5, max_connections=10
                            ),
                        )
                        anthropic_params["http_client"] = http_client
                    except ImportError:
                        # If httpx is not available, let Anthropic use its default client
                        pass

                self.anthropic_client = Anthropic(**anthropic_params)

                # Verify the client works with a minimal test
                # This helps catch configuration issues early
                try:
                    # Test with a minimal completion to verify connectivity
                    self.anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",  # Use the cheapest model for testing
                        max_tokens=1,
                        messages=[{"role": "user", "content": "test"}],
                    )
                    logger.info("Anthropic client initialized successfully")
                except Exception as test_exc:
                    logger.warning(
                        "Anthropic client test failed, disabling: %s", test_exc
                    )
                    self.anthropic_client = None

            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Anthropic client initialisation failed; continuing without it: %s",
                    exc,
                )
                self.anthropic_client = None

        google_key = self.secrets.get("GOOGLE_API_KEY")
        if GenerativeModel and google_key and google_key != "STUBBED_FALLBACK":
            try:
                import google.generativeai as genai

                genai.configure(api_key=google_key)
                self.gemini_client = GenerativeModel("gemini-1.5-pro-latest")
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Gemini client initialisation failed; continuing without it: %s",
                    exc,
                )
                self.gemini_client = None

    def _init_roles(self) -> None:
        self.planner_role = Planner(self)
        self.coder_role = Coder(self)
        self.validator_role = Validator(self)
        self.reflector_role = Reflector(self)
        self.reviewer_role = Reviewer(self)

    def _init_graph(self) -> None:
        self.graph = build_graph(
            AgentState,
            planner=self.planner,
            coder=self.coder,
            validator=self.validator,
            reflector=self.reflector,
            reviewer=self.reviewer,
            governance=self._governance_node,
            validate_route=self._validate_route,
            reflect_route=self._reflect_route,
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON with multiple fallback strategies for robustness."""
        if not text:
            return {}

        candidate = text.strip()
        if not candidate:
            return {}

        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")

        # Strategy 2: Extract JSON object from surrounding text
        json_patterns = [
            (r"\{[^{}]*\}", False),  # Simple object
            (r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", False),  # Nested objects (1 level)
            (r"```json\s*(.*?)\s*```", True),  # Markdown code block
            (r"```\s*(.*?)\s*```", True),  # Generic code block
        ]

        for pattern, extract_group in json_patterns:
            import re

            matches = re.findall(pattern, candidate, re.DOTALL)
            for match in matches:
                try:
                    json_str = match if not extract_group else match
                    # Clean up common issues
                    json_str = self._clean_json_string(json_str)
                    return json.loads(json_str)
                except (json.JSONDecodeError, AttributeError):
                    continue

        # Strategy 3: Fix common JSON issues
        cleaned = self._repair_json(candidate)
        if cleaned:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Extract key-value pairs from structured text
        extracted = self._extract_structured_data(candidate)
        if extracted:
            return extracted

        # Final fallback: Return raw text
        logger.info("Using raw_text fallback after all JSON parse strategies failed")
        return {"raw_text": candidate}

    def _clean_json_string(self, json_str: str) -> str:
        """Clean common issues in JSON strings."""
        # Remove trailing commas
        import re

        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Fix single quotes (carefully, avoiding strings with apostrophes)
        # This is a simplified approach - a full parser would be better
        if '"' not in json_str and "'" in json_str:
            json_str = json_str.replace("'", '"')

        # Remove comments
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

        return json_str.strip()

    def _repair_json(self, text: str) -> Optional[str]:
        """Attempt to repair malformed JSON."""

        # Find the most likely JSON boundaries
        start_patterns = ["{", "["]
        end_patterns = ["}", "]"]

        for start_char, end_char in zip(start_patterns, end_patterns):
            start = text.find(start_char)
            if start == -1:
                continue

            # Find matching end bracket
            depth = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

            if end != -1:
                candidate = text[start : end + 1]
                cleaned = self._clean_json_string(candidate)
                try:
                    json.loads(cleaned)  # Validate
                    return cleaned
                except json.JSONDecodeError:
                    continue

        return None

    def _extract_structured_data(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from text even if not valid JSON."""
        import re

        result = {}

        # Look for key: value patterns
        patterns = [
            r'"?(\w+)"?\s*:\s*"([^"]*)"',  # String values
            r'"?(\w+)"?\s*:\s*(\d+\.?\d*)',  # Numeric values
            r'"?(\w+)"?\s*:\s*(true|false)',  # Boolean values
            r'"?(\w+)"?\s*:\s*\[([^\]]*)\]',  # Array values
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                key = match[0]
                value = match[1] if len(match) > 1 else ""

                # Convert value to appropriate type
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                    value = float(value) if "." in value else int(value)
                elif "," in value:  # Possible array
                    value = [v.strip().strip('"') for v in value.split(",")]

                result[key] = value

        return result if result else None

    def _require_model(self, model: str, stage: str) -> str:
        if not model:
            raise RuntimeError(f"No available model client for stage '{stage}'")
        return model

    def _stubbed_model_output(self, role: str, operation: str, prompt: str) -> str:
        if role == "Planner":
            return "1. Review task requirements\n2. Outline implementation steps\n3. Validate deliverables"
        if role == "Coder":
            return "# Stubbed output based on plan."
        if role == "Validator":
            return '{"passes": true, "coverage": 0.99}'
        if role == "Reflector":
            return '{"analysis": "stub", "fixes": [], "selected_fix": 0, "revised_output": "", "confidence": 0.8}'
        if role == "Reviewer":
            return '{"score": 0.9, "rationale": "stubbed"}'
        return f"Stubbed response for {role}/{operation}."

    @retry_on_timeout
    def _call_model(
        self,
        model: str,
        prompt: str,
        role: str,
        operation: str,
        max_tokens: int = 8192,
        use_cache: bool = True,
        project_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        # Check if we should use Claude Code CLI instead of API
        if (
            self._use_claude_code
            and self._claude_code_provider
            and "claude" in model.lower()
        ):
            try:
                logger.debug(f"Using Claude Code CLI for {role}/{operation}")

                # Use provided project context or build one
                if not project_context:
                    project_context = self._build_project_context()

                response = self._claude_code_provider.call_model(
                    prompt=prompt,
                    model=model,
                    role=role,
                    operation=operation,
                    session_id=getattr(self, "_session_id", None),
                    use_cache=use_cache,
                    max_tokens=max_tokens,
                    project_context=project_context,
                )
                # Track as zero cost since it's included in subscription
                self.cost_estimator.track(0, role, operation, model=model)
                return scrub_pii(response)
            except Exception as exc:
                logger.warning(f"Claude Code CLI failed, falling back to API: {exc}")

                # Enhanced fallback notification
                notify_cli_failure(f"{role}/{operation}", str(exc), fallback_used=True)

                # Store failure info for monitoring
                if hasattr(self, "_claude_validation"):
                    self._claude_validation["last_cli_failure"] = {
                        "timestamp": time.time(),
                        "error": str(exc),
                        "role": role,
                        "operation": operation,
                    }

                # Continue to API fallback

        # Check cache first (for API calls)
        if use_cache:
            cached_response = self._model_cache.get_response(
                model=model, prompt=prompt, max_tokens=max_tokens
            )
            if cached_response:
                logger.debug(f"Cache hit for {role}/{operation}")
                return cached_response

        enhanced = self._enhance_prompt(prompt, role)
        if enhanced != prompt:
            extra_tokens = max(1, (len(enhanced) - len(prompt)) // 4)
            self.cost_estimator.track_estimated(
                extra_tokens, role, f"{operation}_enhance"
            )
            prompt = enhanced

        provider = _detect_provider(model)
        resolved_model = _resolve_model_alias(model)
        client_missing = (
            not resolved_model
            or (provider == "openai" and not self.openai_client)
            or (provider == "anthropic" and not self.anthropic_client)
            or (provider == "gemini" and not self.gemini_client)
        )

        if client_missing:
            stub = self._stubbed_model_output(role, operation, prompt)
            self.cost_estimator.track_estimated(
                len(prompt) // 4, role, f"{operation}_stub", model="stub"
            )
            return scrub_pii(stub)

        self._require_model(resolved_model, role)
        try:
            if provider == "openai" and self.openai_client:
                max_allowed = min(max_tokens, 4096)
                response = self.openai_client.chat.completions.create(
                    model=resolved_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_allowed,
                    timeout=2100,
                )
                output = response.choices[0].message.content or ""
                tokens = getattr(response.usage, "total_tokens", 0)
            elif provider == "anthropic" and self.anthropic_client:
                max_allowed = min(max_tokens, 4000)
                response = self.anthropic_client.messages.create(
                    model=resolved_model,
                    max_tokens=max_allowed,
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                )
                blocks = getattr(response, "content", []) or []
                output_parts = []
                for block in blocks:
                    text_part = getattr(block, "text", None) or getattr(
                        block, "value", None
                    )
                    if text_part:
                        output_parts.append(text_part)
                output = "".join(output_parts)
                usage = getattr(response, "usage", None)
                tokens = 0
                if usage:
                    tokens = getattr(usage, "input_tokens", 0) + getattr(
                        usage, "output_tokens", 0
                    )
            elif provider == "gemini" and self.gemini_client:
                response = self.gemini_client.generate_content(prompt)
                output = getattr(response, "text", "")
                tokens = len(prompt) // 4 + len(output) // 4
            else:
                output = self._stubbed_model_output(role, operation, prompt)
                tokens = len(prompt) // 4

            self.cost_estimator.track(tokens, role, operation, model=resolved_model)

            # Cache the response for future use
            if use_cache and output:
                # Longer TTL for Claude responses to maximize subscription value
                if "claude" in resolved_model.lower():
                    ttl = 1800 if role == "Planner" else 900  # 30min/15min for Claude
                else:
                    ttl = 600 if role == "Planner" else 300  # 10min/5min for others

                self._model_cache.cache_response(
                    model=model,
                    prompt=prompt,
                    response=output,
                    max_tokens=max_tokens,
                    ttl=ttl,
                )

            return scrub_pii(output)
        except Exception as exc:
            # Create structured error with context
            error_context = {
                "model": resolved_model,
                "role": role,
                "operation": operation,
                "provider": provider,
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
            }

            # Determine specific error code based on exception type
            error_code = ErrorCode.MODEL_CALL_FAILED
            if "timeout" in str(exc).lower():
                error_code = ErrorCode.MODEL_TIMEOUT
            elif "rate" in str(exc).lower() and "limit" in str(exc).lower():
                error_code = ErrorCode.MODEL_RATE_LIMITED
            elif "auth" in str(exc).lower() or "unauthorized" in str(exc).lower():
                error_code = ErrorCode.MODEL_AUTHENTICATION_FAILED
            elif "quota" in str(exc).lower() or "usage" in str(exc).lower():
                error_code = ErrorCode.MODEL_QUOTA_EXCEEDED

            model_error = create_model_error(
                f"Model call failed for {role}/{operation}: {str(exc)}",
                model=resolved_model,
                error_code=error_code,
                context=error_context,
                cause=exc,
            )

            # Add recovery suggestions based on error type
            if error_code == ErrorCode.MODEL_RATE_LIMITED:
                model_error.add_recovery_suggestion("Reduce request frequency")
                model_error.add_recovery_suggestion("Implement exponential backoff")
            elif error_code == ErrorCode.MODEL_AUTHENTICATION_FAILED:
                model_error.add_recovery_suggestion("Check API key configuration")
                model_error.add_recovery_suggestion("Verify authentication credentials")
            elif error_code == ErrorCode.MODEL_QUOTA_EXCEEDED:
                model_error.add_recovery_suggestion("Check usage limits")
                model_error.add_recovery_suggestion("Consider upgrading plan")

            handle_error(model_error, error_context)
            raise model_error

    async def _call_model_async(
        self,
        model: str,
        prompt: str,
        role: str,
        operation: str,
        max_tokens: int = 8192,
        use_cache: bool = True,
        project_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of _call_model for improved performance.

        Args:
            model: Model name
            prompt: Input prompt
            role: Role context
            operation: Operation context
            max_tokens: Maximum tokens
            use_cache: Whether to use cache
            project_context: Optional project context
            **kwargs: Additional arguments

        Returns:
            Model response
        """
        if self._async_orchestrator:
            try:
                return await self._async_orchestrator.call_model(
                    model=model,
                    prompt=prompt,
                    role=role,
                    operation=operation,
                    max_tokens=max_tokens,
                    use_cache=use_cache,
                    project_context=project_context or self._build_project_context(),
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"Async call failed, falling back to sync: {e}")

        # Fallback to sync version
        return self._call_model(
            model=model,
            prompt=prompt,
            role=role,
            operation=operation,
            max_tokens=max_tokens,
            use_cache=use_cache,
            project_context=project_context,
            **kwargs,
        )

    def _invoke_codex_cli(self, task_type: str, params: List[str], domain: str) -> str:
        enhanced_params = self._enhance_cli_params(params, "Coder")
        return invoke_codex_cli(task_type, enhanced_params, domain)

    # ------------------------------------------------------------------- routing
    def route_to_model(self, text: str, domain: str, vuln_flag: bool = False) -> str:
        complexity = len(text or "")

        # Prioritize Claude for maximum value from Anthropic Max subscription
        if self.anthropic_client:
            # Use Opus for complex/security-sensitive tasks
            if vuln_flag or complexity > 5000 or domain == "trading":
                return "claude_opus_4"
            # Use Sonnet for everything else (faster and included in Max plan)
            return "claude_sonnet_4"

        # Fall back to OpenAI only if Claude is unavailable
        if self.openai_client:
            # Use basic GPT-3.5 to minimize OpenAI costs
            return "openai_gpt_5"  # This now maps to gpt-3.5-turbo

        # Last resort: Gemini
        if self.gemini_client:
            return "gemini-2.5-pro"

        return ""

    # --------------------------------------------------------------------- nodes
    def planner(self, state: AgentState) -> AgentState:
        # Store current state for role context access
        self._current_state = state

        with timer(
            "planner_execution", tags={"domain": state.get("domain", "unknown")}
        ):
            plan = self.planner_role.decompose(
                state.get("task", ""), state.get("domain", "")
            )

        state["plan"] = plan.get("text", "")
        state["plan_epics"] = plan.get("epics", [])
        state["plan_model"] = plan.get("model")

        # Record metrics
        record_counter("planner_executions", tags={"domain": state.get("domain")})
        record_gauge("plan_epics_count", len(state["plan_epics"]))
        record_event(
            "planner_completed",
            MetricSeverity.INFO,
            domain=state.get("domain"),
            epics=len(state["plan_epics"]),
        )

        self._emit_event(
            "planner.completed",
            domain=state.get("domain"),
            epics=len(state["plan_epics"]),
        )
        return state

    def coder(self, state: AgentState) -> AgentState:
        # Store current state for role context access
        self._current_state = state

        with timer("coder_execution", tags={"domain": state.get("domain", "unknown")}):
            result = self.coder_role.generate(
                state.get("plan", ""), state.get("domain", "")
            )

        state["code"] = result.get("output", "")
        state["code_source"] = result.get("source")
        state["code_model"] = result.get("model")

        # Record metrics
        record_counter("coder_executions", tags={"domain": state.get("domain")})
        record_gauge("code_length", len(state["code"]))
        record_event(
            "coder_completed",
            MetricSeverity.INFO,
            domain=state.get("domain"),
            source=state.get("code_source"),
        )

        self._emit_event(
            "coder.completed",
            domain=state.get("domain"),
            source=state.get("code_source"),
        )
        return state

    def validator(self, state: AgentState) -> AgentState:
        # Store current state for role context access
        self._current_state = state

        result = self.validator_role.validate(
            state.get("code", ""), state.get("domain", "")
        )
        parsed = result.get("parsed", {})
        state["validation"] = parsed
        state["needs_reflect"] = not parsed.get("passes", True)
        self._emit_event(
            "validator.completed",
            domain=state.get("domain"),
            passes=parsed.get("passes", False),
            coverage=parsed.get("coverage"),
        )

        # Provide quality feedback based on validation results
        validation_score = 0.8 if parsed.get("passes", False) else 0.3
        self._provide_quality_feedback(
            validation_score,
            {
                "validation_result": parsed,
                "domain": state.get("domain"),
                "role": "Coder",
                "coverage": parsed.get("coverage"),
            },
        )

        return state

    def _validate_route(self, state: AgentState) -> str:
        return "reflector" if state.get("needs_reflect") else "reviewer"

    def reflector(self, state: AgentState) -> AgentState:
        # Store current state for role context access
        self._current_state = state
        result = self.reflector_role.reflect(
            state.get("validation", {}),
            state.get("code", ""),
            state.get("domain", ""),
            state.get("iterations", 0),
            state.get("vuln_flag", False),
        )
        state["code"] = result.get("output", state.get("code", ""))
        state["iterations"] = result.get("iterations", state.get("iterations", 0))
        state["confidence"] = result.get("confidence", state.get("confidence", 0.0))
        state["reflection_analysis"] = result.get("analysis")
        state["halted"] = result.get("halt")
        state["needs_reflect"] = not result.get("halt")
        self._emit_event(
            "reflector.completed",
            domain=state.get("domain"),
            iterations=state.get("iterations", 0),
        )
        return state

    def _reflect_route(self, state: AgentState) -> str:
        iterations = state.get("iterations", 0)
        confidence = state.get("confidence", 0.0)

        # Get configurable max iterations and confidence threshold
        reflection_cfg = self.agent_cfg.get("reflecting", {})
        max_iterations = reflection_cfg.get("max_iterations", 5)
        confidence_threshold = self.agent_cfg.get("reviewing", {}).get(
            "confidence_threshold", 0.8
        )

        # Support environment variable overrides
        env_max_iterations = os.getenv("REFLECTION_MAX_ITERATIONS")
        if env_max_iterations:
            try:
                max_iterations = int(env_max_iterations)
            except ValueError:
                pass  # Use config default

        env_confidence_threshold = os.getenv("REFLECTION_CONFIDENCE_THRESHOLD")
        if env_confidence_threshold:
            try:
                confidence_threshold = float(env_confidence_threshold)
            except ValueError:
                pass  # Use config default

        if iterations < max_iterations and confidence < confidence_threshold:
            return "coder"
        return "reviewer"

    def reviewer(self, state: AgentState) -> AgentState:
        # Store current state for role context access
        self._current_state = state

        result = self.reviewer_role.review(
            state.get("code", ""),
            state.get("domain", ""),
            state.get("vuln_flag", False),
        )
        state["confidence"] = result.get("confidence", 0.0)
        state["review_scores"] = result.get("scores", [])
        state["review_models"] = result.get("models", [])
        state["needs_reflect"] = state["confidence"] < 0.8
        record_metric(
            "review.confidence", state["confidence"], domain=state.get("domain")
        )
        self._emit_event(
            "reviewer.completed",
            domain=state.get("domain"),
            confidence=state["confidence"],
        )

        # Provide quality feedback to Claude Code provider if available
        self._provide_quality_feedback(
            state["confidence"],
            {
                "review_scores": result.get("scores", []),
                "rationales": result.get("rationales", []),
                "domain": state.get("domain"),
                "role": "Coder",
            },
        )

        return state

    def _run_offline_pipeline(self, state: AgentState) -> AgentState:
        """Run the offline pipeline with enhanced role coordination.

        Args:
            state: Initial agent state

        Returns:
            Final agent state after pipeline execution
        """
        state = AgentState(state)

        # Enhanced role coordination for Claude Code
        state = self._enrich_state_with_context(state)

        # Planning phase
        state = self.planner(state)

        # Main execution loop with reflection
        state = self._execute_main_loop(state)

        # Review and governance
        state = self._finalize_execution(state)

        return state

    def _execute_main_loop(self, state: AgentState) -> AgentState:
        """Execute the main coding loop with validation and reflection.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        # Get max iterations from config with fallback to default
        reflection_cfg = self.agent_cfg.get("reflecting", {})
        max_iterations = reflection_cfg.get("max_iterations", 5)

        # Support environment variable override
        env_max_iterations = os.getenv("REFLECTION_MAX_ITERATIONS")
        if env_max_iterations:
            try:
                max_iterations = int(env_max_iterations)
                logger.info(
                    f"Using reflection max_iterations from environment: {max_iterations}"
                )
            except ValueError:
                logger.warning(
                    f"Invalid REFLECTION_MAX_ITERATIONS value: {env_max_iterations}, using config default: {max_iterations}"
                )

        try:
            # Pass planner context to coder for better reasoning
            state = self._transfer_role_context(state, "planner", "coder")
            state = self.coder(state)

            # Pass coder context to validator
            state = self._transfer_role_context(state, "coder", "validator")
            state = self.validator(state)

            # Reflection loop if needed
            if state.get("needs_reflect"):
                state = self._execute_reflection_loop(state, max_iterations)

        except Exception as e:
            logger.error(f"Error in main execution loop: {e}")
            state["error"] = str(e)
            state["needs_reflect"] = True

        return state

    def _execute_reflection_loop(
        self, state: AgentState, max_iterations: int
    ) -> AgentState:
        """Execute the reflection loop for iterative improvement with comprehensive audit logging.

        Args:
            state: Current agent state
            max_iterations: Maximum number of reflection iterations

        Returns:
            Updated agent state
        """
        # Start reflection audit session
        domain = state.get("domain", "unknown")
        task = state.get("task", "reflection")
        initial_confidence = state.get("confidence", 0.0)

        audit_session_id = start_reflection_session(
            domain=domain,
            task=task,
            initial_confidence=initial_confidence,
            max_iterations=max_iterations,
            configuration={
                "early_termination_enabled": True,
                "domain": domain,
                "validation_issues": state.get("validation", {}),
            },
        )

        # Get early termination configuration
        reflection_cfg = self.agent_cfg.get("reflecting", {})
        early_termination_cfg = reflection_cfg.get("early_termination", {})

        enable_early_termination = early_termination_cfg.get("enable", True)
        stagnation_threshold = early_termination_cfg.get("stagnation_threshold", 3)
        min_iterations = early_termination_cfg.get("min_iterations", 1)
        progress_threshold = early_termination_cfg.get("progress_threshold", 0.1)

        # Track confidence history for early termination
        confidence_history = []
        stagnation_count = 0

        logger.info(
            f"Starting reflection loop: max_iterations={max_iterations}, early_termination={enable_early_termination}"
        )

        # Log initial validation analysis
        validation_issues = self._analyze_validation_issues(state.get("validation", {}))
        log_reflection_step(
            audit_session_id,
            ReflectionPhase.VALIDATION_ANALYSIS,
            input_data={"validation": state.get("validation", {})},
            output_data={"issues_identified": len(validation_issues)},
            confidence_before=initial_confidence,
            issues=validation_issues,
        )

        final_decision = ReflectionDecision.CONTINUE_REFLECTION
        termination_reason = None

        for iteration in range(max_iterations):
            try:
                previous_confidence = state.get("confidence", 0.0)

                # Log iteration start
                log_reflection_step(
                    audit_session_id,
                    ReflectionPhase.ISSUE_IDENTIFICATION,
                    input_data={
                        "iteration": iteration,
                        "confidence": previous_confidence,
                    },
                    confidence_before=previous_confidence,
                    iteration=iteration,
                )

                # Pass validation context to reflector
                state = self._transfer_role_context(state, "validator", "reflector")

                # Execute reflector with audit logging
                reflection_start = time.time()
                state = self.reflector(state)
                reflection_duration = time.time() - reflection_start

                current_confidence = state.get("confidence", 0.0)
                confidence_history.append(current_confidence)

                # Log reflection results
                reflection_analysis = state.get("reflection_analysis", {})
                fixes_generated = (
                    reflection_analysis.get("fixes", [])
                    if isinstance(reflection_analysis, dict)
                    else []
                )
                selected_fix = (
                    reflection_analysis.get("selected_fix")
                    if isinstance(reflection_analysis, dict)
                    else None
                )

                log_reflection_step(
                    audit_session_id,
                    ReflectionPhase.FIX_GENERATION,
                    input_data={"previous_confidence": previous_confidence},
                    output_data={
                        "confidence": current_confidence,
                        "fixes_count": len(fixes_generated),
                        "selected_fix_index": selected_fix,
                    },
                    confidence_before=previous_confidence,
                    confidence_after=current_confidence,
                    fixes=fixes_generated,
                    selected_fix=str(selected_fix)
                    if selected_fix is not None
                    else None,
                    duration=reflection_duration,
                    iteration=iteration,
                )

                logger.debug(
                    f"Reflection iteration {iteration}: confidence {previous_confidence:.3f} -> {current_confidence:.3f}"
                )

                # Check if reflection indicates halt
                if not state.get("needs_reflect"):
                    final_decision = ReflectionDecision.HALT_REFLECTION
                    termination_reason = "reflector_decision"
                    logger.info(
                        f"Reflection loop halted by reflector after {iteration + 1} iterations"
                    )
                    break

                # Early termination checks
                if enable_early_termination and iteration >= min_iterations:
                    # Check for stagnation (no significant improvement)
                    confidence_improvement = current_confidence - previous_confidence
                    if confidence_improvement < progress_threshold:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0

                    # Terminate if stagnating for too long
                    if stagnation_count >= stagnation_threshold:
                        final_decision = ReflectionDecision.EARLY_TERMINATION
                        termination_reason = "stagnation"
                        logger.info(
                            f"Early termination due to stagnation after {iteration + 1} iterations (stagnation_count={stagnation_count})"
                        )

                        log_reflection_step(
                            audit_session_id,
                            ReflectionPhase.TERMINATION_DECISION,
                            input_data={
                                "stagnation_count": stagnation_count,
                                "threshold": stagnation_threshold,
                            },
                            output_data={
                                "decision": "early_termination",
                                "reason": "stagnation",
                            },
                            confidence_after=current_confidence,
                            decisions=["early_termination"],
                            stagnation_count=stagnation_count,
                        )

                        state["early_termination_reason"] = "stagnation"
                        state["needs_reflect"] = False
                        break

                    # Check for regression (confidence getting worse)
                    if len(confidence_history) >= 2:
                        recent_trend = confidence_history[-1] - confidence_history[-2]
                        if recent_trend < -0.2:  # Significant regression
                            final_decision = ReflectionDecision.EARLY_TERMINATION
                            termination_reason = "regression"
                            logger.info(
                                f"Early termination due to confidence regression after {iteration + 1} iterations"
                            )

                            log_reflection_step(
                                audit_session_id,
                                ReflectionPhase.TERMINATION_DECISION,
                                input_data={"confidence_trend": recent_trend},
                                output_data={
                                    "decision": "early_termination",
                                    "reason": "regression",
                                },
                                confidence_after=current_confidence,
                                decisions=["early_termination"],
                                confidence_trend=recent_trend,
                            )

                            state["early_termination_reason"] = "regression"
                            state["needs_reflect"] = False
                            break

                # Log iteration completion
                self._reflection_auditor.log_iteration_complete(
                    audit_session_id,
                    iteration + 1,
                    current_confidence,
                    state.get("needs_reflect", False),
                    termination_reason,
                )

                # Pass reflection context back to coder
                state = self._transfer_role_context(state, "reflector", "coder")

                # Execute coder and validator with timing
                coder_start = time.time()
                state = self.coder(state)
                coder_duration = time.time() - coder_start

                validator_start = time.time()
                state = self.validator(state)
                validator_duration = time.time() - validator_start

                # Log code modification
                log_reflection_step(
                    audit_session_id,
                    ReflectionPhase.CODE_MODIFICATION,
                    input_data={"iteration": iteration},
                    output_data={
                        "code_length": len(state.get("code", "")),
                        "coder_duration": coder_duration,
                        "validator_duration": validator_duration,
                    },
                    confidence_after=state.get("confidence", current_confidence),
                    iteration=iteration,
                    coder_duration=coder_duration,
                    validator_duration=validator_duration,
                )

            except Exception as e:
                # Handle reflection loop errors with structured error handling
                reflection_error = EnterpriseAgentError(
                    ErrorCode.REFLECTION_LOOP_FAILED,
                    f"Error in reflection loop iteration {iteration}: {str(e)}",
                    context={
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "current_confidence": state.get("confidence", 0.0),
                        "needs_reflect": state.get("needs_reflect", False),
                    },
                    recovery_suggestions=[
                        "Check validation results for issues",
                        "Review reflection configuration",
                        "Consider reducing max_iterations",
                    ],
                    cause=e,
                )

                handle_error(reflection_error)

                # Log error in audit trail
                log_reflection_step(
                    audit_session_id,
                    ReflectionPhase.TERMINATION_DECISION,
                    input_data={"iteration": iteration},
                    output_data={"decision": "error_termination"},
                    error=str(e),
                    confidence_after=state.get("confidence", 0.0),
                    decisions=["error_termination"],
                    iteration=iteration,
                )

                state["reflection_error"] = reflection_error.details.to_dict()
                state["early_termination_reason"] = "error"
                final_decision = ReflectionDecision.ERROR_TERMINATION
                termination_reason = "error"
                break

        # Handle max iterations reached
        if (
            len(confidence_history) >= max_iterations
            and final_decision == ReflectionDecision.CONTINUE_REFLECTION
        ):
            final_decision = ReflectionDecision.MAX_ITERATIONS_REACHED
            termination_reason = "max_iterations"

        # Record reflection loop statistics
        final_confidence = state.get("confidence", 0.0)
        confidence_gain = final_confidence - initial_confidence

        state["reflection_stats"] = {
            "iterations_completed": len(confidence_history),
            "max_iterations": max_iterations,
            "initial_confidence": initial_confidence,
            "final_confidence": final_confidence,
            "confidence_gain": confidence_gain,
            "confidence_history": confidence_history,
            "early_termination": state.get("early_termination_reason") is not None,
            "termination_reason": state.get("early_termination_reason", "completed"),
            "audit_session_id": audit_session_id,
        }

        # Finish audit session
        finish_reflection_session(
            audit_session_id,
            final_decision,
            final_confidence,
            outcome={
                "iterations_completed": len(confidence_history),
                "confidence_gain": confidence_gain,
                "termination_reason": termination_reason,
                "success": confidence_gain > 0,
            },
        )

        logger.info(
            f"Reflection loop completed: {len(confidence_history)} iterations, confidence {initial_confidence:.3f} -> {final_confidence:.3f} (gain: {confidence_gain:+.3f})"
        )

        return state

    def _analyze_validation_issues(
        self, validation_data: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Analyze validation data to extract issues for audit logging.

        Args:
            validation_data: Validation results

        Returns:
            List of validation issues
        """
        issues = []

        if not isinstance(validation_data, dict):
            return issues

        # Extract general validation failures
        if not validation_data.get("passes", True):
            issues.append(
                ValidationIssue(
                    issue_type="validation_failure",
                    severity="high",
                    description="General validation failure",
                    confidence=1.0 - validation_data.get("coverage", 0.0),
                )
            )

        # Extract coverage issues
        coverage = validation_data.get("coverage", 1.0)
        if coverage < 0.8:
            issues.append(
                ValidationIssue(
                    issue_type="low_coverage",
                    severity="medium" if coverage > 0.5 else "high",
                    description=f"Low test coverage: {coverage:.1%}",
                    confidence=1.0 - coverage,
                )
            )

        # Extract specific errors or warnings if available
        errors = validation_data.get("errors", [])
        if isinstance(errors, list):
            for error in errors[:5]:  # Limit to first 5 errors
                issues.append(
                    ValidationIssue(
                        issue_type="validation_error",
                        severity="high",
                        description=str(error)[:200],  # Truncate long descriptions
                        confidence=0.9,
                    )
                )

        return issues

    def _finalize_execution(self, state: AgentState) -> AgentState:
        """Finalize execution with review and governance.

        Args:
            state: Current agent state

        Returns:
            Final agent state
        """
        try:
            # Pass final context to reviewer
            state = self._transfer_role_context(state, "validator", "reviewer")
            state = self.reviewer(state)

            if state.get("needs_reflect"):
                state = self.reflector(state)

            state = self._governance_node(state)

        except Exception as e:
            logger.error(f"Error in finalization: {e}")
            state["finalization_error"] = str(e)

        return state

    def _governance_node(self, state: AgentState) -> AgentState:
        approved = self.governance.check(state)
        self._emit_event(
            "governance.checked",
            domain=state.get("domain"),
            approved=approved,
            metrics=self.governance.latest_metrics,
        )
        if not approved and not self.governance.hitl_check("high", state):
            state["governance_blocked"] = True
        return state

    # ------------------------------------------------------------------- public
    def run_mode(
        self, domain: str, task: str, vuln_flag: bool = False
    ) -> Dict[str, Any]:
        """Run the agent in the specified mode with enhanced error handling.

        Args:
            domain: The domain for the task (e.g., 'coding', 'content')
            task: The task description
            vuln_flag: Whether to enable vulnerability scanning

        Returns:
            Dictionary containing the execution results

        Raises:
            ValueError: If domain or task is invalid
            RuntimeError: If execution fails critically
        """
        # Input validation
        if not domain or not isinstance(domain, str):
            raise ValueError("Domain must be a non-empty string")
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        try:
            initial = AgentState(
                {
                    "task": task,
                    "domain": domain,
                    "iterations": 0,
                    "confidence": 0.0,
                    "vuln_flag": vuln_flag,
                }
            )

            # Execute pipeline with fallback handling
            result = self._execute_pipeline(initial)

            # Post-process results
            result = self._post_process_results(result, domain)

            # Record completion event
            self._record_completion_event(result, domain)

            # Cleanup
            self.memory.prune()

            return dict(result)

        except Exception as e:
            logger.error(f"Critical error in run_mode: {e}")
            raise RuntimeError(f"Agent execution failed: {e}") from e

    async def run_mode_async(
        self, domain: str, task: str, vuln_flag: bool = False
    ) -> Dict[str, Any]:
        """Async version of run_mode for improved performance.

        Args:
            domain: The domain for the task (e.g., 'coding', 'content')
            task: The task description
            vuln_flag: Whether to enable vulnerability scanning

        Returns:
            Dictionary containing the execution results

        Raises:
            ValueError: If domain or task is invalid
            RuntimeError: If execution fails critically
        """
        # Input validation
        if not domain or not isinstance(domain, str):
            raise ValueError("Domain must be a non-empty string")
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        try:
            if self._async_orchestrator:
                logger.info("Using async orchestrator for enhanced performance")

                # Use async orchestrator's batch processing capabilities
                requests = [
                    {
                        "role": "Planner",
                        "operation": "decompose",
                        "prompt": f"Plan for task: {task} in domain: {domain}",
                        "model": self.route_to_model(task, domain, vuln_flag),
                    }
                ]

                # Execute planning asynchronously
                results = await self._async_orchestrator.batch_call_models(requests)

                if results and not results[0].startswith("Error:"):
                    # Build initial state with planning result
                    initial = AgentState(
                        {
                            "task": task,
                            "domain": domain,
                            "iterations": 0,
                            "confidence": 0.0,
                            "vuln_flag": vuln_flag,
                            "plan": results[0],
                        }
                    )
                else:
                    # Fallback to sync planning
                    initial = AgentState(
                        {
                            "task": task,
                            "domain": domain,
                            "iterations": 0,
                            "confidence": 0.0,
                            "vuln_flag": vuln_flag,
                        }
                    )
            else:
                initial = AgentState(
                    {
                        "task": task,
                        "domain": domain,
                        "iterations": 0,
                        "confidence": 0.0,
                        "vuln_flag": vuln_flag,
                    }
                )

            # Execute pipeline (mix of async and sync based on capabilities)
            result = await self._execute_pipeline_async(initial)

            # Post-process results
            result = self._post_process_results(result, domain)

            # Record completion event
            self._record_completion_event(result, domain)

            # Cleanup
            self.memory.prune()

            return dict(result)

        except Exception as e:
            logger.error(f"Critical error in async run_mode: {e}")
            raise RuntimeError(f"Async agent execution failed: {e}") from e

    async def _execute_pipeline_async(self, initial: AgentState) -> AgentState:
        """Execute pipeline with async optimizations where possible.

        Args:
            initial: Initial agent state

        Returns:
            Final agent state
        """
        if self._async_orchestrator and self._async_enabled:
            try:
                # Use async orchestrator for parallel role execution
                if not initial.get("plan"):
                    # Execute planning with async orchestrator
                    plan_result = await self._async_orchestrator.call_model(
                        model=self.route_to_model(
                            initial["task"],
                            initial["domain"],
                            initial.get("vuln_flag", False),
                        ),
                        prompt=f"Create a detailed plan for: {initial['task']} in domain: {initial['domain']}",
                        role="Planner",
                        operation="decompose",
                        project_context=self._build_project_context(),
                    )
                    initial["plan"] = plan_result

                # Continue with sync pipeline for complex workflow
                return self._run_offline_pipeline(initial)

            except Exception as e:
                logger.warning(f"Async pipeline failed, falling back to sync: {e}")
                return self._run_offline_pipeline(initial)
        else:
            return self._execute_pipeline(initial)

    def _execute_pipeline(self, initial: AgentState) -> AgentState:
        """Execute the appropriate pipeline based on available clients.

        Args:
            initial: Initial agent state

        Returns:
            Final agent state
        """
        if not any([self.openai_client, self.anthropic_client, self.gemini_client]):
            logger.info("No API clients available, using offline pipeline")
            return self._run_offline_pipeline(initial)
        else:
            try:
                result = self.graph.invoke(initial)
                if result is None:
                    logger.warning(
                        "Graph invocation returned None; falling back to offline pipeline."
                    )
                    return self._run_offline_pipeline(initial)
                return result
            except Exception as e:
                logger.error(
                    f"Graph execution failed: {e}, falling back to offline pipeline"
                )
                return self._run_offline_pipeline(initial)

    def _post_process_results(self, result: AgentState, domain: str) -> AgentState:
        """Post-process the execution results.

        Args:
            result: Execution result state
            domain: Task domain

        Returns:
            Processed result state
        """
        # Normalize plan structure
        plan_text = result.get("plan")
        plan_epics = result.get("plan_epics")
        if not isinstance(plan_text, dict) and (plan_text or plan_epics):
            result["plan"] = {
                "text": plan_text or "",
                "epics": plan_epics or [],
                "model": result.get("plan_model"),
            }

        # Add domain and cost information
        result["domain"] = domain
        result["cost_summary"] = self.cost_estimator.summary()

        return result

    def _record_completion_event(self, result: AgentState, domain: str) -> None:
        """Record the completion event with telemetry.

        Args:
            result: Execution result state
            domain: Task domain
        """
        try:
            record_event(
                "agent.run_completed",
                domain=domain,
                confidence=result.get("confidence", 0.0),
                cost=result["cost_summary"].get("total_cost", 0.0),
            )
        except Exception as e:
            logger.warning(f"Failed to record completion event: {e}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered during orchestrator operation.

        Returns:
            Dictionary containing error statistics and recent errors
        """
        return self._error_handler.get_error_summary()

    def reset_error_history(self) -> None:
        """Reset the error history for clean state (useful for testing)."""
        self._error_handler.error_history.clear()
        self._error_handler.error_counts.clear()


__all__ = ["AgentOrchestrator"]
