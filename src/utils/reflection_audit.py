"""Reflection audit trail logging system for Enterprise Agent.

This module provides comprehensive logging and auditing capabilities for the
reflection loop process, enabling detailed analysis and debugging of reflection
behavior and decision-making.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReflectionPhase(Enum):
    """Phases of the reflection process."""

    VALIDATION_ANALYSIS = "validation_analysis"
    ISSUE_IDENTIFICATION = "issue_identification"
    FIX_GENERATION = "fix_generation"
    FIX_SELECTION = "fix_selection"
    CODE_MODIFICATION = "code_modification"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    TERMINATION_DECISION = "termination_decision"


class ReflectionDecision(Enum):
    """Types of decisions made during reflection."""

    CONTINUE_REFLECTION = "continue_reflection"
    HALT_REFLECTION = "halt_reflection"
    EARLY_TERMINATION = "early_termination"
    ERROR_TERMINATION = "error_termination"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


@dataclass
class ValidationIssue:
    """Individual validation issue identified during reflection."""

    issue_type: str
    severity: str
    description: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ReflectionStep:
    """A single step in the reflection process."""

    step_id: str
    phase: ReflectionPhase
    timestamp: float
    duration: Optional[float] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    decisions: List[str] = field(default_factory=list)
    confidence_before: Optional[float] = None
    confidence_after: Optional[float] = None
    issues_identified: List[ValidationIssue] = field(default_factory=list)
    fixes_generated: List[str] = field(default_factory=list)
    selected_fix: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "phase": self.phase.value,
            "issues_identified": [issue.to_dict() for issue in self.issues_identified],
        }


@dataclass
class ReflectionSession:
    """Complete reflection session audit trail."""

    session_id: str
    domain: str
    task: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    iteration_count: int = 0
    max_iterations: int = 5
    final_decision: Optional[ReflectionDecision] = None
    initial_confidence: float = 0.0
    final_confidence: float = 0.0
    confidence_improvement: float = 0.0
    steps: List[ReflectionStep] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish_session(
        self,
        final_decision: ReflectionDecision,
        final_confidence: float,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finish the reflection session."""
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        self.final_decision = final_decision
        self.final_confidence = final_confidence
        self.confidence_improvement = final_confidence - self.initial_confidence
        self.outcome = outcome or {}

    def add_step(self, step: ReflectionStep) -> None:
        """Add a reflection step to the session."""
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "final_decision": self.final_decision.value
            if self.final_decision
            else None,
            "steps": [step.to_dict() for step in self.steps],
        }


class ReflectionAuditor:
    """Manages reflection audit trail logging and analysis."""

    def __init__(
        self,
        enabled: bool = True,
        log_level: str = "INFO",
        audit_path: Optional[Path] = None,
        max_sessions: int = 1000,
        auto_export: bool = True,
    ):
        """Initialize reflection auditor.

        Args:
            enabled: Whether audit logging is enabled
            log_level: Logging level for audit messages
            audit_path: Path to store audit files
            max_sessions: Maximum sessions to keep in memory
            auto_export: Whether to automatically export sessions
        """
        self.enabled = enabled
        self.audit_path = audit_path or Path(".reflection_audit")
        self.max_sessions = max_sessions
        self.auto_export = auto_export

        # Active sessions
        self._active_sessions: Dict[str, ReflectionSession] = {}
        self._completed_sessions: List[ReflectionSession] = []

        # Setup logging and directories
        if self.enabled:
            self.audit_path.mkdir(exist_ok=True)
            self._setup_logging(log_level)

    def _setup_logging(self, log_level: str) -> None:
        """Setup audit logging."""
        self.logger = logging.getLogger(f"{__name__}.audit")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Create audit log handler
        log_file = self.audit_path / "reflection_audit.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def start_session(
        self,
        domain: str,
        task: str,
        initial_confidence: float = 0.0,
        max_iterations: int = 5,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new reflection session.

        Args:
            domain: Task domain
            task: Task description
            initial_confidence: Initial confidence score
            max_iterations: Maximum reflection iterations
            configuration: Reflection configuration

        Returns:
            Session ID
        """
        if not self.enabled:
            return ""

        session_id = f"refl_{int(time.time() * 1000000)}"
        session = ReflectionSession(
            session_id=session_id,
            domain=domain,
            task=task,
            start_time=time.time(),
            initial_confidence=initial_confidence,
            max_iterations=max_iterations,
            configuration=configuration or {},
        )

        self._active_sessions[session_id] = session

        self.logger.info(
            f"Started reflection session {session_id} for domain '{domain}'"
        )

        return session_id

    def log_step(
        self,
        session_id: str,
        phase: ReflectionPhase,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        confidence_before: Optional[float] = None,
        confidence_after: Optional[float] = None,
        decisions: Optional[List[str]] = None,
        issues: Optional[List[ValidationIssue]] = None,
        fixes: Optional[List[str]] = None,
        selected_fix: Optional[str] = None,
        error: Optional[str] = None,
        **metadata: Any,
    ) -> str:
        """Log a reflection step.

        Args:
            session_id: Session ID
            phase: Reflection phase
            input_data: Input data for this step
            output_data: Output data from this step
            confidence_before: Confidence before this step
            confidence_after: Confidence after this step
            decisions: Decisions made in this step
            issues: Issues identified
            fixes: Fixes generated
            selected_fix: Selected fix
            error: Error message if any
            **metadata: Additional metadata

        Returns:
            Step ID
        """
        if not self.enabled or session_id not in self._active_sessions:
            return ""

        step_id = f"{session_id}_step_{len(self._active_sessions[session_id].steps)}"
        step = ReflectionStep(
            step_id=step_id,
            phase=phase,
            timestamp=time.time(),
            input_data=input_data or {},
            output_data=output_data or {},
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            decisions=decisions or [],
            issues_identified=issues or [],
            fixes_generated=fixes or [],
            selected_fix=selected_fix,
            error=error,
            metadata=metadata,
        )

        self._active_sessions[session_id].add_step(step)

        # Log step details
        log_msg = f"Session {session_id} - {phase.value}"
        if confidence_before is not None and confidence_after is not None:
            confidence_change = confidence_after - confidence_before
            log_msg += f" (confidence: {confidence_before:.3f} → {confidence_after:.3f}, Δ{confidence_change:+.3f})"
        if error:
            log_msg += f" [ERROR: {error}]"

        self.logger.info(log_msg)

        return step_id

    def log_iteration_complete(
        self,
        session_id: str,
        iteration: int,
        confidence: float,
        needs_continue: bool,
        termination_reason: Optional[str] = None,
    ) -> None:
        """Log completion of a reflection iteration.

        Args:
            session_id: Session ID
            iteration: Iteration number
            confidence: Current confidence
            needs_continue: Whether reflection should continue
            termination_reason: Reason for termination if applicable
        """
        if not self.enabled or session_id not in self._active_sessions:
            return

        session = self._active_sessions[session_id]
        session.iteration_count = iteration

        decision = "continue" if needs_continue else "halt"
        log_msg = f"Session {session_id} - Iteration {iteration} complete: {decision} (confidence: {confidence:.3f})"
        if termination_reason:
            log_msg += f" - {termination_reason}"

        self.logger.info(log_msg)

        # Log step for iteration completion
        self.log_step(
            session_id,
            ReflectionPhase.TERMINATION_DECISION,
            input_data={"iteration": iteration, "confidence": confidence},
            output_data={"decision": decision, "reason": termination_reason},
            confidence_after=confidence,
            decisions=[decision],
            iteration=iteration,
            termination_reason=termination_reason,
        )

    def finish_session(
        self,
        session_id: str,
        final_decision: ReflectionDecision,
        final_confidence: float,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Optional[ReflectionSession]:
        """Finish a reflection session.

        Args:
            session_id: Session ID
            final_decision: Final decision
            final_confidence: Final confidence score
            outcome: Session outcome data

        Returns:
            Completed session
        """
        if not self.enabled or session_id not in self._active_sessions:
            return None

        session = self._active_sessions.pop(session_id)
        session.finish_session(final_decision, final_confidence, outcome)

        self._completed_sessions.append(session)

        # Trim sessions if needed
        if len(self._completed_sessions) > self.max_sessions:
            self._completed_sessions = self._completed_sessions[
                -self.max_sessions // 2 :
            ]

        self.logger.info(
            f"Finished reflection session {session_id} - "
            f"Decision: {final_decision.value}, "
            f"Final confidence: {final_confidence:.3f}, "
            f"Improvement: {session.confidence_improvement:+.3f}, "
            f"Duration: {session.total_duration:.2f}s"
        )

        # Auto-export if enabled
        if self.auto_export:
            self._export_session(session)

        return session

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a session.

        Args:
            session_id: Session ID

        Returns:
            Session summary
        """
        # Check active sessions
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
        else:
            # Check completed sessions
            for session in self._completed_sessions:
                if session.session_id == session_id:
                    break
            else:
                return None

        return {
            "session_id": session.session_id,
            "domain": session.domain,
            "status": "active" if session.end_time is None else "completed",
            "iteration_count": session.iteration_count,
            "max_iterations": session.max_iterations,
            "initial_confidence": session.initial_confidence,
            "final_confidence": session.final_confidence,
            "confidence_improvement": session.confidence_improvement,
            "steps_count": len(session.steps),
            "duration": session.total_duration,
            "final_decision": session.final_decision.value
            if session.final_decision
            else None,
        }

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions summary.

        Args:
            limit: Maximum number of sessions

        Returns:
            List of recent session summaries
        """
        recent_sessions = []

        # Add active sessions
        for session in self._active_sessions.values():
            recent_sessions.append(self.get_session_summary(session.session_id))

        # Add completed sessions (most recent first)
        for session in reversed(self._completed_sessions[-limit:]):
            recent_sessions.append(self.get_session_summary(session.session_id))

        return recent_sessions[:limit]

    def get_reflection_analytics(self) -> Dict[str, Any]:
        """Get analytics about reflection performance.

        Returns:
            Analytics data
        """
        if not self._completed_sessions:
            return {"message": "No completed sessions available"}

        # Calculate statistics
        sessions = self._completed_sessions
        total_sessions = len(sessions)

        # Confidence improvements
        improvements = [s.confidence_improvement for s in sessions]
        avg_improvement = sum(improvements) / len(improvements)
        successful_sessions = len([i for i in improvements if i > 0])

        # Iteration counts
        iterations = [s.iteration_count for s in sessions]
        avg_iterations = sum(iterations) / len(iterations)

        # Durations
        durations = [s.total_duration for s in sessions if s.total_duration]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Decision types
        decisions = {}
        for session in sessions:
            if session.final_decision:
                decisions[session.final_decision.value] = (
                    decisions.get(session.final_decision.value, 0) + 1
                )

        # Domain breakdown
        domains = {}
        for session in sessions:
            domains[session.domain] = domains.get(session.domain, 0) + 1

        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": successful_sessions / total_sessions
            if total_sessions > 0
            else 0,
            "average_confidence_improvement": avg_improvement,
            "average_iterations": avg_iterations,
            "average_duration": avg_duration,
            "decision_breakdown": decisions,
            "domain_breakdown": domains,
            "recent_sessions": self.get_recent_sessions(5),
        }

    def _export_session(self, session: ReflectionSession) -> None:
        """Export session to file.

        Args:
            session: Session to export
        """
        try:
            date_str = datetime.fromtimestamp(session.start_time).strftime("%Y%m%d")
            export_file = (
                self.audit_path / f"session_{date_str}_{session.session_id}.json"
            )

            with export_file.open("w") as f:
                json.dump(session.to_dict(), f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to export session {session.session_id}: {e}")

    def export_analytics(self, output_path: Optional[Path] = None) -> Path:
        """Export analytics to file.

        Args:
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        output_path = output_path or (
            self.audit_path / f"analytics_{int(time.time())}.json"
        )

        analytics = self.get_reflection_analytics()
        with output_path.open("w") as f:
            json.dump(analytics, f, indent=2)

        return output_path


# Global auditor instance
_global_auditor: Optional[ReflectionAuditor] = None


def get_reflection_auditor() -> ReflectionAuditor:
    """Get the global reflection auditor instance."""
    global _global_auditor
    if _global_auditor is None:
        enabled = os.getenv("REFLECTION_AUDIT_ENABLED", "true").lower() == "true"
        audit_path = os.getenv("REFLECTION_AUDIT_PATH")
        _global_auditor = ReflectionAuditor(
            enabled=enabled, audit_path=Path(audit_path) if audit_path else None
        )
    return _global_auditor


def initialize_auditor(
    enabled: bool = True, audit_path: Optional[Path] = None, **kwargs
) -> ReflectionAuditor:
    """Initialize the global reflection auditor.

    Args:
        enabled: Whether audit logging is enabled
        audit_path: Path to store audit files
        **kwargs: Additional configuration

    Returns:
        Initialized auditor
    """
    global _global_auditor
    _global_auditor = ReflectionAuditor(
        enabled=enabled, audit_path=audit_path, **kwargs
    )
    return _global_auditor


# Convenience functions
def start_reflection_session(domain: str, task: str, **kwargs) -> str:
    """Start a reflection session."""
    return get_reflection_auditor().start_session(domain, task, **kwargs)


def log_reflection_step(session_id: str, phase: ReflectionPhase, **kwargs) -> str:
    """Log a reflection step."""
    return get_reflection_auditor().log_step(session_id, phase, **kwargs)


def finish_reflection_session(
    session_id: str, decision: ReflectionDecision, **kwargs
) -> Optional[ReflectionSession]:
    """Finish a reflection session."""
    return get_reflection_auditor().finish_session(session_id, decision, **kwargs)


__all__ = [
    "ReflectionPhase",
    "ReflectionDecision",
    "ValidationIssue",
    "ReflectionStep",
    "ReflectionSession",
    "ReflectionAuditor",
    "get_reflection_auditor",
    "initialize_auditor",
    "start_reflection_session",
    "log_reflection_step",
    "finish_reflection_session",
]
