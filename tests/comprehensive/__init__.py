"""Comprehensive test suite for Enterprise Agent.

This package contains a 7-layer comprehensive testing framework that validates
the Enterprise Agent from every conceivable angle:

Layer 1: Component Unit Tests - Individual module functionality
Layer 2: Integration Tests - Component interactions and workflows
Layer 3: End-to-End Functional Tests - Complete user workflows
Layer 4: Performance & Scalability Tests - System performance under load
Layer 5: Security & Resilience Tests - Security hardening and fault tolerance
Layer 6: Environment & Configuration Tests - Deployment and configuration scenarios
Layer 7: Real-World Scenario Tests - Production-like usage patterns

Usage:
    python -m tests.comprehensive.test_complete_functionality
"""

__version__ = "1.0.0"
__all__ = ["TestFramework", "ComprehensiveTestRunner"]
