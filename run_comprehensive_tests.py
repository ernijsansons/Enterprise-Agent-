#!/usr/bin/env python3
"""Quick runner for comprehensive Enterprise Agent tests.

This script demonstrates the comprehensive test suite by running
a subset of tests to validate core functionality.

Usage:
    python run_comprehensive_tests.py
    python run_comprehensive_tests.py --layer 1
    python run_comprehensive_tests.py --quick
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s - %(message)s"  # Reduce noise
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Enterprise Agent comprehensive tests"
    )
    parser.add_argument(
        "--layer",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run specific test layer only",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick subset of tests"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


async def run_quick_tests():
    """Run a quick subset of tests for demonstration."""
    print("🚀 Running Quick Comprehensive Test Demonstration")
    print("=" * 55)

    # Import here to avoid issues if modules missing
    try:
        from tests.comprehensive.layer1_unit.test_async_complete import (
            TestAsyncComplete,
        )
        from tests.comprehensive.layer1_unit.test_orchestrator_complete import (
            TestOrchestratorComplete,
        )
        from tests.comprehensive.test_framework import (
            TestFramework,
            TestLayer,
            TestSuite,
        )
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        print("Make sure you're running from the project root directory")
        return False

    framework = TestFramework()

    # Create a quick test layer with essential tests
    quick_layer = TestLayer(
        name="Quick Validation Tests",
        description="Essential functionality validation",
        enabled=True,
    )

    # Add orchestrator tests
    orch_test_class = TestOrchestratorComplete()
    orch_suite = TestSuite(
        name="Core Orchestrator Tests",
        description="Essential orchestrator functionality",
        tests=[
            orch_test_class.test_orchestrator_initialization,
            orch_test_class.test_model_routing,
            orch_test_class.test_call_model_functionality,
        ],
        setup=orch_test_class.setup_method,
        teardown=orch_test_class.teardown_method,
    )
    quick_layer.suites.append(orch_suite)

    # Add async tests
    async_test_class = TestAsyncComplete()
    async_suite = TestSuite(
        name="Core Async Tests",
        description="Essential async functionality",
        tests=[
            async_test_class.test_async_cache_functionality,
            async_test_class.test_async_memory_store_functionality,
        ],
        setup=async_test_class.setup_method,
    )
    quick_layer.suites.append(async_suite)

    framework.add_layer(quick_layer)

    # Run tests
    print("Running essential functionality tests...\n")
    start_time = time.time()

    try:
        report = await framework.run_all()
        duration = time.time() - start_time

        # Display results
        print("\n" + "=" * 55)
        print("QUICK TEST RESULTS")
        print("=" * 55)

        summary = report.get("summary", {})
        print(f"Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(
            f"Tests: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed"
        )
        print(f"Duration: {duration:.2f} seconds")

        if summary.get("critical_failures", 0) > 0:
            print(f"🔴 Critical failures: {summary['critical_failures']}")

        if summary.get("failed", 0) > 0:
            print("\nFailures:")
            for failure in report.get("failures", [])[:3]:
                print(f"  - {failure['name']}: {failure['message']}")

        success = summary.get("overall_status") == "PASSED"
        if success:
            print("\n✅ Quick tests PASSED - Core functionality verified!")
        else:
            print("\n❌ Some quick tests FAILED - Issues detected!")

        return success

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        return False


async def run_full_tests(layer_filter=None):
    """Run the full comprehensive test suite."""
    print("🔬 Running Full Comprehensive Test Suite")
    print("=" * 55)

    try:
        from tests.comprehensive.test_complete_functionality import (
            ComprehensiveTestRunner,
        )
    except ImportError as e:
        print(f"❌ Failed to import comprehensive tests: {e}")
        return False

    runner = ComprehensiveTestRunner()

    # Filter to specific layer if requested
    if layer_filter:
        print(f"Running Layer {layer_filter} only...")
        for i, layer in enumerate(runner.framework.layers):
            layer.enabled = (i + 1) == layer_filter

    try:
        report = await runner.run_comprehensive_tests()

        summary = report.get("summary", {})
        success = (
            summary.get("overall_status") == "PASSED"
            and summary.get("critical_failures", 0) == 0
        )

        return success

    except Exception as e:
        print(f"\n💥 Comprehensive test execution failed: {e}")
        return False


def check_environment():
    """Check if environment is set up for testing."""
    print("🔍 Checking test environment...")

    # Check if we're in the right directory
    if not Path("src/agent_orchestrator.py").exists():
        print("❌ Not in Enterprise Agent root directory")
        print("Please run from the project root directory")
        return False

    # Check for test framework
    if not Path("tests/comprehensive/test_framework.py").exists():
        print("❌ Test framework not found")
        return False

    print("✅ Environment check passed")
    return True


async def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Check environment
    if not check_environment():
        return 1

    print("Enterprise Agent Comprehensive Test Runner")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        if args.quick:
            success = await run_quick_tests()
        else:
            success = await run_full_tests(args.layer)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
