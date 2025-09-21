"""Comprehensive test framework for Enterprise Agent."""
import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestSeverity(Enum):
    """Test failure severity."""
    CRITICAL = "critical"     # System unusable
    HIGH = "high"             # Major functionality broken
    MEDIUM = "medium"         # Important features affected
    LOW = "low"               # Minor issues or edge cases
    INFO = "info"             # Informational only


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    status: TestStatus
    severity: TestSeverity
    duration: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None
    traceback: Optional[str] = None


@dataclass
class TestSuite:
    """Test suite containing multiple tests."""
    name: str
    description: str
    tests: List[Callable] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None


@dataclass
class TestLayer:
    """Test layer containing multiple test suites."""
    name: str
    description: str
    suites: List[TestSuite] = field(default_factory=list)
    enabled: bool = True


class TestFramework:
    """Comprehensive test framework for Enterprise Agent."""

    def __init__(self):
        """Initialize test framework."""
        self.layers: List[TestLayer] = []
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.config = self._load_test_config()

    def _load_test_config(self) -> Dict[str, Any]:
        """Load test configuration."""
        return {
            "timeout_per_test": 300,  # 5 minutes per test
            "max_retries": 2,
            "parallel_execution": True,
            "stop_on_critical": True,
            "generate_report": True,
            "coverage_threshold": 0.95,
            "performance_baselines": {
                "max_startup_time": 10.0,
                "max_response_time": 30.0,
                "max_memory_mb": 1000,
            }
        }

    def add_layer(self, layer: TestLayer) -> None:
        """Add test layer to framework."""
        self.layers.append(layer)
        logger.info(f"Added test layer: {layer.name}")

    def add_suite_to_layer(self, layer_name: str, suite: TestSuite) -> None:
        """Add test suite to specific layer."""
        for layer in self.layers:
            if layer.name == layer_name:
                layer.suites.append(suite)
                logger.info(f"Added suite '{suite.name}' to layer '{layer_name}'")
                return
        raise ValueError(f"Layer '{layer_name}' not found")

    async def run_test(self, test_func: Callable, test_name: str,
                      severity: TestSeverity = TestSeverity.MEDIUM) -> TestResult:
        """Run individual test with error handling and timing."""
        result = TestResult(name=test_name, status=TestStatus.RUNNING, severity=severity)
        start_time = time.time()

        try:
            logger.info(f"Running test: {test_name}")

            # Handle both sync and async tests
            if asyncio.iscoroutinefunction(test_func):
                test_result = await asyncio.wait_for(
                    test_func(),
                    timeout=self.config["timeout_per_test"]
                )
            else:
                test_result = test_func()

            result.duration = time.time() - start_time

            # Interpret test result
            if isinstance(test_result, bool):
                result.status = TestStatus.PASSED if test_result else TestStatus.FAILED
            elif isinstance(test_result, dict):
                result.status = TestStatus.PASSED if test_result.get("success", True) else TestStatus.FAILED
                result.message = test_result.get("message", "")
                result.details = test_result.get("details", {})
            else:
                result.status = TestStatus.PASSED
                result.message = str(test_result) if test_result else "Test completed"

            if result.status == TestStatus.PASSED:
                logger.info(f"✓ {test_name} passed ({result.duration:.2f}s)")
            else:
                logger.warning(f"✗ {test_name} failed ({result.duration:.2f}s): {result.message}")

        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.message = f"Test timeout after {self.config['timeout_per_test']}s"
            result.duration = time.time() - start_time
            logger.error(f"✗ {test_name} timed out")

        except Exception as e:
            result.status = TestStatus.ERROR
            result.message = str(e)
            result.exception = e
            result.traceback = traceback.format_exc()
            result.duration = time.time() - start_time
            logger.error(f"✗ {test_name} error: {e}")

        return result

    async def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run test suite with setup/teardown."""
        logger.info(f"Running test suite: {suite.name}")
        results = []

        try:
            # Setup
            if suite.setup:
                logger.debug(f"Running setup for {suite.name}")
                if asyncio.iscoroutinefunction(suite.setup):
                    await suite.setup()
                else:
                    suite.setup()

            # Run tests
            for test_func in suite.tests:
                test_name = f"{suite.name}.{test_func.__name__}"

                # Determine severity from test function attributes
                severity = getattr(test_func, '_test_severity', TestSeverity.MEDIUM)

                result = await self.run_test(test_func, test_name, severity)
                results.append(result)
                suite.results.append(result)

                # Stop on critical failure if configured
                if (result.status in [TestStatus.FAILED, TestStatus.ERROR] and
                    result.severity == TestSeverity.CRITICAL and
                    self.config["stop_on_critical"]):
                    logger.critical(f"Critical test failure in {test_name}, stopping suite")
                    break

        except Exception as e:
            logger.error(f"Suite {suite.name} setup/teardown error: {e}")

        finally:
            # Teardown
            if suite.teardown:
                try:
                    logger.debug(f"Running teardown for {suite.name}")
                    if asyncio.iscoroutinefunction(suite.teardown):
                        await suite.teardown()
                    else:
                        suite.teardown()
                except Exception as e:
                    logger.error(f"Teardown error for {suite.name}: {e}")

        return results

    async def run_layer(self, layer: TestLayer) -> List[TestResult]:
        """Run all test suites in a layer."""
        if not layer.enabled:
            logger.info(f"Skipping disabled layer: {layer.name}")
            return []

        logger.info(f"Running test layer: {layer.name}")
        all_results = []

        for suite in layer.suites:
            suite_results = await self.run_suite(suite)
            all_results.extend(suite_results)

        return all_results

    async def run_all(self) -> Dict[str, Any]:
        """Run all test layers and generate comprehensive report."""
        logger.info("Starting comprehensive test execution")
        self.start_time = time.time()
        self.results = []

        try:
            for layer in self.layers:
                layer_results = await self.run_layer(layer)
                self.results.extend(layer_results)

                # Check for critical failures
                critical_failures = [r for r in layer_results
                                   if r.status in [TestStatus.FAILED, TestStatus.ERROR]
                                   and r.severity == TestSeverity.CRITICAL]

                if critical_failures and self.config["stop_on_critical"]:
                    logger.critical(f"Critical failures in {layer.name}, stopping execution")
                    break

        finally:
            self.end_time = time.time()

        # Generate report
        report = self.generate_report()

        if self.config["generate_report"]:
            self.save_report(report)

        return report

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.start_time or not self.end_time:
            return {"error": "Test execution not completed"}

        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.results if r.status == TestStatus.FAILED])
        errors = len([r for r in self.results if r.status == TestStatus.ERROR])
        skipped = len([r for r in self.results if r.status == TestStatus.SKIPPED])

        # Severity breakdown
        critical_failures = len([r for r in self.results
                               if r.status in [TestStatus.FAILED, TestStatus.ERROR]
                               and r.severity == TestSeverity.CRITICAL])

        high_failures = len([r for r in self.results
                            if r.status in [TestStatus.FAILED, TestStatus.ERROR]
                            and r.severity == TestSeverity.HIGH])

        # Performance metrics
        total_duration = self.end_time - self.start_time
        avg_test_duration = sum(r.duration for r in self.results) / max(total_tests, 1)
        slowest_tests = sorted(self.results, key=lambda r: r.duration, reverse=True)[:5]

        # Layer breakdown
        layer_results = {}
        for layer in self.layers:
            layer_tests = []
            for suite in layer.suites:
                layer_tests.extend(suite.results)

            layer_passed = len([r for r in layer_tests if r.status == TestStatus.PASSED])
            layer_total = len(layer_tests)

            layer_results[layer.name] = {
                "total": layer_total,
                "passed": layer_passed,
                "failed": layer_total - layer_passed,
                "success_rate": layer_passed / max(layer_total, 1)
            }

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "success_rate": passed / max(total_tests, 1),
                "critical_failures": critical_failures,
                "high_failures": high_failures,
                "overall_status": "PASSED" if critical_failures == 0 and failed == 0 else "FAILED"
            },
            "performance": {
                "total_duration": total_duration,
                "avg_test_duration": avg_test_duration,
                "slowest_tests": [
                    {"name": t.name, "duration": t.duration}
                    for t in slowest_tests
                ]
            },
            "layer_breakdown": layer_results,
            "failures": [
                {
                    "name": r.name,
                    "severity": r.severity.value,
                    "message": r.message,
                    "duration": r.duration
                }
                for r in self.results
                if r.status in [TestStatus.FAILED, TestStatus.ERROR]
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config
        }

        return report

    def save_report(self, report: Dict[str, Any]) -> None:
        """Save test report to file."""
        try:
            import json
            report_path = Path("test_results") / f"comprehensive_test_report_{int(time.time())}.json"
            report_path.parent.mkdir(exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Test report saved to: {report_path}")

            # Also create a summary report
            self._create_summary_report(report, report_path.parent / "test_summary.txt")

        except Exception as e:
            logger.error(f"Failed to save test report: {e}")

    def _create_summary_report(self, report: Dict[str, Any], path: Path) -> None:
        """Create human-readable summary report."""
        try:
            with open(path, 'w') as f:
                f.write("ENTERPRISE AGENT COMPREHENSIVE TEST REPORT\n")
                f.write("=" * 50 + "\n\n")

                summary = report["summary"]
                f.write(f"Overall Status: {summary['overall_status']}\n")
                f.write(f"Test Results: {summary['passed']}/{summary['total_tests']} passed ")
                f.write(f"({summary['success_rate']:.1%} success rate)\n")
                f.write(f"Critical Failures: {summary['critical_failures']}\n")
                f.write(f"High Priority Failures: {summary['high_failures']}\n\n")

                f.write("Layer Results:\n")
                f.write("-" * 20 + "\n")
                for layer_name, layer_data in report["layer_breakdown"].items():
                    f.write(f"{layer_name}: {layer_data['passed']}/{layer_data['total']} ")
                    f.write(f"({layer_data['success_rate']:.1%})\n")

                f.write(f"\nExecution Time: {report['performance']['total_duration']:.2f} seconds\n")

                if report["failures"]:
                    f.write("\nFailures:\n")
                    f.write("-" * 20 + "\n")
                    for failure in report["failures"]:
                        f.write(f"[{failure['severity'].upper()}] {failure['name']}: {failure['message']}\n")

            logger.info(f"Summary report saved to: {path}")

        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")


def test_severity(severity: TestSeverity):
    """Decorator to mark test severity."""
    def decorator(func):
        func._test_severity = severity
        return func
    return decorator


def critical_test(func):
    """Decorator for critical tests."""
    return test_severity(TestSeverity.CRITICAL)(func)


def high_priority_test(func):
    """Decorator for high priority tests."""
    return test_severity(TestSeverity.HIGH)(func)


def medium_priority_test(func):
    """Decorator for medium priority tests."""
    return test_severity(TestSeverity.MEDIUM)(func)


def low_priority_test(func):
    """Decorator for low priority tests."""
    return test_severity(TestSeverity.LOW)(func)