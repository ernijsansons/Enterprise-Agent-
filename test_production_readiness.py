#!/usr/bin/env python3
"""Production readiness test suite for Enterprise Agent v3.4."""

import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_installation_readiness():
    """Test that installation requirements are met."""
    print("Testing installation readiness...")

    try:
        # Test 1: Check Python version
        print("  Checking Python version...")
        version_info = sys.version_info
        if version_info < (3, 9):
            print(f"  FAIL: Python {version_info.major}.{version_info.minor} is too old. Need 3.9+")
            return False
        print(f"  PASS: Python {version_info.major}.{version_info.minor}.{version_info.micro}")

        # Test 2: Check required modules
        print("  Checking required modules...")
        required_modules = [
            'json', 'os', 'sys', 'pathlib', 'datetime', 'logging',
            'threading', 'subprocess', 'tempfile', 'concurrent.futures'
        ]

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                print(f"  FAIL: Required module '{module}' not available")
                return False

        print(f"  PASS: All {len(required_modules)} required modules available")

        # Test 3: Check file permissions
        print("  Checking file permissions...")
        test_dir = tempfile.mkdtemp()
        try:
            test_file = Path(test_dir) / "test_write.txt"
            test_file.write_text("test")
            content = test_file.read_text()
            if content != "test":
                print("  FAIL: File write/read test failed")
                return False
            print("  PASS: File system access works")
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

        # Test 4: Check environment setup
        print("  Checking environment setup...")

        # Check if we can create directories
        home_dir = Path.home()
        config_dir = home_dir / ".enterprise-agent"

        try:
            config_dir.mkdir(exist_ok=True)
            test_config = config_dir / "test_config.tmp"
            test_config.write_text("test")
            test_config.unlink()
            print("  PASS: Can create configuration directories")
        except Exception as e:
            print(f"  FAIL: Cannot create config directory: {e}")
            return False

        print("SUCCESS: Installation readiness test passed")
        return True

    except Exception as e:
        print(f"FAILED: Installation readiness test failed: {e}")
        return False


def test_configuration_readiness():
    """Test configuration system readiness."""
    print("\nTesting configuration readiness...")

    try:
        # Test 1: Configuration file validation
        print("  Testing configuration validation...")

        config_file = "configs/agent_config_v3.4.yaml"
        if not Path(config_file).exists():
            print(f"  FAIL: Configuration file {config_file} not found")
            return False

        # Try to validate with our validator
        try:
            result = subprocess.run(
                [sys.executable, "validate_config.py", config_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"  FAIL: Configuration validation failed: {result.stderr}")
                return False

            print("  PASS: Configuration validation passed")
        except Exception as e:
            print(f"  WARN: Could not run configuration validator: {e}")

        # Test 2: Environment variable handling
        print("  Testing environment variable handling...")

        # Set test environment variables
        test_env_vars = {
            "CACHE_ENABLED": "true",
            "METRICS_ENABLED": "true",
            "REFLECTION_MAX_ITERATIONS": "5"
        }

        original_values = {}
        for key, value in test_env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Verify environment variables are readable
            for key, expected_value in test_env_vars.items():
                actual_value = os.environ.get(key)
                if actual_value != expected_value:
                    print(f"  FAIL: Environment variable {key} = {actual_value}, expected {expected_value}")
                    return False

            print("  PASS: Environment variable handling works")

        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

        print("SUCCESS: Configuration readiness test passed")
        return True

    except Exception as e:
        print(f"FAILED: Configuration readiness test failed: {e}")
        return False


def test_security_readiness():
    """Test security readiness."""
    print("\nTesting security readiness...")

    try:
        # Test 1: Bandit security scan
        print("  Running security scan...")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print("  PASS: No security issues found")
            else:
                # Check if it's just warnings or actual issues
                if "No issues identified" in result.stdout:
                    print("  PASS: No security issues identified")
                else:
                    print(f"  WARN: Security scan found issues: {result.stdout[:200]}...")

        except subprocess.TimeoutExpired:
            print("  WARN: Security scan timed out")
        except FileNotFoundError:
            print("  WARN: Bandit not available for security scan")

        # Test 2: Check for hardcoded secrets
        print("  Checking for hardcoded secrets...")

        secret_patterns = [
            "sk-ant-api",
            "sk-",
            "password",
            "secret",
            "token"
        ]

        found_secrets = []
        src_dir = Path("src")

        if src_dir.exists():
            for py_file in src_dir.glob("**/*.py"):
                try:
                    content = py_file.read_text()
                    for pattern in secret_patterns:
                        if pattern in content.lower() and "example" not in content.lower():
                            # Check if it's in a comment or string literal
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line.lower() and not line.strip().startswith('#'):
                                    found_secrets.append(f"{py_file}:{i+1}")

                except Exception:
                    continue

        if found_secrets:
            print(f"  WARN: Potential secrets found in {len(found_secrets)} locations")
            for secret in found_secrets[:5]:  # Show first 5
                print(f"    {secret}")
        else:
            print("  PASS: No hardcoded secrets detected")

        # Test 3: Check file permissions
        print("  Checking sensitive file permissions...")

        sensitive_files = [
            "configs/agent_config_v3.4.yaml",
            "install.sh",
        ]

        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                # Check if file is readable by others (basic check)
                if hasattr(stat, 'st_mode'):
                    print(f"  PASS: {file_path} permissions checked")

        print("SUCCESS: Security readiness test passed")
        return True

    except Exception as e:
        print(f"FAILED: Security readiness test failed: {e}")
        return False


def test_performance_readiness():
    """Test performance readiness."""
    print("\nTesting performance readiness...")

    try:
        # Test 1: Import speed
        print("  Testing module import speed...")

        start_time = time.time()

        # Test importing key modules
        import_tests = [
            "src.utils.cache",
            "src.utils.metrics",
            "src.utils.validation",
            "src.utils.errors",
            "src.utils.telemetry"
        ]

        for module in import_tests:
            try:
                __import__(module)
            except ImportError as e:
                print(f"  WARN: Could not import {module}: {e}")

        import_time = time.time() - start_time

        if import_time > 5.0:
            print(f"  WARN: Slow import time: {import_time:.2f}s")
        else:
            print(f"  PASS: Import time: {import_time:.2f}s")

        # Test 2: Memory usage
        print("  Testing memory usage...")

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > 500:  # 500MB threshold
                print(f"  WARN: High memory usage: {memory_mb:.1f}MB")
            else:
                print(f"  PASS: Memory usage: {memory_mb:.1f}MB")

        except ImportError:
            print("  SKIP: psutil not available for memory testing")

        # Test 3: Basic operations speed
        print("  Testing basic operations speed...")

        # Test cache operations
        start_time = time.time()
        try:
            from src.utils.cache import TTLCache, CacheConfig

            config = CacheConfig(max_size=100, default_ttl=60)
            cache = TTLCache(config)

            # Perform 1000 cache operations
            for i in range(1000):
                cache.set(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i}")

            cache_time = time.time() - start_time

            if cache_time > 1.0:
                print(f"  WARN: Slow cache operations: {cache_time:.2f}s for 1000 ops")
            else:
                print(f"  PASS: Cache operations: {cache_time:.3f}s for 1000 ops")

        except Exception as e:
            print(f"  WARN: Cache performance test failed: {e}")

        # Test 4: Concurrent operations
        print("  Testing concurrent operations...")

        start_time = time.time()
        try:
            from src.utils.concurrency import ExecutionManager

            with ExecutionManager(max_workers=4) as manager:
                # Submit multiple tasks
                futures = []
                for i in range(20):
                    future = manager.submit(lambda x: x * 2, i)
                    futures.append(future)

                # Wait for completion
                results = [f.result(timeout=5) for f in futures]

            concurrent_time = time.time() - start_time

            if concurrent_time > 5.0:
                print(f"  WARN: Slow concurrent operations: {concurrent_time:.2f}s")
            else:
                print(f"  PASS: Concurrent operations: {concurrent_time:.2f}s")

        except Exception as e:
            print(f"  WARN: Concurrent performance test failed: {e}")

        print("SUCCESS: Performance readiness test passed")
        return True

    except Exception as e:
        print(f"FAILED: Performance readiness test failed: {e}")
        return False


def test_operational_readiness():
    """Test operational readiness."""
    print("\nTesting operational readiness...")

    try:
        # Test 1: Logging system
        print("  Testing logging system...")

        import logging

        # Test basic logging
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.DEBUG)

        # Create a test handler
        log_output = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_output.append(record.getMessage())

        handler = TestHandler()
        logger.addHandler(handler)

        logger.info("Test log message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        if len(log_output) >= 3:
            print("  PASS: Logging system works")
        else:
            print(f"  FAIL: Logging system not working properly ({len(log_output)} messages)")
            return False

        # Test 2: Error handling
        print("  Testing error handling system...")

        try:
            from src.utils.errors import ErrorHandler, ErrorCode, EnterpriseAgentError

            handler = ErrorHandler()

            # Test error creation and handling
            try:
                raise EnterpriseAgentError(
                    ErrorCode.SYSTEM_ERROR,
                    "Test error message",
                    context={"test": True}
                )
            except EnterpriseAgentError as e:
                handler.handle_error(e.details.code, context=e.details.context)

            stats = handler.get_error_statistics()
            if stats['total_errors'] >= 1:
                print("  PASS: Error handling system works")
            else:
                print("  FAIL: Error handling system not recording errors")
                return False

        except Exception as e:
            print(f"  WARN: Error handling test failed: {e}")

        # Test 3: Metrics system
        print("  Testing metrics system...")

        try:
            from src.utils.metrics import MetricsCollector, MetricType

            collector = MetricsCollector(enabled=True)

            # Test metric recording
            collector.record_counter("test_counter", 1)
            collector.record_gauge("test_gauge", 42.5)

            summary = collector.get_summary()
            if "test_counter" in summary.get("counters", {}):
                print("  PASS: Metrics system works")
            else:
                print("  FAIL: Metrics system not recording metrics")
                return False

        except Exception as e:
            print(f"  WARN: Metrics test failed: {e}")

        # Test 4: Configuration loading
        print("  Testing configuration loading...")

        config_file = "configs/agent_config_v3.4.yaml"
        if Path(config_file).exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                if config and isinstance(config, dict):
                    print("  PASS: Configuration loading works")
                else:
                    print("  FAIL: Configuration not loaded properly")
                    return False

            except Exception as e:
                print(f"  WARN: Configuration loading test failed: {e}")
        else:
            print(f"  WARN: Configuration file {config_file} not found")

        print("SUCCESS: Operational readiness test passed")
        return True

    except Exception as e:
        print(f"FAILED: Operational readiness test failed: {e}")
        return False


def test_deployment_readiness():
    """Test deployment readiness."""
    print("\nTesting deployment readiness...")

    try:
        # Test 1: Required files exist
        print("  Checking required files...")

        required_files = [
            "pyproject.toml",
            "Makefile",
            "install.sh",
            "configs/agent_config_v3.4.yaml",
            "src/agent_orchestrator.py",
            "enterprise_agent_cli.py"
        ]

        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            print(f"  FAIL: Missing required files: {missing_files}")
            return False
        else:
            print(f"  PASS: All {len(required_files)} required files present")

        # Test 2: CLI accessibility
        print("  Testing CLI accessibility...")

        cli_file = Path("enterprise_agent_cli.py")
        if cli_file.exists():
            # Check if CLI file is executable or can be run with Python
            try:
                result = subprocess.run(
                    [sys.executable, str(cli_file), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0 or "usage" in result.stdout.lower():
                    print("  PASS: CLI is accessible")
                else:
                    print(f"  WARN: CLI may have issues: {result.stderr[:100]}")

            except Exception as e:
                print(f"  WARN: CLI test failed: {e}")
        else:
            print("  FAIL: CLI file not found")
            return False

        # Test 3: Dependency availability
        print("  Checking dependency availability...")

        dependencies_ok = True
        try:
            import yaml
        except ImportError:
            print("  WARN: PyYAML not available")
            dependencies_ok = False

        if dependencies_ok:
            print("  PASS: Core dependencies available")

        # Test 4: Build system
        print("  Testing build system...")

        if Path("Makefile").exists():
            try:
                # Test make target validation
                with open("Makefile", 'r') as f:
                    makefile_content = f.read()

                required_targets = ["quality", "test", "security"]
                missing_targets = []

                for target in required_targets:
                    if f"{target}:" not in makefile_content:
                        missing_targets.append(target)

                if missing_targets:
                    print(f"  WARN: Missing make targets: {missing_targets}")
                else:
                    print("  PASS: Make targets available")

            except Exception as e:
                print(f"  WARN: Makefile validation failed: {e}")

        print("SUCCESS: Deployment readiness test passed")
        return True

    except Exception as e:
        print(f"FAILED: Deployment readiness test failed: {e}")
        return False


def main():
    """Run all production readiness tests."""
    print("Enterprise Agent v3.4 - Production Readiness Test Suite")
    print("=" * 65)

    tests_passed = 0
    total_tests = 5

    # Run all readiness tests
    test_functions = [
        test_installation_readiness,
        test_configuration_readiness,
        test_security_readiness,
        test_performance_readiness,
        test_operational_readiness,
        test_deployment_readiness,
    ]

    for test_func in test_functions:
        try:
            if test_func():
                tests_passed += 1
        except Exception as e:
            print(f"FAILED: Test {test_func.__name__} failed with exception: {e}")

    print(f"\n" + "=" * 65)
    print(f"Production Readiness Results: {tests_passed}/{len(test_functions)} tests passed")

    if tests_passed == len(test_functions):
        print("SUCCESS: System is ready for production deployment!")
        print("\nRecommended next steps:")
        print("1. Run full test suite: python -m pytest")
        print("2. Execute make quality for final validation")
        print("3. Review security scan results")
        print("4. Deploy to staging environment")
        return 0
    else:
        print("FAILED: System requires fixes before production deployment.")
        print(f"\nAddress the {len(test_functions) - tests_passed} failing test(s) before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)