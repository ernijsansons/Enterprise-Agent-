#!/usr/bin/env python3
"""Test script to validate metrics collection and observability system."""
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metrics_collector():
    """Test basic metrics collector functionality."""
    print("Testing metrics collector...")

    try:
        from src.utils.metrics import MetricsCollector, MetricSeverity

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(
                enabled=True,
                buffer_size=100,
                flush_interval=5.0,
                export_path=Path(temp_dir)
            )

            # Test counter metrics
            collector.record_counter("test_counter", 1, tags={"test": "true"})
            collector.record_counter("test_counter", 2, tags={"test": "true"})

            # Test gauge metrics
            collector.record_gauge("test_gauge", 42.5, tags={"metric_type": "gauge"})

            # Test histogram metrics
            for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
                collector.record_histogram("test_histogram", value)

            # Test timer metrics
            collector.record_timer("test_timer", 1.234, tags={"operation": "test"})

            # Test events
            collector.record_event("test_event", MetricSeverity.INFO, message="Test event")

            # Test summary
            summary = collector.get_summary()
            assert summary["counters"]["test_counter"] == 3.0
            assert summary["gauges"]["test_gauge"] == 42.5
            assert "test_histogram" in summary["histograms"]
            assert "test_timer" in summary["timers"]

            print("✅ Basic metrics collector test passed")
            return True

    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        return False

def test_timer_context():
    """Test timer context manager."""
    print("\nTesting timer context manager...")

    try:
        from src.utils.metrics import MetricsCollector

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(export_path=Path(temp_dir))

            # Test successful timing
            with collector.timer("test_operation", tags={"type": "success"}):
                time.sleep(0.1)

            # Test timing with exception
            try:
                with collector.timer("test_operation", tags={"type": "error"}):
                    time.sleep(0.05)
                    raise ValueError("Test error")
            except ValueError:
                pass  # Expected

            summary = collector.get_summary()
            assert "test_operation" in summary["timers"]
            assert summary["timers"]["test_operation"]["count"] == 2

            print("✅ Timer context manager test passed")
            return True

    except Exception as e:
        print(f"❌ Timer context manager test failed: {e}")
        return False

def test_performance_events():
    """Test performance event tracking."""
    print("\nTesting performance events...")

    try:
        from src.utils.metrics import MetricsCollector

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(export_path=Path(temp_dir))

            # Start performance event
            event_id = collector.start_performance_event(
                "model_call",
                context={"model": "claude-3", "domain": "coding"}
            )

            time.sleep(0.1)

            # Finish performance event
            event = collector.finish_performance_event(
                event_id,
                model_response="Test response",
                tokens=100
            )

            assert event is not None
            assert event.name == "model_call"
            assert event.duration > 0
            assert event.success == True
            assert event.context["model"] == "claude-3"

            # Test error event
            error_event_id = collector.start_performance_event("error_operation")
            error_event = collector.finish_performance_event(
                error_event_id,
                error="Test error occurred"
            )

            assert error_event.success == False
            assert error_event.error == "Test error occurred"

            print("✅ Performance events test passed")
            return True

    except Exception as e:
        print(f"❌ Performance events test failed: {e}")
        return False

def test_metrics_config():
    """Test metrics configuration."""
    print("\nTesting metrics configuration...")

    try:
        from src.utils.metrics import MetricsConfig

        # Test environment configuration
        os.environ["METRICS_ENABLED"] = "false"
        os.environ["METRICS_BUFFER_SIZE"] = "5000"
        os.environ["METRICS_FLUSH_INTERVAL"] = "30.0"

        env_config = MetricsConfig.from_env()
        assert env_config.enabled == False
        assert env_config.buffer_size == 5000
        assert env_config.flush_interval == 30.0

        # Test dictionary configuration
        dict_config = MetricsConfig.from_dict({
            "enabled": True,
            "buffer_size": 8000,
            "export_path": "/tmp/metrics"
        })
        assert dict_config.enabled == True
        assert dict_config.buffer_size == 8000
        assert str(dict_config.export_path) == "/tmp/metrics"

        print("✅ Metrics configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Metrics configuration test failed: {e}")
        return False

    finally:
        # Cleanup environment variables
        os.environ.pop("METRICS_ENABLED", None)
        os.environ.pop("METRICS_BUFFER_SIZE", None)
        os.environ.pop("METRICS_FLUSH_INTERVAL", None)

def test_metrics_export():
    """Test metrics export functionality."""
    print("\nTesting metrics export...")

    try:
        from src.utils.metrics import MetricsCollector

        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(
                enabled=True,
                export_path=Path(temp_dir),
                flush_interval=1.0  # Short interval for testing
            )

            # Add some metrics
            collector.record_counter("export_test", 5)
            collector.record_gauge("export_gauge", 100)
            collector.record_event("export_event", message="Test export")

            # Force flush
            collector.flush()

            # Check that files were created
            export_files = list(Path(temp_dir).glob("*.jsonl"))
            summary_files = list(Path(temp_dir).glob("summary_*.json"))

            assert len(export_files) > 0, "No export files created"
            assert len(summary_files) > 0, "No summary files created"

            # Check file contents
            import json
            with summary_files[0].open() as f:
                summary_data = json.load(f)
                assert "counters" in summary_data
                assert "gauges" in summary_data
                assert summary_data["counters"]["export_test"] == 5

            print("✅ Metrics export test passed")
            return True

    except Exception as e:
        print(f"❌ Metrics export test failed: {e}")
        return False

def test_global_metrics_functions():
    """Test global convenience functions."""
    print("\nTesting global metrics functions...")

    try:
        from src.utils.metrics import (
            record_counter, record_gauge, record_timer, record_event,
            get_metrics_collector, MetricSeverity
        )

        # Test global functions
        record_counter("global_counter", 10)
        record_gauge("global_gauge", 50.5)
        record_timer("global_timer", 2.5)
        record_event("global_event", MetricSeverity.WARNING, message="Global test")

        # Verify metrics were recorded
        collector = get_metrics_collector()
        summary = collector.get_summary()

        assert summary["counters"]["global_counter"] == 10
        assert summary["gauges"]["global_gauge"] == 50.5

        print("✅ Global metrics functions test passed")
        return True

    except Exception as e:
        print(f"❌ Global metrics functions test failed: {e}")
        return False

def test_metrics_with_orchestrator():
    """Test metrics integration with orchestrator (basic import test)."""
    print("\nTesting metrics with orchestrator integration...")

    try:
        # This tests that the imports work correctly
        from src.utils.metrics import get_metrics_collector

        # Check that metrics collector is accessible
        collector = get_metrics_collector()
        assert collector is not None

        print("✅ Metrics orchestrator integration test passed")
        return True

    except Exception as e:
        print(f"❌ Metrics orchestrator integration test failed: {e}")
        return False

def main():
    """Run all metrics system tests."""
    print("Metrics Collection and Observability Test Suite")
    print("=" * 55)

    tests_passed = 0
    total_tests = 7

    if test_metrics_collector():
        tests_passed += 1

    if test_timer_context():
        tests_passed += 1

    if test_performance_events():
        tests_passed += 1

    if test_metrics_config():
        tests_passed += 1

    if test_metrics_export():
        tests_passed += 1

    if test_global_metrics_functions():
        tests_passed += 1

    if test_metrics_with_orchestrator():
        tests_passed += 1

    print("\n" + "=" * 55)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All metrics tests passed! Observability system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the metrics implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)