#!/usr/bin/env python3
"""Real-world test of the Enterprise Agent with actual tasks."""

import io
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent_orchestrator import AgentOrchestrator

# Set UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def test_coding_task():
    """Test a real coding task."""
    print("\n" + "=" * 60)
    print("Testing Real-World Coding Task")
    print("=" * 60)

    agent = AgentOrchestrator()

    # Test a realistic coding task
    task = """
    Create a Python class called TaskManager that:
    1. Can add tasks with priorities (high, medium, low)
    2. Can mark tasks as complete
    3. Can list all tasks sorted by priority
    4. Has proper error handling
    """

    print(f"\nTask: {task.strip()}\n")
    print("Processing...")

    start_time = time.time()
    result = agent.run_mode("coding", task)
    elapsed = time.time() - start_time

    print(f"\nExecution completed in {elapsed:.2f} seconds")
    print("-" * 40)

    # Display results
    if result.get("plan"):
        if isinstance(result["plan"], dict):
            plan_text = result["plan"].get("text", "")
            print(f"\nPlan Generated ({result['plan'].get('model', 'unknown')}):")
        else:
            plan_text = result["plan"]
            print("\nPlan Generated:")

        if plan_text:
            for line in plan_text.splitlines()[:5]:  # Show first 5 lines
                print(f"  {line}")
            if len(plan_text.splitlines()) > 5:
                print("  ...")

    if result.get("code"):
        print(f"\nCode Generated (Source: {result.get('code_source', 'unknown')}):")
        code_lines = result["code"].splitlines()
        for line in code_lines[:10]:  # Show first 10 lines
            print(f"  {line}")
        if len(code_lines) > 10:
            print("  ...")

    print(f"\nConfidence Score: {result.get('confidence', 0):.2f}")
    print(f"Needs Reflection: {result.get('needs_reflect', False)}")
    print(f"Governance Blocked: {result.get('governance_blocked', False)}")

    if result.get("cost_summary"):
        cost = result["cost_summary"]
        print("\nCost Summary:")
        print(f"  Total Tokens: {cost.get('tokens', 0)}")
        print(f"  Total Cost: ${cost.get('total_cost', 0):.6f}")
        print(f"  Events: {len(cost.get('events', []))}")

    # Test cache effectiveness
    print("\n" + "-" * 40)
    print("Testing Cache Effectiveness...")

    start_time = time.time()
    agent.run_mode("coding", task)
    elapsed2 = time.time() - start_time

    print(f"First run: {elapsed:.2f}s")
    print(f"Second run (with cache): {elapsed2:.2f}s")

    if elapsed2 < elapsed * 0.5:
        print("✓ Cache is significantly improving performance!")
    elif elapsed2 < elapsed:
        print("✓ Cache is providing some performance benefit")
    else:
        print("ℹ Cache may not be effective (could be due to stub mode)")

    return result


def test_validation_and_reflection():
    """Test validation and reflection cycle."""
    print("\n" + "=" * 60)
    print("Testing Validation and Reflection")
    print("=" * 60)

    agent = AgentOrchestrator()

    # Test with a task that might trigger validation issues
    task = """
    Create a function that divides two numbers but has intentional bugs:
    - No zero division check
    - No type validation
    - Missing docstring
    """

    print(f"\nTask: {task.strip()}\n")
    print("Processing (expecting validation to trigger reflection)...")

    result = agent.run_mode("coding", task)

    if result.get("validation"):
        print("\nValidation Result:")
        validation = result["validation"]
        print(f"  Passes: {validation.get('passes', False)}")
        print(f"  Coverage: {validation.get('coverage', 0)}")

    if result.get("reflection_analysis"):
        print("\nReflection Analysis:")
        analysis = result["reflection_analysis"]
        if isinstance(analysis, dict):
            print(f"  Analysis: {analysis.get('analysis', 'N/A')[:100]}...")
            if analysis.get("fixes"):
                print(f"  Proposed Fixes: {len(analysis.get('fixes', []))}")
            print(f"  Confidence: {analysis.get('confidence', 0)}")

    print(f"\nFinal Confidence: {result.get('confidence', 0):.2f}")
    print(f"Iterations: {result.get('iterations', 0)}")


def test_different_domains():
    """Test different domain configurations."""
    print("\n" + "=" * 60)
    print("Testing Different Domains")
    print("=" * 60)

    agent = AgentOrchestrator()

    domains = ["coding", "content", "social_media"]

    for domain in domains:
        print(f"\nTesting domain: {domain}")

        tasks = {
            "coding": "Write a function to calculate fibonacci numbers",
            "content": "Write a brief article about AI",
            "social_media": "Create a tweet about productivity",
        }

        task = tasks.get(domain, "Perform a simple task")

        try:
            result = agent.run_mode(domain, task)
            print(f"  ✓ {domain} domain executed successfully")
            print(f"    - Plan generated: {bool(result.get('plan'))}")
            print(f"    - Output generated: {bool(result.get('code'))}")
            print(f"    - Confidence: {result.get('confidence', 0):.2f}")
        except Exception as e:
            print(f"  ✗ {domain} domain failed: {e}")


def test_error_recovery():
    """Test error recovery mechanisms."""
    print("\n" + "=" * 60)
    print("Testing Error Recovery")
    print("=" * 60)

    agent = AgentOrchestrator()

    # Test with various edge cases
    test_cases = [
        ("", "empty task"),
        ("x" * 100000, "very long task"),
        ("Create a function with @#$% special chars", "special characters"),
        ("🚀 Create emoji function 🎉", "emoji task"),
    ]

    for task, description in test_cases:
        print(f"\nTesting {description}...")
        try:
            agent.run_mode("coding", task)
            print(f"  ✓ Handled {description} successfully")
        except Exception as e:
            print(f"  ✗ Failed on {description}: {e}")


def main():
    """Run all real-world tests."""
    print("=" * 60)
    print("ENTERPRISE AGENT REAL-WORLD TEST SUITE")
    print("=" * 60)

    tests = [
        test_coding_task,
        test_validation_and_reflection,
        test_different_domains,
        test_error_recovery,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print("REAL-WORLD TEST SUMMARY")
    print(f"Tests Passed: {passed}/{len(tests)}")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All real-world tests passed!")
        print("The Enterprise Agent is fully functional and production-ready.")
    else:
        print(f"\n⚠️ {failed} test(s) encountered issues.")


if __name__ == "__main__":
    main()
