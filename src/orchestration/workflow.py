"""Workflow helpers for LangGraph orchestration."""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Try to import from langgraph first, fall back to simple_graph
try:
    from langgraph.graph import END, START, StateGraph

    # Test if the API works by checking for compile method
    test_graph = StateGraph(dict)
    if not hasattr(test_graph, "compile"):
        raise AttributeError("StateGraph missing compile method")
    logger.info("Using langgraph StateGraph")
except (ImportError, AttributeError) as e:
    logger.warning(f"LangGraph import failed ({e}), using simple_graph fallback")
    from .simple_graph import END, START, StateGraph


def build_graph(
    state_cls: type,
    *,
    planner: Callable[[Any], Any],
    coder: Callable[[Any], Any],
    validator: Callable[[Any], Any],
    reflector: Callable[[Any], Any],
    reviewer: Callable[[Any], Any],
    governance: Callable[[Any], Any],
    validate_route: Callable[[Any], str],
    reflect_route: Callable[[Any], str],
):
    graph = StateGraph(state_cls)
    graph.add_node("planner", planner)
    graph.add_node("coder", coder)
    graph.add_node("validator", validator)
    graph.add_node("reflector", reflector)
    graph.add_node("reviewer", reviewer)
    graph.add_node("governance", governance)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "validator")
    graph.add_conditional_edges("validator", validate_route)
    graph.add_conditional_edges("reflector", reflect_route)
    graph.add_edge("reviewer", "governance")
    graph.add_edge("governance", END)

    return graph.compile()


__all__ = ["build_graph"]
