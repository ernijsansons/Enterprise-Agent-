"""Lightweight StateGraph replacement for Python 3.13 compatibility."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Sentinel value for graph termination
END = "END"


class CompiledGraph:
    """Compiled graph that can execute the workflow."""

    def __init__(
        self,
        nodes: Dict[str, Callable],
        edges: Dict[str, str],
        conditional_edges: Dict[str, Callable],
        start_node: str,
    ):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.start_node = start_node

    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the graph starting from the start node."""
        state = initial_state.copy()
        current_node = self.start_node

        while current_node and current_node != END:
            if current_node not in self.nodes:
                logger.error(f"Node '{current_node}' not found in graph")
                break

            # Execute the current node
            try:
                result = self.nodes[current_node](state)
                if isinstance(result, dict):
                    state.update(result)
                elif result is not None:
                    # If node returns non-dict, store as 'output'
                    state["output"] = result
            except Exception as e:
                logger.error(f"Error executing node '{current_node}': {e}")
                state["error"] = str(e)
                break

            # Determine next node
            next_node = None

            # Check conditional edges first
            if current_node in self.conditional_edges:
                try:
                    router_result = self.conditional_edges[current_node](state)
                    if isinstance(router_result, str):
                        next_node = router_result
                    elif isinstance(router_result, dict) and "next" in router_result:
                        next_node = router_result["next"]
                except Exception as e:
                    logger.error(f"Error in conditional edge for '{current_node}': {e}")
                    break

            # Fall back to static edges
            if next_node is None and current_node in self.edges:
                next_node = self.edges[current_node]

            current_node = next_node

        return state


class StateGraph:
    """Lightweight StateGraph replacement."""

    def __init__(self, state_cls: type):
        """Initialize with state class (for compatibility, not used)."""
        self.state_cls = state_cls
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable] = {}
        self.start_node: Optional[str] = None
        self._compiled: Optional[CompiledGraph] = None

    def add_node(self, name: str, func: Callable) -> StateGraph:
        """Add a node to the graph."""
        self.nodes[name] = func
        if self.start_node is None:
            self.start_node = name
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph:
        """Add a static edge between nodes."""
        if src == END:
            raise ValueError("END cannot be a start node")
        if dst == END:
            # Handle END as destination
            self.edges[src] = END
        else:
            self.edges[src] = dst
        return self

    def add_conditional_edges(self, node_name: str, router_fn: Callable) -> StateGraph:
        """Add conditional edges from a node based on router function."""
        self.conditional_edges[node_name] = router_fn
        return self

    def compile(self) -> CompiledGraph:
        """Compile the graph for execution."""
        if self._compiled is None:
            if not self.nodes:
                raise ValueError("Cannot compile empty graph")
            if self.start_node is None:
                raise ValueError("No start node defined")

            self._compiled = CompiledGraph(
                nodes=self.nodes,
                edges=self.edges,
                conditional_edges=self.conditional_edges,
                start_node=self.start_node,
            )

        return self._compiled


# For compatibility with existing code
START = "START"
