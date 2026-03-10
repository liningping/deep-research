"""
Research Graph Module

This module defines the LangGraph-based research workflow.
All node implementations are in the src/nodes/ package.
This file contains only:
- create_graph(): Factory function that assembles the graph
- Backward-compatible re-exports for external code
"""

from langgraph.graph import START, END, StateGraph

from src.state import SummaryState, SummaryStateInput, SummaryStateOutput
from src.configuration import Configuration

# Import all node functions from the nodes package
from src.nodes.utils import (
    get_callback_from_config,
    emit_event,
    get_max_loops,
    reset_state,
    heartbeat_task,
    get_configurable,
)
from src.nodes.search import async_multi_agents_network
from src.nodes.report import (
    generate_report,
    finalize_report,
    route_research,
    route_after_search,
    route_after_multi_agents,
)

# Backward-compatible re-exports used by agent_architecture.py
from src.tools import SearchToolRegistry as ToolRegistry, ToolExecutor


def create_graph():
    """
    Factory function that creates a fresh graph instance.
    """
    builder = StateGraph(
        SummaryState,
        input=SummaryStateInput,
        output=SummaryStateOutput,
        config_schema=Configuration,
    )

    # Add all nodes
    builder.add_node("multi_agents_network", async_multi_agents_network)
    builder.add_node("generate_report", generate_report)
    builder.add_node("finalize_report", finalize_report)

    # === Edges ===
    builder.add_edge(START, "multi_agents_network")

    # After search, always go to report generation
    builder.add_edge("multi_agents_network", "generate_report")

    # === Report Path ===
    # After generating report, decide if we loop back for more research or finish
    builder.add_conditional_edges(
        "generate_report",
        route_research,
        {
            "multi_agents_network": "multi_agents_network",
            "finalize_report": "finalize_report",
        }
    )
    
    builder.add_edge("finalize_report", END)

    return builder.compile()


graph = create_graph()  # Export a compiled instance for LangGraph Studio
create_fresh_graph = create_graph  # Factory function for programmatic use
