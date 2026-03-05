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
    clarify_with_user,
    write_research_brief,
    write_draft_report,
)
from src.nodes.denoise_draft import denoise_draft

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
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)
    builder.add_node("write_draft_report", write_draft_report)
    builder.add_node("multi_agents_network", async_multi_agents_network)
    builder.add_node("denoise_draft", denoise_draft)

    # === Edges ===
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("clarify_with_user", "write_research_brief")
    builder.add_edge("write_research_brief", "write_draft_report")
    builder.add_edge("write_draft_report", "multi_agents_network")
    builder.add_edge("multi_agents_network", "denoise_draft")
    builder.add_edge("denoise_draft", END)


    return builder.compile()


graph = create_graph()  # Export a compiled instance for LangGraph Studio
create_fresh_graph = create_graph  # Factory function for programmatic use
