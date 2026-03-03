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
    reflect_on_report,
    finalize_report,
    generate_markdown_report,
    post_process_report,
    route_research,
    route_after_search,
    route_after_multi_agents,
)
from src.nodes.answer import (
    generate_answer,
    reflect_answer,
    route_after_multi_agents_benchmark,
    route_after_generate_answer,
    route_after_reflect_answer,
    verify_answer,
    finalize_answer,
    post_process_benchmark_answer,
)
from src.nodes.validation import (
    validate_context_sufficiency,
    refine_query,
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
    builder.add_node("reflect_on_report", reflect_on_report)
    builder.add_node("finalize_report", finalize_report)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("reflect_answer", reflect_answer)
    builder.add_node("finalize_answer", finalize_answer)
    builder.add_node("validate_context_sufficiency", validate_context_sufficiency)
    builder.add_node("refine_query", refine_query)

    # === Edges ===
    builder.add_edge(START, "multi_agents_network")

    # Route after search: QA/Benchmark → validation path, Regular → report path
    def route_after_multi_agents_decision(state):
        if state.qa_mode or state.benchmark_mode:
            return "validate_context_sufficiency"
        else:
            return "generate_report"

    builder.add_conditional_edges(
        "multi_agents_network",
        route_after_multi_agents_decision,
        {
            "validate_context_sufficiency": "validate_context_sufficiency",
            "generate_report": "generate_report",
        },
    )

    # === Benchmark/QA Path ===
    builder.add_conditional_edges(
        "validate_context_sufficiency",
        lambda state: "refine_query" if state.needs_refinement else "generate_answer",
        {"refine_query": "refine_query", "generate_answer": "generate_answer"},
    )
    builder.add_edge("refine_query", "multi_agents_network")
    builder.add_edge("generate_answer", "reflect_answer")
    builder.add_conditional_edges(
        "reflect_answer",
        lambda state: route_after_reflect_answer(state, {}),
        {
            "multi_agents_network": "multi_agents_network",
            "finalize_answer": "finalize_answer",
        },
    )
    builder.add_edge("finalize_answer", END)

    # === Regular Report Path ===
    builder.add_edge("generate_report", "reflect_on_report")
    builder.add_conditional_edges(
        "reflect_on_report",
        route_research,
        {
            "multi_agents_network": "multi_agents_network",
            "finalize_report": "finalize_report",
        },
    )
    builder.add_edge("finalize_report", END)

    return builder.compile()


graph = create_graph()  # Export a compiled instance for LangGraph Studio
create_fresh_graph = create_graph  # Factory function for programmatic use
