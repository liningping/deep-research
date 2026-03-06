"""
Research Graph Module

All routing is defined here via explicit add_edge calls.
Nodes only return state update dicts; no node uses Command(goto=...) except
clarify_with_user's conditional END branch.

Pipeline (Option 2 – directional draft before research):
  START
    → clarify_with_user         (conditional END if clarification needed)
    → write_research_brief      writes: research_brief
    → write_draft_report        writes: draft_report  ← from brief only, pre-research
    → decompose_topic           writes: research_plan
    → execute_research          reads:  research_plan + draft_report (gap detection)
                                writes: web_research_results
    → final_report_generation   reads:  draft_report + web_research_results
                                writes: final_report
    → END
"""

from langgraph.graph import START, END, StateGraph

from deep_research.state_scope import AgentState, AgentInputState

from deep_research.research_agent_scope import (
    write_research_brief,
    write_draft_report,
)
from deep_research.research_agent_decompose import decompose_topic
from deep_research.research_agent_execute import execute_research
from deep_research.research_agent_denoise import final_report_generation


def create_graph():
    """Create and compile the research workflow graph."""
    builder = StateGraph(AgentState, input=AgentInputState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("write_research_brief",    write_research_brief)
    builder.add_node("write_draft_report",      write_draft_report)
    builder.add_node("decompose_topic",         decompose_topic)
    builder.add_node("execute_research",        execute_research)
    builder.add_node("final_report_generation", final_report_generation)

    # ── Edges ──────────────────────────────────────────────────────────────────
    builder.add_edge("clarify_with_user",        "write_research_brief")
    builder.add_edge("write_research_brief",     "write_draft_report")
    builder.add_edge("write_draft_report",       "decompose_topic")
    builder.add_edge("decompose_topic",          "execute_research")
    builder.add_edge("execute_research",         "final_report_generation")
    builder.add_edge("final_report_generation",  END)

    return builder.compile()


graph = create_graph()          # Compiled instance for LangGraph Studio
create_fresh_graph = create_graph  # Factory alias
