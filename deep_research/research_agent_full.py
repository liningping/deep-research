
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research.utils import get_today_str
from deep_research.prompts import (
    final_report_generation_step1_prompt,
    final_report_generation_step2_prompt
)
from deep_research.state_scope import AgentState, AgentInputState
from deep_research.research_agent_scope import clarify_with_user, write_research_brief, write_draft_report
from deep_research.multi_agent_supervisor import supervisor_agent

# ===== Config =====

import os
from langchain.chat_models import init_chat_model

writer_model = init_chat_model(
    model=os.getenv("WRITER_MODEL", "openai:gpt-5"),
    model_provider=os.getenv("LLM_PROVIDER", "openai"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_tokens=40000,
    timeout=300.0
)  # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000

# ===== FINAL REPORT GENERATION =====

from deep_research.state_scope import AgentState

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    if os.getenv("DISABLE_REPORT_DENOISING", "false").lower() == "true":
        # Ablation baseline: Directly pass findings without Synthesis (Step 1)
        writer_prompt = final_report_generation_step2_prompt.format(
            research_brief=state.get("research_brief", ""),
            synthesis_or_findings=findings,
            date=get_today_str(),
            draft_report=state.get("draft_report", "")
        )
        final_report = await writer_model.ainvoke([HumanMessage(content=writer_prompt)])
    else:
        # Phase 1: Analysis & Synthesis (Evidence Extraction & Conflict Resolution)
        step1_prompt = final_report_generation_step1_prompt.format(
            research_brief=state.get("research_brief", ""),
            findings=findings,
            date=get_today_str(),
            draft_report=state.get("draft_report", "")
        )
        step1_response = await writer_model.ainvoke([HumanMessage(content=step1_prompt)])
        synthesis = step1_response.content
        
        # Phase 2: Final Report Generation
        step2_prompt = final_report_generation_step2_prompt.format(
            research_brief=state.get("research_brief", ""),
            synthesis_or_findings=synthesis,
            date=get_today_str(),
            draft_report=state.get("draft_report", "")
        )
        final_report = await writer_model.ainvoke([HumanMessage(content=step2_prompt)])

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }

# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()
