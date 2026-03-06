"""State Definitions and Pydantic Schemas for the Research Agent.

Defines AgentState, input/output state types, TypedDicts for structured
node outputs, and Pydantic schemas for with_structured_output() calls.
"""

import operator
from typing import Dict, List, Optional, Any
from typing_extensions import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# State definitions
# ─────────────────────────────────────────────────────────────────────────────

class AgentInputState(MessagesState):
    """Input state for the agent — only the initial user messages."""
    pass


class AgentState(MessagesState):
    """Main state for the full research workflow.

    Field update semantics:
        operator.add  – appended across graph steps (lists)
        plain         – last-write-wins
    """

    # Research brief synthesised from conversation history
    research_brief: Optional[str]

    # Directional draft generated before research (from brief alone)
    draft_report: Optional[str]

    # Structured decomposition of the research topic by decompose_topic node
    research_plan: Optional[Dict[str, Any]]

    # Raw search result dicts accumulated by execute_research
    web_research_results: Annotated[List[Dict[str, Any]], operator.add]

    # Structured notes accumulated by any node (optional enrichment path)
    notes: Annotated[List[str], operator.add]

    # Final publication-ready report from final_report_generation
    final_report: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Topic decomposition TypedDicts
# ─────────────────────────────────────────────────────────────────────────────

class SubtopicPlan(Dict):
    """One subtopic produced by the decompose_topic node (complex path)."""
    name: str            # brief name of the subtopic
    query: str           # search query string (< 400 chars)
    aspect: str          # specific aspect this subtopic covers
    suggested_tool: str  # general_search / academic_search / github_search / linkedin_search


class ResearchPlan(Dict):
    """Standard output of the decompose_topic node.

    Simple path  (topic_complexity == "simple"):
        topic_complexity, query, aspect, rationale, suggested_tool

    Complex path (topic_complexity == "complex"):
        topic_complexity, main_query, main_tool, subtopics: List[SubtopicPlan]
    """


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas for with_structured_output()
# ─────────────────────────────────────────────────────────────────────────────

class ClarifyWithUser(BaseModel):
    """Decision schema for the clarify_with_user node."""

    need_clarification: bool = Field(
        description="True if a clarifying question must be asked before research can begin.",
    )
    question: str = Field(
        description="The clarifying question to ask (empty string if need_clarification=False).",
    )
    verification: str = Field(
        description="Confirmation message that research will start after the user replies.",
    )


class ResearchQuestion(BaseModel):
    """Schema for the write_research_brief node output."""

    research_brief: str = Field(
        description="Detailed research question / brief that will guide the entire research workflow.",
    )


class DraftReport(BaseModel):
    """Schema for the write_draft_report node output."""

    draft_report: str = Field(
        description=(
            "A directional draft report written from the research brief alone, "
            "before any web research is performed. Used to orient the search strategy."
        ),
    )
