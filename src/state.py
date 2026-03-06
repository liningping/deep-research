
"""State Definitions and Pydantic Schemas for Research Scoping.

This defines the state objects and structured schemas used for
the research agent scoping workflow, including researcher state management and output schemas.
"""

import operator
from typing import Dict, List, Optional, Any
from typing_extensions import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    query: str
    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Draft research report
    draft_report: str
    # Final formatted research report
    final_report: str
    # Structured output of topic decomposition node
    research_plan: Optional[Dict[str, Any]]
    # Accumulated search result dicts from all bandit loop iterations
    web_research_results: Annotated[List[Dict[str, Any]], operator.add]

# ===== TOPIC DECOMPOSITION OUTPUT TYPES =====

class SubtopicPlan(TypedDict):
    """One subtopic produced by the decompose_topic node (complex path)."""
    name: str                # brief name of the subtopic
    query: str               # search query string (< 400 chars)
    aspect: str              # specific aspect this subtopic covers
    suggested_tool: str      # one of: general_search / academic_search / github_search / linkedin_search / text2sql

class ResearchPlan(TypedDict, total=False):
    """
    Standard output of the ``decompose_topic`` node.

    Simple path  (topic_complexity == "simple"):
        topic_complexity  : "simple"
        query             : str           – the single search query
        aspect            : str           – the angle / aspect of the query
        rationale         : str           – why this query is relevant
        suggested_tool    : str           – search tool to use

    Complex path (topic_complexity == "complex"):
        topic_complexity  : "complex"
        main_query        : str           – fallback single-query
        main_tool         : str           – tool for the main_query
        subtopics         : List[SubtopicPlan]
    """
    topic_complexity: str
    # --- simple ---
    query: str
    aspect: str
    rationale: str
    suggested_tool: str
    # --- complex ---
    main_query: str
    main_tool: str
    subtopics: List[SubtopicPlan]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class DraftReport(BaseModel):
    """Schema for structured draft report generation."""

    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )

class SubtaskEvaluation(BaseModel):
    """Structured output schema for the bandit subtask quality evaluator."""

    quality_score: float = Field(
        description="Quality score in [0.0, 1.0]: 0=no useful content, 0.5=partial, 1.0=excellent.",
        ge=0.0,
        le=1.0,
    )
    needs_modification: bool = Field(
        description="True if the query should be refined or split into sub-queries.",
    )