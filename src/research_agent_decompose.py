"""Topic decomposition node.

Converts the research brief into a flat list of 1–5 subtopics.
No simple/complex distinction — the LLM naturally outputs 1 subtopic for
simple questions and 3-5 for complex ones.
"""

import os
from datetime import datetime
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

from deep_research.prompts import query_writer_instructions
from deep_research.state_scope import AgentState


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

_model_name = os.environ["OPENAI_MODEL"]
_base_url   = os.environ["OPENAI_BASE_URL"]
model = init_chat_model(model=f"openai:{_model_name}", base_url=_base_url)


# ---------------------------------------------------------------------------
# Output schema  (unified — no simple/complex split)
# ---------------------------------------------------------------------------

class _Subtopic(BaseModel):
    """A single search subtask."""
    query: str = Field(description="Search query string, under 400 characters.")
    aspect: str = Field(description="The specific angle or aspect this query covers.")
    suggested_tool: Literal[
        "general_search", "academic_search", "github_search", "linkedin_search"
    ] = Field(description="Most appropriate search tool for this query.")


class _ResearchPlan(BaseModel):
    """Unified decomposition output: always a list of subtopics (1–5)."""
    subtopics: List[_Subtopic] = Field(
        description=(
            "1–5 focused search subtopics that together provide comprehensive coverage. "
            "Use 1 subtopic for simple factual questions; 3–5 for multi-faceted topics."
        ),
        min_length=1,
        max_length=5,
    )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def decompose_topic(state: AgentState) -> dict:
    """Decompose the research brief into a flat list of search subtopics."""
    research_brief: str = state.get("research_brief", "")

    today = datetime.now()
    formatted_prompt = query_writer_instructions.format(
        research_topic=research_brief,
        current_date=today.strftime("%B %d, %Y"),
        current_year=str(today.year),
        one_year_ago=str(today.year - 1),
    )

    structured_model = model.with_structured_output(_ResearchPlan)
    response: _ResearchPlan = structured_model.invoke([
        HumanMessage(content=formatted_prompt)
    ])

    # Normalise to a plain dict list so state stays JSON-serialisable
    research_plan = {
        "subtopics": [
            {
                "query":          sub.query,
                "aspect":         sub.aspect,
                "suggested_tool": sub.suggested_tool,
            }
            for sub in response.subtopics
        ]
    }

    return {"research_plan": research_plan}
