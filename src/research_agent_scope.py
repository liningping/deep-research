"""User Clarification, Research Brief, and Draft Report Generation.

Scoping phase of the research workflow:
  1. clarify_with_user   – decide whether to ask a clarifying question
  2. write_research_brief – translate conversation into a structured research brief
  3. write_draft_report   – produce a DIRECTIONAL draft from the brief only
                            (placed BEFORE research so execute_research can use it
                             as an "existing summary" for gap detection)
"""

import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import END
from langgraph.types import Command

from deep_research.utils import get_today_str
from deep_research.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_human_msg_prompt,
    draft_report_generation_prompt,
)
from deep_research.state_scope import (
    AgentState,
    ClarifyWithUser,
    ResearchQuestion,
    DraftReport,
)

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

_model_name = os.environ["OPENAI_MODEL"]
_base_url   = os.environ["OPENAI_BASE_URL"]

model          = init_chat_model(model=f"openai:{_model_name}", base_url=_base_url)
creative_model = init_chat_model(model=f"openai:{_model_name}", base_url=_base_url)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def write_research_brief(state: AgentState) -> dict:
    """Transform the conversation history into a structured research brief."""
    structured_model = model.with_structured_output(ResearchQuestion)

    response: ResearchQuestion = structured_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_human_msg_prompt.format(
            messages=state.get("query"),
            date=get_today_str(),
        ))
    ])

    return {"research_brief": response.research_brief}


def write_draft_report(state: AgentState) -> dict:
    """Generate a directional draft report from the research brief alone.

    This runs BEFORE execute_research so that:
    - The search agent can use draft_report as an "existing summary" for gap detection.
    - final_report_generation has a structural skeleton to build on.

    The draft is written purely from the LLM's parametric knowledge —
    it is intentionally incomplete and will be enriched by web research.
    """
    structured_model = creative_model.with_structured_output(DraftReport)
    research_brief = state.get("research_brief", "")

    response: DraftReport = structured_model.invoke([
        HumanMessage(content=draft_report_generation_prompt.format(
            research_brief=research_brief,
            date=get_today_str(),
        ))
    ])

    return {"draft_report": response.draft_report}