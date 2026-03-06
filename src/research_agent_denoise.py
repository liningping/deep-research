"""Final report generation node.

Reads:
    state["research_brief"]      – original research question
    state["draft_report"]        – draft produced by write_draft_report node
    state["notes"]               – accumulated structured research notes
    state["web_research_results"]– raw search result dicts from execute_research
    state["messages"]            – conversation history (for language detection)

Updates:
    state["final_report"]        – publication-ready markdown report
    state["messages"]            – appends assistant message with the final report
"""

import os
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

from deep_research.state_scope import AgentState
from deep_research.prompts import (
    final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt,
)
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

_model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
_base_url   = os.environ["OPENAI_BASE_URL"]

# Writer model – longer context / higher max_tokens for final synthesis
writer_model = init_chat_model(
    model=f"openai:{_model_name}",
    base_url=_base_url,
    max_tokens=16000,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_findings(state: AgentState) -> str:
    """Combine structured notes and raw web_research_results into a single findings string.

    Priority order:
        1. state["notes"] – already processed, higher quality
        2. state["web_research_results"] – raw search content as fallback
    """
    parts: list[str] = []

    # 1. Structured notes from upstream nodes
    notes: list[str] = state.get("notes") or []
    if notes:
        parts.extend(notes)

    # 2. Raw search results (content field only, keyed by query)
    raw_results: list[dict] = state.get("web_research_results") or []
    for r in raw_results:
        if r.get("success") and r.get("content"):
            header = f"[Query: {r['query']}]"
            parts.append(f"{header}\n{r['content']}")

    return "\n\n---\n\n".join(parts) if parts else "(No research findings available)"


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def final_report_generation(state: AgentState) -> dict:
    """Synthesise all research into a publication-ready final report."""
    research_brief: str = state.get("research_brief") or ""
    draft_report:   str = state.get("draft_report")   or ""
    findings:       str = _build_findings(state)
    today:          str = get_today_str()

    logger.info(
        f"📝 [Denoise] Generating final report | "
        f"brief_len={len(research_brief)} "
        f"draft_len={len(draft_report)} "
        f"findings_len={len(findings)}"
    )

    prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=research_brief,
        findings=findings,
        date=today,
        draft_report=draft_report,
        # user_request mirrors research_brief here; override if a separate field exists
        user_request=state.get("user_request", research_brief),
    )

    response = writer_model.invoke([HumanMessage(content=prompt)])
    final_report: str = response.content if hasattr(response, "content") else str(response)

    logger.info(f"✅ [Denoise] Final report generated ({len(final_report)} chars)")

    # Routing to END handled by explicit edge in graph.py
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
    }