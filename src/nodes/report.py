import json
import logging
from typing import Any, Dict, List, Union, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import SummaryState
from src.configuration import Configuration
from llm_clients import get_llm_client, CURRENT_DATE, CURRENT_YEAR, ONE_YEAR_AGO
from src.prompts import (
    summarizer_instructions,
    reflection_instructions,
    finalize_report_instructions,
    clarify_with_user_instructions,
    research_brief_instructions,
)
from src.utils import clean_raw_web_content, extract_citations_from_state, clean_json_response
from src.nodes.utils import get_configurable, get_max_loops

logger = logging.getLogger(__name__)


def clarify_with_user(state: SummaryState, config: RunnableConfig):
    """
    Assess if the user's research request has sufficient detail to proceed.
    
    Currently a pass-through node that always routes to write_research_brief.
    Future expansion: call LLM to evaluate topic clarity and potentially
    return a clarifying question to the user.
    """
    research_topic = state.research_topic
    logger.info(f"[clarify_with_user] Evaluating topic: '{research_topic}'")
    
    # --- Future: Uncomment to enable LLM-based clarification ---
    # configurable = Configuration.from_runnable_config(config)
    # provider = getattr(state, "llm_provider", None) or configurable.llm_provider
    # if not isinstance(provider, str): provider = provider.value
    # model = getattr(state, "llm_model", None) or configurable.llm_model or "gemini-2.5-pro"
    # llm = get_llm_client(provider, model)
    #
    # prompt = clarify_with_user_instructions.format(
    #     research_topic=research_topic,
    #     current_date=CURRENT_DATE,
    #     current_year=CURRENT_YEAR,
    # )
    # response = llm.invoke([HumanMessage(content=prompt)])
    # result = clean_json_response(
    #     response.content if hasattr(response, "content") else str(response)
    # )
    # if result.get("need_clarification", False):
    #     logger.info(f"[clarify_with_user] Clarification needed: {result['question']}")
    #     return {"knowledge_gap": result["question"]}
    # -----------------------------------------------------------
    
    logger.info("[clarify_with_user] Topic is sufficient, proceeding to research brief.")
    return {}  # Pass-through, no state changes


def write_research_brief(state: SummaryState, config: RunnableConfig):
    """
    Transform the research topic into a comprehensive, structured research brief.
    
    This node analyzes the research topic (and any uploaded knowledge) to produce
    a detailed brief covering: objective, scope, key questions, subtopics,
    methodology, and quality criteria.
    """
    research_topic = state.research_topic
    uploaded_knowledge = getattr(state, "uploaded_knowledge", "") or ""
    
    # Configure LLM
    configurable = Configuration.from_runnable_config(config)
    provider = getattr(state, "llm_provider", None) or configurable.llm_provider
    if not isinstance(provider, str):
        provider = provider.value
    model = getattr(state, "llm_model", None) or configurable.llm_model or "gemini-2.5-pro"
    
    logger.info(f"[write_research_brief] Generating brief for: '{research_topic}' with {provider}/{model}")
    llm = get_llm_client(provider, model)
    
    # Build the prompt
    knowledge_context = uploaded_knowledge.strip() if uploaded_knowledge else "No user-provided knowledge available."
    
    prompt = research_brief_instructions.format(
        research_topic=research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        uploaded_knowledge=knowledge_context,
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    research_brief = response.content if hasattr(response, "content") else str(response)
    
    logger.info(f"[write_research_brief] Generated brief ({len(research_brief)} chars)")
    
    return {
        "research_brief": research_brief,
    }


def write_draft_report(state: AgentState) -> Command[Literal["__end__"]]:
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    # Set up structured output model
    structured_output_model = creative_model.with_structured_output(DraftReport)
    research_brief = state.get("research_brief", "")
    draft_report_prompt = draft_report_generation_prompt.format(
        research_brief=research_brief,
        date=get_today_str()
    )

    response = structured_output_model.invoke([HumanMessage(content=draft_report_prompt)])

    return {
        "research_brief": research_brief,
        "draft_report": response.draft_report, 
        "supervisor_messages": ["Here is the draft report: " + response.draft_report, research_brief]
    }
