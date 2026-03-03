"""
Context validation and query refinement nodes.

Contains the nodes for validating research context and refining queries:
- validate_context_sufficiency: Check if gathered context is enough to answer
- refine_query: Improve search query based on gaps identified
"""

import json
import time
import re
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.runnables import RunnableConfig

from src.state import SummaryState
from src.configuration import Configuration
from llm_clients import get_llm_client
from src.prompts_qa import (
    VALIDATE_RETRIEVAL_PROMPT as QA_VALIDATE_RETRIEVAL_PROMPT,
    REFINE_QUERY_PROMPT as QA_REFINE_QUERY_PROMPT,
)
from src.prompts_benchmark import (
    VALIDATE_RETRIEVAL_PROMPT as BENCHMARK_VALIDATE_RETRIEVAL_PROMPT,
    REFINE_QUERY_PROMPT as BENCHMARK_REFINE_QUERY_PROMPT,
)
from src.nodes.utils import get_max_loops

logger = logging.getLogger(__name__)


def validate_context_sufficiency(
    state: SummaryState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Validate if the retrieved context is sufficient to answer the question.
    Updates state with useful information, missing information, and refinement needs.
    Appends useful information to the running_summary.
    """
    print(
        f"--- ENTERING validate_context_sufficiency (Loop {state.research_loop_count}) ---"
    )
    logger.info(
        f"[validate_context_sufficiency] Called at loop {state.research_loop_count}"
    )
    running_summary_preview = (state.running_summary or "")[:100]
    logger.info(
        f"[validate_context_sufficiency] Initial state.running_summary (first 100 chars): '{running_summary_preview}...'"
    )
    logger.info(
        f"[validate_context_sufficiency] Initial state.web_research_results type: {type(state.web_research_results)}"
    )
    if isinstance(state.web_research_results, list):
        logger.info(
            f"[validate_context_sufficiency] Initial state.web_research_results length: {len(state.web_research_results)}"
        )
        if state.web_research_results:
            logger.info(
                f"[validate_context_sufficiency] First item type in web_research_results: {type(state.web_research_results[0])}"
            )

    start_time = time.time()

    configurable = Configuration.from_runnable_config(config)
    provider = configurable.llm_provider or "openai"
    model = configurable.llm_model or "o3-mini-reasoning"

    if hasattr(state, "llm_provider") and state.llm_provider:
        provider = state.llm_provider
    if hasattr(state, "llm_model") and state.llm_model:
        model = state.llm_model

    print(f"[validate_context_sufficiency] Using provider={provider}, model={model}")
    llm = get_llm_client(provider, model)

    # Current web_research_results (list of dicts, each with 'content')
    new_search_results_list = getattr(state, "web_research_results", [])
    new_search_content = "\n\n---\n\n".join(
        item.get("content", "")
        for item in new_search_results_list
        if isinstance(item, dict) and item.get("content")
    )
    logger.debug(
        f"[validate_context_sufficiency] Constructed new_search_content. Length: {len(new_search_content)}. Preview (first 300 chars): '{new_search_content[:300]}...'"
    )

    # Previously accumulated knowledge
    current_running_summary = getattr(state, "running_summary", "")

    # Get uploaded knowledge
    uploaded_knowledge = getattr(state, "uploaded_knowledge", None)

    # Construct the full context for validation: uploaded knowledge + running summary + new results
    # Separator logic
    context_parts = []
    if uploaded_knowledge and uploaded_knowledge.strip():
        context_parts.append(
            f"USER-PROVIDED EXTERNAL KNOWLEDGE (HIGHEST AUTHORITY):\\n{uploaded_knowledge}"
        )
        print(
            f"[validate_context_sufficiency] Including uploaded knowledge in validation: {len(uploaded_knowledge)} characters"
        )
    if current_running_summary.strip():
        context_parts.append(
            f"PREVIOUSLY ACCUMULATED KNOWLEDGE (Running Summary):\\n{current_running_summary}"
        )
    if new_search_content.strip():
        context_parts.append(
            f"NEWLY FETCHED CONTENT (From latest web search):\\n{new_search_content}"
        )

    context_to_validate_llm = "\n\n---\n\n".join(context_parts)

    # If, after all processing, there's still no context to send to the LLM,
    # it means the initial search might have yielded nothing usable, or the running_summary was empty.
    if not context_to_validate_llm.strip():
        print(
            f"[validate_context_sufficiency] No context (running_summary or new web_research_results) to validate. Assuming incomplete."
        )
        running_summary_empty_check = not (
            state.running_summary and state.running_summary.strip()
        )
        new_content_empty_check = (
            not new_search_content
        )  # new_search_content is a string by this point
        logger.warning(
            f"[validate_context_sufficiency] current_context_to_validate is empty. "
            f"Skipping LLM call. Running summary was empty: {running_summary_empty_check}, "
            f"new_search_content was empty: {new_content_empty_check}"
        )
        # Fallback: if no context at all, assume refinement is needed.
        return {
            "running_summary": current_running_summary,  # Preserve existing running_summary
            "useful_information": "",  # No new useful info this round
            "missing_information": "No context was available for validation (neither running_summary nor new search results). The original query needs to be addressed.",
            "refinement_reasoning": "No context provided for validation.",
            "needs_refinement": True,
            "web_research_results": [],
        }

    # Choose the appropriate validation prompt based on mode
    if state.benchmark_mode:
        validate_prompt = BENCHMARK_VALIDATE_RETRIEVAL_PROMPT
    elif state.qa_mode:
        validate_prompt = QA_VALIDATE_RETRIEVAL_PROMPT
    else:
        validate_prompt = QA_VALIDATE_RETRIEVAL_PROMPT  # Fallback

    prompt = validate_prompt.format(
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        question=state.research_topic,
        retrieved_context=context_to_validate_llm,  # Pass combined context
    )

    print(
        f"[validate_context_sufficiency] Sending validation prompt to {provider}/{model}"
    )
    print(
        f"[validate_context_sufficiency] Context for validation LLM (first 300 chars): {context_to_validate_llm[:300]}..."
    )
    response = llm.invoke(prompt)
    raw_llm_output = response.content if hasattr(response, "content") else str(response)
    print(
        f"[validate_context_sufficiency] Raw LLM response preview: {raw_llm_output[:300]}..."
    )

    reasoning = ""
    json_response_str = raw_llm_output.strip()

    if "<think>" in json_response_str and "</think>" in json_response_str:
        parts = json_response_str.split("</think>", 1)
        think_part = parts[0]
        if "<think>" in think_part:
            reasoning = think_part.split("<think>", 1)[1].strip()
        json_response_str = parts[1].strip() if len(parts) > 1 else ""

    if json_response_str.startswith("```json"):
        json_response_str = json_response_str[7:]
    if json_response_str.startswith("```"):
        json_response_str = json_response_str[3:]
    if json_response_str.endswith("```"):
        json_response_str = json_response_str[:-3]
    json_response_str = json_response_str.strip()

    updated_running_summary = current_running_summary  # Default to existing
    newly_identified_useful_info_this_round = ""

    try:
        parsed_output = json.loads(json_response_str)
        status = parsed_output.get("status", "INCOMPLETE").upper()
        llm_identified_missing_info = parsed_output.get(
            "missing_information", "No specific missing information provided by LLM."
        )
        # This is useful info specifically from the *newly_fetched_content*
        llm_identified_useful_info_from_new = parsed_output.get(
            "useful_information", ""
        )
        needs_refinement_flag = status == "INCOMPLETE"

        print(f"[validate_context_sufficiency] Validation Result:")
        print(f"  - Status: {status}")
        print(f"  - Needs Refinement: {needs_refinement_flag}")
        print(f"  - LLM Missing Info: {llm_identified_missing_info[:200]}...")
        print(
            f"  - LLM Useful Info (from new content): {llm_identified_useful_info_from_new[:200]}..."
        )
        print(f"  - Reasoning: {reasoning[:200]}...")

        newly_identified_useful_info_this_round = (
            llm_identified_useful_info_from_new.strip()
        )

        # Append newly identified useful info to the running_summary
        if newly_identified_useful_info_this_round:
            if updated_running_summary:  # If there's existing summary, add a separator
                updated_running_summary += f"\\n\\n---\\nNEW FINDINGS (Loop {state.research_loop_count}):\\n{newly_identified_useful_info_this_round}"
            else:  # First time adding useful info
                updated_running_summary = f"INITIAL FINDINGS (Loop {state.research_loop_count}):\\n{newly_identified_useful_info_this_round}"
            print(
                f"[validate_context_sufficiency] Appended new useful info to running_summary. New length: {len(updated_running_summary)}"
            )
        else:
            print(
                "[validate_context_sufficiency] No new useful information identified by LLM from the latest search results."
            )
            if (
                not needs_refinement_flag
                and not updated_running_summary
                and new_search_content.strip()
            ):  # If complete, and summary is empty, use new content
                updated_running_summary = f"INITIAL FINDINGS (Loop {state.research_loop_count}):\\n{new_search_content.strip()}"
                newly_identified_useful_info_this_round = new_search_content.strip()

        updates = {
            "running_summary": updated_running_summary,
            "useful_information": newly_identified_useful_info_this_round,  # Info from THIS validation step
            "missing_information": llm_identified_missing_info,
            "refinement_reasoning": reasoning,
            "needs_refinement": needs_refinement_flag,
            "web_research_results": [],
        }

    except json.JSONDecodeError as e:
        print(
            f"[validate_context_sufficiency] ERROR: Failed to parse JSON from LLM output: {e}"
        )
        print(
            f"[validate_context_sufficiency] Raw non-JSON output: {json_response_str}"
        )
        # Fallback: assume incomplete, keep existing running_summary, pass raw output as missing info
        updates = {
            "running_summary": current_running_summary,
            "useful_information": "",  # No new useful info parsed
            "missing_information": f"LLM validation output was not valid JSON. Raw output: {raw_llm_output}",
            "refinement_reasoning": reasoning + " (Error parsing LLM JSON response)",
            "needs_refinement": True,
            "web_research_results": [],
        }

    end_time = time.time()
    print(
        f"[validate_context_sufficiency] Processing time: {end_time - start_time:.2f} seconds"
    )
    print(f"--- EXITING validate_context_sufficiency ---")
    return updates


def refine_query(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Refine the search query based on the current state, including running_summary and missing_information.
    """
    print(
        f"--- ENTERING refine_query (Loop {state.research_loop_count}) ---"
    )  # research_loop_count is already incremented before this node
    start_time = time.time()

    next_loop_count = state.research_loop_count + 1
    print(
        f"[refine_query] Incremented research_loop_count from {state.research_loop_count} to {next_loop_count}"
    )

    configurable = Configuration.from_runnable_config(config)
    provider = configurable.llm_provider or "openai"
    model = configurable.llm_model or "o3-mini-reasoning"

    if hasattr(state, "llm_provider") and state.llm_provider:
        provider = state.llm_provider
    if hasattr(state, "llm_model") and state.llm_model:
        model = state.llm_model

    print(f"[refine_query] Using provider={provider}, model={model}")
    llm = get_llm_client(provider, model)

    original_query = state.research_topic
    # This is identified useful info from the *last* validation step
    useful_info_from_last_validation = getattr(state, "useful_information", "")
    # This is what's still missing after the last validation, considering all info so far
    missing_info_after_last_validation = getattr(
        state, "missing_information", "No specific missing information identified yet."
    )
    # This is all useful information accumulated over all previous loops
    current_running_summary = getattr(state, "running_summary", "")
    # Reasoning for why refinement might be needed (from last validation)
    refinement_reasoning_from_last_validation = getattr(
        state, "refinement_reasoning", ""
    )

    # Construct a comprehensive context for the LLM to refine the query
    refinement_context_parts = [f"Original Research Topic/Question: {original_query}"]
    if current_running_summary.strip():
        refinement_context_parts.append(
            f"CUMULATIVE KNOWLEDGE SO FAR (Running Summary):\\n{current_running_summary}"
        )
    if useful_info_from_last_validation.strip():
        refinement_context_parts.append(
            f"NEWLY IDENTIFIED USEFUL INFORMATION (From Last Validation Step):\\n{useful_info_from_last_validation}"
        )
    if missing_info_after_last_validation.strip():
        refinement_context_parts.append(
            f"CURRENTLY MISSING INFORMATION (Based on all knowledge so far):\\n{missing_info_after_last_validation}"
        )
    if refinement_reasoning_from_last_validation.strip():
        refinement_context_parts.append(
            f"REASONING FOR REFINEMENT (From Last Validation Step):\\n{refinement_reasoning_from_last_validation}"
        )

    full_refinement_context_for_llm = "\n\n---\n\n".join(refinement_context_parts)

    # Choose the appropriate refinement prompt based on mode
    if state.benchmark_mode:
        refine_prompt = BENCHMARK_REFINE_QUERY_PROMPT
    elif state.qa_mode:
        refine_prompt = QA_REFINE_QUERY_PROMPT
    else:
        refine_prompt = QA_REFINE_QUERY_PROMPT  # Fallback

    prompt = refine_prompt.format(
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        question=original_query,  # Keep original question for top-level context
        # retrieved_context is now a more structured block of info
        retrieved_context=full_refinement_context_for_llm,
    )

    print(f"[refine_query] Sending refinement prompt to {provider}/{model}")
    print(
        f"[refine_query] Context for LLM (first 300 chars): {full_refinement_context_for_llm[:300]}..."
    )

    invoke_kwargs = {}
    if provider == "openai":
        # GPT-4o has a large context window, but output for a query should be small.
        # The error indicated 16384 was a limit for completion tokens for the model.
        # Let's request far less for the refined query itself.
        invoke_kwargs["max_tokens"] = 500
    # Note: For Google Gemini, max_output_tokens should be set in the client constructor,
    # not passed to invoke(). The invoke() method doesn't accept this parameter.
    # Add other providers and their specific max token parameters if necessary

    try:
        print(
            f"[refine_query] Calling llm.invoke with provider-specific kwargs: {invoke_kwargs}"
        )
        response = llm.invoke(prompt, **invoke_kwargs)
    except TypeError as te:
        # This catch might now be for other unexpected TypeErrors,
        # as we are trying to use correct param names above.
        print(
            f"[refine_query] Warning: llm.invoke failed with TypeError. Provider: {provider}, Error: {te}"
        )
        print(
            f"[refine_query] Attempting invoke without explicit max token override for this call."
        )
        response = llm.invoke(prompt)  # Fallback to calling without it

    raw_llm_output = response.content if hasattr(response, "content") else str(response)
    print(f"[refine_query] Raw LLM response: {raw_llm_output[:300]}...")

    # Attempt to parse JSON, but also handle plain text if JSON fails
    refined_query_str = original_query  # Default to original if parsing fails
    refinement_reasoning_llm = ""

    # Robustly strip ```json and ``` markers and <think> tags
    json_part_str = raw_llm_output.strip()
    if "<think>" in json_part_str and "</think>" in json_part_str:
        parts = json_part_str.split("</think>", 1)
        # We don't need the <think> part for refine_query's primary output
        json_part_str = parts[1].strip() if len(parts) > 1 else ""

    if json_part_str.startswith("```json"):
        json_part_str = json_part_str[7:]
    if json_part_str.startswith("```"):
        json_part_str = json_part_str[3:]
    if json_part_str.endswith("```"):
        json_part_str = json_part_str[:-3]
    json_part_str = json_part_str.strip()

    try:
        parsed_output = json.loads(json_part_str)
        if isinstance(parsed_output, dict):
            refined_query_str = parsed_output.get("refined_query", original_query)
            refinement_reasoning_llm = parsed_output.get("reasoning", "")
            print(
                f"[refine_query] Parsed refined_query from JSON object: '{refined_query_str}'"
            )
            if refinement_reasoning_llm:
                print(
                    f"[refine_query] LLM reasoning for refinement: {refinement_reasoning_llm[:200]}..."
                )
        elif isinstance(parsed_output, str):
            # This handles the case where the LLM returns a JSON string literal, e.g., ""actual query""
            refined_query_str = parsed_output
            refinement_reasoning_llm = ""  # No reasoning if it was just a string
            print(
                f"[refine_query] Parsed refined_query from JSON string literal: '{refined_query_str}'"
            )
        else:
            # Should not happen if json.loads worked, but as a safe fallback
            print(
                f"[refine_query] json.loads returned an unexpected type: {type(parsed_output)}. Falling back to original query."
            )
            refined_query_str = original_query
            refinement_reasoning_llm = "LLM returned an unexpected JSON structure."

    except json.JSONDecodeError:
        print(
            f"[refine_query] Failed to parse JSON from LLM refinement output. Treating raw output as refined query."
        )
        # If not JSON, assume the entire (stripped) response is the refined query
        refined_query_str = raw_llm_output.strip()
        if not refined_query_str:  # Safety net if LLM returns empty or only whitespace
            print(
                "[refine_query] LLM returned empty refined query, falling back to original query."
            )
            refined_query_str = original_query
        else:
            # Extract first non-empty line if output is multi-line non-JSON
            first_line_query = refined_query_str.split("\\n")[0].strip()
            if first_line_query:
                refined_query_str = first_line_query
            else:  # If first line is empty, take original_query
                refined_query_str = original_query

            print(
                f"[refine_query] Using raw output as refined_query: '{refined_query_str}'"
            )

    # Update the research_topic in the state to guide the next round of multi_agents_network
    # Or, if you prefer to keep original research_topic pristine, use a new state field like 'current_search_focus'

    updated_state = {
        # 'research_topic': refined_query_str, # DO NOT UPDATE research_topic here
        "search_query": refined_query_str,  # Update search_query to reflect the new focus for routing
        "research_loop_count": next_loop_count,  # Increment loop count
        "refinement_reasoning": refinement_reasoning_llm,  # Store LLM's reasoning for this refinement
        # research_topic (the original question) is preserved implicitly from the input state
        # Other state fields like running_summary, useful_information, missing_information are preserved implicitly
    }

    end_time = time.time()
    print(f"[refine_query] Original Query: {original_query}")
    print(f"[refine_query] Refined Query for next loop: {refined_query_str}")
    print(f"[refine_query] Processing time: {end_time - start_time:.2f} seconds")
    print(f"--- EXITING refine_query ---")
    return updated_state
