"""
Report generation, reflection, and finalization nodes.

Contains the nodes for the regular (non-benchmark) research flow:
- generate_report: Generate research report from gathered information
- reflect_on_report: Analyze report quality and identify knowledge gaps
- finalize_report: Produce final publication-quality document
- generate_markdown_report: Convert report to clean markdown
- post_process_report: Ensure citation consistency
- route_research: Determine if research should continue
- route_after_search: Route based on search results
- route_after_multi_agents: Route after multi-agent search
"""

import json
import time
import re
import os
import traceback
import logging
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional

from typing_extensions import Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel

from src.state import SummaryState
from src.configuration import Configuration
from llm_clients import get_llm_client, CURRENT_DATE, CURRENT_YEAR, ONE_YEAR_AGO
from src.utils import (
    deduplicate_and_format_sources,
    format_sources,
    deduplicate_sources_list,
    generate_numbered_sources,
    extract_domain,
)
from src.prompts import (
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions,
    finalize_report_instructions,
)
from src.nodes.utils import get_callback_from_config, emit_event, get_max_loops, get_configurable

logger = logging.getLogger(__name__)


def generate_report(state: SummaryState, config: RunnableConfig):
    """
    Generate a research report based on the current state.

    This function takes the current research state and generates a comprehensive
    report by summarizing the findings and integrating new web research results.

    Args:
        state: The current state with research results
        config: Configuration for the report generation

    Returns:
        Updated state with the generated report
    """
    print(f"[UPLOAD_TRACE] generate_report: Function called")
    print(f"[UPLOAD_TRACE] generate_report: State type: {type(state)}")
    print(
        f"[UPLOAD_TRACE] generate_report: State has uploaded_knowledge attr: {hasattr(state, 'uploaded_knowledge')}"
    )
    if hasattr(state, "uploaded_knowledge"):
        print(
            f"[UPLOAD_TRACE] generate_report: State.uploaded_knowledge value: {getattr(state, 'uploaded_knowledge', 'MISSING')}"
        )

    # CRITICAL DEBUG: Check steering state at function entry
    logger.info(f"[generate_report] ===== STEERING DEBUG =====")
    logger.info(
        f"[generate_report] state.steering_enabled: {getattr(state, 'steering_enabled', 'MISSING')}"
    )
    logger.info(
        f"[generate_report] state.steering_todo: {getattr(state, 'steering_todo', 'MISSING')}"
    )
    logger.info(f"[generate_report] ===========================")

    # Get the current research loop count
    research_loop_count = getattr(state, "research_loop_count", 0)
    print(f"--- ENTERING generate_report (Loop {research_loop_count}) ---")

    # Step 1: Check if we have any new web research content
    # Combine ALL raw content strings from the last loop's results
    all_new_raw_content = (
        "\n\n---\n\n".join(
            item.get("content", "")
            for item in state.web_research_results
            if isinstance(item, dict) and item.get("content")
        )
        if state.web_research_results
        else ""
    )

    # Optionally remove base64 images from the textual content
    base64_pattern = r"data:image/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+"
    cleaned_web_research = re.sub(
        base64_pattern, "[Image Data Removed]", all_new_raw_content
    )

    existing_summary = state.running_summary or ""
    knowledge_gap = getattr(state, "knowledge_gap", "")

    print(
        f"DEBUG: Length of combined cleaned_web_research: {len(cleaned_web_research)}"
    )
    print(f"DEBUG: Length of existing_summary: {len(existing_summary)}")
    print(f"DEBUG: Length of knowledge_gap: {len(knowledge_gap)}")

    # Ensure we have source_citations
    source_citations = getattr(state, "source_citations", {})
    if (
        not source_citations
        and hasattr(state, "web_research_results")
        and state.web_research_results
    ):
        # Extract sources from web_research_results if source_citations is empty
        print(
            "SOURCE EXTRACTION: No source_citations found, extracting from web_research_results"
        )
        source_citations = {}
        citation_index = 1

        # Extract sources from each web_research_result
        for result in state.web_research_results:
            if isinstance(result, dict) and "sources" in result:
                result_sources = result.get("sources", [])
                for source in result_sources:
                    if (
                        isinstance(source, dict)
                        and "title" in source
                        and "url" in source
                    ):
                        # Add to source_citations if not already present
                        source_url = source.get("url")
                        found = False
                        for citation_key, citation_data in source_citations.items():
                            if citation_data.get("url") == source_url:
                                found = True
                                break

                        if not found:
                            citation_key = str(citation_index)
                            source_dict = {
                                "title": source["title"],
                                "url": source["url"],
                            }
                            # Preserve source_type if it exists
                            if "source_type" in source:
                                source_dict["source_type"] = source["source_type"]
                            source_citations[citation_key] = source_dict
                            citation_index += 1

        # Update state.source_citations with the newly extracted sources
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Extracted {len(source_citations)} sources from web_research_results"
        )

    # If source_citations is still empty, check sources_gathered
    if (
        not source_citations
        and hasattr(state, "sources_gathered")
        and state.sources_gathered
    ):
        print(
            "SOURCE EXTRACTION: No source_citations found, creating from sources_gathered"
        )
        source_citations = {}
        for idx, source_str in enumerate(state.sources_gathered):
            if isinstance(source_str, str) and " : " in source_str:
                try:
                    title, url = source_str.split(" : ", 1)
                    source_citations[str(idx + 1)] = {"title": title, "url": url}
                except Exception as e:
                    print(
                        f"SOURCE EXTRACTION: Error parsing source {source_str}: {str(e)}"
                    )

        # Update state.source_citations
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Created {len(source_citations)} source citations from sources_gathered"
        )

    uploaded_knowledge_content = getattr(state, "uploaded_knowledge", None)
    external_knowledge_section = ""

    # Enhanced logging for uploaded knowledge
    print(f"[UPLOAD_TRACE] generate_report: Checking for uploaded_knowledge")
    print(
        f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content = {uploaded_knowledge_content}"
    )
    print(
        f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content type = {type(uploaded_knowledge_content)}"
    )

    if uploaded_knowledge_content:
        print(f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content is truthy")
        print(
            f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content.strip() = '{uploaded_knowledge_content.strip()}'"
        )

    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        print(
            f"DEBUG: Including uploaded_knowledge in generate_report. Length: {len(uploaded_knowledge_content)}"
        )
        print(f"[UPLOAD_TRACE] generate_report: Creating external_knowledge_section")
        external_knowledge_section = f"""User-Provided External Knowledge:
------------------------------------------------------------
{uploaded_knowledge_content}

"""
        print(
            f"[UPLOAD_TRACE] generate_report: external_knowledge_section created, length: {len(external_knowledge_section)}"
        )
    else:
        print("DEBUG: No uploaded_knowledge to include in generate_report.")
        print(f"[UPLOAD_TRACE] generate_report: No external knowledge section created")

    if source_citations:
        print(f"Using {len(source_citations)} source citations for summarizer")

        # Check if we have database query results
        database_sources = [
            source
            for source in source_citations.values()
            if source.get("source_type") == "database"
        ]
        # Database results (from text2sql) are already well-formatted HTML tables
        # They flow through normal report generation just like web search results
        print(
            f"[DEBUG] Total sources: {len(source_citations)}, Database sources: {len([s for s in source_citations if 'database://' in s])}"
        )
    else:
        print("WARNING: No source citations found for summarizer. We'll still proceed.")

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    if isinstance(configurable.llm_provider, str):
        provider = configurable.llm_provider
    else:
        provider = configurable.llm_provider.value

    # If user set llm_provider in state, prefer that
    if hasattr(state, "llm_provider") and state.llm_provider:
        provider = state.llm_provider
    else:
        # Default to Google Gemini for report generation
        provider = "google"

    if hasattr(state, "llm_model") and state.llm_model:
        model = state.llm_model
    else:
        model = configurable.llm_model or "gemini-2.5-pro"

    print(f"[generate_report] Summarizing with provider={provider}, model={model}")
    from llm_clients import get_llm_client

    llm = get_llm_client(provider, model)

    # Build the system prompt from summarizer_instructions
    AUGMENT_KNOWLEDGE_CONTEXT = ""
    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should be treated as highly authoritative and trustworthy. This uploaded knowledge should form the foundation of your research synthesis, with web search results used to complement, validate, or provide recent updates.

Uploaded Knowledge Preview: {uploaded_knowledge_content[:500]}{'...' if len(uploaded_knowledge_content) > 500 else ''}
"""
    else:
        AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Rely on web search results as primary sources."

    system_prompt = summarizer_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
    )

    # Provide the existing summary, new content, knowledge_gap for merging
    human_message = f"""Please integrate newly fetched content and any user-provided external knowledge into our running summary.

{external_knowledge_section}Existing Summary (previous round):
------------------------------------------------------------
{existing_summary}

Newly Fetched Web Research (current round):
------------------------------------------------------------
{cleaned_web_research}

Knowledge Gap or Additional Context:
------------------------------------------------------------
{knowledge_gap}

Citations in state: {json.dumps(source_citations, indent=2)}

Generate an updated summary that merges the newly fetched content into the existing summary, removing redundancies while retaining important details. Keep or add citation markers as needed, but do not finalize references section here. This is an internal incremental summary.

Return the updated summary as plain text:
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]
    )

    if hasattr(response, "content"):
        updated_summary = response.content
    else:
        updated_summary = str(response)

    # Replace the old running_summary with the updated merged summary
    state.running_summary = updated_summary

    # Clear the web_research_results so we don't keep huge raw content
    # The summary now incorporates important details from these search results
    cleared_web_research_results = []

    # Return updated state
    return {
        "running_summary": state.running_summary,
        "web_research_results": cleared_web_research_results,  # Cleared after use
        "knowledge_gap": knowledge_gap,
        "source_citations": source_citations,  # Use the updated source_citations
        "research_loop_count": state.research_loop_count,
        "research_topic": state.research_topic,
        "formatted_sources": getattr(
            state, "formatted_sources", ""
        ),  # Preserve formatted_sources if it exists
        "sources_gathered": getattr(
            state, "sources_gathered", []
        ),  # Preserve sources_gathered
        "visualizations": getattr(state, "visualizations", []),
        "base64_encoded_images": getattr(state, "base64_encoded_images", []),
        "visualization_paths": getattr(state, "visualization_paths", []),
        "selected_search_tool": state.selected_search_tool,
        "code_snippets": getattr(state, "code_snippets", []),
    }


def reflect_on_report(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Reflect on the current research report and decide if further research is needed.
    This function analyzes the current research, identifies knowledge gaps,
    and determines the next search query.

    Args:
        state: The current state with research results
        config: Configuration for the reflection process

    Returns:
        Updated state with research_loop_count incremented and possibly new search_query
    """
    try:
        # CRITICAL DEBUG: Check steering state at function entry
        logger.info(f"[reflect_on_report] ===== STEERING DEBUG =====")
        logger.info(
            f"[reflect_on_report] state.steering_enabled: {getattr(state, 'steering_enabled', 'MISSING')}"
        )
        logger.info(
            f"[reflect_on_report] state.steering_todo: {getattr(state, 'steering_todo', 'MISSING')}"
        )
        logger.info(
            f"[reflect_on_report] state.steering_todo type: {type(getattr(state, 'steering_todo', None))}"
        )
        if hasattr(state, "steering_todo") and state.steering_todo:
            logger.info(
                f"[reflect_on_report] steering_todo.tasks count: {len(state.steering_todo.tasks)}"
            )
            logger.info(
                f"[reflect_on_report] steering_todo.pending_messages: {len(state.steering_todo.pending_messages)}"
            )
        logger.info(f"[reflect_on_report] ===========================")

        # IMPORTANT: If we have a database report, skip LLM reflection and mark complete
        # All reports go through normal reflection, including database results

        configurable = get_configurable(config)

        # Get current research state
        research_loop_count = getattr(state, "research_loop_count", 0)
        extra_effort = getattr(state, "extra_effort", False)
        minimum_effort = getattr(
            state, "minimum_effort", False
        )  # Get minimum_effort flag
        max_research_loops = get_max_loops(
            configurable, extra_effort, minimum_effort
        )
        research_topic = state.research_topic

        print(f"\n--- REFLECTION START (Loop {research_loop_count+1}) ---")
        print(f"  - Current Loop Count: {research_loop_count}")
        print(
            f"  - Max Research Loops: {max_research_loops} (extra_effort={extra_effort}, minimum_effort={minimum_effort})"
        )
        print(f"  - Research Topic: {research_topic}")

        # Increment the research loop counter
        next_research_loop_count = research_loop_count + 1

        # Check if we've reached the maximum number of research loops
        if next_research_loop_count > max_research_loops:
            # PRIORITY CHECK: Only steering messages prevent max_loop stop (NOT tasks!)
            # User requirement: "if no steering messages left to process, but if max_loops reached then research stops"
            pending_messages_count = 0

            if (
                hasattr(state, "pending_steering_messages")
                and state.pending_steering_messages
            ):
                pending_messages_count = len(state.pending_steering_messages)

            # Also check todo manager for pending messages
            if hasattr(state, "steering_todo") and state.steering_todo:
                pending_messages_count += len(state.steering_todo.pending_messages)

            if pending_messages_count > 0:
                logger.warning(
                    f"[REFLECT] 🚨 Max loops reached but {pending_messages_count} steering messages still need processing - MUST continue"
                )
                research_complete = False  # MUST process all steering messages
            else:
                logger.info(
                    f"[REFLECT] 🛑 Max loops reached with no steering messages - STOPPING research"
                )
                research_complete = True  # Hard stop at max loops

            print(
                f"REFLECTION DECISION: Reached maximum research loops ({max_research_loops}). Pending steering messages: {pending_messages_count}, research_complete={research_complete}"
            )
            return {
                # Fields calculated/updated by this node
                "research_loop_count": next_research_loop_count,
                "research_complete": research_complete,
                "knowledge_gap": "",
                "search_query": "",
                "extra_effort": extra_effort,
                "minimum_effort": minimum_effort,
                # Reflection metadata (max loops reached)
                "priority_section": None,
                "section_gaps": None,
                "evaluation_notes": "Max loops reached, forcing completion",
                # <<< START FIX: Preserve fields from input state >>>
                "research_topic": state.research_topic,
                "running_summary": state.running_summary,
                "sources_gathered": state.sources_gathered,
                "source_citations": state.source_citations,
                "visualization_paths": getattr(state, "visualization_paths", []),
                "web_research_results": getattr(state, "web_research_results", []),
                "visualizations": getattr(state, "visualizations", []),
                "base64_encoded_images": getattr(state, "base64_encoded_images", []),
                "code_snippets": getattr(
                    state, "code_snippets", []
                ),  # Preserve code_snippets
                # <<< END FIX >>>
            }

        # Get LLM client - use Gemini for reflection
        provider = configurable.llm_provider or "google"
        model = configurable.llm_model or "gemini-2.5-pro"

        # Prioritize provider and model from state if they exist
        if hasattr(state, "llm_provider") and state.llm_provider:
            provider = state.llm_provider

        if hasattr(state, "llm_model") and state.llm_model:
            model = state.llm_model

        logger.info(f"[reflect_on_report] Using provider: {provider}, model: {model}")

        # Import the get_llm_client function
        from llm_clients import get_llm_client

        # Call with correct parameters
        llm = get_llm_client(provider, model)

        # Format the reflection_instructions with the appropriate context
        uploaded_knowledge_content = getattr(state, "uploaded_knowledge", None)
        AUGMENT_KNOWLEDGE_CONTEXT = ""
        if uploaded_knowledge_content and uploaded_knowledge_content.strip():
            AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should be considered when evaluating research completeness. This uploaded knowledge is highly authoritative and may already cover significant portions of the research topic.

Uploaded Knowledge Preview: {uploaded_knowledge_content[:500]}{'...' if len(uploaded_knowledge_content) > 500 else ''}

When evaluating completeness, consider:
- What aspects are already well-covered by the uploaded knowledge
- Whether web research has successfully complemented the uploaded knowledge
- Focus knowledge gaps on areas not covered by either uploaded knowledge or web research
"""
        else:
            AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Evaluate completeness based solely on web research results."

        # CRITICAL: Collect todo context and steering messages for unified reflection
        # NEW ARCHITECTURE: Send ONLY pending tasks for completion evaluation
        # Completed tasks sent separately when creating NEW tasks (to avoid duplicates)
        pending_tasks_for_reflection = ""
        completed_tasks_context = ""
        steering_messages = ""
        print(f"[reflect_on_report] state.steering_todo: {state.steering_todo}")
        if hasattr(state, "steering_todo") and state.steering_todo:
            # Get ONLY pending tasks for LLM to evaluate completion
            pending_tasks_for_reflection = (
                state.steering_todo.get_pending_tasks_for_llm()
            )
            logger.info(
                f"[reflect_on_report] pending_tasks_for_reflection length: {len(pending_tasks_for_reflection)}"
            )
            logger.debug(
                f"[reflect_on_report] pending_tasks_for_reflection content:\n{pending_tasks_for_reflection[:500]}"
            )

            # Get completed tasks context (for creating new tasks without duplicates)
            # IMPORTANT: Show ALL completed tasks to prevent duplicates!
            completed_tasks_context = state.steering_todo.get_completed_tasks_for_llm(
                limit=None  # Show ALL completed tasks, not just 10
            )
            completed_count = len(state.steering_todo.get_completed_tasks())
            logger.info(
                f"[reflect_on_report] Showing {completed_count} completed tasks to LLM (length: {len(completed_tasks_context)} chars)"
            )
            logger.info(
                f"[reflect_on_report] completed_tasks_context preview:\n{completed_tasks_context[:800]}"
            )

            # CRITICAL: Snapshot the message queue to prevent race conditions
            # If user sends messages DURING reflection, we need to preserve them
            messages_snapshot = list(state.steering_todo.pending_messages)

            # Get pending steering messages (queued by prepare_steering_for_next_loop)
            # Index messages with [0], [1], etc. for LLM to reference in clear_messages
            if messages_snapshot:
                steering_messages = "\n".join(
                    [f'[{i}] "{msg}"' for i, msg in enumerate(messages_snapshot)]
                )
                logger.info(
                    f"[reflect_on_report] Snapshotted {len(messages_snapshot)} steering messages for LLM processing"
                )
            else:
                steering_messages = "No new steering messages this loop"

            logger.info(
                f"[reflect_on_report] Pending tasks: {len(state.steering_todo.get_pending_tasks())}"
            )
            logger.info(
                f"[reflect_on_report] Completed tasks: {len(state.steering_todo.get_completed_tasks())}"
            )
            logger.info(
                f"[reflect_on_report] Steering messages: {len(state.steering_todo.pending_messages)}"
            )
        else:
            pending_tasks_for_reflection = "No todo list active (steering disabled)"
            completed_tasks_context = ""
            steering_messages = "No steering system active"

        formatted_prompt = reflection_instructions.format(
            research_topic=research_topic,
            current_date=CURRENT_DATE,
            current_year=CURRENT_YEAR,
            one_year_ago=ONE_YEAR_AGO,
            AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
            pending_tasks=pending_tasks_for_reflection,
            completed_tasks=completed_tasks_context,
            steering_messages=steering_messages,
        )

        # Prepare the current summary for analysis
        current_summary = (
            state.running_summary if hasattr(state, "running_summary") else ""
        )

        # Call LLM with the properly formatted system prompt
        response = llm.invoke(
            [
                SystemMessage(content=formatted_prompt),
                HumanMessage(
                    content=f"Analyze this research summary and determine if more research is needed:\n\n{current_summary}"
                ),
            ]
        )

        print("  - Raw LLM Reflection Response:")
        print(f"    {response}")

        # Extract content based on the response type
        if hasattr(response, "content"):
            content = response.content
        else:
            content = (
                response  # SimpleOpenAIClient or Claude3ExtendedClient returns a string
            )

        # Parse the response - extract JSON from <answer> tags
        def parse_wrapped_response(reg_exp, text_phrase):
            match = re.search(reg_exp, text_phrase, re.DOTALL)
            if match:
                return match.group(1)
            return ""

        try:
            # First try to extract from <answer> tags
            json_str = parse_wrapped_response(r"<answer>\s*(.*?)\s*</answer>", content)

            if json_str:
                # Clean up the JSON string
                json_str = json_str.strip()
                result = json.loads(json_str)
            else:
                # Fallback: Look for JSON block in markdown code blocks
                json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                else:
                    # Last resort: Try parsing the entire content as JSON
                    result = json.loads(content)

            # Log the reflection result
            print("  - Parsed LLM Reflection Result:")
            print(f"    {json.dumps(result, indent=4)}")

            # Extract the key information from the result
            research_complete = result.get("research_complete", False)
            knowledge_gap = result.get("knowledge_gap", "")
            search_query = result.get("follow_up_query", "")

            # Check for pending steering messages - override research_complete if messages pending
            has_pending_steering = (
                hasattr(state, "pending_steering_messages")
                and state.pending_steering_messages
            )

            # CRITICAL: Check for pending steering messages in todo manager
            if hasattr(state, "steering_todo") and state.steering_todo:
                todo_pending_messages = len(state.steering_todo.pending_messages)
                if todo_pending_messages > 0:
                    has_pending_steering = True
                    logger.info(
                        f"[REFLECT] Found {todo_pending_messages} pending steering messages in todo manager"
                    )

            if has_pending_steering and research_complete:
                logger.info(
                    f"[REFLECT] LLM marked complete but {len(state.pending_steering_messages) if hasattr(state, 'pending_steering_messages') else 0} steering messages pending - continuing research"
                )
                research_complete = False
                # Create knowledge gap for steering messages
                if not knowledge_gap:
                    knowledge_gap = "Process pending steering messages"

            # Extract the research_topic field, or use the original if not provided
            preserved_research_topic = result.get("research_topic", research_topic)

            # If research is complete, ensure search_query is empty
            if research_complete:
                print(
                    "  - LLM determined research is complete. Clearing knowledge gap and search query."
                )
                search_query = ""
                knowledge_gap = ""
            else:
                print("  - LLM determined research should continue.")

            # CRITICAL: Process todo_updates from LLM response
            if hasattr(state, "steering_todo") and state.steering_todo:
                todo_updates = result.get("todo_updates", {})
                logger.info(
                    f"[reflect_on_report] LLM result keys: {list(result.keys())}"
                )
                logger.info(
                    f"[reflect_on_report] todo_updates present: {bool(todo_updates)}"
                )
                logger.info(f"[reflect_on_report] todo_updates content: {todo_updates}")

                if todo_updates:

                    # Mark tasks as completed (only if they're currently PENDING or IN_PROGRESS)
                    from src.simple_steering import TaskStatus

                    mark_completed_list = todo_updates.get("mark_completed", [])
                    logger.info(
                        f"[reflect_on_report] mark_completed list: {mark_completed_list}"
                    )

                    for task_id in mark_completed_list:
                        logger.info(f"🔍 [DEBUG] Processing task_id: {task_id}")

                        if task_id in state.steering_todo.tasks:
                            logger.info(
                                f"🔍 [DEBUG] Task {task_id} found in tasks dict"
                            )
                            task = state.steering_todo.tasks[task_id]
                            logger.info(f"🔍 [DEBUG] Task status: {task.status}")

                            # Only mark as completed if it's NOT already completed
                            if task.status == TaskStatus.COMPLETED:
                                logger.debug(
                                    f"[reflect_on_report] Task {task_id} already COMPLETED, skipping"
                                )
                                continue

                            # Only mark as completed if it's PENDING or IN_PROGRESS
                            if task.status in [
                                TaskStatus.PENDING,
                                TaskStatus.IN_PROGRESS,
                            ]:
                                state.steering_todo.mark_task_completed(
                                    task_id,
                                    completion_note="Addressed in research loop",
                                )
                                logger.info(
                                    f"[reflect_on_report] ✓ Marked task {task_id} as completed"
                                )
                            else:
                                logger.debug(
                                    f"[reflect_on_report] Task {task_id} has status {task.status.name}, not marking completed"
                                )
                        else:
                            logger.warning(
                                f"[reflect_on_report] ⚠️ Task {task_id} not found in tasks dict. Available tasks: {list(state.steering_todo.tasks.keys())[:5]}"
                            )

                    # Cancel tasks
                    cancel_tasks_list = todo_updates.get("cancel_tasks", [])
                    logger.info(
                        f"[reflect_on_report] cancel_tasks list: {cancel_tasks_list}"
                    )
                    for task_id in cancel_tasks_list:
                        if task_id in state.steering_todo.tasks:
                            state.steering_todo.mark_task_cancelled(
                                task_id, reason="No longer relevant based on findings"
                            )
                            logger.info(
                                f"[reflect_on_report] ✗ Cancelled task {task_id}"
                            )

                    # Add new tasks with source-based priority
                    # Priority mapping: steering_message=10, original_query=9, knowledge_gap=7
                    SOURCE_PRIORITY = {
                        "steering_message": 10,  # User explicitly requested
                        "original_query": 9,  # From initial research query
                        "knowledge_gap": 7,  # System-identified gaps
                    }

                    add_tasks_list = todo_updates.get("add_tasks", [])
                    logger.info(
                        f"[reflect_on_report] add_tasks list length: {len(add_tasks_list)}"
                    )
                    for i, new_task in enumerate(add_tasks_list):
                        source = new_task.get("source", "knowledge_gap")
                        priority = SOURCE_PRIORITY.get(source, 8)

                        logger.info(
                            f"[reflect_on_report] Processing new task {i+1}/{len(add_tasks_list)}: {new_task}"
                        )
                        task_id = state.steering_todo.create_task(
                            description=new_task.get("description", ""),
                            priority=priority,
                            source=source,
                            created_from_message=new_task.get(
                                "rationale", "Added by reflection"
                            ),
                        )
                        logger.info(
                            f"[reflect_on_report] + Added task {task_id} (source: {source}, priority: {priority}): {new_task.get('description', '')[:60]}"
                        )

                    # SMART MESSAGE CLEARING: Only clear messages LLM says are fully addressed
                    # Use the snapshot to avoid race conditions with messages added during reflection
                    clear_message_indices = todo_updates.get("clear_messages", [])
                    if clear_message_indices:
                        original_snapshot_count = len(messages_snapshot)

                        # Clear from snapshot (not live list!)
                        remaining_snapshot_messages = [
                            msg
                            for i, msg in enumerate(messages_snapshot)
                            if i not in clear_message_indices
                        ]

                        # Now merge: Keep any NEW messages added during reflection + remaining snapshot messages
                        current_live_messages = state.steering_todo.pending_messages
                        new_messages_during_reflection = [
                            msg
                            for msg in current_live_messages
                            if msg not in messages_snapshot
                        ]

                        # Final queue = remaining snapshot + new messages
                        state.steering_todo.pending_messages = (
                            remaining_snapshot_messages + new_messages_during_reflection
                        )

                        cleared_count = original_snapshot_count - len(
                            remaining_snapshot_messages
                        )
                        if new_messages_during_reflection:
                            logger.info(
                                f"[reflect_on_report] ⚡ {len(new_messages_during_reflection)} new messages arrived during reflection - preserved!"
                            )
                        logger.info(
                            f"[reflect_on_report] Cleared {cleared_count}/{original_snapshot_count} steering messages: indices {clear_message_indices}"
                        )
                    else:
                        logger.info(
                            f"[reflect_on_report] No messages cleared (LLM didn't specify any in clear_messages)"
                        )

                    # Session store is automatically synced since we modified state.steering_todo.pending_messages directly
                    # The UI polling endpoint reads from state.steering_todo.pending_messages
                    # No need to manually update session store

                    # Update todo version
                    state.steering_todo.todo_version += 1
                    logger.info(
                        f"[reflect_on_report] Updated todo version to {state.steering_todo.todo_version}"
                    )

                    # CRITICAL: Update session store so UI polling picks up the changes
                    # Retry logic to ensure UI gets the update
                    session_update_success = False
                    max_retries = 3

                    for attempt in range(max_retries):
                        try:
                            from routers.simple_steering_api import (
                                active_research_sessions,
                            )

                            # Get session ID directly from config (much cleaner!)
                            session_id = config.get("configurable", {}).get("stream_id")

                            if not session_id:
                                logger.warning(
                                    f"[reflect_on_report] No session ID in config (attempt {attempt + 1}). Config keys: {list(config.get('configurable', {}).keys())}"
                                )
                                break  # No point retrying if no session ID

                            if session_id not in active_research_sessions:
                                logger.warning(
                                    f"[reflect_on_report] Session {session_id} not in active_research_sessions (attempt {attempt + 1}). Active sessions: {list(active_research_sessions.keys())}"
                                )
                                if attempt < max_retries - 1:
                                    from time import sleep

                                    sleep(0.1)  # Wait for registration
                                continue

                            # IMPORTANT: Update the state reference in the session
                            # LangGraph creates new state instances, so the stored reference gets stale
                            active_research_sessions[session_id]["state"] = state
                            session_update_success = True
                            logger.info(
                                f"[reflect_on_report] ✅ Updated session {session_id} state reference for UI polling (attempt {attempt + 1})"
                            )
                            logger.info(
                                f"[reflect_on_report] Pending messages after update: {len(state.steering_todo.pending_messages)}"
                            )
                            break  # Success - exit retry loop

                        except ImportError as e:
                            logger.error(
                                f"[reflect_on_report] Failed to import active_research_sessions (attempt {attempt + 1}): {e}"
                            )
                            break  # Import error won't fix itself
                        except KeyError as e:
                            logger.error(
                                f"[reflect_on_report] KeyError accessing session (attempt {attempt + 1}): {e}"
                            )
                            if attempt < max_retries - 1:
                                from time import sleep

                                sleep(0.1)
                        except Exception as e:
                            logger.error(
                                f"[reflect_on_report] Unexpected error updating session (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                            )
                            if attempt < max_retries - 1:
                                from time import sleep

                                sleep(0.1)  # Brief delay before retry

                    if not session_update_success and clear_message_indices:
                        logger.error(
                            "🚨 [reflect_on_report] CRITICAL: Session state update failed after all retries - UI queue may not clear!"
                        )

            # CRITICAL: Final validation - check ONLY pending steering messages (NOT tasks!)
            # User requirement: "research can stop only if steered_message_queue is empty"
            final_pending_messages = 0
            if hasattr(state, "steering_todo") and state.steering_todo:
                final_pending_messages = len(state.steering_todo.pending_messages)

                if final_pending_messages > 0 and research_complete:
                    logger.warning(
                        f"[REFLECT] 🚨 LLM marked complete but {final_pending_messages} steering messages still need processing - OVERRIDING to continue"
                    )
                    research_complete = False
                    if not knowledge_gap:
                        knowledge_gap = f"Process {final_pending_messages} pending steering messages"
                    if not search_query:
                        search_query = "Continue research to process steering messages"

            # Log the reflection decision
            print(
                f"REFLECTION DECISION: Proceeding to loop {next_research_loop_count}."
            )
            print(f"  - research_complete set to: {research_complete}")
            print(f"  - Pending steering messages: {final_pending_messages}")
            print(f"  - Identified knowledge gap: '{knowledge_gap}'")
            print(f"  - New search query: '{search_query}'")
            print(f"  - Research topic: '{preserved_research_topic}'")
            print("--- REFLECTION END ---")

            # Log complete reflection execution step (non-invasive, never fails research)
            try:
                if hasattr(state, "log_execution_step"):
                    state.log_execution_step(
                        step_type="llm_call",
                        action="reflect_on_report",
                        input_data={
                            "running_summary": (
                                current_summary[:500] + "..."
                                if len(current_summary) > 500
                                else current_summary
                            )
                        },
                        output_data={
                            "research_complete": research_complete,
                            "knowledge_gap": knowledge_gap,
                            "priority_section": result.get("priority_section"),
                            "section_gaps": result.get("section_gaps"),
                            "evaluation_notes": result.get("evaluation_notes"),
                            "follow_up_query": search_query,
                        },
                        metadata={"provider": provider, "model": model},
                    )
            except Exception:
                pass  # Logging errors should never break research

            # Return updated state
            return {
                # Fields calculated/updated by this node
                "research_loop_count": next_research_loop_count,
                "knowledge_gap": knowledge_gap,
                "search_query": search_query,
                "research_complete": research_complete,
                "research_topic": preserved_research_topic,
                "extra_effort": extra_effort,
                "minimum_effort": minimum_effort,
                # Reflection metadata (for trajectory capture)
                "priority_section": result.get("priority_section"),
                "section_gaps": result.get("section_gaps"),
                "evaluation_notes": result.get("evaluation_notes"),
                # <<< START FIX: Preserve fields from input state >>>
                "research_topic": state.research_topic,
                "running_summary": state.running_summary,
                "sources_gathered": state.sources_gathered,
                "source_citations": state.source_citations,
                "visualization_paths": getattr(state, "visualization_paths", []),
                "web_research_results": getattr(state, "web_research_results", []),
                "visualizations": getattr(state, "visualizations", []),
                "base64_encoded_images": getattr(state, "base64_encoded_images", []),
                "code_snippets": getattr(
                    state, "code_snippets", []
                ),  # Preserve code_snippets
                # CRITICAL: Preserve steering state
                "steering_enabled": getattr(state, "steering_enabled", False),
                "steering_todo": getattr(state, "steering_todo", None),
                # <<< END FIX >>>
            }

        except Exception as e:
            print(f"REFLECTION ERROR: Failed to parse LLM response: {str(e)}")
            print(f"  - Raw response: {content}")

            # Fallback to basic approach
            current_sources = getattr(state, "sources_gathered", [])
            num_sources = len(current_sources)

            if num_sources < 3:
                knowledge_gap = "Need more comprehensive sources"
                search_query = f"More detailed information about {research_topic}"
            else:
                knowledge_gap = "Need specific examples and case studies"
                search_query = f"Examples and case studies of {research_topic}"

            print(f"REFLECTION DECISION: Using fallback approach due to parsing error.")
            print(f"  - Identified knowledge gap: '{knowledge_gap}'")
            print(f"  - New search query: '{search_query}'")
            print(f"  - Research topic: '{research_topic}' (preserved from original)")
            print("--- REFLECTION END (Fallback) ---")

            # Return updated state with fallback values
            return {
                # Fields calculated/updated by this node
                "research_loop_count": next_research_loop_count,
                "knowledge_gap": knowledge_gap,
                "search_query": search_query,
                "research_complete": False,
                "research_topic": research_topic,
                "extra_effort": extra_effort,
                "minimum_effort": minimum_effort,
                # Reflection metadata (fallback - empty for trajectory capture)
                "priority_section": None,
                "section_gaps": None,
                "evaluation_notes": "Reflection parsing failed, using fallback",
                # <<< START FIX: Preserve fields from input state >>>
                "research_topic": state.research_topic,
                "running_summary": state.running_summary,
                "sources_gathered": state.sources_gathered,
                "source_citations": state.source_citations,
                "visualization_paths": getattr(state, "visualization_paths", []),
                "web_research_results": getattr(state, "web_research_results", []),
                "visualizations": getattr(state, "visualizations", []),
                "base64_encoded_images": getattr(state, "base64_encoded_images", []),
                "code_snippets": getattr(
                    state, "code_snippets", []
                ),  # Preserve code_snippets
                # <<< END FIX >>>
            }

    except Exception as e:
        print(f"REFLECTION FATAL ERROR: {str(e)}")
        # On error, increment research loop but mark as complete to avoid infinite loops
        print("  - Marking research as complete to avoid infinite loops.")
        print("--- REFLECTION END (Fatal Error) ---")
        return {
            # Fields calculated/updated by this node
            "research_loop_count": getattr(state, "research_loop_count", 0) + 1,
            "research_complete": True,
            "knowledge_gap": "",
            "search_query": "",
            "research_topic": research_topic,
            "extra_effort": extra_effort,
            "minimum_effort": minimum_effort,
            # <<< START FIX: Preserve fields from input state >>>
            "research_topic": state.research_topic,
            "running_summary": state.running_summary,
            "sources_gathered": state.sources_gathered,
            "source_citations": state.source_citations,
            "visualization_paths": getattr(state, "visualization_paths", []),
            "web_research_results": getattr(state, "web_research_results", []),
            "visualizations": getattr(state, "visualizations", []),
            "base64_encoded_images": getattr(state, "base64_encoded_images", []),
            "code_snippets": getattr(
                state, "code_snippets", []
            ),  # Preserve code_snippets
            # <<< END FIX >>>
        }


def finalize_report(state: SummaryState, config: RunnableConfig):
    """Finalize the summary into a publication-quality document"""

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Note: Steering messages that arrive during finalization are queued but don't interrupt the finalization process.
    if hasattr(state, "steering_todo") and state.steering_todo:
        pending_count = len(state.steering_todo.pending_messages)
        if pending_count > 0:
            logger.info(
                f"[finalize_report] {pending_count} steering message(s) queued but not interrupting finalization."
            )

    # All reports go through normal finalization, including database results
    current_summary = state.running_summary or ""
    web_research_results = state.web_research_results or []

    input_content_for_finalization = ""
    using_raw_content = False

    # If running summary is empty, fallback to using raw web results
    if not current_summary.strip() and web_research_results:
        print(
            "FINALIZE_REPORT: Running summary is empty. Using raw web research results."
        )
        using_raw_content = True
        # Combine and clean raw content, similar to generate_report
        all_new_raw_content = (
            "\n\n---\n\n".join(
                item.get("content", "")
                for item in web_research_results
                if isinstance(item, dict) and item.get("content")
            )
            if isinstance(web_research_results, list)
            else "\n\n---\n\n".join(web_research_results)
        )

        base64_pattern = r"data:image/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+"
        input_content_for_finalization = re.sub(
            base64_pattern, "[Image Data Removed]", all_new_raw_content
        )
        print(
            f"FINALIZE_REPORT: Using combined raw content of length {len(input_content_for_finalization)}"
        )
    else:
        print(
            f"FINALIZE_REPORT: Using existing running summary of length {len(current_summary)}"
        )
        input_content_for_finalization = current_summary

    # If even raw content is empty, we might have an issue, but proceed anyway
    if not input_content_for_finalization.strip():
        print(
            "FINALIZE_REPORT WARNING: Both running summary and raw results are empty!"
        )
        # Assign an empty string or some placeholder if necessary
        input_content_for_finalization = "(No content available to finalize)"

    # Ensure we have source_citations
    source_citations = getattr(state, "source_citations", {})
    if (
        not source_citations
        and hasattr(state, "web_research_results")
        and state.web_research_results
    ):
        # Extract sources from web_research_results if source_citations is empty
        print(
            "SOURCE EXTRACTION: No source_citations found, extracting from web_research_results"
        )
        source_citations = {}
        citation_index = 1

        # Extract sources from each web_research_result
        for result in state.web_research_results:
            if isinstance(result, dict) and "sources" in result:
                result_sources = result.get("sources", [])
                for source in result_sources:
                    if (
                        isinstance(source, dict)
                        and "title" in source
                        and "url" in source
                    ):
                        # Add to source_citations if not already present
                        source_url = source.get("url")
                        found = False
                        for citation_key, citation_data in source_citations.items():
                            if citation_data.get("url") == source_url:
                                found = True
                                break

                        if not found:
                            citation_key = str(citation_index)
                            source_dict = {
                                "title": source["title"],
                                "url": source["url"],
                            }
                            # Preserve source_type if it exists
                            if "source_type" in source:
                                source_dict["source_type"] = source["source_type"]
                            source_citations[citation_key] = source_dict
                            citation_index += 1

        # Update state.source_citations with the newly extracted sources
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Extracted {len(source_citations)} sources from web_research_results"
        )

    # If source_citations is still empty, check sources_gathered
    if (
        not source_citations
        and hasattr(state, "sources_gathered")
        and state.sources_gathered
    ):
        print(
            "SOURCE EXTRACTION: No source_citations found, creating from sources_gathered"
        )
        source_citations = {}
        for idx, source_str in enumerate(state.sources_gathered):
            if isinstance(source_str, str) and " : " in source_str:
                try:
                    title, url = source_str.split(" : ", 1)
                    source_citations[str(idx + 1)] = {"title": title, "url": url}
                except Exception as e:
                    print(
                        f"SOURCE EXTRACTION: Error parsing source {source_str}: {str(e)}"
                    )

        # Update state.source_citations
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Created {len(source_citations)} source citations from sources_gathered"
        )

    # Create a properly formatted references section
    if source_citations:
        # Format the source citations into a numbered list (These are already deduplicated by generate_numbered_sources)
        numbered_sources = [
            f"{num}. {src['title']}, [{src['url']}]"
            for num, src in sorted(source_citations.items())
        ]
        formatted_sources_for_prompt = "\n".join(numbered_sources)
        print(f"USING {len(numbered_sources)} UNIQUE NUMBERED SOURCES IN FINAL REPORT")

        # Also log sources that were gathered but not included in citations
        cited_urls = set(src["url"] for src in source_citations.values())
        all_source_texts = (
            state.sources_gathered
        )  # This list might still contain duplicates
        unused_sources = []
        seen_unused_urls = set()  # Deduplicate unused sources as well for logging
        for source_text in all_source_texts:
            if " : " in source_text:
                url = source_text.split(" : ", 1)[1].strip()
                if url not in cited_urls and url not in seen_unused_urls:
                    unused_sources.append(source_text)
                    seen_unused_urls.add(url)
        if unused_sources:
            print(
                f"NOTE: {len(unused_sources)} unique sources were gathered but not cited in the final report"
            )

    else:
        # Fallback to simple formatting if no citations were tracked
        print(
            "WARNING: No source citations found, using basic source list from sources_gathered."
        )
        all_sources_raw = state.sources_gathered
        print(
            f"DEBUG: Fallback - processing {len(all_sources_raw)} raw gathered sources."
        )

        # --- START FIX: Deduplicate sources_gathered in fallback ---
        deduplicated_sources = []
        seen_urls_fallback = set()
        for source in all_sources_raw:
            if source and ":" in source:
                try:
                    url = source.split(" : ", 1)[1].strip()
                    if url not in seen_urls_fallback:
                        deduplicated_sources.append(source)  # Keep original format
                        seen_urls_fallback.add(url)
                except Exception:
                    print(f"DEBUG: Fallback - could not parse source for URL: {source}")
                    deduplicated_sources.append(source)  # Keep unparsable ones?
            elif source:  # Keep non-empty, non-parsable sources
                deduplicated_sources.append(source)
        # --- END FIX ---

        formatted_sources_for_prompt = "\n".join(deduplicated_sources)
        print(
            f"DEBUG: Fallback - using {len(deduplicated_sources)} deduplicated sources for prompt."
        )

    # Handle both cases for llm_provider:
    # 1. When selected in Studio UI -> returns a string (e.g. "openai")
    # 2. When using default -> returns an Enum (e.g. LLMProvider.OPENAI)
    if isinstance(configurable.llm_provider, str):
        provider = configurable.llm_provider
    else:
        provider = configurable.llm_provider.value

    # Prioritize provider and model from state if they exist
    if hasattr(state, "llm_provider") and state.llm_provider:
        provider = state.llm_provider
    else:
        # Default to Google Gemini for report finalization
        provider = "google"

    # Use Gemini 2.5 Pro for final summary
    if hasattr(state, "llm_model") and state.llm_model:
        model = state.llm_model
    else:
        model = "gemini-2.5-pro"

    logger.info(f"[finalize_report] Using provider: {provider}, model: {model}")
    llm = get_llm_client(provider, model)
    print(
        f"Using cloud LLM provider: {provider} with model: {model} for finalizing summary"
    )

    # Generate the finalized summary with date information
    uploaded_knowledge_content = getattr(state, "uploaded_knowledge", None)
    AUGMENT_KNOWLEDGE_CONTEXT = ""
    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should form the authoritative foundation of your final report. This uploaded knowledge is highly trustworthy and should be given precedence over web search results.

Uploaded Knowledge Preview: {uploaded_knowledge_content[:500]}{'...' if len(uploaded_knowledge_content) > 500 else ''}

Integration Instructions:
- Use uploaded knowledge as the primary structural foundation
- Integrate web research to enhance, validate, or update uploaded knowledge
- Clearly distinguish between uploaded knowledge and web source information
- Give precedence to uploaded knowledge when conflicts arise
"""
    else:
        AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Base the final report on web research results."

    system_prompt = finalize_report_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
    )

    # Construct the human message based on whether we used raw content or the summary
    uploaded_knowledge_section = ""
    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        uploaded_knowledge_section = f"""

USER-PROVIDED EXTERNAL KNOWLEDGE (HIGHEST AUTHORITY):
------------------------------------------------------------
{uploaded_knowledge_content}

INTEGRATION INSTRUCTIONS: Use the above uploaded knowledge as your primary foundation. The web research content below should complement, validate, or provide recent updates to this authoritative knowledge.

"""

    # CRITICAL: Include todo completion status and user steering intentions in final report
    todo_completion_section = ""
    if hasattr(state, "steering_todo") and state.steering_todo:
        completed_tasks = state.steering_todo.get_completed_tasks()
        pending_tasks = state.steering_todo.get_pending_tasks()
        all_messages = getattr(state.steering_todo, "all_user_messages", [])

        if completed_tasks or pending_tasks or all_messages:
            todo_completion_section = f"""

USER STEERING AND TODO COMPLETION STATUS:
------------------------------------------------------------
The user provided {len(all_messages)} steering messages during research to guide the process.
These messages represent the user's TRUE NEEDS and PRIORITIES for the final report.

COMPLETED RESEARCH TASKS ({len(completed_tasks)} tasks):
"""
            for task in completed_tasks[-10:]:  # Last 10 completed tasks
                todo_completion_section += f"✓ {task.description}\n"
                if task.completed_note:
                    todo_completion_section += f"  └─ {task.completed_note}\n"

            if pending_tasks:
                todo_completion_section += (
                    f"\n\nREMAINING TASKS NOT COMPLETED ({len(pending_tasks)} tasks):\n"
                )
                for task in pending_tasks[:5]:  # First 5 pending tasks
                    todo_completion_section += (
                        f"⚠ {task.description} (Priority: {task.priority})\n"
                    )

            if all_messages:
                todo_completion_section += (
                    f"\n\nUSER'S STEERING MESSAGES (in chronological order):\n"
                )
                for i, msg in enumerate(all_messages[-10:], 1):  # Last 10 messages
                    todo_completion_section += f'{i}. "{msg}"\n'

            todo_completion_section += f"""

CRITICAL INSTRUCTION FOR FINAL REPORT:
Your final report MUST address all completed tasks and respect all user steering messages above.
The report should reflect the user's refined intentions throughout the research process.
If any high-priority pending tasks remain, acknowledge them as areas for future research.
The report should feel like it was written specifically to answer the user's evolving needs.

"""

    if using_raw_content:
        human_message = (
            f"Please create a polished final report with a descriptive title based on the following sources.\n\n"
            f"IMPORTANT: Begin your report with a clear, descriptive title in the format 'Profile of [Person]: [Role/Position]' or similar format appropriate to the topic. For example: 'Profile of Dr. Caiming Xiong: AI Research Leader at Salesforce' or 'State-of-the-Art Data Strategies for Pretraining 7B Parameter LLMs from Scratch'.\n\n"
            f"{uploaded_knowledge_section}"
            f"{todo_completion_section}"
            f"Raw Research Content from Web Search:\n{input_content_for_finalization}\n\n"
            f"Numbered Sources for Citation:\n{formatted_sources_for_prompt}"
        )
    else:
        human_message = (
            f"Please finalize this research summary into a polished document with a descriptive title.\n\n"
            f"IMPORTANT: Begin your report with a clear, descriptive title in the format 'Profile of [Person]: [Role/Position]' or similar format appropriate to the topic. For example: 'Profile of Dr. Caiming Xiong: AI Research Leader at Salesforce' or 'State-of-the-Art Data Strategies for Pretraining 7B Parameter LLMs from Scratch'.\n\n"
            f"{uploaded_knowledge_section}"
            f"{todo_completion_section}"
            f"Working Summary from Web Research:\n{input_content_for_finalization}\n\n"
            f"Numbered Sources for Citation:\n{formatted_sources_for_prompt}"
        )

    # Add visualization information to the prompt if available
    if (
        state.base64_encoded_images
        or hasattr(state, "visualizations")
        and state.visualizations
    ):
        visualization_info = []
        seen_filenames = set()  # Track seen filenames to avoid duplication

        # Add base64 encoded image info
        for idx, img in enumerate(
            state.base64_encoded_images[:5]
        ):  # Limit to first 5 to avoid overwhelming prompt
            # Skip if we've already processed this image
            filename = img.get("filename", "")
            if filename and filename in seen_filenames:
                continue

            if filename:
                seen_filenames.add(filename)

            title = img.get("title", f"Visualization {idx+1}")
            description = img.get("description", "")
            visualization_info.append(
                f"Image {len(visualization_info) + 1}: {title}\nDescription: {description}"
            )

        # Use full visualization objects instead of just paths
        if hasattr(state, "visualizations"):
            for idx, viz in enumerate(state.visualizations[:5]):  # Limit to first 5
                # Skip if we've already processed this image
                if "filename" in viz and viz["filename"] in seen_filenames:
                    continue

                if "filename" in viz:
                    seen_filenames.add(viz["filename"])

                # Extract title from subtask_name or filename
                title = viz.get("subtask_name", "")
                if not title and "filename" in viz:
                    # Generate title from filename if no subtask_name
                    filename = viz["filename"]
                    title_base = os.path.splitext(filename)[0].replace("_", " ").title()
                    title = title_base

                # Build enhanced description with whatever metadata is available
                description_parts = []
                if "description" in viz and viz["description"]:
                    description_parts.append(viz["description"])
                if "chart_type" in viz and viz["chart_type"]:
                    description_parts.append(f"Chart type: {viz['chart_type']}")
                if "data_summary" in viz and viz["data_summary"]:
                    description_parts.append(f"Data summary: {viz['data_summary']}")

                description = "\n".join(description_parts) if description_parts else ""

                # Add to visualization info
                visualization_info.append(
                    f"Image {len(visualization_info) + 1}: {title}\nDescription: {description}"
                )

        # Also add visualization_paths as fallback if we have no visualizations object
        # (This covers the transition period where old code might still be using visualization_paths)
        elif hasattr(state, "visualization_paths") and state.visualization_paths:
            for idx, path in enumerate(state.visualization_paths[:5]):
                # Extract just the filename to check for duplicates
                filename = os.path.basename(path)
                if filename in seen_filenames:
                    continue

                seen_filenames.add(filename)

                title_base = os.path.splitext(filename)[0]
                if len(title_base) > 6 and title_base[-6:].isalnum():
                    title_base = title_base[:-6]

                # Capitalize and replace underscores
                title = title_base.replace("_", " ").title()

                # Use the current count of visualization_info for image numbering to avoid gaps
                visualization_info.append(
                    f"Image {len(visualization_info) + 1}: {title}"
                )

        if visualization_info:
            visualization_prompt = "\n\nAvailable Visualizations:\n" + "\n\n".join(
                visualization_info
            )
            visualization_prompt += "\n\nPlease indicate where these visualizations should be placed in the report by adding [INSERT IMAGE X] markers at appropriate locations in your text. Choose the most relevant locations based on content."
            human_message += visualization_prompt

    result = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]
    )

    # Extract content based on the client type
    if hasattr(result, "content"):
        finalized_summary = result.content
    else:
        finalized_summary = (
            result  # SimpleOpenAIClient or Claude3ExtendedClient returns a string
        )

    # Post-process the final report to ensure citation consistency and include References section
    finalized_summary = post_process_report(finalized_summary, source_citations)

    # Enhance the markdown conversion for better HTML styling, particularly for the Table of Contents
    finalized_summary = re.sub(
        r"^## Table of Contents",
        "<h2>Table of Contents</h2>",
        finalized_summary,
        flags=re.MULTILINE,
    )

    # Format date on a new line with proper spacing
    finalized_summary = re.sub(
        r"<h1>(.*?)</h1>\s*(\w+ \d+, \d{4})",
        r'<h1>\1</h1>\n<p class="report-date">\2</p>',
        finalized_summary,
    )

    # Convert horizontal table of contents to vertical list format
    # Handle both bullet-separated and dash-separated formats
    toc_pattern = r"<h2>Table of Contents</h2>\s*\n*([^#<]+?)(?=\n\n|<h2>|##)"
    toc_match = re.search(toc_pattern, finalized_summary, re.DOTALL)
    if toc_match:
        toc_content = toc_match.group(1).strip()

        # First, split by newlines to get individual lines
        lines = [line.strip() for line in toc_content.split("\n") if line.strip()]

        # Then process each line - split by " - " if present
        all_toc_items = []
        for line in lines:
            # Remove leading bullet markers (-, •, *, etc.)
            line = re.sub(r"^[\-•*]\s*", "", line)

            # Split by " - " to handle concatenated sections
            if " - " in line:
                items = [item.strip() for item in line.split(" - ") if item.strip()]
                all_toc_items.extend(items)
            elif line:
                all_toc_items.append(line)

        # Create properly formatted vertical TOC
        if all_toc_items:
            vertical_toc = (
                "<ul>\n"
                + "\n".join([f"<li>{item}</li>" for item in all_toc_items])
                + "\n</ul>"
            )
            # Replace the entire TOC section
            finalized_summary = re.sub(
                toc_pattern,
                f"<h2>Table of Contents</h2>\n{vertical_toc}",
                finalized_summary,
                flags=re.DOTALL,
            )

    # create a copy for markdown report
    import copy

    markdown_final_summary = copy.deepcopy(finalized_summary)

    # ---- START NEW APPROACH: Process LLM-directed visualization placement ----
    # Define the maximum number of visualizations to embed in the LLM prompt
    MAX_VISUALIZATIONS_TO_EMBED = 5

    # Prepare visualization items for insertion into the main content
    inline_visualizations = []

    # --- START FIX: Re-add definition of base64_images ---
    # Check for base64 encoded images stored in the state
    base64_images = []
    # Try to get base64_encoded_images directly from state
    base64_images = getattr(state, "base64_encoded_images", [])
    if base64_images:
        print(
            f"🖼️ Found {len(base64_images)} base64-encoded images directly from state for final report"
        )
    else:
        # Try to extract from result_combiner in state (fallback)
        result_combiner = getattr(state, "result_combiner", None)
        if result_combiner and hasattr(result_combiner, "_base64_encoded_images"):
            base64_images = result_combiner._base64_encoded_images
            print(
                f"🖼️ Found {len(base64_images)} base64-encoded images from ResultCombiner instance (fallback)"
            )
    # --- END FIX ---

    # Add CSS for styling the report
    visualization_css = """
    <style>
    .report-container {
        font-family: Arial, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .report-container h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .report-container h2 {
        font-size: 2rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .report-container h3 {
        font-size: 1.5rem;
        margin-top: 1.5rem;
    }
    .report-container ul {
        list-style-type: none;
        margin-left: 1rem;
        padding-left: 0.5rem;
    }
    .report-container ul li {
        margin-bottom: 0.25rem;
        position: relative;
    }
    .report-container ul li:before {
        content: "•";
        position: absolute;
        left: -1rem;
    }
    .report-container ul ul li:before {
        content: "◦";
    }
    .report-date {
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 2rem;
        color: #555;
    }
    .inline-visualization {
        margin: 1.5rem 0;
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 8px;
    }
    .inline-visualization h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .inline-visualization img {
        max-width: 100%;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        display: block;
        margin: 0 auto;
    }
    </style>
    """

    # Process base64 and file path visualizations into a list
    all_visualizations = []

    # First add any base64 encoded images (these are most reliable)
    for idx, img in enumerate(base64_images):
        try:
            filename = img.get("filename", "")
            title = img.get("title", f"Visualization {idx+1}")
            img_data = img.get("base64_data", "")
            img_format = img.get("format", "png")

            if img_data:
                all_visualizations.append(
                    {
                        "id": idx + 1,
                        "title": title,
                        "html": f'<div class="inline-visualization">'
                        f"<h4>{title}</h4>"
                        f'<img src="data:image/{img_format};base64,{img_data}" alt="{title}" />'
                        f"</div>",
                    }
                )
        except Exception as e:
            print(f"Error processing base64 image: {e}")

    # Process images from the visualizations object (preferred way)
    seen_filenames = set()
    if hasattr(state, "visualizations") and state.visualizations:
        for idx, viz in enumerate(state.visualizations):
            try:
                # Skip if we've already processed this file
                if "filename" in viz and viz["filename"] in seen_filenames:
                    continue

                if "filename" in viz:
                    seen_filenames.add(viz["filename"])

                # Get filepath
                filepath = viz.get("filepath", "")
                if not filepath and "filename" in viz:
                    # Try to reconstruct path if only filename is available
                    filepath = os.path.join("visualizations", viz["filename"])

                if not filepath or not os.path.exists(filepath):
                    print(f"Warning: Visualization file not found: {filepath}")
                    continue

                # Get title from metadata or fallback to filename
                title = viz.get("subtask_name", "")
                if not title and "filename" in viz:
                    # Generate title from filename
                    filename = viz["filename"]
                    title_base = os.path.splitext(filename)[0].replace("_", " ").title()
                    title = title_base

                # Try to read file and encode as base64
                with open(filepath, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")

                # Determine image format
                img_format = os.path.splitext(filepath)[1][1:].lower()
                if img_format not in ["png", "jpg", "jpeg", "gif", "svg"]:
                    img_format = "png"  # Default to png

                # Include description if available
                description_html = ""
                if "description" in viz and viz["description"]:
                    description_html = (
                        f'<p class="visualization-description">{viz["description"]}</p>'
                    )

                # Add visualization to our collection
                all_visualizations.append(
                    {
                        "id": len(base64_images) + idx + 1,
                        "title": title,
                        "html": f'<div class="inline-visualization">'
                        f"<h4>{title}</h4>"
                        f'<img src="data:image/{img_format};base64,{img_data}" alt="{title}" />'
                        f"{description_html}"
                        f"</div>",
                    }
                )
            except Exception as e:
                print(f"Error processing visualization: {e}")

    # Fallback to visualization_paths if no visualizations were processed
    if not any(
        viz.get("id", 0) > len(base64_images) for viz in all_visualizations
    ) and hasattr(state, "visualization_paths"):
        for idx, path in enumerate(state.visualization_paths[:5]):
            # Skip if we've already processed this file
            filename = os.path.basename(path)
            if filename in seen_filenames:
                continue

            seen_filenames.add(filename)

            try:
                # Extract a title from the filename
                title_base = os.path.splitext(filename)[0]
                if len(title_base) > 6 and title_base[-6:].isalnum():
                    title_base = title_base[:-6]

                # Capitalize and replace underscores
                title = title_base.replace("_", " ").title()

                # Use the current count of visualization_info for image numbering to avoid gaps
                visualization_info.append(
                    f"Image {len(visualization_info) + 1}: {title}"
                )
            except Exception as e:
                print(f"Error processing visualization path {path}: {e}")

    print(f"Total visualizations prepared for embedding: {len(all_visualizations)}")

    # Look for LLM-provided placement markers [INSERT IMAGE X] and replace them with visualizations
    placed_visualization_ids = set()

    for viz in all_visualizations:
        viz_id = viz["id"]
        marker = f"[INSERT IMAGE {viz_id}]"

        if marker in finalized_summary:
            # Replace the marker with the visualization HTML
            finalized_summary = finalized_summary.replace(marker, viz["html"])
            placed_visualization_ids.add(viz_id)
            print(f"✅ Placed visualization {viz_id} at LLM-specified location")

    # MODIFIED: Only place visualizations via explicit markers to prevent duplicates
    # Visualizations not placed via markers will be handled by the activity events system
    # and don't need to be added to the report content directly
    if not placed_visualization_ids:
        print(
            f"⚠️ No visualizations were placed via markers. They will be shown through the activity events system instead."
        )
    elif len(placed_visualization_ids) < len(all_visualizations):
        unplaced_count = len(all_visualizations) - len(placed_visualization_ids)
        print(
            f"ℹ️ {unplaced_count} visualizations were not placed via markers and will be shown through the activity events system instead."
        )

    # CLEANUP: Remove any unreplaced [INSERT IMAGE X] markers from the report
    # This prevents placeholder text from appearing in the final output
    marker_pattern = r"\[INSERT IMAGE \d+\]"
    original_markers = re.findall(marker_pattern, finalized_summary)
    if original_markers:
        finalized_summary = re.sub(marker_pattern, "", finalized_summary)
        print(
            f"🧹 Cleaned up {len(original_markers)} unreplaced image markers: {original_markers}"
        )

    # Check if we need a report container
    has_container = '<div class="report-container">' in finalized_summary

    # Make sure we have the container and CSS
    if '<div class="report-container">' not in finalized_summary:
        finalized_summary = f'<div class="report-container">{visualization_css}{finalized_summary}</div>'

    # ---- END NEW APPROACH ----

    # Generate clean markdown version of the report
    markdown_report = generate_markdown_report(markdown_final_summary)
    print(f"Generated markdown report with length: {len(markdown_report)}")
    # Ensure correct indentation for the return statement
    return {
        "running_summary": finalized_summary,
        "markdown_report": markdown_report,  # Add the clean markdown version
        "web_research_results": [],  # Completely clear web_research_results as it's no longer needed
        "selected_search_tool": state.selected_search_tool,
        "source_citations": source_citations,  # Preserve the source citations
        "visualization_paths": getattr(
            state, "visualization_paths", []
        ),  # Preserve visualization paths from state
        "extra_effort": getattr(state, "extra_effort", False),  # Preserve extra_effort
        "minimum_effort": getattr(
            state, "minimum_effort", False
        ),  # Preserve minimum_effort
        # <<< START FIX: Preserve fields from input state >>>
        "research_topic": state.research_topic,
        "research_loop_count": state.research_loop_count,
        "sources_gathered": state.sources_gathered,
        "knowledge_gap": getattr(state, "knowledge_gap", ""),
        "visualizations": getattr(state, "visualizations", []),
        "base64_encoded_images": getattr(state, "base64_encoded_images", []),
        "code_snippets": getattr(state, "code_snippets", []),  # Preserve code_snippets
        # <<< END FIX >>>
    }


def generate_markdown_report(report):
    """
    Generate a clean markdown version of the report without HTML elements.
    The output is formatted as a JSON-serializable string suitable for dumping as a JSON field.

    Args:
        report (str): The generated report (may contain HTML)

    Returns:
        str: Clean markdown version of the report formatted for JSON serialization
    """
    # Start with the original report
    markdown_report = report

    # Remove HTML tags and convert to clean markdown
    # Convert HTML headers back to markdown
    markdown_report = re.sub(
        r"<h1[^>]*>(.*?)</h1>", r"# \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h2[^>]*>(.*?)</h2>", r"## \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h3[^>]*>(.*?)</h3>", r"### \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h4[^>]*>(.*?)</h4>", r"#### \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h5[^>]*>(.*?)</h5>", r"##### \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h6[^>]*>(.*?)</h6>", r"###### \1", markdown_report, flags=re.DOTALL
    )

    # Convert HTML lists to markdown
    markdown_report = re.sub(r"<ul[^>]*>", "", markdown_report)
    markdown_report = re.sub(r"</ul>", "", markdown_report)
    markdown_report = re.sub(r"<ol[^>]*>", "", markdown_report)
    markdown_report = re.sub(r"</ol>", "", markdown_report)
    markdown_report = re.sub(
        r"<li[^>]*>(.*?)</li>", r"* \1", markdown_report, flags=re.DOTALL
    )

    # Convert HTML formatting to markdown
    markdown_report = re.sub(
        r"<strong[^>]*>(.*?)</strong>", r"**\1**", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<b[^>]*>(.*?)</b>", r"**\1**", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<em[^>]*>(.*?)</em>", r"*\1*", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<i[^>]*>(.*?)</i>", r"*\1*", markdown_report, flags=re.DOTALL
    )

    # Convert HTML code blocks to markdown
    markdown_report = re.sub(
        r"<code[^>]*>(.*?)</code>", r"`\1`", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<pre[^>]*>(.*?)</pre>", r"```\n\1\n```", markdown_report, flags=re.DOTALL
    )

    # Remove other HTML tags
    markdown_report = re.sub(
        r'<p[^>]*class="report-date"[^>]*>(.*?)</p>',
        r"\1",
        markdown_report,
        flags=re.DOTALL,
    )
    markdown_report = re.sub(
        r"<div[^>]*>(.*?)</div>", r"\1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<p[^>]*>(.*?)</p>", r"\1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(r"<br\s*/?>", "\n", markdown_report)

    # Remove any remaining HTML tags (after table conversion)
    markdown_report = re.sub(r"<[^>]+>", "", markdown_report)

    # Remove any leftover HTML escape sequences or artifacts
    markdown_report = re.sub(r"&[a-zA-Z0-9#]+;", "", markdown_report)

    # Clean up any CSS or JavaScript that might have snuck in
    markdown_report = re.sub(
        r"<style[^>]*>.*?</style>", "", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<script[^>]*>.*?</script>", "", markdown_report, flags=re.DOTALL
    )

    # Clean up extra whitespace and newlines
    markdown_report = re.sub(
        r"\n\s*\n\s*\n+", "\n\n", markdown_report
    )  # Multiple empty lines to double
    markdown_report = re.sub(
        r"^\s+", "", markdown_report, flags=re.MULTILINE
    )  # Leading whitespace
    markdown_report = markdown_report.strip()

    # Remove base64 image data that might be embedded in the text (if any)
    base64_pattern = r"data:image/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+"
    markdown_report = re.sub(base64_pattern, "", markdown_report)

    # Remove any HTML attributes that might be left behind
    markdown_report = re.sub(r'class="[^"]*"', "", markdown_report)
    markdown_report = re.sub(r'style="[^"]*"', "", markdown_report)
    markdown_report = re.sub(r'id="[^"]*"', "", markdown_report)

    # Remove HTML entities
    markdown_report = markdown_report.replace("&nbsp;", " ")
    markdown_report = markdown_report.replace("&amp;", "&")
    markdown_report = markdown_report.replace("&lt;", "<")
    markdown_report = markdown_report.replace("&gt;", ">")
    markdown_report = markdown_report.replace("&quot;", '"')
    markdown_report = markdown_report.replace("&#39;", "'")

    # Remove any remaining HTML comments
    markdown_report = re.sub(r"<!--.*?-->", "", markdown_report, flags=re.DOTALL)

    # Convert HTML tables to markdown tables (preserve table structure)
    def convert_html_table_to_markdown(match):
        """Convert a single HTML table to markdown format"""
        table_html = match.group(0)

        # Extract rows
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
        if not rows:
            return ""

        markdown_rows = []
        is_header_row = True

        for row in rows:
            # Extract cells (th or td)
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL)
            if not cells:
                continue

            # Clean cell content of any remaining HTML
            clean_cells = []
            for cell in cells:
                clean_cell = re.sub(r"<[^>]+>", "", cell).strip()
                clean_cells.append(clean_cell)

            # Create markdown row
            markdown_row = "| " + " | ".join(clean_cells) + " |"
            markdown_rows.append(markdown_row)

            # Add separator after header row
            if is_header_row and clean_cells:
                separator = "| " + " | ".join(["---"] * len(clean_cells)) + " |"
                markdown_rows.append(separator)
                is_header_row = False

        return "\n" + "\n".join(markdown_rows) + "\n"

    # Apply table conversion
    markdown_report = re.sub(
        r"<table[^>]*>.*?</table>",
        convert_html_table_to_markdown,
        markdown_report,
        flags=re.DOTALL,
    )

    # Final cleanup: ensure proper JSON escaping for special characters
    # Replace problematic characters that might break JSON
    markdown_report = markdown_report.replace("\r\n", "\n")  # Normalize line endings
    markdown_report = markdown_report.replace("\r", "\n")  # Normalize line endings

    # Clean up excessive whitespace while preserving intentional formatting
    markdown_report = re.sub(
        r"\n\s*\n\s*\n+", "\n\n", markdown_report
    )  # Multiple empty lines to double
    markdown_report = re.sub(
        r"[ \t]+$", "", markdown_report, flags=re.MULTILINE
    )  # Trailing whitespace
    markdown_report = markdown_report.strip()

    return markdown_report

def post_process_report(report, source_citations):
    """
    Post-process the report to ensure citation consistency and include a References section
    if it's missing.

    Args:
        report (str): The generated report
        source_citations (dict): Dictionary mapping citation numbers to source metadata

    Returns:
        str: The post-processed report
    """
    # Convert markdown headers to proper HTML for better styling

    # First identify and convert the main title (first # header)
    import re

    title_pattern = r"^#\s+(.*?)$"
    match = re.search(title_pattern, report, re.MULTILINE)
    if match:
        title = match.group(1)
        report = re.sub(
            title_pattern, f"<h1>{title}</h1>", report, count=1, flags=re.MULTILINE
        )

        # Remove any duplicate titles that match exactly
        report = re.sub(f"<h1>{re.escape(title)}</h1>", "", report)
        report = re.sub(f"#\\s+{re.escape(title)}\\s*\n", "", report)

        # Also remove similar titles (those that contain the main title)
        similar_title_pattern = f"<h1>.*?{re.escape(title)}.*?</h1>"
        report = re.sub(similar_title_pattern, "", report)
        report = re.sub(f"#\\s+.*?{re.escape(title)}.*?\\s*\n", "", report)

    # Fix the table of contents formatting if needed

    if not source_citations:
        return report  # No citations to check or add

    # Check if a References section already exists in the report
    references_section_patterns = [
        "References",
        "References:",
        "## References",
        "# References",
    ]

    has_references_section = any(
        pattern in report for pattern in references_section_patterns
    )

    # Create the references section if needed
    if not has_references_section:
        print("Adding missing References section to the report")
        # Format the references section
        references_section = "\n\n──────────────────────────────\nReferences\n\n"

        # Add each reference in order
        for num, src in sorted(source_citations.items()):
            references_section += f"{num}. {src['title']}, [{src['url']}]\n"

        # Append to the report
        report += references_section

    # Fix any generic citations that might have been generated
    # Multiple patterns to catch different variations of generic citations
    generic_citation_patterns = [
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+research",
        r"\[(\d+)\]\s*Source\s+\d+",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited",
    ]

    all_matches = []
    for pattern in generic_citation_patterns:
        matches = re.findall(pattern, report, re.IGNORECASE)
        all_matches.extend(matches)

    matches = list(set(all_matches))  # Remove duplicates

    if matches:
        print(f"Fixing {len(matches)} generic citations in report")
        for citation_num in matches:
            if citation_num in source_citations:
                src = source_citations[citation_num]
                title = src.get("title", "Unknown Title")
                url = src.get("url", "")

                # Replace with proper format
                replacement = f"[{citation_num}] {title}"

                # Replace all variations of generic citations
                for pattern in generic_citation_patterns:
                    report = re.sub(
                        pattern.replace(r"(\d+)", citation_num),
                        replacement,
                        report,
                        flags=re.IGNORECASE,
                    )

    # Check for citation consistency
    # Get all citation numbers used in the report
    citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    found_citations = set()

    for match in re.finditer(citation_pattern, report):
        # Handle both single citations [1] and multiple citations [1,2,3]
        for citation in re.split(r",\s*", match.group(1)):
            if citation.isdigit():
                found_citations.add(citation)  # Keep as string

    # Check which citations were used but not in source_citations
    # Ensure source_citations keys are strings if they aren't already (they should be)
    source_citations_keys = set(str(k) for k in source_citations.keys())
    missing_citations = [c for c in found_citations if c not in source_citations_keys]
    if missing_citations:
        print(
            f"WARNING: Report contains citations {missing_citations} not found in source_citations"
        )

    # Check which citations from source_citations were not used
    unused_citations = [c for c in source_citations_keys if c not in found_citations]
    if unused_citations:
        print(
            f"WARNING: Report doesn't use citations {unused_citations} from source_citations"
        )

    return report

def route_after_search(
    state: SummaryState,
) -> Literal["generate_report", "reflect_on_report"]:
    """Route after search based on whether we have results or not"""

    # Check if the search_results_empty flag is set
    if getattr(state, "search_results_empty", False):
        print(
            "ROUTING: Search returned no results, skipping summarization and going directly to reflection"
        )
        return "reflect_on_report"

    # Normal flow - proceed to summarization
    print("ROUTING: Search returned results, proceeding to summarization")
    return "generate_report"


# NEW ROUTING FUNCTION
def route_after_multi_agents(
    state: SummaryState,
) -> Literal["generate_report", "reflect_on_report", "finalize_report"]:
    """
    Determines the next step after the multi_agents_network based on minimum_effort
    and search results.
    """
    minimum_effort = getattr(state, "minimum_effort", False)
    if minimum_effort:
        # If minimum effort is requested, skip reflection and go directly to finalize
        print(
            "ROUTING: Minimum effort requested, skipping reflection, finalizing report"
        )
        return "finalize_report"
    else:
        # Otherwise, use the existing routing logic based on search results
        return route_after_search(state)


def route_research(state: SummaryState, config: RunnableConfig):
    """Determines if research is complete or should continue"""

    configurable = Configuration.from_runnable_config(config)

    # Debug logging
    print(f"ROUTING STATE EXAMINATION:")
    print(f"  - research_complete: {state.research_complete}")
    print(
        f"  - search_query: '{state.search_query if hasattr(state, 'search_query') else ''}'"
    )
    print(
        f"  - research_loop_count: {state.research_loop_count}/{getattr(configurable, 'max_web_research_loops', 'N/A')}"
    )
    print(
        f"  - running_summary length: {len(state.running_summary) if state.running_summary else 0} chars"
    )

    # Check if we've reached the maximum number of research loops
    # Get effort flags from state
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)

    # Get max_loops using the utility function
    max_loops = get_max_loops(configurable, extra_effort, minimum_effort)
    if state.research_loop_count >= max_loops:
        print(f"ROUTING OVERRIDE: Max loops reached ({max_loops}), finalizing report")
        return "finalize_report"

    # BUGFIX: Check LLM's decision about research completeness first
    if state.research_complete:
        print("ROUTING DECISION: Research marked as complete by LLM, finalizing report")
        return "finalize_report"

    # First iteration: always continue research
    if state.research_loop_count == 1:
        print(
            "ROUTING OVERRIDE: First iteration - forcing research to continue regardless of flags"
        )
        return "multi_agents_network"

    # If no follow-up query was generated, finalize the report
    if (
        not hasattr(state, "search_query")
        or not state.search_query
        or len(state.search_query.strip()) == 0
    ):
        print("ROUTING DECISION: No follow-up query generated, finalizing report")
        return "finalize_report"

    # Otherwise, continue with research
    print(
        "ROUTING DECISION: Continuing with research, going to multi-agent network with reflection's query"
    )
    return "multi_agents_network"
