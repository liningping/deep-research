import json
import logging
from typing import Any, Dict, List, Union, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import SummaryState
from src.configuration import Configuration
from llm_clients import get_llm_client, CURRENT_DATE, CURRENT_YEAR, ONE_YEAR_AGO
from src.prompts import summarizer_instructions, reflection_instructions, finalize_report_instructions
from src.utils import clean_raw_web_content, extract_citations_from_state, clean_json_response
from src.nodes.utils import get_configurable, get_max_loops

logger = logging.getLogger(__name__)


def generate_report(state: SummaryState, config: RunnableConfig):
    """
    Generate a research report based on the current state.
    Refactored to remove redundant logging and delegate complex processing to utils.
    """
    # 1. Prepare Content: Clean and merge new web research results
    # Encapsulated logic for joining content and removing Base64 patterns
    cleaned_web_research = clean_raw_web_content(state.web_research_results)
    
    existing_summary = state.running_summary or ""
    knowledge_gap = getattr(state, "knowledge_gap", "")

    # 2. Manage Citations: Unified extraction logic
    # Encapsulated logic checking source_citations -> web_results -> sources_gathered
    source_citations = extract_citations_from_state(state)
    
    # 3. Handle Uploaded Knowledge (Simplified logic without trace logs)
    uploaded_knowledge = getattr(state, "uploaded_knowledge", "")
    external_knowledge_section = ""
    augment_knowledge_context = "No user-provided external knowledge available. Rely on web search results as primary sources."

    if uploaded_knowledge and uploaded_knowledge.strip():
        external_knowledge_section = f"""User-Provided External Knowledge:
------------------------------------------------------------
{uploaded_knowledge}

"""
        augment_knowledge_context = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge that should be treated as authoritative.
Uploaded Knowledge Preview: {uploaded_knowledge[:500]}{'...' if len(uploaded_knowledge) > 500 else ''}
"""

    # 4. Configure LLM
    configurable = Configuration.from_runnable_config(config)
    
    # Priority: State Override > Config > Default
    provider = getattr(state, "llm_provider", None) or configurable.llm_provider
    if not isinstance(provider, str): provider = provider.value # Handle Enum if needed
    
    model = getattr(state, "llm_model", None) or configurable.llm_model or "gemini-2.5-pro"
    
    logger.info(f"[generate_report] Summarizing with {provider}/{model}")
    llm = get_llm_client(provider, model)

    # 5. Construct Prompts
    system_prompt = summarizer_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=augment_knowledge_context,
    )

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

Generate an updated summary that merges the newly fetched content into the existing summary.
Return the updated summary as plain text:
"""

    # 6. Execute
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]
    )
    updated_summary = response.content if hasattr(response, "content") else str(response)

    # 7. Update State
    # Returning only modified fields (assuming LangGraph merges updates)
    return {
        "running_summary": updated_summary,
        "web_research_results": [],  # Clear buffer after processing
        "source_citations": source_citations, # Persist the normalized citations
        # Pass through strictly necessary context for next nodes
        "research_loop_count": state.research_loop_count, 
        "knowledge_gap": knowledge_gap 
    }


def reflect_on_report(state: SummaryState, config: RunnableConfig) -> dict[str, any]:
    """
    Reflect on the current research report, manage the Todo list, and decide next steps.
    """
    configurable = get_configurable(config)
    research_loop_count = getattr(state, "research_loop_count", 0)
    next_research_loop_count = research_loop_count + 1
    
    # 1. Check Max Loops Constraints
    # ------------------------------------------------------------------
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)
    max_loops = get_max_loops(configurable, extra_effort, minimum_effort)

    # Check for pending steering messages (User overrides max loops if they are talking)
    pending_steering_count = 0
    if hasattr(state, "steering_todo") and state.steering_todo:
        pending_steering_count = len(state.steering_todo.pending_messages)

    if next_research_loop_count > max_loops:
        if pending_steering_count > 0:
            logger.info(f"[Reflect] 🚨 Max loops reached but {pending_steering_count} msgs pending - CONTINUING.")
        else:
            logger.info(f"[Reflect] 🛑 Max loops reached ({max_loops}) - STOPPING.")
            return {
                "research_loop_count": next_research_loop_count,
                "research_complete": True,
                "evaluation_notes": "Max loops reached."
            }

    # 2. Prepare Context for LLM (Todo Status & Messages)
    # ------------------------------------------------------------------
    pending_tasks_context = "No todo list active"
    completed_tasks_context = ""
    steering_messages_context = "No steering system active"
    messages_snapshot = []

    if hasattr(state, "steering_todo") and state.steering_todo:
        # Get contexts
        pending_tasks_context = state.steering_todo.get_pending_tasks_for_llm()
        completed_tasks_context = state.steering_todo.get_completed_tasks_for_llm(limit=None)
        
        # Snapshot messages to prevent race conditions during async execution
        messages_snapshot = list(state.steering_todo.pending_messages)
        if messages_snapshot:
            steering_messages_context = "\n".join(
                [f'[{i}] "{msg}"' for i, msg in enumerate(messages_snapshot)]
            )
        else:
            steering_messages_context = "No new steering messages this loop"

    # 3. LLM Execution
    # ------------------------------------------------------------------
    # Priority: State Override > Config > Default
    provider = getattr(state, "llm_provider", None) or configurable.llm_provider
    if not isinstance(provider, str): provider = provider.value
    
    model = getattr(state, "llm_model", None) or configurable.llm_model or "gemini-2.5-pro"
    
    llm = get_llm_client(provider, model)

    # Prepare User Knowledge Context
    uploaded_knowledge = getattr(state, "uploaded_knowledge", "")
    augment_knowledge_context = "No user-provided external knowledge available."
    if uploaded_knowledge and uploaded_knowledge.strip():
        augment_knowledge_context = f"User Uploaded Knowledge Preview: {uploaded_knowledge[:500]}..."

    formatted_prompt = reflection_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=augment_knowledge_context,
        pending_tasks=pending_tasks_context,
        completed_tasks=completed_tasks_context,
        steering_messages=steering_messages_context,
    )

    response = llm.invoke([
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Analyze summary and determine next steps:\n\n{state.running_summary[:5000] if state.running_summary else 'No summary yet.'}")
    ])

    # 4. Parse & Process Results
    # ------------------------------------------------------------------
    content = response.content if hasattr(response, "content") else str(response)
    
    try:
        # Use a utility function to extract JSON from ```json blocks or raw text
        # Assumed signature: clean_json_response(text) -> dict
        result = clean_json_response(content) 
        
        research_complete = result.get("research_complete", False)
        knowledge_gap = result.get("knowledge_gap", "")
        search_query = result.get("follow_up_query", "")
        todo_updates = result.get("todo_updates", {})

        # 5. Apply Todo Updates (The "Agentic" Side Effects)
        # ------------------------------------------------------------------
        if hasattr(state, "steering_todo") and state.steering_todo and todo_updates:
            
            # A. Mark Completed
            for task_id in todo_updates.get("mark_completed", []):
                if task_id in state.steering_todo.tasks:
                    state.steering_todo.mark_task_completed(task_id, completion_note="Addressed in loop")
                    logger.info(f"[Reflect] ✓ Task {task_id} completed")

            # B. Cancel Tasks
            for task_id in todo_updates.get("cancel_tasks", []):
                state.steering_todo.mark_task_cancelled(task_id, reason="LLM Cancelled")

            # C. Add New Tasks
            for new_task in todo_updates.get("add_tasks", []):
                source = new_task.get("source", "knowledge_gap")
                prio = 10 if source == "steering_message" else 8
                
                tid = state.steering_todo.create_task(
                    description=new_task.get("description", ""),
                    priority=prio,
                    source=source
                )
                logger.info(f"[Reflect] + New Task {tid}: {new_task.get('description', '')[:40]}...")

            # D. Clear Processed Steering Messages
            # Only clear messages the LLM explicitly says it handled
            clear_indices = todo_updates.get("clear_messages", [])
            if clear_indices and messages_snapshot:
                # Filter out cleared messages from the snapshot
                remaining_snapshot = [
                    msg for i, msg in enumerate(messages_snapshot) 
                    if i not in clear_indices
                ]
                
                # Re-merge: Remaining snapshot + New messages that arrived during LLM call
                current_queue = state.steering_todo.pending_messages
                new_arrivals = [msg for msg in current_queue if msg not in messages_snapshot]
                
                state.steering_todo.pending_messages = remaining_snapshot + new_arrivals
                
                # Bump version for UI polling
                state.steering_todo.todo_version += 1
                logger.info(f"[Reflect] Cleared {len(clear_indices)} messages. Queue size: {len(state.steering_todo.pending_messages)}")

        # 6. Final Decision Logic
        # ------------------------------------------------------------------
        # Override: If we still have pending messages, we CANNOT stop.
        final_pending_count = 0
        if hasattr(state, "steering_todo") and state.steering_todo:
            final_pending_count = len(state.steering_todo.pending_messages)

        if final_pending_count > 0 and research_complete:
            logger.info(f"[Reflect] ⚡ Override: {final_pending_count} msgs pending. Research continues.")
            research_complete = False
            if not search_query:
                search_query = "Process pending user steering messages"

        return {
            "research_loop_count": next_research_loop_count,
            "research_complete": research_complete,
            "knowledge_gap": knowledge_gap,
            "search_query": search_query,
            "extra_effort": extra_effort,
            "minimum_effort": minimum_effort,
            # Metadata for trajectory analysis
            "priority_section": result.get("priority_section"),
            "section_gaps": result.get("section_gaps"),
            "evaluation_notes": result.get("evaluation_notes"),
        }

    except Exception as e:
        logger.error(f"[Reflect] ❌ Error parsing/processing: {e}")
        # Fallback safety mode
        return {
            "research_loop_count": next_research_loop_count,
            "research_complete": False, # Safer to continue than stop prematurely on error
            "knowledge_gap": "Error in reflection, retrying exploration",
            "search_query": f"Continue research on {state.research_topic}",
        }


import logging
import re
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import SummaryState
from src.configuration import Configuration
from src.utils import (
    extract_citations_from_state,
    clean_raw_web_content
)

logger = logging.getLogger(__name__)

def finalize_report(state: SummaryState, config: RunnableConfig):
    """
    Finalize the summary for BENCHMARKING.
    Produces a clean, pure Markdown report without HTML/CSS artifacts.
    """
    # 1. Prepare Content (Logic preserved)
    # ------------------------------------------------------------------
    current_summary = state.running_summary or ""
    
    if current_summary.strip():
        input_content = current_summary
        logger.info(f"[Finalize] Using running summary ({len(input_content)} chars)")
    else:
        input_content = clean_raw_web_content(state.web_research_results)
        logger.info(f"[Finalize] Using raw web content ({len(input_content)} chars)")

    if not input_content.strip():
        input_content = "(No content available to finalize)"

    # 2. Extract Citations
    # ------------------------------------------------------------------
    source_citations = extract_citations_from_state(state)
    
    # Format numbered sources for Prompt
    # Benchmark 关键点：确保 Prompt 里的引用格式清晰，方便模型引用
    numbered_sources = [
        f"{num}. {src['title']}, [{src['url']}]"
        for num, src in sorted(source_citations.items())
    ]
    formatted_sources_prompt = "\n".join(numbered_sources)

    # 3. Prepare User Context (Steering)
    # ------------------------------------------------------------------
    # Benchmark 场景下，uploaded_knowledge 和 steering 仍然重要，因为它们定义了 Ground Truth
    uploaded_knowledge = getattr(state, "uploaded_knowledge", "")
    augment_context = "No user-provided external knowledge available."
    uploaded_section = ""
    
    if uploaded_knowledge and uploaded_knowledge.strip():
        uploaded_section = f"\n\nUSER-PROVIDED KNOWLEDGE (HIGHEST AUTHORITY):\n{uploaded_knowledge}\n"
        augment_context = f"User Uploaded Knowledge Preview: {uploaded_knowledge[:500]}..."

    todo_section = ""
    if hasattr(state, "steering_todo") and state.steering_todo:
        # 即使是评测，保留用户意图上下文也有助于模型生成更准确的内容
        completed = state.steering_todo.get_completed_tasks()
        all_messages = getattr(state.steering_todo, "all_user_messages", [])
        
        if completed or all_messages:
            todo_section = "\n\nRESEARCH CONTEXT & GOALS:\n"
            if all_messages:
                todo_section += "USER INSTRUCTIONS:\n" + "\n".join([f"- {msg}" for msg in all_messages])
            if completed:
                todo_section += "\nCOVERED TOPICS:\n" + "\n".join([f"- {t.description}" for t in completed])

    # 4. Configure LLM
    # ------------------------------------------------------------------
    configurable = Configuration.from_runnable_config(config)
    provider = getattr(state, "llm_provider", None) or configurable.llm_provider
    if not isinstance(provider, str): provider = provider.value
    
    model = getattr(state, "llm_model", None) or configurable.llm_model or "gemini-2.5-pro"
    llm = get_llm_client(provider, model)

    # 5. Execution (Optimized for Pure Markdown)
    # ------------------------------------------------------------------
    system_prompt = finalize_report_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=augment_context,
    )

    # 明确要求 Markdown 格式
    human_message = (
        f"Please finalize this research summary into a polished report using standard Markdown.\n"
        f"DO NOT use HTML tags (like <div>, <h1>, <style>). Use strictly Markdown (# Title, **bold**, - list).\n"
        f"{uploaded_section}"
        f"{todo_section}\n\n"
        f"DRAFT CONTENT:\n{input_content}\n\n"
        f"AVAILABLE SOURCES:\n{formatted_sources_prompt}"
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_message)])
    final_text = response.content if hasattr(response, "content") else str(response)

    # 6. Post-Processing (Benchmark Safe)
    # ------------------------------------------------------------------
    # 不调用 post_process_report_formatting，因为它会引入 HTML。
    # 我们只在本地做最必要的引用修复。
    
    # A. 修复通用引用描述 (e.g. "[1] Source 1" -> "[1] Real Title")
    final_text = _fix_citations_for_benchmark(final_text, source_citations)
    
    # B. 确保参考文献部分存在 (Markdown 格式)
    final_text = _ensure_markdown_references(final_text, source_citations)

    # 7. Return State
    # ------------------------------------------------------------------
    return {
        "running_summary": final_text,
        "markdown_report": final_text, # 两个字段内容一致
        "web_research_results": [],
        "research_topic": state.research_topic,
        "source_citations": source_citations,
        # 评测不需要图片数据，清空以节省内存/Token
        "base64_encoded_images": [],
        "visualizations": []
    }

# =============================================================================
# Local Helpers (Inline to avoid dependency on formatters.py which has HTML)
# =============================================================================

def _fix_citations_for_benchmark(report: str, source_citations: dict) -> str:
    """Fix generic citations without introducing HTML."""
    if not source_citations:
        return report
        
    generic_patterns = [
        r"\[(\d+)\]\s*Source\s+\d+",
        r"\[(\d+)\]\s*source\s+\d+",
    ]
    
    # 简单的替换逻辑：查找文中类似 "[1] Source 1" 的文本并替换为 "[1] Title"
    for citation_id, src in source_citations.items():
        title = src.get("title", "Unknown Title")
        replacement = f"[{citation_id}] {title}"
        
        for pattern in generic_patterns:
            # 构造特定 ID 的正则
            specific_pattern = pattern.replace(r"(\d+)", citation_id)
            report = re.sub(specific_pattern, replacement, report, flags=re.IGNORECASE)
            
    return report

def _ensure_markdown_references(report: str, source_citations: dict) -> str:
    """Ensure a References section exists in pure Markdown."""
    if not source_citations:
        return report

    patterns = ["# References", "## References", "### References", "**References**"]
    has_section = any(p in report for p in patterns)

    if not has_section:
        refs = "\n\n---\n## References\n\n"
        # 按序号排序
        sorted_citations = sorted(source_citations.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        for num, src in sorted_citations:
            refs += f"{num}. [{src['title']}]({src['url']})\n"
        report += refs
        
    return report

def route_after_search(
    state: SummaryState,
) -> Literal["generate_report", "reflect_on_report"]:
    """Route after search based on whether we have results or not"""

    # Check if the search_results_empty flag is set
    if getattr(state, "search_results_empty", False):
        logger.info(
            "ROUTING: Search returned no results, skipping summarization and going directly to reflection"
        )
        return "reflect_on_report"

    # Normal flow - proceed to summarization
    logger.info("ROUTING: Search returned results, proceeding to summarization")
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
        logger.info(
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
    logger.info(f"ROUTING STATE EXAMINATION:")
    logger.info(f"  - research_complete: {state.research_complete}")
    logger.info(
        f"  - search_query: '{state.search_query if hasattr(state, 'search_query') else ''}'"
    )
    logger.info(
        f"  - research_loop_count: {state.research_loop_count}/{getattr(configurable, 'max_web_research_loops', 'N/A')}"
    )
    logger.info(
        f"  - running_summary length: {len(state.running_summary) if state.running_summary else 0} chars"
    )

    # Check if we've reached the maximum number of research loops
    # Get effort flags from state
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)

    # Get max_loops using the utility function
    max_loops = get_max_loops(configurable, extra_effort, minimum_effort)
    if state.research_loop_count >= max_loops:
        logger.info(f"ROUTING OVERRIDE: Max loops reached ({max_loops}), finalizing report")
        return "finalize_report"

    # First iteration: always continue research (PRIORITY 1)
    if state.research_loop_count == 1:
        logger.info(
            "ROUTING OVERRIDE: First iteration - forcing research to continue regardless of flags"
        )
        return "multi_agents_network"

    # LLM completeness check (PRIORITY 2)
    if state.research_complete:
        logger.info("ROUTING DECISION: Research marked as complete by LLM, finalizing report")
        return "finalize_report"

    # If no follow-up query was generated, finalize the report
    if (
        not hasattr(state, "search_query")
        or not state.search_query
        or len(state.search_query.strip()) == 0
    ):
        logger.info("ROUTING DECISION: No follow-up query generated, finalizing report")
        return "finalize_report"

    # Otherwise, continue with research
    logger.info(
        "ROUTING DECISION: Continuing with research, going to multi-agent network with reflection's query"
    )
    return "multi_agents_network"
