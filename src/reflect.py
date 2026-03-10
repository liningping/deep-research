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


def reflect_on_report(state: SummaryState, config: RunnableConfig) -> dict[str, any]:
    """
    Reflect on the current research report, manage the Todo list, and decide next steps.
    """
    configurable = get_configurable(config)
    research_loop_count = getattr(state, "research_loop_count", 0)
    next_research_loop_count = research_loop_count + 1
    
    # 1. Check Max Loops Constraints 
    # (Moved to execute_research for architectural separation)
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)

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
