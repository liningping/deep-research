"""
Search node for the research graph.

Contains the multi-agent search network that coordinates research agents
to execute search queries and gather information.
"""

import logging
import asyncio
import traceback
import uuid
from datetime import datetime

from src.state import SummaryState
from src.nodes.utils import heartbeat_task, emit_event

logger = logging.getLogger(__name__)


async def async_multi_agents_network(state: SummaryState, callbacks=None):
    """
    Asynchronously execute research using the new agent-based architecture.
    This function represents the multi-agent network entry point.

    Args:
        state: The current state containing research parameters

    Returns:
        Updated state with research results
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("[async_multi_agents_network] Starting research")
    logger.info(f"[async_multi_agents_network] state type: {type(state)}")
    logger.info(
        f"[async_multi_agents_network] state.steering_enabled: {getattr(state, 'steering_enabled', 'MISSING')}"
    )
    logger.info(
        f"[async_multi_agents_network] state.steering_todo: {getattr(state, 'steering_todo', 'MISSING')}"
    )
    logger.info(
        f"[async_multi_agents_network] hasattr steering_enabled: {hasattr(state, 'steering_enabled')}"
    )
    logger.info(
        f"[async_multi_agents_network] hasattr steering_todo: {hasattr(state, 'steering_todo')}"
    )
    logger.info("=" * 70)

    logger.info(
        "[async_multi_agents_network] Starting research with agent architecture"
    )
    logger.info(f"callbacks at entry: {'present' if callbacks else 'none'}")
    logger.info(
        f"[async_multi_agents_network] Research loop count: {state.research_loop_count}"
    )

    try:
        # Process steering messages and update todo.md at the start of each research loop
        if hasattr(state, "steering_todo") and state.steering_todo:
            from src.steering_integration import (
                integrate_steering_with_research_loop,
                get_steering_summary_for_agent,
            )

            logger.info(
                "[STEERING] Processing steering messages and updating todo.md before research loop"
            )

            # Process queued steering messages and update todo.md
            steering_result = await state.prepare_steering_for_next_loop()
            if steering_result.get("steering_enabled"):
                logger.info(
                    f"[STEERING] Todo.md updated to version {steering_result.get('todo_version')}"
                )
                logger.info(
                    f"[STEERING] Pending tasks: {steering_result.get('pending_tasks')}, "
                    f"Completed tasks: {steering_result.get('completed_tasks')}"
                )

                # Emit steering update event for UI
                if callbacks:
                    await callbacks.emit_event(
                        "steering_updated",
                        {
                            "todo_version": steering_result.get("todo_version"),
                            "current_plan": steering_result.get("current_plan"),
                            "pending_tasks": steering_result.get("pending_tasks"),
                            "completed_tasks": steering_result.get("completed_tasks"),
                            "loop_guidance": steering_result.get("loop_guidance"),
                            "research_loop_count": state.research_loop_count,
                        },
                    )

            # Get steering summary for agent context
            steering_context = get_steering_summary_for_agent(state)
            if steering_context:
                logger.info(
                    f"[STEERING] Active constraints: {steering_context.strip()}"
                )

        # Import the master agent
        from src.agent_architecture import MasterResearchAgent

        # Initialize the master agent with config from state
        # Use state for configuration: Create a config object that contains the llm_provider and llm_model
        config = getattr(state, "config", None)
        if not config:
            config = {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "llm_provider": state.llm_provider,
                    "llm_model": state.llm_model,
                }
            }
        elif "configurable" not in config:
            config["configurable"] = {
                "thread_id": str(uuid.uuid4()),
                "llm_provider": state.llm_provider,
                "llm_model": state.llm_model,
            }
        else:
            # Ensure llm_provider and llm_model are in configurable
            config["configurable"]["llm_provider"] = state.llm_provider
            config["configurable"]["llm_model"] = state.llm_model

        # Log the provider and model being used
        logger.info(
            f"[async_multi_agents_network] Using provider: {state.llm_provider}, model: {state.llm_model}"
        )

        master_agent = MasterResearchAgent(config)

        # Start heartbeat
        heartbeat = asyncio.create_task(heartbeat_task(callbacks))
        # Execute research using the master agent asynchronously
        # The 'results' from master_agent.execute_research should be a list of dictionaries,
        # where each dictionary is a search result.
        # MODIFICATION: master_agent_output is now a dictionary
        # WORKAROUND: LangGraph is losing the database_info field during state serialization
        # Use session-specific global variable instead of trying to get from state.config
        stream_id = None
        if config and "configurable" in config:
            stream_id = config["configurable"].get("stream_id")
        
        master_agent_output = await master_agent.execute_research(
            state, callbacks=callbacks
        )
        # Cancel heartbeat when done
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        logger.info(
            "[async_multi_agents_network] Research completed successfully by master_agent."
        )

        # Initialize results dictionary that will be returned.
        # It should preserve existing state fields and update with new research data.
        current_state_dict = state.__dict__.copy() if state else {}
        updated_results = current_state_dict  # Start with all current state

        # Process master_agent_output which is now a dictionary
        if isinstance(master_agent_output, dict):
            logger.info(
                f"[async_multi_agents_network] master_agent returned a dictionary. Keys: {list(master_agent_output.keys())}"
            )

            # Primary search results
            raw_agent_results = master_agent_output.get("web_research_results", [])
            if isinstance(raw_agent_results, list):
                updated_results["web_research_results"] = raw_agent_results
                if raw_agent_results:
                    logger.info(
                        f"[async_multi_agents_network] First item in web_research_results is of type: {type(raw_agent_results[0])}"
                    )
                    if isinstance(raw_agent_results[0], dict):
                        logger.info(
                            f"[async_multi_agents_network] Keys in first search result item: {raw_agent_results[0].keys()}"
                        )

                    sources_gathered = []
                    source_citations = {}
                    citation_index = 1

                    existing_source_citations = getattr(state, "source_citations", {})
                    if existing_source_citations:
                        source_citations.update(existing_source_citations)
                        highest_index = 0
                        for key in source_citations.keys():
                            if (
                                isinstance(key, str)
                                and key.isdigit()
                                and int(key) > highest_index
                            ):
                                highest_index = int(key)
                            elif (
                                isinstance(key, int) and key > highest_index
                            ):  # handle int keys too
                                highest_index = key
                        citation_index = highest_index + 1

                    existing_urls_in_citations = {
                        sc_data.get("url")
                        for sc_data in source_citations.values()
                        if isinstance(sc_data, dict)
                    }

                    for result in raw_agent_results:
                        if isinstance(result, dict):
                            if "sources" in result and isinstance(
                                result["sources"], list
                            ):
                                for source in result["sources"]:
                                    if isinstance(source, dict):
                                        if "title" in source and "url" in source:
                                            source_str = (
                                                f"{source['title']} : {source['url']}"
                                            )
                                            if (
                                                source_str not in sources_gathered
                                            ):  # Basic dedupe for sources_gathered
                                                sources_gathered.append(source_str)

                                            source_url = source["url"]
                                            if (
                                                source_url
                                                not in existing_urls_in_citations
                                            ):
                                                citation_key = str(citation_index)
                                                source_dict = {
                                                    "title": source["title"],
                                                    "url": source["url"],
                                                }
                                                # Preserve source_type if it exists
                                                if "source_type" in source:
                                                    source_dict["source_type"] = source[
                                                        "source_type"
                                                    ]
                                                source_citations[citation_key] = (
                                                    source_dict
                                                )
                                                existing_urls_in_citations.add(
                                                    source_url
                                                )
                                                citation_index += 1

                    if sources_gathered:
                        logger.info(
                            f"[async_multi_agents_network] Extracted {len(sources_gathered)} sources from search results"
                        )
                        # Append to existing sources_gathered, ensuring uniqueness
                        current_sources_gathered = updated_results.get(
                            "sources_gathered", []
                        )
                        for sg in sources_gathered:
                            if sg not in current_sources_gathered:
                                current_sources_gathered.append(sg)
                        updated_results["sources_gathered"] = current_sources_gathered

                    if (
                        source_citations
                    ):  # source_citations already includes existing ones
                        logger.info(
                            f"[async_multi_agents_network] Updated source_citations, total: {len(source_citations)}"
                        )
                        updated_results["source_citations"] = source_citations
            else:
                logger.warning(
                    "[async_multi_agents_network] 'web_research_results' from master_agent was not a list."
                )
                updated_results["web_research_results"] = []

            # Merge visualization outputs
            new_visualizations = master_agent_output.get(
                "visualizations_generated_this_loop", []
            )
            if new_visualizations:
                current_visualizations = updated_results.get("visualizations", [])
                if not isinstance(current_visualizations, list):
                    current_visualizations = []
                current_visualizations.extend(new_visualizations)
                updated_results["visualizations"] = current_visualizations
                logger.info(
                    f"[async_multi_agents_network] Added {len(new_visualizations)} new visualizations. Total: {len(current_visualizations)}"
                )

            new_base64_images = master_agent_output.get(
                "base64_images_generated_this_loop", []
            )
            if new_base64_images:
                current_base64_images = updated_results.get("base64_encoded_images", [])
                if not isinstance(current_base64_images, list):
                    current_base64_images = []
                current_base64_images.extend(new_base64_images)
                updated_results["base64_encoded_images"] = current_base64_images
                logger.info(
                    f"[async_multi_agents_network] Added {len(new_base64_images)} new base64 images. Total: {len(current_base64_images)}"
                )

            # Update visualization_paths from the 'filepath' attribute of new_visualizations
            new_viz_paths = [
                viz.get("filepath")
                for viz in new_visualizations
                if isinstance(viz, dict) and viz.get("filepath")
            ]
            if new_viz_paths:
                current_viz_paths = updated_results.get("visualization_paths", [])
                if not isinstance(current_viz_paths, list):
                    current_viz_paths = []
                for path in new_viz_paths:
                    if path not in current_viz_paths:  # Ensure uniqueness
                        current_viz_paths.append(path)
                updated_results["visualization_paths"] = current_viz_paths
                logger.info(
                    f"[async_multi_agents_network] Added {len(new_viz_paths)} new visualization paths. Total unique: {len(current_viz_paths)}"
                )

            new_code_snippets = master_agent_output.get(
                "code_snippets_generated_this_loop", []
            )
            if new_code_snippets:
                current_code_snippets = updated_results.get("code_snippets", [])
                if not isinstance(current_code_snippets, list):
                    current_code_snippets = []
                # Simple deduplication for code snippets based on code content
                existing_code_hashes = {
                    hash(cs.get("code"))
                    for cs in current_code_snippets
                    if isinstance(cs, dict) and cs.get("code")
                }
                for snippet in new_code_snippets:
                    if isinstance(snippet, dict) and snippet.get("code"):
                        if hash(snippet.get("code")) not in existing_code_hashes:
                            current_code_snippets.append(snippet)
                            existing_code_hashes.add(hash(snippet.get("code")))
                updated_results["code_snippets"] = current_code_snippets
                logger.info(
                    f"[async_multi_agents_network] Added/updated code snippets. Total: {len(current_code_snippets)}"
                )

            # Preserve research_plan if master_agent_output contains it
            if "research_plan" in master_agent_output:
                updated_results["research_plan"] = master_agent_output["research_plan"]
                logger.info(
                    "[async_multi_agents_network] Updated research_plan from master_agent_output."
                )

        else:
            logger.warning(
                f"[async_multi_agents_network] master_agent returned an unexpected type: {type(master_agent_output)}. Converting to string and placing in web_research_results."
            )
            updated_results["web_research_results"] = (
                [str(master_agent_output)] if master_agent_output else []
            )

        # Log what visualization data is available (if master_agent provided it directly in a dict)
        if "visualization_html" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Visualization HTML is present: {len(updated_results['visualization_html'])} chars"
            )
        if "base64_encoded_images" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Base64 images: {len(updated_results.get('base64_encoded_images', []))} items"
            )
        if "visualization_paths" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Visualization paths: {len(updated_results.get('visualization_paths', []))} items"
            )
        if "visualizations" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Visualizations: {len(updated_results.get('visualizations', []))} items"
            )

        # Ensure research_loop_count is preserved (it should be part of current_state_dict already)
        if (
            "research_loop_count" not in updated_results
        ):  # Should not happen if current_state_dict was used as base
            updated_results["research_loop_count"] = current_state_dict.get(
                "research_loop_count", 0
            )

        # Explicitly preserve benchmark fields from the original state, as they are critical for flow
        benchmark_fields = [
            "benchmark_mode",
            "benchmark_result",
            "previous_answers",
            "reflection_history",
            "config",
        ]
        for field in benchmark_fields:
            if field in current_state_dict:  # Prioritize original state for these
                if updated_results.get(field) != current_state_dict[field]:
                    logger.info(
                        f"[async_multi_agents_network] Preserving benchmark field '{field}' from original state."
                    )
                updated_results[field] = current_state_dict[field]
            elif (
                field not in updated_results
            ):  # If not in current_state_dict and not set by agent
                updated_results[field] = (
                    None  # Or some default like [] for lists, {} for dicts
                )

        # Visualization fields should ideally be part of raw_agent_results if it's a dict,
        # or handled within MasterResearchAgent to be part of its structured output.
        # For now, we assume they might be top-level keys in raw_agent_results if it was a dict.
        # If raw_agent_results was a list, visualization data would need to be part of the search result items
        # or handled differently by MasterResearchAgent.

        logger.info(
            f"[async_multi_agents_network] Final web_research_results going to next node is a list of {len(updated_results.get('web_research_results', []))} items."
        )
        if updated_results.get("web_research_results"):
            logger.info(
                f"[async_multi_agents_network] Type of first item in final web_research_results: {type(updated_results['web_research_results'][0])}"
            )

        # CRITICAL: Ensure steering fields are preserved
        if hasattr(state, "steering_enabled"):
            updated_results["steering_enabled"] = state.steering_enabled
        if hasattr(state, "steering_todo"):
            updated_results["steering_todo"] = state.steering_todo

        logger.info(
            f"[async_multi_agents_network] Preserving steering_enabled: {updated_results.get('steering_enabled', 'MISSING')}"
        )
        logger.info(
            f"[async_multi_agents_network] Preserving steering_todo: {updated_results.get('steering_todo', 'MISSING')}"
        )

        return updated_results

    except asyncio.CancelledError as ce:
        # Gracefully handle client disconnection (e.g., laptop lid close)
        logger.warning(
            f"[async_multi_agents_network] Research cancelled due to client disconnection: {str(ce)}"
        )

        # Create a partial result state with what we have so far
        # This maintains the current state while marking it as interrupted
        interrupted_state = {
            "status": "interrupted",
            "error": f"Research was interrupted: {str(ce)}",
            "interrupted_at": datetime.now().isoformat(),
            "research_topic": state.research_topic,
            "running_summary": (
                state.running_summary if hasattr(state, "running_summary") else ""
            ),
            "research_loop_count": state.research_loop_count,
            "sources_gathered": getattr(state, "sources_gathered", []),
            "web_research_results": getattr(state, "web_research_results", []),
            "selected_search_tool": getattr(
                state, "selected_search_tool", "general_search"
            ),
            # Preserve visualization fields
            "visualization_html": getattr(state, "visualization_html", ""),
            "base64_encoded_images": getattr(state, "base64_encoded_images", []),
            "visualization_paths": getattr(state, "visualization_paths", []),
            "visualizations": getattr(state, "visualizations", []),
            "code_snippets": getattr(
                state, "code_snippets", []
            ),  # Preserve code_snippets
        }

        # Merge with existing state to avoid losing fields
        current_state_dict = state.__dict__ if state else {}
        return {**current_state_dict, **interrupted_state}

    except Exception as e:
        logger.error(
            f"[async_multi_agents_network] Error in agent-based research: {str(e)}"
        )
        logger.error(traceback.format_exc())

        # Return an error state
        error_result = {
            "error": f"Async multi-agent network failed: {str(e)}",
            "status": "failed",
            "research_topic": state.research_topic if state else "Unknown",
            "running_summary": (
                state.running_summary if state else ""
            ),  # Keep existing summary
            "research_loop_count": state.research_loop_count if state else 0,
            # Preserve visualization fields even in error cases
            "visualization_html": getattr(state, "visualization_html", ""),
            "base64_encoded_images": getattr(state, "base64_encoded_images", []),
            "visualization_paths": getattr(state, "visualization_paths", []),
            "visualizations": getattr(state, "visualizations", []),
            "code_snippets": getattr(
                state, "code_snippets", []
            ),  # Preserve code_snippets
        }
        # Merge error dict with existing state to avoid losing other fields
        current_state_dict = state.__dict__ if state else {}
        updated_state = {**current_state_dict, **error_result}
        # Ensure the state remains valid, e.g., don't clear essential fields needed later
        return updated_state
