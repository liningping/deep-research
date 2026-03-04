"""
Search node for the research graph.

Contains the multi-agent search network that coordinates research agents
to execute search queries and gather information.
"""

import logging
import asyncio
import traceback
import uuid

from src.state import SummaryState
from src.nodes.utils import heartbeat_task

logger = logging.getLogger(__name__)


def _merge_list_field(target: dict, key: str, new_items: list, dedupe_key: str = None):
    """Merge new_items into target[key] with optional deduplication.

    Args:
        target: The result dict to update in-place.
        key: The key in target whose value is a list.
        new_items: New items to append.
        dedupe_key: If set, deduplicate by this dict key within each item.
    """
    if not new_items:
        return
    current = target.get(key, [])
    if not isinstance(current, list):
        current = []
    if dedupe_key:
        existing = {hash(item.get(dedupe_key)) for item in current if isinstance(item, dict) and item.get(dedupe_key)}
        for item in new_items:
            if isinstance(item, dict) and item.get(dedupe_key):
                h = hash(item.get(dedupe_key))
                if h not in existing:
                    current.append(item)
                    existing.add(h)
    else:
        current.extend(new_items)
    target[key] = current


def _extract_sources(raw_results: list, existing_citations: dict):
    """Extract sources_gathered and source_citations from search results.

    Returns:
        (sources_gathered, source_citations) — both cumulative.
    """
    sources_gathered = []
    source_citations = dict(existing_citations) if existing_citations else {}

    # Determine next citation index from existing citations
    citation_index = 1
    for key in source_citations:
        idx = int(key) if isinstance(key, int) or (isinstance(key, str) and key.isdigit()) else 0
        if idx >= citation_index:
            citation_index = idx + 1

    existing_urls = {
        v.get("url") for v in source_citations.values() if isinstance(v, dict)
    }

    for result in raw_results:
        if not isinstance(result, dict):
            continue
        for source in result.get("sources", []):
            if not isinstance(source, dict) or "title" not in source or "url" not in source:
                continue
            # Dedupe for sources_gathered
            source_str = f"{source['title']} : {source['url']}"
            if source_str not in sources_gathered:
                sources_gathered.append(source_str)
            # Dedupe for citations
            if source["url"] not in existing_urls:
                entry = {"title": source["title"], "url": source["url"]}
                if "source_type" in source:
                    entry["source_type"] = source["source_type"]
                source_citations[str(citation_index)] = entry
                existing_urls.add(source["url"])
                citation_index += 1

    return sources_gathered, source_citations


def _build_error_state(state, error_msg: str, status: str = "failed"):
    """Build an error/interrupted result dict preserving existing state."""
    base = state.__dict__.copy() if state else {}
    base.update({
        "error": error_msg,
        "status": status,
    })
    return base


async def async_multi_agents_network(state: SummaryState, callbacks=None):
    """
    Execute research using the multi-agent network.

    Args:
        state: The current state containing research parameters

    Returns:
        Updated state with research results
    """
    logger.info(
        f"[search] Starting research loop {state.research_loop_count} "
        f"| provider={state.llm_provider} model={state.llm_model}"
    )

    try:
        # --- Steering integration (modifies research behavior) ---
        if hasattr(state, "steering_todo") and state.steering_todo:
            from src.steering_integration import get_steering_summary_for_agent
            steering_result = await state.prepare_steering_for_next_loop()
            if steering_result.get("steering_enabled"):
                logger.info(
                    f"[steering] todo v{steering_result.get('todo_version')} | "
                    f"pending={steering_result.get('pending_tasks')} done={steering_result.get('completed_tasks')}"
                )
            steering_context = get_steering_summary_for_agent(state)
            if steering_context:
                logger.info(f"[steering] Active constraints: {steering_context.strip()}")

        # --- Initialize master agent ---
        from src.agent_architecture import MasterResearchAgent

        config = getattr(state, "config", None) or {}
        if "configurable" not in config:
            config["configurable"] = {"thread_id": str(uuid.uuid4())}
        config["configurable"]["llm_provider"] = state.llm_provider
        config["configurable"]["llm_model"] = state.llm_model

        master_agent = MasterResearchAgent(config)

        # --- Execute research with heartbeat ---
        heartbeat = asyncio.create_task(heartbeat_task(callbacks))
        try:
            master_agent_output = await master_agent.execute_research(state)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

        logger.info("[search] Research completed successfully.")

        # --- Assemble results (start from current state) ---
        results = state.__dict__.copy()

        if not isinstance(master_agent_output, dict):
            logger.warning(f"[search] Unexpected output type: {type(master_agent_output)}")
            results["web_research_results"] = [str(master_agent_output)] if master_agent_output else []
            return results

        # Merge web_research_results
        raw_results = master_agent_output.get("web_research_results", [])
        if isinstance(raw_results, list):
            results["web_research_results"] = raw_results

            # Extract and merge sources + citations
            new_sources, merged_citations = _extract_sources(
                raw_results, getattr(state, "source_citations", {})
            )
            if new_sources:
                existing = results.get("sources_gathered", [])
                for s in new_sources:
                    if s not in existing:
                        existing.append(s)
                results["sources_gathered"] = existing
            if merged_citations:
                results["source_citations"] = merged_citations
        else:
            results["web_research_results"] = []

        # Merge visualization outputs
        _merge_list_field(results, "visualizations",
                          master_agent_output.get("visualizations_generated_this_loop", []))
        _merge_list_field(results, "base64_encoded_images",
                          master_agent_output.get("base64_images_generated_this_loop", []))
        _merge_list_field(results, "code_snippets",
                          master_agent_output.get("code_snippets_generated_this_loop", []),
                          dedupe_key="code")

        # Merge visualization paths (extract from visualizations)
        new_viz_paths = [
            v.get("filepath") for v in master_agent_output.get("visualizations_generated_this_loop", [])
            if isinstance(v, dict) and v.get("filepath")
        ]
        if new_viz_paths:
            current_paths = results.get("visualization_paths", [])
            if not isinstance(current_paths, list):
                current_paths = []
            for p in new_viz_paths:
                if p not in current_paths:
                    current_paths.append(p)
            results["visualization_paths"] = current_paths

        # Preserve research_plan if provided
        if "research_plan" in master_agent_output:
            results["research_plan"] = master_agent_output["research_plan"]

        # Preserve steering fields
        if hasattr(state, "steering_enabled"):
            results["steering_enabled"] = state.steering_enabled
        if hasattr(state, "steering_todo"):
            results["steering_todo"] = state.steering_todo

        logger.info(f"[search] Done. {len(results.get('web_research_results', []))} results, "
                    f"{len(results.get('sources_gathered', []))} sources.")
        return results

    except asyncio.CancelledError as ce:
        logger.warning(f"[search] Cancelled (client disconnect): {ce}")
        return _build_error_state(state, f"Research interrupted: {ce}", status="interrupted")

    except Exception as e:
        logger.error(f"[search] Error: {e}")
        logger.error(traceback.format_exc())
        return _build_error_state(state, f"Research failed: {e}", status="failed")
