"""
Utility functions for graph nodes.

Contains helper functions used across multiple node modules:
- Callback handling (get_callback_from_config, emit_event)
- Research loop configuration (get_max_loops)
- State management (reset_state)
- Heartbeat for long-running tasks
- Configuration extraction (get_configurable)
"""

import os
import asyncio
import logging

from src.state import SummaryState
from src.configuration import Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_callback_from_config(config):
    """
    Get the appropriate callback object from config, handling both single callback
    objects and lists of callbacks.

    Args:
        config: Configuration object which may contain callbacks

    Returns:
        tuple: (callback_obj, has_callbacks) where callback_obj is the object to call
               on_event on, and has_callbacks is a boolean indicating if a valid
               callback was found.
    """
    # If config is None, return a default empty callback
    if config is None:
        return {"on_event": lambda event_type, data: None}, False

    callbacks = config.get("callbacks", [])

    # First check if callbacks is a single object with on_event
    if callbacks and hasattr(callbacks, "on_event"):
        return callbacks, True

    # Then check if it's a list with at least one item that has on_event
    if isinstance(callbacks, list) and callbacks and hasattr(callbacks[-1], "on_event"):
        return callbacks[-1], True

    # No valid callback found
    return {"on_event": lambda event_type, data: None}, False


def emit_event(callbacks, event_type, data=None, error_message=None):
    """
    Emit an event to the specified callbacks.

    Args:
        callbacks: Callback object or None
        event_type: Type of event to emit
        data: Data to include with the event
        error_message: Custom error message to display if emission fails
    """
    logger.info(
        f"emit_event called: event_type={event_type}, data={data}, callbacks={'present' if callbacks else 'none'}"
    )
    try:
        if callbacks and isinstance(callbacks, dict) and "on_event" in callbacks:
            callbacks["on_event"](event_type, data or {})
        elif callbacks and hasattr(callbacks, "on_event"):
            callbacks.on_event(event_type, data or {})
    except Exception as e:
        error_msg = error_message or f"Warning: Failed to emit event {event_type}"
        logger.error(f"{error_msg}: {str(e)}")


def get_max_loops(
    configurable,
    extra_effort=False,
    minimum_effort=False,
):
    """Get maximum number of research loops with consistent handling of effort flags.

    Args:
        configurable: Configuration object containing max_web_research_loops
        extra_effort: Boolean flag indicating if extra effort (more loops) should be used
        minimum_effort: Boolean flag indicating if minimum effort (1 loop) should be used

    Returns:
        int: Maximum number of research loops to perform
    """
    # Minimum effort or QA mode overrides everything - use only 1 loop
    if minimum_effort:
        logger.info("  - Using minimum effort (1 loop)")
        return 1

    env_max_loops = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
    base_max_loops = (
        int(env_max_loops)
        if env_max_loops
        else int(configurable.max_web_research_loops)
    )
    logger.info(f"  - Reading MAX_WEB_RESEARCH_LOOPS from environment: {env_max_loops}")

    max_loops = base_max_loops

    logger.info(
        f"  - Using max_loops={max_loops} (extra_effort={extra_effort}, base={base_max_loops})"
    )
    return max_loops


def reset_state(state: SummaryState):
    """Reset the state for a new research topic"""

    # Use the research_topic from the input but completely reset everything else
    # We need to create a brand new state instead of modifying the existing one
    return {
        "research_loop_count": 0,
        "sources_gathered": [],  # Empty array, not appending
        "web_research_results": [],  # Empty array, not appending
        "running_summary": "",
        "search_query": "",
        "research_complete": False,  # Initialize research_complete flag
        "knowledge_gap": "",  # Initialize knowledge_gap
        "search_results_empty": False,  # Initialize search_results_empty flag
        "selected_search_tool": "general_search",  # Initialize search tool tracker
        # Copy the research_topic from the input state
        "research_topic": state.research_topic,
    }


async def heartbeat_task(callbacks, interval=5):
    try:
        while True:
            emit_event(callbacks, "heartbeat", {"message": "still working"})
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


def get_configurable(config):
    """
    Helper function to extract configuration from the runnable config.

    Args:
        config: The runnable config object or None

    Returns:
        Configuration object with the settings from config
    """
    # If config is None, return a default configuration
    if config is None:
        return Configuration()

    # If config already has a configurable, use it
    if "configurable" in config and isinstance(config["configurable"], Configuration):
        return config["configurable"]

    # Otherwise create a new configurable from the config
    return Configuration.from_runnable_config(config)
