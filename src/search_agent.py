"""Search Agent.

Dispatches search queries to the appropriate tool backend and normalises
the raw results into a uniform ``SearchResult`` dict.

Standard output of ``execute_task``
-------------------------------------
::

    {
        "query":     str,          # the original search query
        "tool_used": str,          # tool name that was invoked
        "success":   bool,         # True when content or sources were found
        "content":   str,          # joined text content from the tool
        "sources":   List[Dict],   # list of {"title": str, "url": str}
        "error":     str | None,   # error message, or None on success
    }
"""

import logging
from typing import Any, Dict, List, Optional

from deep_research.tools.executor import ToolExecutor
from deep_research.tools.registry import SearchToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# source_type → tool name mapping
# ---------------------------------------------------------------------------

_SOURCE_TO_TOOL: Dict[str, str] = {
    "academic": "academic_search",
    "github":   "github_search",
    "linkedin": "linkedin_search",
    "general":  "general_search",
}


# ---------------------------------------------------------------------------
# SearchAgent
# ---------------------------------------------------------------------------

class SearchAgent:
    """Dispatch search queries to the right tool and normalise results.

    ``execute_task`` is the public entry-point. It accepts a subtask dict
    produced by the bandit scheduler and returns a normalised ``SearchResult``
    dict (see module docstring for schema).
    """

    def __init__(self, config=None):
        self.config = config
        try:
            tool_registry = SearchToolRegistry(self.config)
            self.tool_executor: Optional[ToolExecutor] = ToolExecutor(tool_registry)
        except Exception as exc:
            logger.warning(
                f"⚠️ [SearchAgent] Failed to init ToolExecutor: {exc}",
                exc_info=True,
            )
            self.tool_executor = None

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def execute_task(self, subtask: Dict) -> Dict:
        """Dispatch a subtask to the appropriate search tool.

        Args:
            subtask: Dict with at minimum a ``"query"`` key and optionally
                     a ``"source_type"`` key (``"general"`` | ``"academic"``
                     | ``"github"`` | ``"linkedin"``).

        Returns:
            Normalised ``SearchResult`` dict (see module docstring).
        """
        query: str = subtask.get("query", "")
        source_type: str = subtask.get("source_type", "general")
        tool_name: str = _SOURCE_TO_TOOL.get(source_type, "general_search")

        logger.info(f"🔍 [Search] '{tool_name}' -> '{query}'")

        try:
            raw_result = await self._dispatch(tool_name, query)
            result = _normalize(raw_result, query, tool_name)
            logger.info(f"   -> Found {len(result['sources'])} sources.")
            return result

        except Exception as exc:
            logger.error(f"❌ [Search] Failed '{query}': {exc}")
            return {
                "query":     query,
                "tool_used": tool_name,
                "success":   False,
                "content":   "",
                "sources":   [],
                "error":     str(exc),
            }

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    async def _dispatch(self, tool_name: str, query: str) -> Any:
        """Route to the correct tool executor call."""
        if not self.tool_executor:
            return {"error": "No executor available"}

        return await self.tool_executor.execute_tool(
            tool_name=tool_name,
            params={"query": query, "top_k": 5},
        )


# ---------------------------------------------------------------------------
# Result normalisation (module-level, pure function)
# ---------------------------------------------------------------------------

def _normalize(raw_result: Any, query: str, tool_name: str) -> Dict:
    """Normalise raw tool output into a uniform ``SearchResult`` dict.

    Handles three raw formats:
    - ``dict``  – most tools return a dict with ``content``/``sources``/etc.
    - ``str``   – simple tools or plain error strings.
    - ``list``  – rare; items are joined with newlines.
    """
    content: str = ""
    sources: List[Dict] = []
    error_msg: Optional[str] = None

    if isinstance(raw_result, dict):
        error_msg = raw_result.get("error")

        # Sources: prefer formatted_sources ("Title : URL"), fallback to sources list
        if "formatted_sources" in raw_result:
            for s in raw_result["formatted_sources"]:
                if isinstance(s, str) and " : " in s:
                    title, url = s.split(" : ", 1)
                    sources.append({"title": title.strip(), "url": url.strip()})
        elif isinstance(raw_result.get("sources"), list):
            sources = raw_result["sources"]

        # Content: prefer explicit content key, then raw_contents chunks
        if "content" in raw_result:
            content = raw_result["content"]
        elif "raw_contents" in raw_result:
            content = "\n\n".join(str(c) for c in raw_result["raw_contents"] if c)
        elif not error_msg:
            content = str(raw_result)

    elif isinstance(raw_result, str):
        content = raw_result

    elif isinstance(raw_result, list):
        content = "\n".join(str(item) for item in raw_result)

    has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
    has_sources = bool(sources)

    return {
        "query":     query,
        "tool_used": tool_name,
        "success":   bool(not error_msg and (has_content or has_sources)),
        "content":   content,
        "sources":   sources,
        "error":     error_msg,
    }