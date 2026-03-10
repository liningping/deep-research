"""
Agent Architecture for Deep Research (Refactored)

Key Features:
- Master Agent: Orchestrates planning, execution, and task management.
- Todo Manager Integration: Drives research via persistent tasks (steering_todo).
- OpenAI Exclusive: Optimized for OpenAI's function calling format.
- Optimized Logging: Concise, informative, and traceable.
"""

import os
import re
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

# Internal imports (Assumed available in the environment)
from llm_clients import get_async_llm_client, MODEL_CONFIGS
from src.tools.executor import ToolExecutor
from src.graph import ToolRegistry
from src.prompts import query_writer_instructions
from src.tools.tool_schema import TOPIC_DECOMPOSITION_FUNCTION

logger = logging.getLogger(__name__)

class SearchAgent:
    """
    Specialized Agent for executing search queries across different domains.
    Restored full functionality for Result Normalization and Tool Dispatching.
    """

    def __init__(self, config=None):
        self.config = config
        self.logger = logger
        # Initialize executor lazily or here
        try:
            tool_registry = ToolRegistry(self.config)
            self.tool_executor = ToolExecutor(tool_registry)
        except Exception as e:
            self.logger.warning(f"⚠️ [SearchAgent] Failed to init ToolExecutor (might be missing config): {e}", exc_info=True)
            self.tool_executor = None

    async def execute_task(self, subtask: Dict) -> Dict:
        """
        Smart Dispatcher: Determines the right tool and cleans the result.
        
        Args:
            subtask: Dict containing 'query' and optional 'source_type' or 'tool'
        """
        query = subtask.get("query")
        # Map source_type to actual tool names
        source_map = {
            "academic": "academic_search",
            "github": "github_search",
            "linkedin": "linkedin_search",
            "general": "general_search"
        }
        
        # Determine tool name
        requested_source = subtask.get("source_type", "general")
        tool_name = source_map.get(requested_source, "general_search")
        
        self.logger.info(f"🔍 [Search] '{tool_name}' -> '{query}'")

        try:
            # Execute specific search method
            raw_result = None
            if tool_name == "academic_search":
                raw_result = await self.academic_search(query)
            elif tool_name == "github_search":
                raw_result = await self.github_search(query)
            elif tool_name == "linkedin_search":
                raw_result = await self.linkedin_search(query)
            else:
                raw_result = await self.general_search(query)

            # --- CRITICAL: Result Normalization ---
            # Search tools return messy data. We must standardize it here.
            normalized_result = self._normalize_search_result(raw_result, query, tool_name)
            
            # Log success metrics
            source_count = len(normalized_result["sources"])
            self.logger.info(f"   -> Found {source_count} sources.")
            
            return normalized_result

        except Exception as e:
            self.logger.error(f"❌ [Search] Failed '{query}': {e}")
            return {
                "query": query,
                "tool_used": tool_name,
                "success": False,
                "content": "",
                "sources": [],
                "error": str(e)
            }

    def _normalize_search_result(self, raw_result: Any, query: str, tool_name: str) -> Dict:
        """
        Standardizes the output from various search tools into a uniform format.
        This handles the complexity of different provider return types.
        """
        content = ""
        sources = []
        error_msg = None
        
        # Case 1: Result is a Dictionary (Most common for proper tools)
        if isinstance(raw_result, dict):
            if "error" in raw_result:
                error_msg = raw_result["error"]
                
            # Extract Sources
            if "formatted_sources" in raw_result:
                # Format: "Title : URL" string list
                for s in raw_result["formatted_sources"]:
                    if isinstance(s, str) and " : " in s:
                        parts = s.split(" : ", 1)
                        sources.append({"title": parts[0].strip(), "url": parts[1].strip()})
            elif "sources" in raw_result and isinstance(raw_result["sources"], list):
                # Format: List of dicts
                sources = raw_result["sources"]
            
            # Extract Content
            if "content" in raw_result:
                content = raw_result["content"]
            elif "raw_contents" in raw_result:
                # Join raw chunks
                content = "\n\n".join([str(c) for c in raw_result["raw_contents"] if c])
            else:
                # Fallback
                content = str(raw_result)

        # Case 2: Result is a String (Simple tools or errors)
        elif isinstance(raw_result, str):
            content = raw_result
        
        # Case 3: Result is List (Rare, but possible)
        elif isinstance(raw_result, list):
            content = "\n".join([str(item) for item in raw_result])

        has_content = bool(str(content).strip() if isinstance(content, str) else content)
        has_sources = bool(sources)

        return {
            "query": query,
            "tool_used": tool_name,
            "success": True if (not error_msg and (has_content or has_sources)) else False,
            "content": content,
            "sources": sources,
            "error": error_msg
        }

    # --- Specific Search Implementations ---
    # These act as wrappers around the generic executor to enforce specific params (like top_k)

    async def general_search(self, query):
        if not self.tool_executor: return {"error": "No executor"}
        return await self.tool_executor.execute_tool(
            tool_name="general_search", params={"query": query, "top_k": 5}
        )

    async def academic_search(self, query):
        if not self.tool_executor: return {"error": "No executor"}
        return await self.tool_executor.execute_tool(
            tool_name="academic_search", params={"query": query, "top_k": 5}
        )

    async def github_search(self, query):
        if not self.tool_executor: return {"error": "No executor"}
        return await self.tool_executor.execute_tool(
            tool_name="github_search", params={"query": query, "top_k": 5}
        )

    async def linkedin_search(self, query):
        if not self.tool_executor: return {"error": "No executor"}
        return await self.tool_executor.execute_tool(
            tool_name="linkedin_search", params={"query": query, "top_k": 5}
        )