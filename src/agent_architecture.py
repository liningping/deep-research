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

class MasterResearchAgent:
    """
    Master agent responsible for planning research, managing the Todo list,
    and coordinating search execution.
    """

    def __init__(self, config=None):
        self.config = config
        self.logger = logger
        self.state = None # Will be set during execution

    async def _get_llm(self):
        """Helper to get the OpenAI LLM client."""
        # Hardcoded to OpenAI as requested
        model = os.environ.get("LLM_MODEL", "gpt-4o")
        return await get_async_llm_client("openai", model)

    async def decompose_topic(self, query, knowledge_gap, existing_tasks=None):
        """
        Decompose the research topic into sub-queries using OpenAI.
        """
        try:
            llm = await self._get_llm()
            
            # Prepare Context
            today = datetime.now()
            context = {
                "research_topic": f"{query} - {knowledge_gap}" if knowledge_gap else query,
                "research_context": f"Knowledge gap: {knowledge_gap}" if knowledge_gap else "",
                "current_date": today.strftime("%B %d, %Y"),
                "current_year": str(today.year),
                "one_year_ago": str(today.year - 1),
                "AUGMENT_KNOWLEDGE_CONTEXT": "No user-provided external knowledge.",
                "steering_context": "Focus on actionable research tasks.",
                "DATABASE_CONTEXT": None
            }
            
            # Format Prompt
            formatted_prompt = query_writer_instructions.format(**context)
            if existing_tasks:
                formatted_prompt += f"\n\nEXISTING TASKS (Avoid Duplicates): {[t.description for t in existing_tasks]}"

            # Bind Tools (OpenAI Format)
            llm_with_tool = llm.bind_tools(
                tools=[{"type": "function", "function": TOPIC_DECOMPOSITION_FUNCTION}],
                tool_choice={"type": "function", "function": {"name": "decompose_research_topic"}}
            )

            self.logger.info(f"🔎 [Decompose] Analyzing complexity for: '{query}'")
            response = await llm_with_tool.ainvoke([
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": f"Decompose: {context['research_topic']}"}
            ])

            # Extract Arguments
            tool_call = response.tool_calls[0]
            
            if isinstance(tool_call, dict):
                args = tool_call.get("args", {})
            else:
                args = getattr(tool_call, "args", {})
            
            complexity = args.get("topic_complexity", "simple")
            self.logger.info(f"🧠 [Decompose] Topic identified as: {complexity.upper()}")

            if complexity == "complex":
                complex_data = args.get("complex_topic", {})
                return {
                    "topic_complexity": "complex",
                    "main_query": complex_data.get("main_query", query),
                    "subtopics": complex_data.get("subtopics", []),
                    "visualization_type": complex_data.get("recommended_visualizations", [])
                }
            else:
                simple_data = args.get("simple_topic", {})
                return {
                    "topic_complexity": "simple",
                    "query": simple_data.get("query", query),
                    "suggested_tool": simple_data.get("suggested_tool", "general_search")
                }

        except Exception as e:
            self.logger.error(f"❌ [Decompose] Error: {e}")
            return {"topic_complexity": "simple", "query": query, "suggested_tool": "general_search"}

    async def plan_research_from_tasks(self, query, tasks, state):
        """
        Generate specific queries for specific pending Todo tasks.
        This represents the 'Agentic' behavior of mapping Intent -> Action.
        """
        self.logger.info(f"📋 [Planning] Creating plan for {len(tasks)} pending tasks...")
        
        task_list_str = "\n".join([f"{i+1}. [{t.id}] {t.description}" for i, t in enumerate(tasks)])
        
        prompt = f"""
        You are a task-driven research agent.
        MAIN TOPIC: {query}
        
        PENDING TASKS:
        {task_list_str}
        
        Generate ONE search query for EACH task.
        Return JSON format: {{ "queries": [ {{ "query": "...", "tool": "general_search", "completes_task_id": "..." }} ] }}
        """

        try:
            llm = await self._get_llm()
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            
            # Robust JSON extraction
            content = response.content if hasattr(response, "content") else str(response)
            match = re.search(r"```(?:json)?\n*(.*?)\n*```", content, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
            
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
                
            plan_data = json.loads(content)
            
            subtasks = []
            for item in plan_data.get("queries", []):
                subtasks.append({
                    "type": "search",
                    "query": item["query"],
                    "description": f"Task execution: {item.get('completes_task_id')}",
                    "source_type": "general",
                    "completes_task_id": item.get("completes_task_id") # Critical for tracking
                })
            
            return {
                "topic_complexity": "task_driven",
                "subtasks": subtasks,
                "tasks_targeted": [t.id for t in tasks]
            }
        except Exception as e:
            self.logger.warning(f"⚠️ [Planning] Task-driven planning failed, falling back to simple extraction. Error: {e}")
            # Fallback: Just search for the task description
            return {
                "topic_complexity": "task_driven",
                "subtasks": [{
                    "type": "search",
                    "query": t.description, 
                    "completes_task_id": t.id,
                    "source_type": "general"
                } for t in tasks]
            }

    async def execute_research(self, state):
        """
        Main Execution Loop.
        Integrates Todo Management -> Planning -> Execution -> Feedback.
        """
        self.state = state
        loop_count = getattr(state, "research_loop_count", 0)
        query = getattr(state, "search_query", state.research_topic)

        self.logger.info(f"🚀 [Execution] Starting Loop {loop_count} for query: '{query}'")

        # --- STEP 1: INITIALIZE TODO (Loop 0 only) ---
        if loop_count == 0 and hasattr(state, "steering_todo"):
             await self._create_initial_research_plan(query, state)

        # --- STEP 2: DETERMINE STRATEGY ---
        research_plan = None
        
        # Check for Pending Tasks in Todo Manager
        pending_tasks = []
        if hasattr(state, "steering_todo"):
            pending_tasks = state.steering_todo.get_pending_tasks()
            # Sort by priority
            pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        if pending_tasks:
            # Strategy A: Task-Driven (We have explicit things to do)
            # Take top 3 tasks to avoid context overflow
            top_tasks = pending_tasks[:3]
            self.logger.info(f"🎯 [Strategy] Focusing on top {len(top_tasks)} high-priority tasks.")
            
            # Mark as in-progress
            for t in top_tasks:
                state.steering_todo.mark_task_in_progress(t.id)
            
            research_plan = await self.plan_research_from_tasks(query, top_tasks, state)
        else:
            # Strategy B: Exploration/Decomposition (No tasks, or tasks exhausted)
            self.logger.info(f"🧭 [Strategy] No pending tasks. Performing standard decomposition.")
            decomposition = await self.decompose_topic(query, getattr(state, "knowledge_gap", ""))
            
            # Convert decomposition to a plan format
            subtasks = []
            if decomposition["topic_complexity"] == "complex":
                for sub in decomposition.get("subtopics", []):
                    subtasks.append({"type": "search", "query": sub, "source_type": "general"})
            else:
                subtasks.append({"type": "search", "query": decomposition["query"], "source_type": "general"})
            
            research_plan = {"topic_complexity": decomposition["topic_complexity"], "subtasks": subtasks}

        # --- STEP 3: EXECUTE SEARCH ---
        search_results = await self._execute_search_tasks(research_plan, state)

        # --- STEP 4: UPDATE TODO & FEEDBACK ---
        if hasattr(state, "steering_todo") and research_plan.get("topic_complexity") == "task_driven":
            await self._update_todo_based_on_results(search_results, state)

        # --- STEP 5: VISUALIZATION (Simplified) ---
        # (Assuming Visualization Logic exists elsewhere or is simplified here)
        # For brevity, returning the search results primarily.
        
        return {
            "web_research_results": search_results,
            "research_plan": research_plan
        }

    async def _execute_search_tasks(self, plan, state):
        """Execute searches with steering checks (deduplication/cancellation)."""
        search_agent = SearchAgent(self.config)
        results = []
        
        tasks = plan.get("subtasks", [])
        if not tasks: return []

        self.logger.info(f"⚡ [Search] Executing {len(tasks)} queries...")

        for task in tasks:
            query = task.get("query")
            
            # Steering Check: Deduplication
            if hasattr(state, "steering_todo") and state.steering_todo.is_query_duplicate(query):
                self.logger.info(f"⏭️ [Search] Skipping duplicate: '{query}'")
                continue

            # Execute
            try:
                # Execute task securely (includes exact dispatch and normalization)
                res = await search_agent.execute_task(task)
                
                # Mark executed
                if hasattr(state, "steering_todo"):
                    state.steering_todo.mark_query_executed(query)

                # Format Result (execute_task returns normalized format)
                success = res.get("success", False)
                results.append({
                    "query": query,
                    "success": success,
                    "content": res.get("content", ""),
                    "sources": res.get("sources", []),
                    "error": res.get("error"),
                    "completes_task_id": task.get("completes_task_id") # Pass through for feedback
                })
                
                log_icon = "✅" if success else "⚠️"
                self.logger.info(f"{log_icon} [Search] '{query}' -> {len(res.get('sources', []))} sources")

            except Exception as e:
                self.logger.error(f"❌ [Search] Failed '{query}': {e}", exc_info=True)
                results.append({
                    "query": query, 
                    "success": False, 
                    "content": "", 
                    "sources": [], 
                    "error": str(e),
                    "completes_task_id": task.get("completes_task_id")
                })

        return results

    async def _create_initial_research_plan(self, query, state):
        """Decompose original query into the first set of Todo items."""
        self.logger.info("📝 [Todo] Initializing Research Plan...")
        decomposition = await self.decompose_topic(query, "")
        
        # Add "Main Goal"
        state.steering_todo.create_task(f"Comprehensive Report: {query}", priority=10)
        
        # Add Subtasks
        subtopics = decomposition.get("subtopics", []) if decomposition.get("topic_complexity") == "complex" else [decomposition.get("query")]
        
        for sub in subtopics:
            state.steering_todo.create_task(f"Research: {sub}", priority=5)
        
        self.logger.info(f"📝 [Todo] Created {len(subtopics) + 1} initial tasks.")

    async def _update_todo_based_on_results(self, results, state):
        """Mark tasks as completed based on search success."""
        completed_count = 0
        for res in results:
            task_id = res.get("completes_task_id")
            if task_id and res.get("success"):
                sources_list = res.get("sources", [])
                query_str = res.get("query", "")
                state.steering_todo.mark_task_completed(
                    task_id=task_id,
                    completion_note=f"Found {len(sources_list)} sources via query '{query_str}'"
                )
                completed_count += 1
        
        if completed_count > 0:
            self.logger.info(f"🏁 [Todo] Automatically marked {completed_count} tasks as COMPLETED.")

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