"""
Agent Architecture for Deep Research

This module implements a modular, agent-based architecture for research:
- Master Agent: Planning, query decomposition, and coordination
- Search Agent: Specialized for executing search queries
- Visualization Agent: Creates data visualizations from search results
- Result Combiner: Integrates findings from multiple sources

The architecture provides better separation of concerns while
maintaining backward compatibility with the existing codebase.
"""

import os
import json
import logging
import traceback
import re
import base64
from typing import Dict, List, Any, Optional
from openai import OpenAI
import asyncio
import uuid
from src.tools.executor import ToolExecutor

import time


class MasterResearchAgent:
    """
    Master agent responsible for planning research, decomposing topics,
    and coordinating specialized agents.

    This agent serves as the "central brain" that breaks down complex queries
    into manageable subtasks and delegates them to specialized agents.
    """

    def __init__(self, config=None):
        """
        Initialize the Master Research Agent.

        Args:
            config: Configuration object containing LLM settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def decompose_topic(
        self,
        query,
        knowledge_gap,
        research_loop_count,
        uploaded_knowledge=None,
        existing_tasks=None,
    ):
        """
        Analyze a topic and decompose it if complex.
        Uses query_writer_instructions as the central brain for decomposition.

        Args:
            query: The main research query or topic
            knowledge_gap: Additional context about knowledge gaps to address
            research_loop_count: Current iteration of research loop
            uploaded_knowledge: User-provided external knowledge (optional)
            existing_tasks: List of existing pending tasks to avoid duplicates (optional)

        Returns:
            Dict containing topic_complexity, query/subtopics, and analysis
        """
        # Import necessary functions
        import os
        import json
        import traceback
        from llm_clients import get_llm_client, MODEL_CONFIGS

        try:
            # Get configuration
            if self.config is not None:
                from src.graph import get_configurable

                configurable = get_configurable(self.config)
                provider = configurable.llm_provider
                model = configurable.llm_model
            else:
                # Try environment variables first
                provider = os.environ.get("LLM_PROVIDER")
                model = os.environ.get("LLM_MODEL")

                # Use defaults if not found in environment
                if not provider:
                    provider = "openai"
                # Will use the default model from llm_clients.py if model is None

            # Get the model name from the configuration or default
            if provider == "openai" and not model:
                model = MODEL_CONFIGS["openai"]["default_model"]
            elif not model and provider in MODEL_CONFIGS:
                model = MODEL_CONFIGS[provider]["default_model"]

            self.logger.info(
                f"[MasterAgent] Using {provider} model {model} for topic decomposition"
            )

            # Generate date constants
            from datetime import datetime

            today = datetime.now()
            CURRENT_DATE = today.strftime("%B %d, %Y")
            CURRENT_YEAR = str(today.year)
            ONE_YEAR_AGO = str(today.year - 1)

            # Create a combined query that incorporates knowledge gaps if available
            combined_topic = query
            research_context = ""
            if knowledge_gap and knowledge_gap.strip():
                combined_topic = f"{query} - {knowledge_gap}"
                research_context = (
                    f"Knowledge gap identified in previous research: {knowledge_gap}"
                )

            # Prepare uploaded knowledge context
            AUGMENT_KNOWLEDGE_CONTEXT = ""
            if uploaded_knowledge and uploaded_knowledge.strip():
                AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should be treated as highly authoritative and trustworthy. This uploaded knowledge should guide query generation to complement rather than duplicate existing information.

Uploaded Knowledge Preview: {uploaded_knowledge[:500]}{'...' if len(uploaded_knowledge) > 500 else ''}

Query Generation Strategy:
- Identify what information is already covered in the uploaded knowledge
- Generate queries that fill gaps or provide additional context to the uploaded knowledge
- Focus on recent developments, validation, or areas not covered in uploaded knowledge
- Use uploaded knowledge to inform more targeted and specific search queries
"""
            else:
                AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Generate queries based on the research topic and knowledge gaps."

            # Import the query_writer_instructions prompt
            from src.prompts import query_writer_instructions

            # Import async client getter
            from llm_clients import get_async_llm_client

            # No steering context in DRB mode
            steering_context = "No steering instructions provided. Follow standard research approach."

            # No database context in DRB mode
            database_context = "No database files are available. Use standard web search tools."

            # Build existing tasks context for duplicate prevention
            existing_tasks_context = ""
            if existing_tasks and len(existing_tasks) > 0:
                self.logger.info(
                    f"[MasterAgent] Including {len(existing_tasks)} existing tasks for duplicate prevention"
                )
                existing_tasks_context = "\n\nIMPORTANT - EXISTING PENDING TASKS:\n"
                existing_tasks_context += "The following research tasks are already pending or in-progress. DO NOT create duplicate or highly similar tasks:\n\n"
                for task in existing_tasks:
                    existing_tasks_context += f"- [{task.id}] {task.description} (Priority: {task.priority}, Source: {task.source})\n"
                existing_tasks_context += "\nQuery Generation Strategy:\n"
                existing_tasks_context += "- Only generate queries for NEW aspects NOT covered by existing tasks\n"
                existing_tasks_context += "- Avoid creating semantically similar queries to existing task descriptions\n"
                existing_tasks_context += "- If existing tasks already cover the knowledge gap, generate fewer or zero new queries\n"

            # Format the prompt with the appropriate context
            formatted_prompt = query_writer_instructions.format(
                research_topic=combined_topic,
                research_context=research_context,
                current_date=CURRENT_DATE,
                current_year=CURRENT_YEAR,
                one_year_ago=ONE_YEAR_AGO,
                AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
                DATABASE_CONTEXT=database_context,
                steering_context=steering_context,
            )

            # Append existing_tasks_context after formatting (so it's not part of the template)
            if existing_tasks_context:
                formatted_prompt += existing_tasks_context

            self.logger.info(
                f"[MasterAgent.decompose_topic] Database context being passed to LLM: {database_context[:200]}..."
            )

            # Get the appropriate ASYNC LLM client based on provider
            # llm = get_llm_client(provider, model) # Old sync call
            llm = await get_async_llm_client(provider, model)  # Use async client

            # Format messages for the LLM client (Using dictionary format for broader compatibility)
            messages = [
                {"role": "system", "content": formatted_prompt},
                {
                    "role": "user",
                    "content": f"Generate search queries for the following research topic: {combined_topic}",
                },
            ]

            # Import the topic decomposition function schema
            from src.tools.tool_schema import TOPIC_DECOMPOSITION_FUNCTION

            # Bind the decomposition tool
            langchain_tools = [
                {"type": "function", "function": TOPIC_DECOMPOSITION_FUNCTION}
            ]

            # Handle tool choice format based on provider
            if provider == "anthropic":
                tool_choice = {"type": "tool", "name": "decompose_research_topic"}
            elif provider == "google":
                # Use string format for Google Generative AI
                tool_choice = "decompose_research_topic"
            elif provider == "openai":
                # Use standard dictionary format for OpenAI
                tool_choice = {
                    "type": "function",
                    "function": {"name": "decompose_research_topic"},
                }
            else:
                # Default or fallback - assuming OpenAI-like structure might work for some
                # or default to no specific tool choice if unsure.
                # For now, let's default to the OpenAI format, but log a warning.
                self.logger.warning(
                    f"[MasterAgent] Using default OpenAI tool_choice format for provider '{provider}'. This might not be correct."
                )
                tool_choice = {
                    "type": "function",
                    "function": {"name": "decompose_research_topic"},
                }

            llm_with_tool = llm.bind_tools(
                tools=langchain_tools, tool_choice=tool_choice
            )

            # Call LLM API with function calling ASYNCHRONOUSLY
            model_str = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
            self.logger.info(
                f"[MasterAgent] Making ASYNC tool call with model {model_str} to decompose topic..."
            )
            response = await llm_with_tool.ainvoke(messages)  # Use await and ainvoke
            self.logger.info(
                f"[MasterAgent] Raw ASYNC LLM response for decomposition: {response}"
            )

            # Process the response (consistent across providers due to Langchain abstraction)
            function_args = None  # Initialize to None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                self.logger.info(f"[MasterAgent] Extracted tool call: {tool_call}")
                try:
                    # Standard Langchain way: access parsed args directly
                    if hasattr(tool_call, "args") and isinstance(tool_call.args, dict):
                        function_args = tool_call.args
                        self.logger.info(
                            f"[MasterAgent] Using pre-parsed args (from tool_call.args): {function_args}"
                        )
                    # Fallback for string arguments (less common)
                    elif (
                        hasattr(tool_call, "function")
                        and hasattr(tool_call.function, "arguments")
                        and isinstance(tool_call.function.arguments, str)
                    ):
                        raw_args = tool_call.function.arguments
                        self.logger.warning(
                            f"[MasterAgent] Received string arguments, attempting JSON parse: {raw_args}"
                        )
                        function_args = json.loads(raw_args)
                    else:
                        # Handle unexpected structures, including the dict case if needed
                        if (
                            isinstance(tool_call, dict)
                            and "args" in tool_call
                            and isinstance(tool_call["args"], dict)
                        ):
                            function_args = tool_call["args"]
                            self.logger.info(
                                f"[MasterAgent] Using args from dict key (standard structure): {function_args}"
                            )
                        else:
                            self.logger.warning(
                                f"[MasterAgent] Tool call structure not recognized or args not found/parsed: {tool_call}"
                            )

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"[MasterAgent] Failed to parse JSON arguments: {getattr(locals(), 'raw_args', 'N/A')} - Error: {e}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"[MasterAgent] Error processing tool call arguments: {e}"
                    )

            if function_args:
                self.logger.info(f"[MasterAgent] Parsed function_args: {function_args}")

                # Log what LLM returned for duplicate tracking
            else:
                # Fallback if no tool call or parsing failed
                self.logger.warning(
                    "[MasterAgent] No valid tool call arguments found. Falling back to simple search."
                )
                function_args = {
                    "topic_complexity": "simple",
                    "simple_topic": {
                        "query": query,
                        "aspect": "general information",
                        "rationale": "Fallback: No tool call or argument parsing failed",
                        "suggested_tool": "general_search",
                    },
                }

            # Process function call result based on topic complexity
            if function_args.get("topic_complexity") == "simple":
                simple_topic = function_args.get("simple_topic", {})
                query_text = simple_topic.get("query", query)
                suggested_tool = simple_topic.get("suggested_tool", "general_search")

                result = {
                    "topic_complexity": "simple",
                    "query": query_text,
                    "aspect": simple_topic.get("aspect", "general information"),
                    "rationale": simple_topic.get("rationale", ""),
                    "suggested_tool": suggested_tool,
                }
                # Log tool call for trajectory capture (non-invasive, never fails research)
                try:
                    if hasattr(self, "state") and self.state:
                        self.state.log_tool_call(
                            tool_name="decompose_research_topic",
                            params={"query": query, "knowledge_gap": knowledge_gap},
                            result_summary=f"simple topic: {result['query']}",
                        )
                        # Log complete execution step
                        self.state.log_execution_step(
                            step_type="llm_call",
                            action="decompose_query",
                            input_data={"query": query, "knowledge_gap": knowledge_gap},
                            output_data=result,
                            metadata={"provider": provider, "model": model},
                        )
                except Exception:
                    pass  # Logging errors should never break research
                return result
            elif function_args.get("topic_complexity") == "complex":
                complex_topic = function_args.get("complex_topic", {})
                self.logger.info(
                    f"[MasterAgent] Topic decomposed into {len(complex_topic.get('subtopics', []))} subtopics."
                )  # Added log
                result = {
                    "topic_complexity": "complex",
                    "main_query": complex_topic.get("main_query", query),
                    "main_tool": complex_topic.get("main_tool", "general_search"),
                    "subtopics": complex_topic.get("subtopics", []),
                }
                # Log tool call for trajectory capture (non-invasive, never fails research)
                try:
                    if hasattr(self, "state") and self.state:
                        self.state.log_tool_call(
                            tool_name="decompose_research_topic",
                            params={"query": query, "knowledge_gap": knowledge_gap},
                            result_summary=f"complex topic: {len(result['subtopics'])} subtopics",
                        )
                        # Log complete execution step
                        self.state.log_execution_step(
                            step_type="llm_call",
                            action="decompose_query",
                            input_data={"query": query, "knowledge_gap": knowledge_gap},
                            output_data=result,
                            metadata={
                                "provider": provider,
                                "model": model,
                                "num_subtopics": len(result["subtopics"]),
                            },
                        )
                except Exception:
                    pass  # Logging errors should never break research
                return result
            else:
                # Fallback if the function call didn't return expected format
                self.logger.warning(
                    f"[MasterAgent] Function call did not return expected format (complexity: {function_args.get('topic_complexity')}). Falling back to simple topic."
                )
                return {
                    "topic_complexity": "simple",
                    "query": query,
                    "aspect": "general information",
                    "rationale": "Fallback due to unexpected function call format",
                    "suggested_tool": "general_search",
                }

        except Exception as e:
            self.logger.error(f"[MasterAgent] Error in topic decomposition: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "topic_complexity": "simple",
                "query": query,
                "aspect": "general information",
                "rationale": "Error during execution",
                "suggested_tool": "general_search",
            }

    async def plan_research(
        self,
        query,
        knowledge_gap,
        research_loop_count,
        uploaded_knowledge=None,
        existing_tasks=None,
    ):
        """
        Create a research plan based on the topic decomposition.

        Args:
            query: The main research query or topic
            knowledge_gap: Additional context about knowledge gaps to address
            research_loop_count: Current iteration of research loop
            uploaded_knowledge: User-provided external knowledge (optional)
            existing_tasks: List of existing pending tasks to avoid duplicates (optional)

        Returns:
            Dict containing the research plan with tasks for specialized agents
        """
        # Decompose the topic (pass existing_tasks for LLM awareness)
        topic_info = await self.decompose_topic(
            query,
            knowledge_gap,
            research_loop_count,
            uploaded_knowledge,
            existing_tasks,
        )

        # Log decomposition in benchmark mode
        if hasattr(self, "state") and getattr(self.state, "benchmark_mode", False):
            print(f"[plan_research] Benchmark mode: Question decomposition")
            if "subtopics" in topic_info:
                print(f"[plan_research] Identified subtopics:")
                for i, subtopic in enumerate(topic_info.get("subtopics", [])):
                    print(f"  {i+1}. {subtopic}")
            if "key_entities" in topic_info:
                print(f"[plan_research] Key entities:")
                for entity in topic_info.get("key_entities", []):
                    print(f"  - {entity}")

        # Create a research plan based on the topic complexity
        if topic_info.get("topic_complexity") == "complex":
            # For complex topics, create a plan with subtasks
            subtasks = []

            # Add search tasks based on subtopics
            for i, subtopic in enumerate(topic_info.get("subtopics", [])):
                # Use subtopic name/aspect for concise description instead of full dict
                if isinstance(subtopic, dict):
                    desc = (
                        subtopic.get("name")
                        or subtopic.get("aspect")
                        or subtopic.get("query", "")
                    )
                else:
                    desc = str(subtopic)

                subtasks.append(
                    {
                        "index": i,
                        "type": "search",
                        "query": subtopic,
                        "description": desc,
                        "source_type": topic_info.get(
                            "recommended_sources", ["general"]
                        )[
                            min(
                                i,
                                len(topic_info.get("recommended_sources", ["general"]))
                                - 1,
                            )
                        ],
                    }
                )
            # Skip visualization tasks if in benchmark mode
            if not (
                hasattr(self, "state")
                and (
                    getattr(self.state, "benchmark_mode", False)
                    or getattr(self.state, "visualization_disabled", False)
                )
            ):
                # Add visualization tasks based on recommended visualizations
                for i, viz in enumerate(
                    topic_info.get("recommended_visualizations", [])
                ):
                    search_task_index = viz.get("search_task_index")
                    if search_task_index is not None:
                        viz_task = {
                            "index": len(subtasks),
                            "type": "visualization",
                            "description": viz.get(
                                "description",
                                f"Create visualization based on subtopic {search_task_index}",
                            ),
                            "visualization_type": viz.get("type", "default"),
                            "search_task_index": search_task_index,
                        }
                        subtasks.append(viz_task)

            # Create the plan
            plan = {
                "title": topic_info.get("title", query),
                "description": topic_info.get(
                    "description", "A comprehensive research plan"
                ),
                "subtasks": subtasks,
                "subtopics": topic_info.get("subtopics", []),
            }

        else:
            # For simple topics, create a basic plan with a single search task
            search_query = topic_info.get("query", query)
            suggested_tool = topic_info.get("suggested_tool", "general_search")

            # Map suggested_tool to source_type (for backwards compatibility)
            source_type_map = {
                "text2sql": "text2sql",
                "general_search": "general",
                "academic_search": "academic",
                "github_search": "github",
                "linkedin_search": "linkedin",
            }
            source_type = source_type_map.get(suggested_tool, "general")

            plan = {
                "title": topic_info.get("title", query),
                "description": topic_info.get("description", "A simple research plan"),
                "subtasks": [
                    {
                        "index": 0,
                        "type": "search",
                        "query": search_query,
                        "description": f"Research information about {search_query}",
                        "source_type": source_type,
                    }
                ],
                "subtopics": [search_query],
            }

            # Skip visualization tasks if in benchmark mode or visualization mode is disabled
            if not (
                hasattr(self, "state")
                and (
                    getattr(self.state, "benchmark_mode", False)
                    or getattr(self.state, "visualization_disabled", False)
                )
            ):
                # Add a generic visualization task for simple topics
                if topic_info.get("recommended_visualizations"):
                    viz = topic_info.get("recommended_visualizations")[0]
                    viz_task = {
                        "index": 1,
                        "type": "visualization",
                        "description": viz.get(
                            "description", "Create visualization of the topic"
                        ),
                        "visualization_type": viz.get("type", "default"),
                        "search_task_index": 0,  # Reference the single search task
                    }
                    plan["subtasks"].append(viz_task)

        # Log the plan
        subtask_count = len(plan.get("subtasks", []))
        search_task_count = len(
            [t for t in plan.get("subtasks", []) if t.get("type") == "search"]
        )
        visualization_task_count = len(
            [t for t in plan.get("subtasks", []) if t.get("type") == "visualization"]
        )

        self.logger.info(
            f"[MasterAgent] Created research plan with {subtask_count} subtasks:"
        )
        self.logger.info(f"[MasterAgent] - {search_task_count} search tasks")
        self.logger.info(
            f"[MasterAgent] - {visualization_task_count} visualization tasks"
        )

        if hasattr(self, "state") and getattr(self.state, "benchmark_mode", False):
            print(f"[plan_research] Benchmark mode: Research plan created")
            print(f"[plan_research] Plan contains {search_task_count} search tasks")
            if search_task_count > 0:
                print(f"[plan_research] Search queries:")
                for i, task in enumerate(
                    [t for t in plan.get("subtasks", []) if t.get("type") == "search"]
                ):
                    print(
                        f"  {i+1}. \"{task.get('query')}\" (source: {task.get('source_type', 'general')})"
                    )

        return plan


    async def execute_research(self, state, callbacks=None, database_info=None):
        """
        Execute research tasks based on the research plan.
        """
        # Store state for access by all methods in this execution
        self.state = state
        self.database_info = database_info

        try:
            # Always get query, knowledge_gap, and loop_count from state for planning
            query = (
                state.search_query
                if hasattr(state, "search_query") and state.search_query
                else state.research_topic
            )
            knowledge_gap = getattr(state, "knowledge_gap", "")
            research_loop_count = getattr(state, "research_loop_count", 0)
            uploaded_knowledge = getattr(state, "uploaded_knowledge", None)

            self.logger.info(
                f"[MasterAgent.execute_research] Planning with query: '{query[:100]}...'"
            )

            research_plan = None
            if (
                hasattr(state, "research_plan")
                and state.research_plan
                and research_loop_count == 0
            ):
                # Only use existing plan if it's the very first loop AND a plan was somehow pre-loaded
                research_plan = state.research_plan
                print(
                    "[execute_research] Using pre-existing research plan for initial loop"
                )

            # If no pre-existing plan for loop 0, or if it's a subsequent loop, always (re)plan.
            if not research_plan or research_loop_count > 0:
                if research_loop_count > 0:
                    print(
                        f"[execute_research] Re-planning research for loop {research_loop_count}."
                    )
                else:
                    print("[execute_research] Creating initial research plan.")

                research_plan = await self.plan_research(
                    query, knowledge_gap, research_loop_count, uploaded_knowledge
                )

                # Add/update the research plan in state if possible
                try:
                    state.research_plan = research_plan
                    print("[execute_research] Updated research plan in state")
                except (ValueError, AttributeError):
                    print(
                        "[execute_research] Unable to update research plan in state, using local copy"
                    )

            # Execute search tasks
            search_results_list = await self._execute_search_tasks(
                research_plan, state
            )

            successful_search_indices = [
                i
                for i, result in enumerate(search_results_list)
                if result.get("success", False)
            ]

            self.logger.info(
                f"[MasterAgent] Completed {len(successful_search_indices)} successful search tasks out of {len(search_results_list)}."
            )

            # Prepare the return dictionary
            return_value = {
                "web_research_results": search_results_list,
                "visualizations_generated_this_loop": [],
                "base64_images_generated_this_loop": [],
                "code_snippets_generated_this_loop": [],
            }
            # Preserve essential state fields that might have been updated by planning
            if hasattr(state, "research_plan"):
                return_value["research_plan"] = state.research_plan

            return return_value

        except Exception as e:
            self.logger.error(f"[MasterAgent] Error in research execution: {str(e)}")
            self.logger.error(f"[MasterAgent] {traceback.format_exc()}")
            return {
                "web_research_results": getattr(
                    self, "search_results_list", []
                ),
                "error": f"MasterAgent execution failed: {str(e)}",
                "visualizations_generated_this_loop": [],
                "base64_images_generated_this_loop": [],
                "code_snippets_generated_this_loop": [],
            }



    async def _execute_search_tasks(self, research_plan, state):
        """
        Execute search tasks from the research plan.

        Args:
            research_plan: Dictionary containing the research plan
            state: Current state object

        Returns:
            list: Results from search tasks
        """
        # Start timer for performance logging
        search_start_time = time.time()

        # State is accessible via self.state set in execute_research

        # Track if we're in benchmark mode
        benchmark_mode = getattr(state, "benchmark_mode", False)
        if benchmark_mode:
            print(f"[_execute_search_tasks] Executing search tasks in benchmark mode")

        visualization_disabled = getattr(state, "visualization_disabled", False)
        if visualization_disabled:
            print(
                f"[_execute_search_tasks] Executing search tasks in visualization disabled mode"
            )

        # Initialize specialized agents and tools
        search_agent = SearchAgent(self.config, database_info=self.database_info)

        # Initialize task results list to track what's completed
        task_results = []

        # If there are no research tasks, exit early
        if "subtasks" not in research_plan or not research_plan["subtasks"]:
            print("No search tasks found in research plan.")
            return task_results

        # Loop through each search task
        for task in research_plan["subtasks"]:
            if task.get("type") == "search":
                task_index = task.get("index", 0)
                task_query = task.get("query", {})

                # Handle both string and dict query formats
                if isinstance(task_query, str):
                    # If query is a string, use it directly
                    query_text = task_query
                    tool_name = task.get("source_type", "general_search")
                    # Map source_type to tool names
                    if tool_name == "general":
                        tool_name = "general_search"
                    elif tool_name == "academic":
                        tool_name = "academic_search"
                    elif tool_name == "github":
                        tool_name = "github_search"
                    elif tool_name == "linkedin":
                        tool_name = "linkedin_search"
                    elif tool_name == "text2sql":
                        tool_name = "text2sql"
                    else:
                        tool_name = "general_search"  # Default fallback

                    # WORKAROUND: If database_info is available and query contains SQL, use text2sql
                    if (
                        hasattr(self, "database_info")
                        and self.database_info
                        and (
                            "SELECT" in query_text.upper()
                            or "FROM" in query_text.upper()
                            or "JOIN" in query_text.upper()
                        )
                    ):
                        self.logger.info(
                            f"[MasterAgent._execute_search_tasks] Detected SQL query, switching to text2sql tool"
                        )
                        tool_name = "text2sql"
                elif isinstance(task_query, dict):
                    # If query is a dict, extract the query text and tool
                    query_text = task_query.get("query", "")
                    tool_name = task_query.get("suggested_tool", "general_search")
                else:
                    # Fallback for unexpected format
                    query_text = str(task_query)
                    tool_name = "general_search"



                # Log this to help with tracing
                print(f"Executing search task {task_index} with query: '{query_text}'")
                print(f"Using search tool: {tool_name}")

                try:
                    # Execute the search based on the tool_name
                    search_result = None
                    if tool_name == "general_search":
                        search_result = await search_agent.general_search(query_text)
                    elif tool_name == "academic_search":
                        search_result = await search_agent.academic_search(query_text)
                    elif tool_name == "github_search":
                        search_result = await search_agent.github_search(query_text)
                    elif tool_name == "linkedin_search":
                        search_result = await search_agent.linkedin_search(query_text)
                    elif tool_name == "text2sql":
                        # Handle text2sql tool execution
                        search_result = await search_agent.text2sql_search(query_text)
                    else:
                        # Default to general search if tool is unknown
                        search_result = await search_agent.general_search(query_text)

                    # Log search tool call for trajectory capture (non-invasive, never fails research)
                    try:
                        if hasattr(self, "state") and self.state:
                            num_sources = 0
                            sources_list = []
                            if isinstance(search_result, dict):
                                if "formatted_sources" in search_result:
                                    sources_list = search_result.get(
                                        "formatted_sources", []
                                    )
                                    num_sources = len(sources_list)
                                elif "sources" in search_result:
                                    sources_list = search_result.get("sources", [])
                                    num_sources = len(sources_list)

                            self.state.log_tool_call(
                                tool_name=tool_name,
                                params={"query": query_text},
                                result_summary=f"{num_sources} sources",
                            )

                            # Log complete execution step
                            self.state.log_execution_step(
                                step_type="tool_execution",
                                action=tool_name,
                                input_data={"query": query_text},
                                output_data={
                                    "num_sources": num_sources,
                                    "sources": (
                                        sources_list[:10]
                                        if len(sources_list) > 10
                                        else sources_list
                                    ),  # First 10 sources
                                },
                                metadata={"total_sources": num_sources},
                            )
                    except Exception:
                        pass  # Logging errors should never break research

                    # Extract sources from search_result for easy access in results
                    sources = []

                    # Handle different search_result formats
                    if isinstance(search_result, dict):
                        # Format from newer search tools that return a dict with 'formatted_sources'
                        if "formatted_sources" in search_result:
                            formatted_sources = search_result.get(
                                "formatted_sources", []
                            )
                            # Extract source details from formatted_sources
                            for src in formatted_sources:
                                if isinstance(src, str) and " : " in src:
                                    # Parse title and URL from source string (format: "title : url")
                                    parts = src.split(" : ", 1)
                                    if len(parts) == 2:
                                        title, url = parts
                                        sources.append({"title": title, "url": url})

                        # If search_result directly contains 'sources' key with structured data
                        if "sources" in search_result and isinstance(
                            search_result["sources"], list
                        ):
                            for src in search_result["sources"]:
                                if (
                                    isinstance(src, dict)
                                    and "title" in src
                                    and "url" in src
                                ):
                                    sources.append(
                                        {"title": src["title"], "url": src["url"]}
                                    )

                        # Extract content from search_result
                        if "content" in search_result:
                            content = search_result["content"]
                        elif "raw_contents" in search_result:
                            # Join multiple raw contents into a single string
                            raw_contents = search_result.get("raw_contents", [])
                            if isinstance(raw_contents, list):
                                content = "\n\n".join(
                                    [str(item) for item in raw_contents if item]
                                )
                            else:
                                content = str(raw_contents)
                        else:
                            # Fallback: convert the entire result to a string
                            content = str(search_result)
                    else:
                        # Fallback for unexpected search_result type
                        content = str(search_result)
                        # No sources can be extracted in this case

                    # Build the result object with both content and sources
                    task_result = {
                        "index": task_index,
                        "query": query_text,  # Store the actual query text, not the original task_query
                        "success": True,
                        "content": content,
                        "sources": sources,
                        "error": None,
                        "tool_used": tool_name,  # Include the tool used for tracking
                    }

                    print(
                        f"Search task {task_index} completed with {len(sources)} sources"
                    )
                    task_results.append(task_result)



                except Exception as e:
                    # Log the error and continue with other tasks
                    print(f"Error in search task {task_index}: {str(e)}")
                    error_result = {
                        "index": task_index,
                        "query": query_text,  # Store the actual query text, not the original task_query
                        "success": False,
                        "content": "",
                        "sources": [],
                        "error": str(e),
                        "tool_used": tool_name,  # Include the tool used for tracking even on error
                    }
                    task_results.append(error_result)

        # Log performance
        search_end_time = time.time()
        search_duration = search_end_time - search_start_time
        print(f"All search tasks completed in {search_duration:.2f} seconds")
        print(f"Total search tasks completed: {len(task_results)}")

        return task_results



class SearchAgent:
    """
    Specialized agent for executing search queries.

    This agent is responsible for executing specific search tasks
    using the appropriate search tool based on the subtopic.
    """

    def __init__(self, config=None, database_info=None):
        """
        Initialize the Search Agent.

        Args:
            config: Configuration object containing search settings
            database_info: Database context information for text2sql queries
        """
        self.config = config
        self.database_info = database_info
        self.logger = logging.getLogger(__name__)

    async def general_search(self, query):
        """Execute a general search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.general_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="general_search", params={"query": query, "top_k": 5}
        )

    async def academic_search(self, query):
        """Execute an academic search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.academic_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="academic_search", params={"query": query, "top_k": 5}
        )

    async def github_search(self, query):
        """Execute a GitHub search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.github_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="github_search", params={"query": query, "top_k": 5}
        )

    async def linkedin_search(self, query):
        """Execute a LinkedIn search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.linkedin_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="linkedin_search", params={"query": query, "top_k": 5}
        )

    async def text2sql_search(self, query):
        """Execute a text2sql query"""
        # Import the global text2sql tool instance from the database router
        from routers.database import text2sql_tool

        self.logger.info(f"SearchAgent.text2sql_search called with query: {query}")

        # Get database context from the agent's database_info
        db_id = None
        if hasattr(self, "database_info") and self.database_info:
            # Use the first database from the database_info
            if isinstance(self.database_info, list) and len(self.database_info) > 0:
                db_id = self.database_info[0].get("database_id")
                self.logger.info(f"Using database_id from database_info: {db_id}")

        # Use the global text2sql tool instance that has access to uploaded databases
        try:
            result = text2sql_tool._run(query, db_id=db_id)

            self.logger.info(f"text2sql_tool._run returned: {result}")

            # Format the result to match the expected search result format
            if "error" in result:
                return {
                    "index": 0,
                    "query": query,
                    "success": False,
                    "content": f"Error: {result['error']}",
                    "sources": [],
                    "error": result["error"],
                }
            else:
                # Format the successful result with HTML for better display
                content = "<div class='database-results' style='margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;'>\n"
                content += "<h3 style='color: #2c3e50; margin-top: 0;'>📊 Database Analysis Results</h3>\n"
                content += (
                    f"<p style='color: #555;'><strong>Query:</strong> {query}</p>\n"
                )

                if "sql" in result:
                    content += "<div style='background-color: #f4f4f4; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 15px 0; font-family: monospace; font-size: 14px; overflow-x: auto;'>\n"
                    content += "<h4 style='margin-top: 0; color: #555;'>Generated SQL Query:</h4>\n"
                    content += f"<pre style='margin: 0; white-space: pre-wrap;'><code>{result['sql']}</code></pre>\n"
                    content += "</div>\n"

                if "results" in result and result["results"]:
                    results_data = result["results"]
                    if results_data.get("type") == "select" and results_data.get(
                        "rows"
                    ):
                        content += f"<h4 style='color: #2c3e50; margin-top: 20px;'>📈 Query Results ({results_data.get('row_count', 0)} rows)</h4>\n"

                        # Create HTML table with inline styles
                        columns = results_data.get("columns", [])
                        if columns:
                            content += "<table style='width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); background-color: white;'>\n"
                            content += "<thead><tr>\n"
                            for col in columns:
                                content += f"<th style='background-color: #3498db; color: white; padding: 12px; text-align: left; font-weight: bold; border-bottom: 2px solid #2980b9;'>{col}</th>\n"
                            content += "</tr></thead>\n"
                            content += "<tbody>\n"

                            # Add data rows
                            for idx, row in enumerate(results_data.get("rows", [])):
                                bg_color = "#f8f9fa" if idx % 2 == 0 else "white"
                                content += (
                                    f"<tr style='background-color: {bg_color};'>\n"
                                )
                                for col in columns:
                                    value = row.get(col, "")
                                    # Format numbers nicely
                                    if isinstance(value, float):
                                        value = f"{value:.2f}"
                                    content += f"<td style='padding: 10px; border-bottom: 1px solid #ddd;'>{value}</td>\n"
                                content += "</tr>\n"

                            content += "</tbody>\n"
                            content += "</table>\n"
                    else:
                        content += f"<p><strong>Results:</strong> {results_data}</p>\n"

                if "database" in result:
                    content += f"<p style='color: #555; margin-top: 15px;'><strong>📁 Source Database:</strong> {result['database']}</p>\n"

                if "executed_at" in result:
                    content += f"<p style='color: #888; font-size: 0.9em;'><strong>⏰ Executed at:</strong> {result['executed_at']}</p>\n"

                content += "</div>\n"

                self.logger.info(f"text2sql formatted content: {content[:200]}...")

                return {
                    "index": 0,
                    "query": query,
                    "success": True,
                    "content": content,
                    "sources": [
                        {
                            "title": f'Database Query Results - {result.get("database", "Unknown Database")}',
                            "url": f'database://{result.get("database", "unknown")}',
                            "snippet": f'SQL: {result.get("sql", "N/A")}\nResults: {len(result.get("results", {}).get("rows", []))} rows returned',
                            "source_type": "database",
                        }
                    ],
                    "error": None,
                }

        except Exception as e:
            self.logger.error(f"Error in text2sql_search: {e}")
            return {
                "index": 0,
                "query": query,
                "success": False,
                "content": f"Error executing text2sql query: {str(e)}",
                "sources": [],
                "error": str(e),
            }

    async def execute(self, subtask, tool_executor=None):
        """
        Execute a search for a specific subtask asynchronously using the provided tool executor.
        This method directly uses the query and tool name from the subtask description
        without making an additional LLM call.

        Args:
            subtask: Dict containing task details (query, tool, name, aspect, etc.)
            tool_executor: An initialized ToolExecutor instance.

        Returns:
            Dict containing the search results or an error.
        """
        # Import necessary components
        from src.graph import ToolRegistry, ToolExecutor  # Keep ToolExecutor import
        import traceback

        # Ensure tool_executor is provided (as it's essential now)
        if tool_executor is None:
            self.logger.error("[SearchAgent] ToolExecutor instance must be provided.")
            # Consider raising an error or handling appropriately
            # For now, let's try creating one, but this indicates a potential design issue
            try:
                from src.graph import ToolRegistry

                tool_registry = ToolRegistry(self.config)
                tool_executor = ToolExecutor(tool_registry)
                self.logger.warning(
                    "[SearchAgent] Created a default ToolExecutor instance as none was provided."
                )
            except Exception as te_err:
                self.logger.error(
                    f"[SearchAgent] Failed to create default ToolExecutor: {te_err}"
                )
                return {
                    "error": "ToolExecutor instance required but not provided and creation failed."
                }

        # Directly extract parameters from the subtask dictionary
        query = subtask.get("query")
        # Default to general_search if not specified in subtask
        tool_name = subtask.get("suggested_tool", subtask.get("tool", "general_search"))
        subtask_name = subtask.get("name", "Unnamed subtask")

        # Validate extracted parameters
        if not query:
            self.logger.error(
                f"[SearchAgent] Subtask '{subtask_name}' is missing a query."
            )
            return {"error": f"Subtask '{subtask_name}' missing query."}
        if not tool_name:
            self.logger.error(
                f"[SearchAgent] Subtask '{subtask_name}' is missing a tool name."
            )
            # Defaulting again just in case, but log the error
            tool_name = "general_search"

        self.logger.info(
            f"[SearchAgent] Directly executing tool '{tool_name}' for subtask '{subtask_name}' with query: '{query}'"
        )

        try:
            # --- LLM Call Removed ---
            # No need to call LLM again to determine parameters; use directly from subtask.

            # Define standard parameters (can be extended if MasterAgent provides more)
            # Example: if MasterAgent decided top_k based on decomposition:
            # top_k = subtask.get("parameters", {}).get("top_k", 5)
            params = {
                "query": query,
                "top_k": 5,  # Using a default, could be passed in subtask if needed
            }

            # Execute search with the specified tool asynchronously using ToolExecutor
            self.logger.info(
                f"[SearchAgent] Calling tool_executor.execute_tool with tool_name='{tool_name}', params={params}"
            )
            search_result = await tool_executor.execute_tool(
                tool_name=tool_name, params=params
            )

            # Ensure search_result is a dictionary (ToolExecutor should ideally ensure this)
            if not isinstance(search_result, dict):
                self.logger.warning(
                    f"[SearchAgent] ToolExecutor result is not a dictionary: {type(search_result)}. Converting."
                )
                search_result = {
                    "formatted_sources": str(search_result),
                    "search_string": query,
                    "domains": [],
                }

            # Add metadata to the result
            search_result["subtask"] = subtask  # Include original subtask for context
            search_result["tool_used"] = tool_name  # Log the tool used
            # No "tool_used_llm" as LLM didn't choose the tool here

            self.logger.info(
                f"[SearchAgent] Successfully executed tool '{tool_name}' for subtask '{subtask_name}'"
            )
            return search_result

        except Exception as e:
            self.logger.error(
                f"[SearchAgent] Error executing tool '{tool_name}' for subtask '{subtask_name}': {str(e)}"
            )
            self.logger.error(traceback.format_exc())

            # Return error result
            return {
                "error": str(e),
                "formatted_sources": f"Error executing search: {str(e)}",
                "search_string": query,
                "domains": [],
                "subtask": subtask,  # Include original subtask even on error
                "tool_used": tool_name,
            }

