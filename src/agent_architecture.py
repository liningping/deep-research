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
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

try:
    from scipy.stats import beta as beta_dist
except ImportError:  # graceful fallback – scipy may not be installed
    import random
    class _BetaFallback:
        @staticmethod
        def rvs(a, b):
            # Very rough approximation: mean + small jitter
            return a / (a + b) + random.gauss(0, 0.05)
    beta_dist = _BetaFallback()

# Internal imports (Assumed available in the environment)
from llm_clients import get_async_llm_client
from src.tools.executor import ToolExecutor
from src.graph import ToolRegistry
from src.prompts import query_writer_instructions
from src.tools.tool_schema import TOPIC_DECOMPOSITION_FUNCTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thompson Sampling Bandit Scheduler
# ---------------------------------------------------------------------------

class ThompsonBanditScheduler:
    """
    Multi-armed bandit scheduler using Thompson Sampling (Beta distribution).

    Each subtask is treated as an "arm". On each round:
      1. Draw a Beta(α, β) sample for every arm.
      2. Select the arm with the highest sample.
      3. Execute it, evaluate quality → reward ∈ [0, 1].
      4. Update α += reward, β += (1 - reward).

    Subtask identity (the arm key) is the `query` field of the subtask dict.
    """

    def __init__(self, subtasks: List[Dict]):
        self.subtasks: List[Dict] = list(subtasks)  # mutable – arms may be added
        # Beta distribution params: initialise with (α=1, β=1) → Uniform prior
        self.distributions: Dict[str, Dict[str, float]] = {
            self._key(t): {"alpha": 1.0, "beta": 1.0} for t in subtasks
        }
        self.satisfaction: Dict[str, float] = {self._key(t): 0.0 for t in subtasks}
        self.n: Dict[str, int] = {self._key(t): 0 for t in subtasks}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(subtask: Dict) -> str:
        """Canonical identifier for a subtask arm."""
        return subtask.get("query", str(subtask))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_subtask(self) -> Optional[Dict]:
        """Thompson-sample all arms and return the subtask with the highest draw."""
        if not self.subtasks:
            return None
        samples = {
            self._key(t): beta_dist.rvs(
                self.distributions[self._key(t)]["alpha"],
                self.distributions[self._key(t)]["beta"]
            )
            for t in self.subtasks
        }
        best_key = max(samples, key=samples.get)
        # Return the first subtask dict whose key matches
        for t in self.subtasks:
            if self._key(t) == best_key:
                return t
        return self.subtasks[0]  # fallback (should not happen)

    def update(self, subtask: Dict, reward: float):
        """Update Beta distribution parameters after observing `reward`."""
        key = self._key(subtask)
        if key not in self.distributions:
            return
        self.distributions[key]["alpha"] += reward
        self.distributions[key]["beta"] += max(0.0, 1.0 - reward)
        self.satisfaction[key] = self.satisfaction.get(key, 0.0) + reward
        self.n[key] = self.n.get(key, 0) + 1

    def add_subtask(self, subtask: Dict, warm_start_from: Optional[Dict] = None):
        """
        Dynamically add a new arm (e.g., a refined/evolved subtask).
        If `warm_start_from` is given the new arm inherits its parent's
        distribution (with slight exploration boost) rather than the flat prior.
        """
        key = self._key(subtask)
        if key in self.distributions:
            return  # already registered
        if warm_start_from is not None:
            parent_key = self._key(warm_start_from)
            parent = self.distributions.get(parent_key, {"alpha": 1.0, "beta": 1.0})
            alpha = parent["alpha"] * 1.2   # mild exploration boost
            beta_v = parent["beta"] * 0.8
        else:
            alpha, beta_v = 2.0, 2.0  # slightly more informed than flat prior
        self.distributions[key] = {"alpha": alpha, "beta": beta_v}
        self.satisfaction[key] = 0.0
        self.n[key] = 0
        self.subtasks.append(subtask)
        logger.info(f"🎰 [Bandit] Added new arm: '{key}' (α={alpha:.2f}, β={beta_v:.2f})")

    def all_executed(self) -> bool:
        """True when every arm has been pulled at least once."""
        return all(self.n.get(self._key(t), 0) > 0 for t in self.subtasks)

    def best_subtask(self) -> Optional[Dict]:
        """Return the arm with the highest accumulated satisfaction."""
        if not self.subtasks:
            return None
        return max(self.subtasks, key=lambda t: self.satisfaction.get(self._key(t), 0.0))


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

    async def execute_research(self, state):
        """
        Main Execution Loop.
        Integrates Todo Management -> Planning -> Bandit-driven Execution -> Feedback.

        Instead of blindly executing all subtasks, a Thompson Sampling bandit
        (ThompsonBanditScheduler) iteratively selects the most promising
        subtask to run next, evaluates its result, and updates the arm
        distributions accordingly.
        """
        self.state = state
        loop_count = getattr(state, "research_loop_count", 0)
        query = getattr(state, "search_query", state.research_topic)

        self.logger.info(f"🚀 [Execution] Starting Loop {loop_count} for query: '{query}'")

        # --- STEP 1: INITIALIZE TODO (Loop 0 only) ---
        if loop_count == 0 and hasattr(state, "steering_todo"):
            await self._create_initial_research_plan(query, state)

        # --- STEP 2: DETERMINE STRATEGY & BUILD SUBTASK LIST ---
        research_plan = None
        subtasks = []

        # Check for Pending Tasks in Todo Manager
        pending_tasks = []
        if hasattr(state, "steering_todo") and state.steering_todo is not None:
            pending_tasks = state.steering_todo.get_pending_tasks()
            pending_tasks.sort(key=lambda t: t.priority, reverse=True)

        if pending_tasks:
            # Strategy A: Task-Driven
            top_tasks = pending_tasks[:6]  # allow bandit to pick among top 6
            self.logger.info(f"🎯 [Strategy] Bandit will choose among top {len(top_tasks)} pending tasks.")
            for t in top_tasks:
                state.steering_todo.mark_task_in_progress(t.id)
            research_plan = await self.plan_research_from_tasks(query, top_tasks, state)
            subtasks = research_plan.get("subtasks", [])
        else:
            # Strategy B: Exploration / Decomposition
            self.logger.info("🧭 [Strategy] No pending tasks – performing standard decomposition.")
            decomposition = await self.decompose_topic(query, getattr(state, "knowledge_gap", ""))
            if decomposition["topic_complexity"] == "complex":
                for sub in decomposition.get("subtopics", []):
                    subtasks.append({"type": "search", "query": sub, "source_type": "general"})
            else:
                subtasks.append({"type": "search", "query": decomposition["query"], "source_type": "general"})
            research_plan = {"topic_complexity": decomposition["topic_complexity"], "subtasks": subtasks}

        # --- STEP 3: BANDIT-DRIVEN EXECUTION ---
        # Cap iterations: at most 2× the number of arms (ensures every arm is
        # visited at least once while still allowing the bandit to exploit good arms).
        max_steps = max(len(subtasks) * 2, 6)
        search_results = await self._run_bandit_loop(subtasks, state, max_steps=max_steps)

        # --- STEP 4: UPDATE TODO & FEEDBACK ---
        if hasattr(state, "steering_todo") and state.steering_todo is not None and \
                research_plan.get("topic_complexity") == "task_driven":
            await self._update_todo_based_on_results(search_results, state)

        return {
            "web_research_results": search_results,
            "research_plan": research_plan
        }

    # ------------------------------------------------------------------
    # Thompson Bandit Loop
    # ------------------------------------------------------------------

    async def _run_bandit_loop(
        self,
        subtasks: List[Dict],
        state,
        max_steps: int = 10,
    ) -> List[Dict]:
        """
        Core bandit loop (Algorithm 1 – Thompson Sampling).

        For each step:
          1. Sample all Beta arms → select highest draw.
          2. Execute the chosen subtask via SearchAgent.
          3. Evaluate quality with LLM → reward ∈ [0, 1].
          4. Update the arm's Beta distribution.
          5. If LLM flags modification needed, add an evolved variant arm.

        Args:
            subtasks:  List of subtask dicts (must have "query" key).
            state:     Research state (used for dedup tracking).
            max_steps: Maximum bandit iterations.

        Returns:
            Aggregated list of search result dicts.
        """
        if not subtasks:
            return []

        scheduler = ThompsonBanditScheduler(subtasks)
        search_agent = SearchAgent(self.config)
        all_results: List[Dict] = []

        self.logger.info(
            f"🎰 [Bandit] Starting Thompson Sampling loop | arms={len(subtasks)} | max_steps={max_steps}"
        )

        for step in range(max_steps):
            # 1. Select best arm via Thompson sample
            chosen = scheduler.sample_subtask()
            if chosen is None:
                break

            query = chosen.get("query", "")

            # Deduplication guard (steering_todo tracks executed queries)
            if hasattr(state, "steering_todo") and state.steering_todo is not None:
                if state.steering_todo.is_query_duplicate(query):
                    self.logger.info(f"⏭️  [Bandit] Step {step}: duplicate query skipped – '{query}'")
                    # Penalise arm slightly so it is deprioritised
                    scheduler.update(chosen, reward=0.1)
                    # If all remaining arms are duplicates, stop early
                    if scheduler.all_executed():
                        break
                    continue

            # 2. Execute search
            try:
                res = await search_agent.execute_task(chosen)
                if hasattr(state, "steering_todo") and state.steering_todo is not None:
                    state.steering_todo.mark_query_executed(query)
            except Exception as exc:
                self.logger.error(f"❌ [Bandit] Search failed for '{query}': {exc}")
                res = {"success": False, "content": "", "sources": [], "error": str(exc)}

            result_entry = {
                "query": query,
                "success": res.get("success", False),
                "content": res.get("content", ""),
                "sources": res.get("sources", []),
                "error": res.get("error"),
                "completes_task_id": chosen.get("completes_task_id"),
            }
            all_results.append(result_entry)

            # 3. Evaluate quality → reward
            reward, needs_modification = await self._evaluate_subtask(chosen, res)

            log_icon = "✅" if res.get("success") else "⚠️"
            self.logger.info(
                f"{log_icon} [Bandit] Step {step}: '{query}' | "
                f"reward={reward:.2f} | sources={len(res.get('sources', []))} | "
                f"modify={needs_modification}"
            )

            # 4. Update arm distribution
            scheduler.update(chosen, reward)

            # 5. Evolve arm if LLM suggests modification
            if needs_modification:
                evolved = self._evolve_subtask(chosen)
                scheduler.add_subtask(evolved, warm_start_from=chosen)
                self.logger.info(f"🔬 [Bandit] Evolved arm added: '{evolved.get('query')}'")

            # Early exit: once every original arm has been tried, let the
            # scheduler exploit until all new arms are also exhausted.
            if scheduler.all_executed() and step >= len(subtasks) - 1:
                self.logger.info(f"🏁 [Bandit] All arms executed after {step + 1} steps – stopping early.")
                break

        self.logger.info(
            f"🎰 [Bandit] Loop complete | steps={min(step+1, max_steps)} | results={len(all_results)}"
        )
        return all_results

    async def _evaluate_subtask(
        self,
        subtask: Dict,
        result: Dict,
    ) -> Tuple[float, bool]:
        """
        LLM-based quality evaluator for a completed subtask.

        Prompts the LLM with the query and a summary of results, asking for:
          - A quality score in [0, 1]
          - Whether the subtask needs refinement / modification

        Returns:
            (reward: float, needs_modification: bool)
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.utils import clean_json_response
        from src.nodes.utils import get_configurable

        try:
            configurable = get_configurable(self.config)
            
            provider_attr = getattr(self.state, "llm_provider", None) if self.state else None
            provider = provider_attr or configurable.llm_provider
            if not isinstance(provider, str): provider = provider.value
            
            model_attr = getattr(self.state, "llm_model", None) if self.state else None
            model = model_attr or configurable.llm_model or "gemini-2.5-pro"
            
            llm = await get_async_llm_client(provider, model)
            
            query = subtask.get("query", "")
            content_preview = str(result.get("content", ""))[:800]
            source_count = len(result.get("sources", []))

            eval_prompt = (
                f"You are evaluating the quality of a research search step.\n\n"
                f"Search Query: {query}\n"
                f"Sources Found: {source_count}\n"
                f"Content Preview:\n{content_preview}\n\n"
                f"Rate the result on two dimensions:\n"
                f"1. quality_score: float in [0.0, 1.0] reflecting relevance, depth, and source count.\n"
                f"   - 0.0 = no useful content; 0.5 = partial; 1.0 = excellent.\n"
                f"2. needs_modification: boolean – true if the query should be refined or split.\n\n"
                f"Respond with ONLY a valid JSON object, e.g. "
                f'{{"quality_score": 0.75, "needs_modification": false}}'
            )

            # Using ainvoke mapped to the LangChain interface
            response = await llm.ainvoke([
                SystemMessage(content="You evaluate research search quality. Respond only with JSON."),
                HumanMessage(content=eval_prompt)
            ])

            content = response.content if hasattr(response, "content") else str(response)
            parsed = clean_json_response(content)
            
            score = float(parsed.get("quality_score", 0.5))
            score = max(0.0, min(1.0, score))  # clamp
            needs_mod = bool(parsed.get("needs_modification", False))
            return score, needs_mod

        except Exception as exc:
            self.logger.warning(f"⚠️  [Bandit] Evaluator error for '{subtask.get('query')}': {exc}")
            # Fallback: heuristic reward based on source count
            source_count = len(result.get("sources", []))
            reward = min(1.0, source_count / 5.0) if result.get("success") else 0.0
            return reward, False

    @staticmethod
    def _evolve_subtask(subtask: Dict) -> Dict:
        """
        Create a refined variant of the given subtask when the LLM decides
        the original query needs modification. The variant appends a
        clarifying suffix to the query so it becomes a distinct arm.
        """
        original_query = subtask.get("query", "")
        evolved = dict(subtask)  # shallow copy preserves all metadata
        evolved["query"] = f"{original_query} – detailed analysis"
        return evolved

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