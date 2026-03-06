import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from deep_research.state_scope import AgentState
from deep_research.search_agent import SearchAgent
from deep_research.prompts import (
    bandit_eval_system_prompt,
    bandit_eval_user_prompt,
    bandit_evolve_system_prompt,
    bandit_evolve_user_prompt,
)

try:
    from scipy.stats import beta as beta_dist
except ImportError:
    import random
    class _BetaFallback:
        @staticmethod
        def rvs(a, b):
            return a / (a + b) + random.gauss(0, 0.05)
    beta_dist = _BetaFallback()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model
# ---------------------------------------------------------------------------

_model_name = os.environ["OPENAI_MODEL"]
_base_url = os.environ["OPENAI_BASE_URL"]
model = init_chat_model(model=f"openai:{_model_name}", base_url=_base_url)


# ---------------------------------------------------------------------------
# Evaluator structured output schema (aligned to reflection_instructions)
# ---------------------------------------------------------------------------

class _SubtaskEval(BaseModel):
    """Multi-dimensional quality evaluation for a single bandit search step.
    """
    coverage_pct: float = Field(
        description="Coverage completeness 0-100: how thoroughly the result addresses the query subtopic."
    )
    source_tier: int = Field(
        description="Best source quality tier (1=official/academic, 2=reputable secondary, 3=general)."
    )
    evidence_count: int = Field(
        description="Number of distinct specific data points, examples, or measurements found."
    )
    is_specific: bool = Field(
        description="True if result has specific names, metrics, or quantitative evidence—not just broad overviews."
    )
    fills_existing_gap: bool = Field(
        description=(
            "True if the result covers information NOT already present in the existing draft/notes. "
            "False if it duplicates content already known. "
            "If no existing draft is provided, default to True."
        )
    )
    needs_modification: bool = Field(
        description="True if the query should be refined or split for better results."
    )
    can_denoise_draft: bool = Field(
        description="True if the accumulated research completely answers the question and permits denoising."
    )

_eval_model = model.with_structured_output(_SubtaskEval)


# ---------------------------------------------------------------------------
# Tool → source_type mapping
# ---------------------------------------------------------------------------

_TOOL_TO_SOURCE: Dict[str, str] = {
    "general_search":  "general",
    "academic_search": "academic",
    "github_search":   "github",
    "linkedin_search": "linkedin",
}


def _tool_to_source(suggested_tool: str) -> str:
    return _TOOL_TO_SOURCE.get(suggested_tool, "general")


# ---------------------------------------------------------------------------
# Thompson Sampling Bandit Scheduler
# ---------------------------------------------------------------------------

class ThompsonBanditScheduler:
    """
    Multi-armed bandit scheduler using Thompson Sampling (Beta distribution).
    """

    def __init__(self, subtasks: List[Dict]):
        self.subtasks: List[Dict] = list(subtasks)
        self.distributions: Dict[str, Dict[str, float]] = {
            self._key(t): {"alpha": 1.0, "beta": 1.0} for t in subtasks
        }
        self.satisfaction: Dict[str, float] = {self._key(t): 0.0 for t in subtasks}
        self.n: Dict[str, int] = {self._key(t): 0 for t in subtasks}

    @staticmethod
    def _key(subtask: Dict) -> str:
        return subtask.get("query", str(subtask))

    def sample_subtask(self) -> Optional[Dict]:
        if not self.subtasks:
            return None
        samples = {
            self._key(t): beta_dist.rvs(
                self.distributions[self._key(t)]["alpha"],
                self.distributions[self._key(t)]["beta"],
            )
            for t in self.subtasks
        }
        best_key = max(samples, key=samples.get)
        for t in self.subtasks:
            if self._key(t) == best_key:
                return t
        return self.subtasks[0]

    def update(self, subtask: Dict, reward: float):
        key = self._key(subtask)
        if key not in self.distributions:
            return
        self.distributions[key]["alpha"] += reward
        self.distributions[key]["beta"] += max(0.0, 1.0 - reward)
        self.satisfaction[key] = self.satisfaction.get(key, 0.0) + reward
        self.n[key] = self.n.get(key, 0) + 1

    def add_subtask(self, subtask: Dict, warm_start_from: Optional[Dict] = None):
        key = self._key(subtask)
        if key in self.distributions:
            return
        if warm_start_from is not None:
            parent = self.distributions.get(
                self._key(warm_start_from), {"alpha": 1.0, "beta": 1.0}
            )
            alpha, beta_v = parent["alpha"] * 1.2, parent["beta"] * 0.8
        else:
            alpha, beta_v = 2.0, 2.0
        self.distributions[key] = {"alpha": alpha, "beta": beta_v}
        self.satisfaction[key] = 0.0
        self.n[key] = 0
        self.subtasks.append(subtask)
        logger.info(f"🎰 [Bandit] Added arm: '{key}' (α={alpha:.2f}, β={beta_v:.2f})")

    def all_executed(self) -> bool:
        return all(self.n.get(self._key(t), 0) > 0 for t in self.subtasks)


# ---------------------------------------------------------------------------
# Structured output schema for task mutation
# ---------------------------------------------------------------------------

class _EvolvedQueries(BaseModel):
    """Output of the LLM-driven task evolution step.
    """
    strategy: Literal["specify", "split", "deepen"] = Field(
        description=(
            "'specify' if the original query was too vague and results were shallow; "
            "'split' if the query was too complex and should be decomposed; "
            "'deepen' if the query produced good coverage and needs an advanced follow-up."
        )
    )
    queries: List[str] = Field(
        description=(
            "1-3 evolved search queries. "
            "'specify' and 'deepen' typically yield 1 query; "
            "'split' yields 2-3 focused sub-queries."
        )
    )
    rationale: str = Field(
        description="One-sentence explanation of why this strategy was chosen."
    )

_evolve_model = model.with_structured_output(_EvolvedQueries)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def _evolve_subtask(
    subtask: Dict,
    ev: "_SubtaskEval",
    existing_summary: str = "",
) -> List[Dict]:
    """LLM-driven task mutation aligned to reflection_instructions gap strategy.
    """
    query = subtask.get("query", "")
    source_type = subtask.get("source_type", "general")

    # Heuristic pre-selection to guide the LLM (reduces ambiguity)
    if ev.coverage_pct < 55 and not ev.is_specific:
        hint = "specify"
    elif ev.coverage_pct < 55 and ev.evidence_count >= 4:
        hint = "split"
    elif ev.coverage_pct >= 55 and ev.fills_existing_gap:
        hint = "deepen"
    else:
        hint = "specify"

    draft_preview = existing_summary[:800] if existing_summary else "(none)"

    prompt = bandit_evolve_user_prompt.format(
        query=query,
        coverage_pct=ev.coverage_pct,
        source_tier=ev.source_tier,
        evidence_count=ev.evidence_count,
        is_specific=ev.is_specific,
        fills_existing_gap=ev.fills_existing_gap,
        hint=hint,
        draft_preview=draft_preview,
    )

    try:
        ev_q: _EvolvedQueries = _evolve_model.invoke([
            SystemMessage(content=bandit_evolve_system_prompt),
            HumanMessage(content=prompt),
        ])
        logger.info(
            f"🔬 [Evolve] '{query}' → strategy={ev_q.strategy} | "
            f"{len(ev_q.queries)} query(ies) | {ev_q.rationale}"
        )
        return [
            {**subtask, "query": q.strip(), "source_type": source_type}
            for q in ev_q.queries[:3]
            if q.strip() and q.strip() != query
        ]

    except Exception as exc:
        logger.warning(f"⚠️  [Evolve] LLM error for '{query}': {exc} – falling back to suffix variant")
        # Graceful fallback: simple suffix based on hint
        suffix = {
            "specify":  "specific examples and details",
            "split":    "comprehensive overview",
            "deepen":   "advanced technical deep dive",
        }.get(hint, "detailed analysis")
        return [{**subtask, "query": f"{query} – {suffix}"}]


def _compute_reward(ev: _SubtaskEval) -> float:
    """Compute a scalar reward ∈ [0, 1] from a ``_SubtaskEval`` response.
    """
    gap_score        = 1.0 if ev.fills_existing_gap else 0.0
    coverage_score   = min(ev.coverage_pct, 100.0) / 100.0
    tier_map         = {1: 1.0, 2: 0.65, 3: 0.30}
    source_score     = tier_map.get(max(1, min(ev.source_tier, 3)), 0.30)
    evidence_score   = min(ev.evidence_count, 5) / 5.0
    specificity_score = 1.0 if ev.is_specific else 0.0

    reward = (
        0.30 * gap_score
        + 0.30 * coverage_score
        + 0.20 * source_score
        + 0.15 * evidence_score
        + 0.05 * specificity_score
    )
    return round(max(0.0, min(1.0, reward)), 4)


async def _evaluate_subtask(
    subtask: Dict,
    result: Dict,
    research_brief: str = "",
    draft_and_notes: str = "",
    accumulated_results: List[Dict] = None,
) -> Tuple[float, bool, bool, "_SubtaskEval"]:
    """Multi-dimensional subtask evaluator aligned to reflection_instructions.
    """
    accumulated = accumulated_results or []
    query = subtask.get("query", "")
    content_preview = str(result.get("content", ""))[:1200]
    source_count = len(result.get("sources", []))

    # Build the full accumulated context for the LLM
    # We want to feed both the previous notes/draft and the results accumulated so far in the loop
    accumulated_texts = [draft_and_notes]
    for past_res in accumulated:
        accumulated_texts.append(f"Prev query: {past_res.get('query')}\nContent: {str(past_res.get('content', ''))[:800]}")
    
    full_context = "\n\n---\n\n".join(filter(None, accumulated_texts))
    draft_preview = full_context[:3000] if full_context else "(none \u2013 this is the first search)"

    eval_prompt = bandit_eval_user_prompt.format(
        research_brief=research_brief,
        draft_preview=draft_preview,
        query=query,
        source_count=source_count,
        content_preview=content_preview,
    )

    try:
        ev: _SubtaskEval = _eval_model.invoke([
            SystemMessage(content=bandit_eval_system_prompt),
            HumanMessage(content=eval_prompt),
        ])
        reward = _compute_reward(ev)
        logger.debug(
            f"[Eval] '{query}' | gap={ev.fills_existing_gap} "
            f"coverage={ev.coverage_pct:.0f}% tier={ev.source_tier} "
            f"evidence={ev.evidence_count} specific={ev.is_specific} → reward={reward}"
        )
        return reward, ev.needs_modification, ev.can_denoise_draft, ev

    except Exception as exc:
        logger.warning(f"⚠️  [Bandit] Evaluator error for '{query}': {exc}")
        reward = min(1.0, source_count / 5.0) if result.get("success") else 0.0
        return reward, False, False, None


# ---------------------------------------------------------------------------
# Bandit execution loop
# ---------------------------------------------------------------------------

async def _run_bandit_loop(
    subtasks: List[Dict],
    search_agent: SearchAgent,
    research_brief: str = "",
    draft_and_notes: str = "",
    max_steps: int = 10,
) -> List[Dict]:
    """Thompson Sampling bandit loop over search subtasks.
    """
    if not subtasks:
        return []

    scheduler = ThompsonBanditScheduler(subtasks)
    all_results: List[Dict] = []
    executed_queries: set = set()  # in-loop dedup guard

    logger.info(
        f"🎰 [Bandit] Starting | arms={len(subtasks)} | max_steps={max_steps}"
    )

    for step in range(max_steps):
        chosen = scheduler.sample_subtask()
        if chosen is None:
            break

        query = chosen.get("query", "")

        # Skip already-executed queries within this run
        if query in executed_queries:
            logger.info(f"⏭️  [Bandit] Step {step}: duplicate skipped – '{query}'")
            scheduler.update(chosen, reward=0.1)
            if scheduler.all_executed():
                break
            continue

        # Execute search
        try:
            res = await search_agent.execute_task(chosen)
            executed_queries.add(query)
        except Exception as exc:
            logger.error(f"❌ [Bandit] Search failed for '{query}': {exc}")
            res = {"success": False, "content": "", "sources": [], "error": str(exc)}

        all_results.append({
            "query":   query,
            "success": res.get("success", False),
            "content": res.get("content", ""),
            "sources": res.get("sources", []),
            "error":   res.get("error"),
        })

        # Evaluate → update arm (pass accumulated context to verify early stopping)
        reward, needs_modification, can_denoise, ev_obj = await _evaluate_subtask(
            chosen, res, research_brief, draft_and_notes, all_results[:-1]
        )
        log_icon = "✅" if res.get("success") else "⚠️"
        logger.info(
            f"{log_icon} [Bandit] Step {step}: '{query}' | "
            f"reward={reward:.2f} | sources={len(res.get('sources', []))} | "
            f"modify={needs_modification} | can_denoise={can_denoise}"
        )
        scheduler.update(chosen, reward)

        if can_denoise:
            logger.info(f"🎉 [Bandit] Early stop at step {step}: Research is sufficient to denoise the draft!")
            break

        if needs_modification and ev_obj is not None:
            # LLM-driven mutation: pass eval result so the model can pick strategy
            evolved_list = await _evolve_subtask(chosen, ev_obj, draft_and_notes)
            for evo in evolved_list:
                scheduler.add_subtask(evo, warm_start_from=chosen)
                logger.info(f"🔬 [Bandit] Evolved arm added: '{evo.get('query')}'")
        elif needs_modification:
            # Fallback when ev_obj is None (heuristic path)
            evolved_list = await _evolve_subtask(
                chosen,
                _SubtaskEval(
                    coverage_pct=30.0, source_tier=3, evidence_count=2,
                    is_specific=False, fills_existing_gap=True, needs_modification=True,
                    can_denoise_draft=False,
                ),
                draft_and_notes,
            )
            for evo in evolved_list:
                scheduler.add_subtask(evo, warm_start_from=chosen)
                logger.info(f"🔬 [Bandit] Evolved arm (fallback): '{evo.get('query')}'")

        if scheduler.all_executed() and step >= len(subtasks) - 1:
            logger.info(f"🏁 [Bandit] All arms done after {step + 1} steps.")
            break

    logger.info(
        f"🎰 [Bandit] Done | steps={min(step + 1, max_steps)} | results={len(all_results)}"
    )
    return all_results


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def execute_research(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """Execute bandit-driven search from the decomposed research plan.
    """
    research_brief: str = state.get("research_brief", "")
    research_plan: Dict[str, Any] = state.get("research_plan") or {}

    # Build existing summary context for the evaluator
    # Combines draft_report + all accumulated notes so the evaluator knows
    # what is already covered — mirrors reflection_instructions gap assessment
    draft_report: str = state.get("draft_report") or ""
    notes: List[str] = state.get("notes") or []
    existing_summary: str = "\n\n".join(filter(None, [draft_report] + notes))

    logger.info(
        f"🚀 [Execution] research_brief='{research_brief[:80]}...' | "
        f"existing_summary_len={len(existing_summary)}"
    )

    # --- Build subtask list from research_plan --------------------------------
    raw_subtopics = research_plan.get("subtopics") or []
    if raw_subtopics:
        subtasks = [
            {
                "type":        "search",
                "query":       sub["query"],
                "source_type": _tool_to_source(sub.get("suggested_tool", "general_search")),
            }
            for sub in raw_subtopics
        ]
        logger.info(f"🧭 [Strategy] {len(subtasks)} subtopics from research_plan.")
    else:
        # Fallback: research_plan missing — search the brief directly
        subtasks = [{"type": "search", "query": research_brief, "source_type": "general"}]
        logger.warning("⚠️ [Strategy] No subtopics in research_plan — falling back to brief.")

    # --- Run bandit loop (accumulated context for early stopping evaluation) ---
    max_steps = max(len(subtasks) * 2, 6)
    search_results = asyncio.run(
        _run_bandit_loop(
            subtasks,
            SearchAgent(),
            research_brief=research_brief,
            draft_and_notes=existing_summary,
            max_steps=max_steps,
        )
    )

    # Routing to write_draft_report handled by explicit edge in graph.py
    return {"web_research_results": search_results}
