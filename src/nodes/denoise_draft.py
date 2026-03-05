import logging
import re
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import SummaryState
from src.configuration import Configuration
from src.utils import (
    extract_citations_from_state,
    clean_raw_web_content
)

logger = logging.getLogger(__name__)

def denoise_draft(state: SummaryState, config: RunnableConfig):
    """
    Finalize the summary for BENCHMARKING.
    Produces a clean, pure Markdown report without HTML/CSS artifacts.
    """
    # 1. Prepare Content (Logic preserved)
    # ------------------------------------------------------------------
    current_summary = state.running_summary or ""
    
    if current_summary.strip():
        input_content = current_summary
        logger.info(f"[Finalize] Using running summary ({len(input_content)} chars)")
    else:
        input_content = clean_raw_web_content(state.web_research_results)
        logger.info(f"[Finalize] Using raw web content ({len(input_content)} chars)")

    if not input_content.strip():
        input_content = "(No content available to finalize)"

    # 2. Extract Citations
    # ------------------------------------------------------------------
    source_citations = extract_citations_from_state(state)
    
    # Format numbered sources for Prompt
    # Benchmark 关键点：确保 Prompt 里的引用格式清晰，方便模型引用
    numbered_sources = [
        f"{num}. {src['title']}, [{src['url']}]"
        for num, src in sorted(source_citations.items())
    ]
    formatted_sources_prompt = "\n".join(numbered_sources)

    # 3. Prepare User Context (Steering)
    # ------------------------------------------------------------------
    # Benchmark 场景下，uploaded_knowledge 和 steering 仍然重要，因为它们定义了 Ground Truth
    uploaded_knowledge = getattr(state, "uploaded_knowledge", "")
    augment_context = "No user-provided external knowledge available."
    uploaded_section = ""
    
    if uploaded_knowledge and uploaded_knowledge.strip():
        uploaded_section = f"\n\nUSER-PROVIDED KNOWLEDGE (HIGHEST AUTHORITY):\n{uploaded_knowledge}\n"
        augment_context = f"User Uploaded Knowledge Preview: {uploaded_knowledge[:500]}..."

    todo_section = ""
    if hasattr(state, "steering_todo") and state.steering_todo:
        # 即使是评测，保留用户意图上下文也有助于模型生成更准确的内容
        completed = state.steering_todo.get_completed_tasks()
        all_messages = getattr(state.steering_todo, "all_user_messages", [])
        
        if completed or all_messages:
            todo_section = "\n\nRESEARCH CONTEXT & GOALS:\n"
            if all_messages:
                todo_section += "USER INSTRUCTIONS:\n" + "\n".join([f"- {msg}" for msg in all_messages])
            if completed:
                todo_section += "\nCOVERED TOPICS:\n" + "\n".join([f"- {t.description}" for t in completed])

    # 4. Configure LLM
    # ------------------------------------------------------------------
    configurable = Configuration.from_runnable_config(config)
    provider = getattr(state, "llm_provider", None) or configurable.llm_provider
    if not isinstance(provider, str): provider = provider.value
    
    model = getattr(state, "llm_model", None) or configurable.llm_model or "gemini-2.5-pro"
    llm = get_llm_client(provider, model)

    # 5. Execution (Optimized for Pure Markdown)
    # ------------------------------------------------------------------
    system_prompt = finalize_report_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=augment_context,
    )

    # 明确要求 Markdown 格式
    human_message = (
        f"Please finalize this research summary into a polished report using standard Markdown.\n"
        f"DO NOT use HTML tags (like <div>, <h1>, <style>). Use strictly Markdown (# Title, **bold**, - list).\n"
        f"{uploaded_section}"
        f"{todo_section}\n\n"
        f"DRAFT CONTENT:\n{input_content}\n\n"
        f"AVAILABLE SOURCES:\n{formatted_sources_prompt}"
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_message)])
    final_text = response.content if hasattr(response, "content") else str(response)

    # 6. Post-Processing (Benchmark Safe)
    # ------------------------------------------------------------------
    # 不调用 post_process_report_formatting，因为它会引入 HTML。
    # 我们只在本地做最必要的引用修复。
    
    # A. 修复通用引用描述 (e.g. "[1] Source 1" -> "[1] Real Title")
    final_text = _fix_citations_for_benchmark(final_text, source_citations)
    
    # B. 确保参考文献部分存在 (Markdown 格式)
    final_text = _ensure_markdown_references(final_text, source_citations)

    # 7. Return State
    # ------------------------------------------------------------------
    return {
        "running_summary": final_text,
        "markdown_report": final_text, # 两个字段内容一致
        "web_research_results": [],
        "research_topic": state.research_topic,
        "source_citations": source_citations,
        # 评测不需要图片数据，清空以节省内存/Token
        "base64_encoded_images": [],
        "visualizations": []
    }

# =============================================================================
# Local Helpers (Inline to avoid dependency on formatters.py which has HTML)
# =============================================================================

def _fix_citations_for_benchmark(report: str, source_citations: dict) -> str:
    """Fix generic citations without introducing HTML."""
    if not source_citations:
        return report
        
    generic_patterns = [
        r"\[(\d+)\]\s*Source\s+\d+",
        r"\[(\d+)\]\s*source\s+\d+",
    ]
    
    # 简单的替换逻辑：查找文中类似 "[1] Source 1" 的文本并替换为 "[1] Title"
    for citation_id, src in source_citations.items():
        title = src.get("title", "Unknown Title")
        replacement = f"[{citation_id}] {title}"
        
        for pattern in generic_patterns:
            # 构造特定 ID 的正则
            specific_pattern = pattern.replace(r"(\d+)", citation_id)
            report = re.sub(specific_pattern, replacement, report, flags=re.IGNORECASE)
            
    return report

def _ensure_markdown_references(report: str, source_citations: dict) -> str:
    """Ensure a References section exists in pure Markdown."""
    if not source_citations:
        return report

    patterns = ["# References", "## References", "### References", "**References**"]
    has_section = any(p in report for p in patterns)

    if not has_section:
        refs = "\n\n---\n## References\n\n"
        # 按序号排序
        sorted_citations = sorted(source_citations.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        for num, src in sorted_citations:
            refs += f"{num}. [{src['title']}]({src['url']})\n"
        report += refs
        
    return report