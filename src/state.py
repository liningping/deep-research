"""
State Module

Defines the state models for the research summary graph.
SummaryState is the core state passed between graph nodes.
SummaryStateInput/SummaryStateOutput define the graph's I/O contract.
"""

import logging
import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


logger = logging.getLogger(__name__)


def replace_list(old_list, new_list):
    """Custom reducer that replaces the old list with the new list completely."""
    return new_list


class SummaryState(BaseModel):
    """
    State for the research summary graph.
    """

    # ===== Core Research Fields =====
    research_topic: str = Field(description="The main research topic")
    search_query: str = Field(default="", description="Current search query")
    running_summary: str = Field(default="", description="Running summary of research")
    research_complete: bool = Field(default=False, description="Whether research is complete")
    knowledge_gap: str = Field(default="", description="Identified knowledge gaps")
    research_loop_count: int = Field(default=0, description="Number of research loops completed")

    # ===== Search Results =====
    sources_gathered: List[str] = Field(default_factory=list, description="List of sources gathered")
    web_research_results: List[Dict[str, Any]] = Field(default_factory=list, description="Web research results")
    search_results_empty: bool = Field(default=False, description="Whether search results were empty")
    selected_search_tool: str = Field(default="general_search", description="Selected search tool")
    source_citations: Dict[str, Any] = Field(default_factory=dict, description="Source citations")

    # ===== Research Planning =====
    subtopic_queries: List[str] = Field(default_factory=list, description="Subtopic queries")
    subtopics_metadata: List[Dict[str, Any]] = Field(default_factory=list, description="Subtopics metadata")
    research_plan: Optional[Dict[str, Any]] = Field(default=None, description="Current research plan")

    # ===== Reflection Metadata (trajectory capture) =====
    priority_section: Optional[str] = Field(default=None, description="Priority section from reflection")
    section_gaps: Optional[Dict[str, str]] = Field(default=None, description="Section gaps from reflection")
    evaluation_notes: Optional[str] = Field(default=None, description="Evaluation notes from reflection")
    tool_calls_log: List[Dict[str, Any]] = Field(default_factory=list, description="Tool call log for trajectory")
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Complete execution trace")

    # ===== Mode Flags =====
    extra_effort: bool = Field(default=False, description="Whether to use extra effort")
    minimum_effort: bool = Field(default=False, description="Whether to use minimum effort")
    qa_mode: bool = Field(default=False, description="Whether in QA mode")
    benchmark_mode: bool = Field(default=False, description="Whether in benchmark mode")

    # ===== LLM Configuration =====
    llm_provider: Optional[str] = Field(default=None, description="LLM provider")
    llm_model: Optional[str] = Field(default=None, description="LLM model")

    # ===== Uploaded Content =====
    uploaded_knowledge: Optional[str] = Field(default=None, description="User-uploaded external knowledge")
    uploaded_files: List[str] = Field(default_factory=list, description="List of uploaded file IDs")
    analyzed_files: List[Dict[str, Any]] = Field(default_factory=list, description="Analysis results from uploaded files")

    # ===== Context Refinement =====
    formatted_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Formatted sources for UI")
    useful_information: str = Field(default="", description="Useful information extracted")
    missing_information: str = Field(default="", description="Missing information identified")
    needs_refinement: bool = Field(default=False, description="Whether query needs refinement")
    current_refined_query: str = Field(default="", description="Current refined query")
    refinement_reasoning: str = Field(default="", description="Reasoning for refinement")
    previous_answers: List[str] = Field(default_factory=list, description="Previous answers")
    reflection_history: List[str] = Field(default_factory=list, description="Reflection history")

    # ===== Visualization (kept for interface compatibility) =====
    visualization_disabled: bool = Field(default=True, description="Whether visualizations are disabled")
    visualizations: List[Dict[str, Any]] = Field(default_factory=list, description="Generated visualizations")
    base64_encoded_images: List[str] = Field(default_factory=list, description="Base64 encoded images")
    visualization_html: str = Field(default="", description="Visualization HTML content")
    visualization_paths: List[str] = Field(default_factory=list, description="Paths to visualization files")
    code_snippets: List[Dict[str, Any]] = Field(default_factory=list, description="Generated code snippets")

    # ===== Report =====
    markdown_report: Optional[str] = Field(default="", description="Plain markdown report")
    benchmark_result: Optional[Dict[str, Any]] = Field(default=None, description="Benchmark result")

    # ===== Configuration =====
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration settings")



    def model_post_init(self, __context):
        """Post-initialization hook."""
        logger.info(f"[STATE_INIT] SummaryState created for topic: {self.research_topic}")
        return super().model_post_init(__context)

    def __init__(self, **data):
        # Auto-fill search_query from research_topic if not provided
        if not data.get("search_query") and data.get("research_topic"):
            data["search_query"] = data["research_topic"]
        super().__init__(**data)

        # Log uploaded knowledge
        if hasattr(self, "uploaded_knowledge") and self.uploaded_knowledge:
            print(f"[STATE] uploaded_knowledge set, length: {len(self.uploaded_knowledge)}")

    # ===== Trajectory Capture Methods =====

    def log_tool_call(self, tool_name: str, params: Dict[str, Any], result_summary: str = None):
        """Log a tool call for trajectory capture (non-invasive)."""
        tool_call_entry = {
            "tool": tool_name,
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "research_loop": self.research_loop_count,
        }
        if result_summary:
            tool_call_entry["result_summary"] = result_summary
        self.tool_calls_log.append(tool_call_entry)

    def log_execution_step(
        self, step_type: str, action: str,
        input_data: Any = None, output_data: Any = None, metadata: Dict[str, Any] = None,
    ):
        """Log a complete execution step in chronological order (trajectory capture only)."""
        try:
            step_entry = {
                "step_type": step_type,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "research_loop": self.research_loop_count,
            }
            if input_data is not None:
                step_entry["input"] = input_data
            if output_data is not None:
                if isinstance(output_data, str) and len(output_data) > 1000:
                    step_entry["output_preview"] = output_data[:1000] + "..."
                    step_entry["output_length"] = len(output_data)
                else:
                    step_entry["output"] = output_data
            if metadata:
                step_entry["metadata"] = metadata
            self.execution_trace.append(step_entry)
        except Exception:
            pass  # Trajectory logging should never break research




class SummaryStateInput(BaseModel):
    """Input model for the research process."""

    research_topic: str
    extra_effort: bool = False
    minimum_effort: bool = False
    qa_mode: bool = Field(default=False, description="Whether to run in QA mode")
    benchmark_mode: bool = Field(default=False, description="Whether to run in benchmark mode")
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    uploaded_knowledge: Optional[str] = None
    uploaded_files: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None



class SummaryStateOutput(BaseModel):
    """Output model for the research process."""

    running_summary: str
    research_complete: bool
    research_loop_count: int
    sources_gathered: List[str]
    web_research_results: List[Dict[str, Any]] = []
    source_citations: Dict[str, Dict[str, str]]
    qa_mode: bool = Field(default=False, description="Whether ran in QA mode")
    benchmark_mode: bool = Field(default=False, description="Whether ran in benchmark mode")
    benchmark_result: Optional[Dict[str, Any]] = Field(default=None, description="Benchmark results")
    visualizations: List[Dict[str, Any]] = []
    base64_encoded_images: List[Dict[str, Any]] = []
    visualization_paths: List[str] = []
    code_snippets: List[Dict[str, Any]] = []
    markdown_report: str = ""
    uploaded_knowledge: Optional[str] = None
    analyzed_files: List[Dict[str, Any]] = []
