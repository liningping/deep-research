"""
Answer generation, reflection, and finalization nodes for QA/Benchmark mode.

Contains the nodes for the benchmark/QA flow:
- post_process_benchmark_answer: Format citations in benchmark answers
- generate_answer: Generate concise answers for QA/benchmark questions
- reflect_answer: Evaluate answer quality and determine if more research needed
- route_after_multi_agents_benchmark: Route after search in benchmark mode
- route_after_generate_answer: Route after answer generation
- route_after_reflect_answer: Determine if research should continue
- verify_answer: Verify answer against expected benchmark answer
- finalize_answer: Produce final answer with all research findings
"""

import json
import time
import re
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from typing_extensions import Literal
from langchain_core.runnables import RunnableConfig

from src.state import SummaryState
from src.configuration import Configuration
from llm_clients import get_llm_client
from src.prompts_qa import (
    ANSWER_GENERATION_PROMPT as QA_ANSWER_GENERATION_PROMPT,
    ANSWER_REFLECTION_PROMPT as QA_ANSWER_REFLECTION_PROMPT,
    FINAL_ANSWER_PROMPT as QA_FINAL_ANSWER_PROMPT,
    ANSWER_VERIFICATION_PROMPT as QA_ANSWER_VERIFICATION_PROMPT,
)
from src.prompts_benchmark import (
    ANSWER_GENERATION_PROMPT as BENCHMARK_ANSWER_GENERATION_PROMPT,
    ANSWER_REFLECTION_PROMPT as BENCHMARK_ANSWER_REFLECTION_PROMPT,
    FINAL_ANSWER_PROMPT as BENCHMARK_FINAL_ANSWER_PROMPT,
    ANSWER_VERIFICATION_PROMPT as BENCHMARK_ANSWER_VERIFICATION_PROMPT,
)
from src.nodes.utils import get_callback_from_config, emit_event, get_max_loops

logger = logging.getLogger(__name__)


def post_process_benchmark_answer(answer, source_citations):
    """
    Post-process the benchmark answer to ensure citation consistency and include a References section
    with the specific format: [cite number] title. authors. [link]

    Args:
        answer (str): The generated benchmark answer
        source_citations (dict): Dictionary mapping citation numbers to source metadata

    Returns:
        str: The post-processed answer with properly formatted citations
    """
    if not source_citations:
        return answer  # No citations to check or add

    # Check if a References section already exists in the answer
    references_section_patterns = [
        "References",
        "References:",
        "## References",
        "# References",
        "**References:**",
    ]

    has_references_section = any(
        pattern in answer for pattern in references_section_patterns
    )

    # Create the references section if needed
    if not has_references_section:
        print("Adding missing References section to the benchmark answer")
        # Format the references section with benchmark-specific format
        references_section = "\n\n**References:**\n"

        # Add each reference in the academic format: [cite number] First Author et al. (year) Title. [link]
        for num, src in sorted(source_citations.items()):
            title = src.get("title", "Unknown Title")
            url = src.get("url", "")
            author = src.get("author")
            year = src.get("year")

            # Format: [cite number] First Author et al. (year) Title. [link]
            if author and year:
                references_section += f"[{num}] {author} et al. ({year}) {title}\n"
            elif author:
                references_section += f"[{num}] {author} et al. {title}\n"
            elif year:
                references_section += f"[{num}] ({year}) {title}\n"
            else:
                # Fallback to original format if no author/year available
                references_section += f"[{num}] {title}\n"

        # Append to the answer
        answer += references_section

    # Fix any generic citations that might have been generated
    import re

    # Multiple patterns to catch different variations of generic citations
    generic_citation_patterns = [
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+research",
        r"\[(\d+)\]\s*Source\s+\d+",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited",
    ]

    all_matches = []
    for pattern in generic_citation_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        all_matches.extend(matches)

    matches = list(set(all_matches))  # Remove duplicates

    if matches:
        print(f"Fixing {len(matches)} generic citations in benchmark answer")
        for citation_num in matches:
            if citation_num in source_citations:
                src = source_citations[citation_num]
                title = src.get("title", "Unknown Title")
                url = src.get("url", "")
                author = src.get("author")
                year = src.get("year")

                # Replace with proper format
                if author and year:
                    replacement = f"[{citation_num}] {author} et al. ({year}) {title}"
                elif author:
                    replacement = f"[{citation_num}] {author} et al. {title}"
                elif year:
                    replacement = f"[{citation_num}] ({year}) {title}"
                else:
                    replacement = f"[{citation_num}] {title}"

                # Replace all variations of generic citations
                for pattern in generic_citation_patterns:
                    answer = re.sub(
                        pattern.replace(r"(\d+)", citation_num),
                        replacement,
                        answer,
                        flags=re.IGNORECASE,
                    )

    # Check for citation consistency (same logic as regular mode)
    # Get all citation numbers used in the answer
    citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    found_citations = set()

    for match in re.finditer(citation_pattern, answer):
        # Handle both single citations [1] and multiple citations [1,2,3]
        for citation in re.split(r",\s*", match.group(1)):
            if citation.isdigit():
                found_citations.add(citation)  # Keep as string

    # Check which citations were used but not in source_citations
    source_citations_keys = set(str(k) for k in source_citations.keys())
    missing_citations = [c for c in found_citations if c not in source_citations_keys]
    if missing_citations:
        print(
            f"WARNING: Benchmark answer contains citations {missing_citations} not found in source_citations"
        )

    # Check which citations from source_citations were not used
    unused_citations = [c for c in source_citations_keys if c not in found_citations]
    if unused_citations:
        print(
            f"WARNING: Benchmark answer doesn't use citations {unused_citations} from source_citations"
        )

    return answer


def generate_answer(state: SummaryState, config: RunnableConfig):
    """
    Generate a concise, fact-based answer for QA and benchmark questions.
    This node is used when qa_mode or benchmark_mode is True.
    """
    print(f"--- ENTERING generate_answer (Loop {state.research_loop_count}) ---")
    print(
        f"[generate_answer] qa_mode={state.qa_mode}, benchmark_mode={state.benchmark_mode}"
    )
    print(f"[generate_answer] research_topic={state.research_topic}")

    # Start timer for performance logging
    start_time = time.time()

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get LLM client
    provider = configurable.llm_provider or "openai"
    model = configurable.llm_model or "o3-mini-reasoning"

    # Prioritize provider and model from state if they exist
    if hasattr(state, "llm_provider") and state.llm_provider:
        provider = state.llm_provider
    if hasattr(state, "llm_model") and state.llm_model:
        model = state.llm_model

    print(f"[generate_answer] Using provider={provider}, model={model}")

    # Get LLM client
    llm = get_llm_client(provider, model)

    # Get previous answers with reasoning if available
    previous_answers = []
    if hasattr(state, "previous_answers") and state.previous_answers:
        previous_answers = state.previous_answers

    previous_answers_text = ""
    if previous_answers:
        previous_answers_text = "\n\n".join(
            [
                f"LOOP {idx+1}:\n{answer.get('answer', '')}\nConfidence: {answer.get('confidence', 'UNKNOWN')}\nReasoning: {answer.get('reasoning', 'None provided')}"
                for idx, answer in enumerate(previous_answers)
            ]
        )

    # Use running_summary as the primary context for answer generation
    accumulated_context = getattr(state, "running_summary", "")
    if not accumulated_context:
        print(
            "[generate_answer] No running_summary available. Using research_topic as minimal context."
        )
        accumulated_context = f"Initial query: {state.research_topic}. No information has been gathered yet."

    # Optionally, include any very recent, unsummarized info from web_research_results if it exists
    # (though ideally, this would have been processed by validate_context_sufficiency already)
    recent_unprocessed_info = getattr(state, "web_research_results", [])
    if recent_unprocessed_info:
        recent_text = "\n\n---\n\n".join(
            item.get("content", "")
            for item in recent_unprocessed_info
            if isinstance(item, dict) and item.get("content")
        )
        if recent_text:
            accumulated_context += f"\n\nADDITIONAL RECENTLY FETCHED CONTENT (May not be fully validated):\n{recent_text}"

    print(
        f"[generate_answer] Context for answer generation (running_summary, first 300 chars): {accumulated_context[:300]}..."
    )

    # Generate date constants for time context
    from datetime import datetime

    today = datetime.now()
    current_date = today.strftime("%B %d, %Y")
    current_year = str(today.year)
    one_year_ago = str(today.year - 1)

    # Choose the appropriate prompt based on mode
    if state.benchmark_mode:
        answer_prompt = BENCHMARK_ANSWER_GENERATION_PROMPT
        print(
            f"[generate_answer] Using BENCHMARK mode prompts with full citation processing"
        )
    elif state.qa_mode:
        answer_prompt = QA_ANSWER_GENERATION_PROMPT
        print(f"[generate_answer] Using QA mode prompts")
    else:
        # Fallback to QA mode if neither is specified
        answer_prompt = QA_ANSWER_GENERATION_PROMPT
        print(f"[generate_answer] Fallback to QA mode prompts")

    # Generate focused answer using the selected prompt
    prompt = answer_prompt.format(
        current_date=current_date,
        current_year=current_year,
        one_year_ago=one_year_ago,
        research_topic=state.research_topic,
        web_research_results=accumulated_context,  # Use accumulated_context (running_summary)
        previous_answers_with_reasoning=previous_answers_text,
    )

    print(f"[generate_answer] Sending prompt to {provider}/{model}")

    response = llm.invoke(prompt)

    # Parse the response to extract structured information
    answer_text = response.content

    # Log the raw response
    print(f"[generate_answer] Raw LLM response preview:")
    preview = answer_text[:300] + "..." if len(answer_text) > 300 else answer_text
    print(f"{preview}")

    # Initialize fields
    direct_answer = None
    confidence_level = None
    supporting_evidence = None
    sources = []
    reasoning = None
    missing_info = None

    # Parse the structured response
    direct_answer = None
    confidence_level = None
    supporting_evidence = None
    sources = []
    reasoning = None
    missing_info = None

    # Split response into lines and process
    lines = answer_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Handle different formats for each section
        if (
            line.startswith("# Direct Answer")
            or line.startswith("1. Direct Answer:")
            or line.startswith("Direct Answer:")
        ):
            # Extract answer from same line if present
            if ":" in line:
                direct_answer = line.split(":", 1)[1].strip()
            # Look for answer on next line if not on same line
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (
                    next_line
                    and not next_line.startswith("#")
                    and not next_line.startswith("2.")
                    and not next_line.startswith("Confidence:")
                ):
                    direct_answer = next_line
                    i += 1  # Skip the next line since we processed it

        elif (
            line.startswith("# Confidence")
            or line.startswith("2. Confidence:")
            or line.startswith("Confidence:")
        ):
            if ":" in line:
                confidence_level = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (
                    next_line
                    and not next_line.startswith("#")
                    and not next_line.startswith("3.")
                    and not next_line.startswith("Supporting Evidence:")
                ):
                    confidence_level = next_line
                    i += 1

        elif (
            line.startswith("# Supporting Evidence")
            or line.startswith("3. Supporting Evidence:")
            or line.startswith("Supporting Evidence:")
        ):
            if ":" in line:
                supporting_evidence = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                # Collect multi-line supporting evidence
                evidence_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if (
                        next_line.startswith("#")
                        or next_line.startswith("4.")
                        or next_line.startswith("Sources:")
                        or next_line.startswith("Reasoning:")
                        or next_line.startswith("Missing Information:")
                    ):
                        break
                    if next_line:
                        evidence_lines.append(next_line)
                    j += 1
                if evidence_lines:
                    supporting_evidence = " ".join(evidence_lines)
                    i = j - 1  # Set i to the last processed line

        elif (
            line.startswith("# Sources")
            or line.startswith("4. Sources:")
            or line.startswith("Sources:")
        ):
            # Collect numbered sources
            sources = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if (
                    next_line.startswith("#")
                    or next_line.startswith("5.")
                    or next_line.startswith("Reasoning:")
                    or next_line.startswith("Missing Information:")
                ):
                    break
                if next_line and (
                    next_line.startswith("1.")
                    or next_line.startswith("2.")
                    or next_line.startswith("3.")
                    or next_line.startswith("4.")
                    or next_line.startswith("5.")
                    or "http" in next_line
                    or next_line.startswith("-")
                    or next_line.startswith("*")
                ):
                    # Clean up source formatting
                    source = next_line.lstrip("1234567890.-* ").strip()
                    if source:
                        sources.append(source)
                j += 1
            i = j - 1

        elif (
            line.startswith("# Reasoning")
            or line.startswith("5. Reasoning:")
            or line.startswith("Reasoning:")
        ):
            if ":" in line:
                reasoning = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                # Collect multi-line reasoning
                reasoning_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if (
                        next_line.startswith("#")
                        or next_line.startswith("6.")
                        or next_line.startswith("Missing Information:")
                    ):
                        break
                    if next_line:
                        reasoning_lines.append(next_line)
                    j += 1
                if reasoning_lines:
                    reasoning = " ".join(reasoning_lines)
                    i = j - 1

        elif (
            line.startswith("# Missing Information")
            or line.startswith("6. Missing Information:")
            or line.startswith("Missing Information:")
        ):
            if ":" in line:
                missing_info = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                # Collect multi-line missing info
                missing_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith("#") or next_line.startswith("7."):
                        break
                    if next_line:
                        missing_lines.append(next_line)
                    j += 1
                if missing_lines:
                    missing_info = " ".join(missing_lines)
                    i = j - 1

        i += 1

    # Fallback: if direct_answer is still None, try to extract from the beginning of the response
    if not direct_answer:
        # Look for the first substantial line that's not a header
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("#")
                and not line.startswith("1.")
                and not line.startswith("Direct Answer:")
                and not line.startswith("Confidence:")
                and len(line) > 10
            ):  # Ensure it's substantial
                direct_answer = line
                break

        # If still no answer, use the first line
        if not direct_answer and lines:
            direct_answer = lines[0].strip()

    # Convert confidence level to numeric value if possible
    numeric_confidence = 0.5  # default mid-range value
    if confidence_level:
        if confidence_level == "HIGH":
            numeric_confidence = 0.9
        elif confidence_level == "MEDIUM":
            numeric_confidence = 0.6
        elif confidence_level == "LOW":
            numeric_confidence = 0.3

    # Create structured answer object
    answer_result = {
        "answer": direct_answer if direct_answer else answer_text.split("\n")[0],
        "confidence": numeric_confidence,
        "confidence_level": confidence_level,
        "supporting_evidence": supporting_evidence,
        "sources": sources,
        "reasoning": reasoning,
        "missing_info": missing_info,
        "full_response": answer_text,
    }

    # Log the structured answer
    print(f"[generate_answer] Structured answer:")
    print(f"  - Answer: {answer_result['answer']}")
    print(
        f"  - Confidence: {answer_result['confidence']} ({answer_result['confidence_level']})"
    )
    print(f"  - Sources: {answer_result['sources']}")
    if reasoning:
        print(
            f"  - Reasoning: {reasoning[:100]}..."
            if len(reasoning) > 100
            else reasoning
        )

    # Create new previous_answers list by copying the existing one and adding the new answer
    previous_answers_updated = list(previous_answers)
    previous_answers_updated.append(answer_result)

    # Log performance timing
    end_time = time.time()
    print(f"[generate_answer] Processing time: {end_time - start_time:.2f} seconds")
    print(f"--- EXITING generate_answer ---\n")

    # Return the updates as a dictionary instead of modifying state directly
    return {
        "benchmark_result": answer_result,
        "qa_mode": state.qa_mode,
        "benchmark_mode": state.benchmark_mode,
        "research_loop_count": getattr(state, "research_loop_count", 0) + 1,
        "previous_answers": previous_answers_updated,
    }


def reflect_answer(state: SummaryState, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflect on the generated answer and determine if more research is needed.
    """
    print(f"--- REFLECTION START (Loop {state.research_loop_count}) ---")
    print(f"[reflect_answer] benchmark_mode={state.benchmark_mode}")

    # Start timer for performance logging
    start_time = time.time()

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get LLM client
    provider = configurable.llm_provider or "openai"
    model = configurable.llm_model or "o3-mini-reasoning"

    # Prioritize provider and model from state if they exist
    if hasattr(state, "llm_provider") and state.llm_provider:
        provider = state.llm_provider
    if hasattr(state, "llm_model") and state.llm_model:
        model = state.llm_model

    print(f"[reflect_answer] Using provider={provider}, model={model}")

    # Get LLM client
    llm = get_llm_client(provider, model)

    # Get current answer and research state
    current_answer = ""
    benchmark_result = getattr(state, "benchmark_result", {})
    if benchmark_result:
        # Use full structured response if available
        if "full_response" in benchmark_result:
            current_answer = benchmark_result["full_response"]
        else:
            # Fall back to simple format
            answer_text = benchmark_result.get("answer", "")
            confidence = benchmark_result.get("confidence", 0.0)
            sources = benchmark_result.get("sources", [])
            current_answer = f"Answer: {answer_text}\nConfidence: {confidence}\nSources: {', '.join(sources)}"

    # Log the current answer for reflection
    print(f"[reflect_answer] Current answer to reflect on:")
    preview = (
        current_answer[:200] + "..." if len(current_answer) > 200 else current_answer
    )
    print(f"  {preview}")

    # Get effort flags and max loops for the combined reflection prompt
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)

    # Get max loops using the utility function
    max_loops = get_max_loops(
        configurable, extra_effort, minimum_effort, state.benchmark_mode, state.qa_mode
    )
    research_loop_count = state.research_loop_count

    print(f"[reflect_answer] Research statistics:")
    print(f"  - Current loop: {research_loop_count} of maximum {max_loops}")
    print(f"  - Extra effort: {extra_effort}")
    print(f"  - Minimum effort: {minimum_effort}")

    # Get web_research_results safely
    web_research_results = getattr(state, "web_research_results", [])

    # Generate date constants for time context
    from datetime import datetime

    today = datetime.now()
    current_date = today.strftime("%B %d, %Y")
    current_year = str(today.year)
    one_year_ago = str(today.year - 1)

    # Choose the appropriate reflection prompt based on mode
    if state.benchmark_mode:
        reflection_prompt = BENCHMARK_ANSWER_REFLECTION_PROMPT
        print(f"[reflect_answer] Using BENCHMARK mode reflection prompts")
    elif state.qa_mode:
        reflection_prompt = QA_ANSWER_REFLECTION_PROMPT
        print(f"[reflect_answer] Using QA mode reflection prompts")
    else:
        # Fallback to QA mode
        reflection_prompt = QA_ANSWER_REFLECTION_PROMPT
        print(f"[reflect_answer] Fallback to QA mode reflection prompts")

    # Prepare reflection prompt with additional parameters including time context
    prompt = reflection_prompt.format(
        current_date=current_date,
        current_year=current_year,
        one_year_ago=one_year_ago,
        research_topic=state.research_topic,
        current_answer=current_answer,
        web_research_results=(
            web_research_results
            if web_research_results
            else "No research results available yet."
        ),
        research_loop_count=research_loop_count,
        max_loops=max_loops,
    )

    print(f"[reflect_answer] Sending reflection prompt to {provider}/{model}")

    response = llm.invoke(prompt)

    # Parse reflection response
    reflection_text = response.content

    # Log the raw reflection response
    print(f"[reflect_answer] Raw reflection response preview:")
    preview = (
        reflection_text[:300] + "..." if len(reflection_text) > 300 else reflection_text
    )
    print(f"{preview}")

    # Initialize the reflection result
    decision = None
    confidence = None
    missing_facts = []
    follow_up = None

    # Parse the structured response - handle both JSON and text formats
    should_continue = False
    justification = ""

    # Try to parse as JSON first (since prompt asks for function call format)
    try:
        import json

        # Look for JSON in the response
        json_start = reflection_text.find("{")
        json_end = reflection_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_text = reflection_text[json_start:json_end]
            parsed_json = json.loads(json_text)

            # Extract decision from JSON structure
            if "evaluation" in parsed_json:
                evaluation = parsed_json["evaluation"]

                # Look for final decision
                if "final_decision" in evaluation:
                    final_decision = evaluation["final_decision"]
                    continue_research = final_decision.get(
                        "continueResearch", final_decision.get("continue_research", "")
                    )

                    if isinstance(continue_research, str):
                        should_continue = continue_research.lower() in [
                            "yes",
                            "true",
                            "1",
                        ]
                    elif isinstance(continue_research, bool):
                        should_continue = continue_research

                    justification = final_decision.get("justification", "")
                    follow_up = final_decision.get(
                        "followUpQuery", final_decision.get("follow_up_query", "")
                    )

                # If no final_decision, look for other decision indicators
                if not justification and "answer_quality" in evaluation:
                    answer_quality = evaluation["answer_quality"]
                    # If answer has logical flaws or inappropriate confidence, continue research
                    logical_flaws = answer_quality.get(
                        "logical_flaws", answer_quality.get("logicalFlaws", "")
                    )
                    confidence_appropriate = answer_quality.get(
                        "confidence_appropriate",
                        answer_quality.get("confidenceAppropriate", ""),
                    )

                    if logical_flaws and logical_flaws.lower() == "yes":
                        should_continue = True
                        justification = (
                            "Answer contains logical flaws that need to be addressed"
                        )
                    elif (
                        confidence_appropriate
                        and confidence_appropriate.lower() == "no"
                    ):
                        should_continue = True
                        justification = "Confidence level is not appropriate for the evidence provided"

                # Extract missing facts if available
                if "evidence_evaluation" in evaluation or "evidence" in evaluation:
                    evidence_eval = evaluation.get(
                        "evidence_evaluation", evaluation.get("evidence", {})
                    )
                    missing_info = evidence_eval.get(
                        "missingCriticalInformation",
                        evidence_eval.get(
                            "missing_critical_information",
                            evidence_eval.get("missing_critical_info", ""),
                        ),
                    )
                    if missing_info and missing_info.lower() == "yes":
                        # Try to extract what's missing from justification or other fields
                        if not justification:
                            justification = (
                                "Critical information is missing from the sources"
                            )
                        if not missing_facts:
                            missing_facts = [justification]

            print(f"[reflect_answer] Successfully parsed JSON response")
            print(f"[reflect_answer] Extracted continue_research: {should_continue}")
            print(f"[reflect_answer] Extracted justification: {justification}")

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[reflect_answer] Failed to parse JSON response: {e}")
        print(f"[reflect_answer] Falling back to text pattern parsing")

        # Fall back to text pattern parsing
        for line in reflection_text.split("\n"):
            line = line.strip()
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip()
            elif line.startswith("Confidence:"):
                confidence = line.replace("Confidence:", "").strip()
            elif line.startswith("Missing facts:"):
                missing_facts_text = line.replace("Missing facts:", "").strip()
                missing_facts = (
                    [fact.strip() for fact in missing_facts_text.split(",")]
                    if missing_facts_text
                    else []
                )
            elif line.startswith("Follow-up query:"):
                follow_up = line.replace("Follow-up query:", "").strip()
            elif "Should research continue?" in line:
                if "Yes" in line:
                    should_continue = True
                elif "No" in line:
                    should_continue = False
            elif line.startswith("Justification:"):
                justification = line.replace("Justification:", "").strip()

        # If no clear decision was found, check for key phrases in the response
        if should_continue is None:
            if (
                "needs more research" in reflection_text.lower()
                or "continue research" in reflection_text.lower()
            ):
                should_continue = True
                print(
                    f"[reflect_answer] No explicit continue/terminate decision found, defaulting to CONTINUE based on text analysis"
                )
            elif (
                "complete" in reflection_text.lower()
                or "sufficient" in reflection_text.lower()
            ):
                should_continue = False
                print(
                    f"[reflect_answer] No explicit continue/terminate decision found, defaulting to TERMINATE based on text analysis"
                )
            else:
                # Default to continuing research if unclear
                should_continue = True
                print(
                    f"[reflect_answer] No explicit continue/terminate decision found, defaulting to CONTINUE as safest option"
                )

    # Create a dictionary for the reflection result
    reflection_result = {
        "should_continue": should_continue,
        "justification": justification,
        "missing_facts": missing_facts,
        "follow_up_query": follow_up,
        "loop_count": research_loop_count,
    }

    # Get existing reflection history safely
    reflection_history = getattr(state, "reflection_history", [])

    # Create a new reflection history list with the current reflection added
    reflection_history_updated = list(reflection_history)
    reflection_history_updated.append(reflection_result)

    # Log the reflection decision and next steps
    print(
        f"[reflect_answer] REFLECTION DECISION: {'Continue research' if should_continue else 'Complete research'}"
    )
    print(f"  - Justification: {justification}")

    if missing_facts:
        print(f"  - Missing facts identified:")
        for fact in missing_facts:
            print(f"    * {fact}")
    else:
        print(f"  - No specific missing facts identified")

    if follow_up:
        print(f'  - Follow-up query: "{follow_up}"')
    else:
        print(f"  - No follow-up query generated")

    print(f"  - Research complete: {not should_continue}")
    print(f"  - Research loop count: {research_loop_count}")

    # Log performance timing
    end_time = time.time()
    print(f"[reflect_answer] Processing time: {end_time - start_time:.2f} seconds")
    print(f"--- REFLECTION END ---\n")

    # Return the updates as a dictionary instead of modifying state directly
    return {
        "research_complete": not should_continue,
        "knowledge_gap": "\n".join(missing_facts) if missing_facts else "",
        "search_query": follow_up if follow_up else "",
        "reflection_history": reflection_history_updated,
    }


def route_after_multi_agents_benchmark(
    state: SummaryState,
) -> Literal["generate_answer", "reflect_answer", "finalize_answer"]:
    """
    Determines the next step after the multi_agents_network in benchmark mode.
    """
    minimum_effort = getattr(state, "minimum_effort", False)
    if minimum_effort:
        print(
            "ROUTING: Minimum effort requested, skipping reflection, finalizing answer"
        )
        return "finalize_answer"
    else:
        # Check if search returned results
        if getattr(state, "search_results_empty", False):
            print("ROUTING: Search returned no results, going directly to reflection")
            return "reflect_answer"
        else:
            print("ROUTING: Search returned results, proceeding to answer generation")
            return "generate_answer"


def route_after_generate_answer(state: SummaryState) -> Literal["reflect_answer"]:
    """Route after generating answer to reflection"""
    print("ROUTING: Moving to answer reflection")
    return "reflect_answer"


def route_after_reflect_answer(state: SummaryState, config: RunnableConfig):
    """Determines if research should continue or answer should be finalized"""
    # Create default config if none provided
    if not config:
        config = {"configurable": {"max_web_research_loops": 3}}

    configurable = Configuration.from_runnable_config(config)

    # Get effort flags from state
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)

    print(f"ROUTING STATE CHECK:")
    print(f"  - research_complete: {state.research_complete}")
    print(f"  - research_loop_count: {state.research_loop_count}")
    print(
        f"  - has search_query: {hasattr(state, 'search_query') and bool(state.search_query)}"
    )
    print(f"  - extra_effort: {extra_effort}")
    print(f"  - minimum_effort: {minimum_effort}")

    # Get max_loops using the utility function
    env_max_loops = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
    if env_max_loops:
        print(f"  - Reading MAX_WEB_RESEARCH_LOOPS from environment: {env_max_loops}")

    max_loops = get_max_loops(
        configurable, extra_effort, minimum_effort, state.benchmark_mode, state.qa_mode
    )
    print(
        f"  - Using max_loops={max_loops} (extra_effort={extra_effort}, base={configurable.max_web_research_loops or 3})"
    )

    # First check if we've hit max loops
    if state.research_loop_count >= max_loops:
        print(f"ROUTING OVERRIDE: Max loops reached ({max_loops}), finalizing answer")
        return "finalize_answer"

    # If research is marked as complete, finalize the answer
    if state.research_complete:
        print("ROUTING DECISION: Research marked as complete, finalizing answer")
        return "finalize_answer"

    # If minimum effort is requested, skip further research and finalize
    if minimum_effort:
        print("ROUTING DECISION: Minimum effort requested, finalizing answer")
        return "finalize_answer"

    # If we have high confidence in our current answer, finalize it
    if hasattr(state, "benchmark_result") and state.benchmark_result:
        confidence = state.benchmark_result.get("confidence", 0)
        confidence_level = state.benchmark_result.get("confidence_level", "")

        # Check if confidence exceeds threshold from config
        confidence_threshold = 0.8  # Default threshold
        if hasattr(state, "config") and state.config:
            benchmark_config = state.config.get("benchmark", {})
            if "confidence_threshold" in benchmark_config:
                confidence_threshold = benchmark_config.get("confidence_threshold")

        if confidence >= confidence_threshold or confidence_level == "HIGH":
            print(
                f"ROUTING DECISION: High confidence answer ({confidence} >= {confidence_threshold}), finalizing"
            )
            return "finalize_answer"

    # Check if we have a search query to continue research
    has_search_query = hasattr(state, "search_query") and bool(state.search_query)

    # If no search query but we need more research, generate a default one based on state
    if not has_search_query:
        # Generate a search query based on the missing information or follow up
        if hasattr(state, "reflection_history") and state.reflection_history:
            latest_reflection = state.reflection_history[-1]
            missing_facts = latest_reflection.get("missing_facts", [])

            if missing_facts:
                # Use missing facts to formulate a query
                missing_facts_text = ", ".join(missing_facts)
                state.search_query = f"{state.research_topic} {missing_facts_text}"
                state.knowledge_gap = missing_facts_text
                print(
                    f"ROUTING DECISION: Generated search query from missing facts: {state.search_query}"
                )
            else:
                # Create a generic search query from the research topic
                state.search_query = f"{state.research_topic} detailed information"
                state.knowledge_gap = "Need more comprehensive information"
                print(
                    f"ROUTING DECISION: Generated generic search query: {state.search_query}"
                )
        else:
            # Create a generic search query from the research topic
            state.search_query = f"{state.research_topic} detailed information"
            state.knowledge_gap = "Need more comprehensive information"
            print(
                f"ROUTING DECISION: Generated default search query: {state.search_query}"
            )

        return "multi_agents_network"

    # Continue research with the existing search query
    print("ROUTING DECISION: Research not complete, continuing with existing query")
    return "multi_agents_network"


def verify_answer(state: SummaryState, config: RunnableConfig):
    """
    Verify the generated answer against the expected benchmark answer.
    """
    print("[verify_answer] Starting answer verification")

    # Get the benchmark result
    result = state.benchmark_result
    if not result:
        print("[verify_answer] No benchmark result found to verify")
        return state

    # Get expected answer from config
    expected = state.config.get("benchmark", {}).get("expected_answer")
    if not expected:
        print("[verify_answer] No expected answer found in config")
        return state

    # Parse the generated answer
    answer_lines = result.get("answer", "").split("\n")
    parsed_answer = None
    confidence = result.get("confidence", 0.0)
    sources = result.get("sources", [])

    for line in answer_lines:
        if line.strip():
            parsed_answer = line.strip()
            break

    if not parsed_answer:
        print("[verify_answer] Could not parse generated answer")
        return state

    # Compare answers
    is_correct = parsed_answer.lower() == expected.lower()
    print(f"[verify_answer] Generated answer: {parsed_answer}")
    print(f"[verify_answer] Expected answer: {expected}")
    print(f"[verify_answer] Match: {is_correct}")
    print(f"[verify_answer] Confidence: {confidence}")
    print(f"[verify_answer] Sources: {sources}")

    # Update state with verification results
    state.benchmark_result = {
        "answer": parsed_answer,
        "expected": expected,
        "is_correct": is_correct,
        "confidence": confidence,
        "sources": sources,
    }

    return state


def finalize_answer(state: SummaryState, config: RunnableConfig):
    """
    Finalize the answer for benchmark questions using all research findings.
    """
    print("[finalize_answer] Starting answer finalization")
    print(f"[finalize_answer] benchmark_mode={state.benchmark_mode}")

    # Start timer for performance logging
    start_time = time.time()

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get LLM client
    provider = configurable.llm_provider or "openai"
    model = configurable.llm_model or "o3-mini-reasoning"

    # Prioritize provider and model from state if they exist
    if hasattr(state, "llm_provider") and state.llm_provider:
        provider = state.llm_provider
    if hasattr(state, "llm_model") and state.llm_model:
        model = state.llm_model

    print(f"[finalize_answer] Using provider={provider}, model={model}")

    # Get LLM client
    llm = get_llm_client(provider, model)

    # Get all previous answers with reasoning from the benchmark workflow
    previous_answers = getattr(state, "previous_answers", [])
    reflection_history = getattr(state, "reflection_history", [])

    print(
        f"[finalize_answer] Processing {len(previous_answers)} previous answers and {len(reflection_history)} reflections"
    )

    # Debug: Log the actual data structures
    if previous_answers:
        print(f"[finalize_answer] Previous answers structure:")
        for i, answer in enumerate(previous_answers):
            print(
                f"  Answer {i+1}: {list(answer.keys()) if isinstance(answer, dict) else type(answer)}"
            )
            if isinstance(answer, dict):
                print(f"    - answer: {answer.get('answer', 'MISSING')[:100]}...")
                print(
                    f"    - confidence_level: {answer.get('confidence_level', 'MISSING')}"
                )
                print(
                    f"    - reasoning: {answer.get('reasoning', 'MISSING')[:100] if answer.get('reasoning') else 'MISSING'}..."
                )
    else:
        print(f"[finalize_answer] No previous answers found in state")

    if reflection_history:
        print(f"[finalize_answer] Reflection history structure:")
        for i, reflection in enumerate(reflection_history):
            print(
                f"  Reflection {i+1}: {list(reflection.keys()) if isinstance(reflection, dict) else type(reflection)}"
            )
            if isinstance(reflection, dict):
                print(
                    f"    - justification: {reflection.get('justification', 'MISSING')[:100]}..."
                )
                print(
                    f"    - missing_facts: {reflection.get('missing_facts', 'MISSING')}"
                )
    else:
        print(f"[finalize_answer] No reflection history found in state")

    # Format all answers with reasoning for the FINAL_ANSWER_PROMPT
    all_answers_with_reasoning = ""
    if previous_answers:
        for i, answer in enumerate(previous_answers):
            loop_num = i + 1

            # Handle both dict and non-dict answer formats
            if isinstance(answer, dict):
                answer_text = answer.get("answer", "No answer provided")
                confidence = answer.get(
                    "confidence_level", answer.get("confidence", "UNKNOWN")
                )
                reasoning = answer.get("reasoning", "No reasoning provided")
                supporting_evidence = answer.get(
                    "supporting_evidence", "No evidence provided"
                )
                sources = answer.get("sources", [])
            else:
                # Fallback for non-dict answers
                answer_text = str(answer)
                confidence = "UNKNOWN"
                reasoning = "No reasoning provided"
                supporting_evidence = "No evidence provided"
                sources = []

            all_answers_with_reasoning += f"RESEARCH LOOP {loop_num}:\n"
            all_answers_with_reasoning += f"Answer: {answer_text}\n"
            all_answers_with_reasoning += f"Confidence: {confidence}\n"
            all_answers_with_reasoning += (
                f"Supporting Evidence: {supporting_evidence}\n"
            )
            all_answers_with_reasoning += (
                f"Sources: {', '.join(sources) if sources else 'No sources listed'}\n"
            )
            all_answers_with_reasoning += f"Reasoning: {reasoning}\n"

            # Add reflection for this loop if available
            if i < len(reflection_history) and isinstance(reflection_history[i], dict):
                reflection = reflection_history[i]
                justification = reflection.get(
                    "justification", "No reflection justification"
                )
                missing_facts = reflection.get("missing_facts", [])
                all_answers_with_reasoning += f"Reflection: {justification}\n"
                if missing_facts:
                    all_answers_with_reasoning += (
                        f"Missing Facts Identified: {', '.join(missing_facts)}\n"
                    )
            else:
                all_answers_with_reasoning += (
                    f"Reflection: No reflection available for this loop\n"
                )

            all_answers_with_reasoning += "\n---\n\n"
    else:
        all_answers_with_reasoning = "No previous research answers available."

    # Get the most recent web research results as final context
    web_research_results = getattr(state, "web_research_results", [])
    final_search_results = ""
    if web_research_results:
        # Convert web_research_results to text format for the prompt
        if isinstance(web_research_results, list):
            final_search_results = "\n\n---\n\n".join(
                item.get("content", str(item))
                for item in web_research_results
                if isinstance(item, dict) and item.get("content")
            )
        else:
            final_search_results = str(web_research_results)

    # Use running_summary if available as additional context
    running_summary = getattr(state, "running_summary", "")
    if running_summary:
        final_search_results = f"ACCUMULATED RESEARCH SUMMARY:\n{running_summary}\n\n---\n\nLATEST SEARCH RESULTS:\n{final_search_results}"

    # Add source citations context for benchmark mode
    source_citations = getattr(state, "source_citations", {})
    if source_citations and state.benchmark_mode:
        citations_context = "\n\n---\n\nAVAILABLE SOURCES FOR CITATION:\n"
        for cite_num, cite_data in sorted(source_citations.items()):
            title = cite_data.get("title", "Unknown Title")
            url = cite_data.get("url", "No URL")
            author = cite_data.get("author")
            year = cite_data.get("year")

            # Format with author and year if available (academic style)
            if author and year:
                citations_context += f"[{cite_num}] {author} et al. ({year}) {title}\n"
            elif author:
                citations_context += f"[{cite_num}] {author} et al. {title}\n"
            elif year:
                citations_context += f"[{cite_num}] ({year}) {title}\n"
            else:
                citations_context += f"[{cite_num}] {title}\n"
        final_search_results += citations_context

    if not final_search_results:
        final_search_results = "No search results available."

    print(f"[finalize_answer] Context lengths:")
    print(f"  - All answers with reasoning: {len(all_answers_with_reasoning)} chars")
    print(f"  - Final search results: {len(final_search_results)} chars")

    # Generate date constants for time context
    from datetime import datetime

    today = datetime.now()
    current_date = today.strftime("%B %d, %Y")
    current_year = str(today.year)
    one_year_ago = str(today.year - 1)

    # Choose the appropriate final answer prompt based on mode
    if state.benchmark_mode:
        final_prompt = BENCHMARK_FINAL_ANSWER_PROMPT
        print(f"[finalize_answer] Using BENCHMARK mode final answer prompts")
    elif state.qa_mode:
        final_prompt = QA_FINAL_ANSWER_PROMPT
        print(f"[finalize_answer] Using QA mode final answer prompts")
    else:
        # Fallback to QA mode
        final_prompt = QA_FINAL_ANSWER_PROMPT
        print(f"[finalize_answer] Fallback to QA mode final answer prompts")

    # Use the selected FINAL_ANSWER_PROMPT
    prompt = final_prompt.format(
        current_date=current_date,
        current_year=current_year,
        one_year_ago=one_year_ago,
        research_topic=state.research_topic,
        all_answers_with_reasoning=all_answers_with_reasoning,
        web_research_results=final_search_results,
    )

    print(f"[finalize_answer] Sending FINAL_ANSWER_PROMPT to {provider}/{model}")

    response = llm.invoke(prompt)

    # Parse the response to extract structured information
    answer_text = response.content if hasattr(response, "content") else str(response)

    # Log the raw response
    print(f"[finalize_answer] Raw LLM response preview:")
    preview = answer_text[:300] + "..." if len(answer_text) > 300 else answer_text
    print(f"{preview}")

    # Initialize fields for parsing the FINAL_ANSWER_PROMPT format
    direct_answer = None
    confidence_level = None
    key_evidence = None
    sources = []
    limitations = None

    # Parse the structured response according to FINAL_ANSWER_PROMPT format
    for line in answer_text.split("\n"):
        line = line.strip()
        if (
            line.startswith("# Direct Answer:")
            or line.startswith("1. Direct Answer:")
            or line.startswith("## Direct Answer:")
            or line.startswith("Direct Answer:")
        ):
            direct_answer = (
                line.replace("# Direct Answer:", "")
                .replace("1. Direct Answer:", "")
                .replace("## Direct Answer:", "")
                .replace("Direct Answer:", "")
                .strip()
            )
        elif (
            line.startswith("# Overall Confidence:")
            or line.startswith("2. Overall Confidence:")
            or line.startswith("## Overall Confidence:")
            or line.startswith("Overall Confidence:")
        ):
            confidence_level = (
                line.replace("# Overall Confidence:", "")
                .replace("2. Overall Confidence:", "")
                .replace("## Overall Confidence:", "")
                .replace("Overall Confidence:", "")
                .strip()
            )
        elif (
            line.startswith("# Key Evidence:")
            or line.startswith("3. Key Evidence:")
            or line.startswith("## Key Evidence:")
            or line.startswith("Key Evidence:")
        ):
            key_evidence = (
                line.replace("# Key Evidence:", "")
                .replace("3. Key Evidence:", "")
                .replace("## Key Evidence:", "")
                .replace("Key Evidence:", "")
                .strip()
            )
        elif (
            line.startswith("# Sources:")
            or line.startswith("4. Sources:")
            or line.startswith("## Sources:")
            or line.startswith("Sources:")
        ):
            sources_text = (
                line.replace("# Sources:", "")
                .replace("4. Sources:", "")
                .replace("## Sources:", "")
                .replace("Sources:", "")
                .strip()
            )
            sources = [src.strip() for src in sources_text.split("\n") if src.strip()]
        elif (
            line.startswith("# Limitations:")
            or line.startswith("5. Limitations:")
            or line.startswith("## Limitations:")
            or line.startswith("Limitations:")
        ):
            limitations = (
                line.replace("# Limitations:", "")
                .replace("5. Limitations:", "")
                .replace("## Limitations:", "")
                .replace("Limitations:", "")
                .strip()
            )

    # If parsing didn't work well, try a more comprehensive approach
    if not direct_answer or not confidence_level:
        lines = answer_text.split("\n")
        current_section = None
        content_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Identify section headers
            if (
                line.startswith("# Direct Answer")
                or line.startswith("## Direct Answer")
                or line.startswith("1. Direct Answer")
                or line.startswith("Direct Answer")
            ):
                if current_section == "direct_answer" and content_lines:
                    direct_answer = " ".join(content_lines).strip()
                current_section = "direct_answer"
                content_lines = []
                # Check if answer is on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Overall Confidence")
                or line.startswith("## Overall Confidence")
                or line.startswith("2. Overall Confidence")
                or line.startswith("Overall Confidence")
            ):
                if current_section == "direct_answer" and content_lines:
                    direct_answer = " ".join(content_lines).strip()
                elif current_section == "confidence" and content_lines:
                    confidence_level = " ".join(content_lines).strip()
                current_section = "confidence"
                content_lines = []
                # Check if confidence is on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Key Evidence")
                or line.startswith("## Key Evidence")
                or line.startswith("3. Key Evidence")
                or line.startswith("Key Evidence")
            ):
                if current_section == "confidence" and content_lines:
                    confidence_level = " ".join(content_lines).strip()
                elif current_section == "evidence" and content_lines:
                    key_evidence = " ".join(content_lines).strip()
                current_section = "evidence"
                content_lines = []
                # Check if evidence is on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Sources")
                or line.startswith("## Sources")
                or line.startswith("4. Sources")
                or line.startswith("Sources")
            ):
                if current_section == "evidence" and content_lines:
                    key_evidence = " ".join(content_lines).strip()
                elif current_section == "sources" and content_lines:
                    sources = [src.strip() for src in content_lines if src.strip()]
                current_section = "sources"
                content_lines = []
                # Check if sources are on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Limitations")
                or line.startswith("## Limitations")
                or line.startswith("5. Limitations")
                or line.startswith("Limitations")
            ):
                if current_section == "sources" and content_lines:
                    sources = [src.strip() for src in content_lines if src.strip()]
                elif current_section == "limitations" and content_lines:
                    limitations = " ".join(content_lines).strip()
                current_section = "limitations"
                content_lines = []
                # Check if limitations are on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif line and not line.startswith("#") and current_section:
                # This is content for the current section
                content_lines.append(line)

            # Handle the last section
        if current_section == "direct_answer" and content_lines:
            direct_answer = " ".join(content_lines).strip()
        elif current_section == "confidence" and content_lines:
            confidence_level = " ".join(content_lines).strip()
        elif current_section == "evidence" and content_lines:
            key_evidence = " ".join(content_lines).strip()
        elif current_section == "sources" and content_lines:
            sources = [src.strip() for src in content_lines if src.strip()]
        elif current_section == "limitations" and content_lines:
            limitations = " ".join(content_lines).strip()

    # Convert confidence level to numeric value if possible
    numeric_confidence = 0.5  # default mid-range value
    if confidence_level:
        if confidence_level.upper() == "HIGH":
            numeric_confidence = 0.9
        elif confidence_level.upper() == "MEDIUM":
            numeric_confidence = 0.6
        elif confidence_level.upper() == "LOW":
            numeric_confidence = 0.3

    # If direct answer wasn't parsed, use the first meaningful line as the answer
    if not direct_answer and answer_text:
        lines = answer_text.split("\n")
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("1.")
                and not line.startswith("2.")
                and not line.startswith("3.")
                and not line.startswith("4.")
                and not line.startswith("5.")
            ):
                direct_answer = line
                break

        # If still no answer, use the first line
        if not direct_answer:
            direct_answer = (
                lines[0].strip() if lines else "No answer could be extracted"
            )

    # Verify against expected answer if available in benchmark config
    expected_answer = None
    is_correct = False

    if hasattr(state, "config") and state.config:
        benchmark_config = state.config.get("benchmark", {})
        expected_answer = benchmark_config.get("expected_answer")

        if expected_answer and direct_answer:
            # Simple string matching for now, could be enhanced with semantic matching
            is_correct = (
                expected_answer.lower() in direct_answer.lower()
                or direct_answer.lower() in expected_answer.lower()
            )
            print(f"[finalize_answer] Comparing answers:")
            print(f"  - Generated: {direct_answer}")
            print(f"  - Expected:  {expected_answer}")
            print(f"  - Match:     {is_correct}")

    # Apply citation processing for benchmark mode
    processed_answer_text = answer_text
    if state.benchmark_mode:
        print(f"[finalize_answer] Applying citation processing for benchmark mode")

        # Get source citations from state
        source_citations = getattr(state, "source_citations", {})

        # Apply benchmark-specific citation processing with custom format
        processed_answer_text = post_process_benchmark_answer(
            answer_text, source_citations
        )

        print(f"[finalize_answer] Citation processing completed")
        print(f"  - Original length: {len(answer_text)} chars")
        print(f"  - Processed length: {len(processed_answer_text)} chars")
        print(f"  - Available citations: {len(source_citations)}")

    # Create the final benchmark result
    final_result = {
        "answer": direct_answer,
        "confidence": numeric_confidence,
        "confidence_level": confidence_level,
        "evidence": key_evidence,
        "sources": sources,
        "limitations": limitations,
        "expected_answer": expected_answer,
        "is_correct": is_correct,
        "full_response": processed_answer_text,  # Use processed text with citations
        "raw_response": answer_text,  # Keep original for debugging
        "synthesis_of_all_loops": all_answers_with_reasoning,  # Include the synthesis for debugging
    }

    # Log the final structured answer
    print(f"[finalize_answer] Final structured answer:")
    print(f"  - Answer: {final_result['answer']}")
    print(
        f"  - Confidence: {final_result['confidence']} ({final_result['confidence_level']})"
    )
    if sources:
        print(f"  - Sources: {sources}")
    if key_evidence:
        print(
            f"  - Evidence: {key_evidence[:100]}..."
            if len(key_evidence) > 100
            else key_evidence
        )
    if limitations:
        print(f"  - Limitations: {limitations}")

    # Log performance timing
    end_time = time.time()
    print(f"[finalize_answer] Processing time: {end_time - start_time:.2f} seconds")
    print(f"[finalize_answer] Complete\n")

    # Return the updated state with the final result
    return {
        "benchmark_result": final_result,
        "qa_mode": state.qa_mode,
        "benchmark_mode": state.benchmark_mode,
        "research_complete": True,
        "previous_answers": previous_answers,  # Preserve previous answers
        "reflection_history": reflection_history,  # Preserve reflection history
        "config": getattr(state, "config", {}),
        "source_citations": getattr(
            state, "source_citations", {}
        ),  # Preserve citations
        "running_summary": getattr(
            state, "running_summary", ""
        ),  # Preserve running summary
        "research_loop_count": getattr(
            state, "research_loop_count", 0
        ),  # Preserve loop count
    }
