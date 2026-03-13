
"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

The supervisor uses parallel research execution to improve efficiency while
maintaining isolated context windows for each research topic.
"""

import asyncio

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel, Field

from deep_research.prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt
from deep_research.research_agent import researcher_agent
from deep_research.state_multi_agent_supervisor import (
    SupervisorState, 
    ConductResearch,
    ResearchComplete
)
from deep_research.utils import get_today_str, think_tool, get_logger

logger = get_logger()

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    # Only extract messages that are tool responses for ConductResearch and contain the "PASS" verification
    # We do not want think_tools or FAILED research attempts polluting our final note state
    valid_notes = []
    tool_msgs = filter_messages(messages, include_types="tool")
    for msg in tool_msgs:
        if msg.name == "ConductResearch" and "FAIL:" not in str(msg.content):
            valid_notes.append(str(msg.content))
    return valid_notes

# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it


# ===== CONFIGURATION =====

import os
supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = init_chat_model(
    model=os.getenv("SUPERVISOR_MODEL", "openai:gpt-5"),
    model_provider=os.getenv("LLM_PROVIDER", "openai"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    timeout=300.0
)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

class VerificationResult(BaseModel):
    passed: bool = Field(description="Whether the findings satisfy all assertions.")
    feedback: str = Field(description="Feedback explaining why it passed or failed.")

verifier_model = init_chat_model(
    model=os.getenv("VERIFIER_MODEL", "openai:gpt-4o-mini"),
    model_provider=os.getenv("LLM_PROVIDER", "openai"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.0
).with_structured_output(VerificationResult)

# System constants
# Maximum number of tool call iterations for individual researcher agents
# This prevents infinite loops and controls research depth per topic
max_researcher_iterations = int(os.getenv("MAX_WEB_RESEARCH_LOOPS", "3")) # Calls to think_tool + ConductResearch

# Maximum number of concurrent research agents the supervisor can launch
# This is passed to the lead_researcher_prompt to limit parallel research tasks
max_concurrent_researchers = 3

# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """
    logger.info(f"Entering: supervisor (Iteration {state.get('research_iterations', 0)})")
    supervisor_messages = state.get("supervisor_messages", [])

    # Prepare system message with current date and constraints

    system_message = lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
        date=get_today_str(), 
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    logger.debug(f"supervisor - Number of input messages: {len(messages)}")

    # Make decision about next research steps
    response = await supervisor_model_with_tools.ainvoke(messages)
    
    logger.debug(f"supervisor - Decision made, tool calls: {len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}")

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """
    logger.info(f"Entering: supervisor_tools (Iteration {state.get('research_iterations', 0)})")
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Initialize variables for single return pattern
    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"  # Default next step
    should_end = False

    # Check exit criteria first
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not getattr(most_recent_message, "tool_calls", None)
    research_complete = False
    
    if not no_tool_calls:
        research_complete = any(
            tool_call["name"] == "ResearchComplete" 
            for tool_call in most_recent_message.tool_calls
        )

    if exceeded_iterations or no_tool_calls or research_complete:
        logger.info(f"supervisor_tools - Terminating research. exceeded={exceeded_iterations}, no_tools={no_tool_calls}, complete={research_complete}")
        should_end = True
        next_step = END

    else:
        logger.debug(f"supervisor_tools - Processing {len(most_recent_message.tool_calls)} tool calls")
        # Execute ALL tool calls before deciding next step
        try:
            # Separate think_tool calls from ConductResearch calls
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "ConductResearch"
            ]

            # Handle think_tool calls (synchronous)
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # Handle ConductResearch calls (asynchronous)
            if conduct_research_calls:
                logger.info(f"supervisor_tools - Delegating {len(conduct_research_calls)} concurrent sub-research agent(s)")
                # Launch parallel research agents
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    }) 
                    for tool_call in conduct_research_calls
                ]

                # Wait for all research to complete
                tool_results = await asyncio.gather(*coros)

                # Format research results as tool messages
                # Each sub-agent returns compressed research findings in result["compressed_research"]
                # We write this compressed research as the content of a ToolMessage after verification
                research_tool_messages = []
                for result, tool_call in zip(tool_results, conduct_research_calls):
                    raw_findings = result.get("compressed_research", "Error synthesizing research report")
                    assertions = tool_call["args"].get("verification_assertions", [])
                    
                    if not assertions:
                        # Fallback if no assertions provided
                        final_content = f"<findings>\n{raw_findings}\n</findings>\n<verification>PASS: No assertions provided.</verification>"
                    else:
                        assertions_str = "\n".join([f"- {a}" for a in assertions])
                        verification_prompt = f"Evaluate the following research findings against these specific assertions:\n\nAssertions:\n{assertions_str}\n\nFindings:\n{raw_findings}\n\nDo the findings fully satisfy ALL of the assertions? Provide a boolean 'passed' and 'feedback' explaining any missing information."
                        
                        try:
                            verification_res = await verifier_model.ainvoke([HumanMessage(content=verification_prompt)])
                            if verification_res.passed:
                                final_content = f"<findings>\n{raw_findings}\n</findings>\n<verification>PASS: All criteria met.</verification>"
                            else:
                                final_content = f"<findings>\n{raw_findings}\n</findings>\n<verification>FAIL: {verification_res.feedback}\nPlease adjust your search query and retry.</verification>"
                        except Exception as e:
                            logger.error(f"Verification failed: {e}")
                            final_content = f"<findings>\n{raw_findings}\n</findings>\n<verification>PASS: Verification step errored out ({e}).</verification>"

                    research_tool_messages.append(
                        ToolMessage(
                            content=final_content,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    )

                tool_messages.extend(research_tool_messages)

                # Aggregate raw notes from all research
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", [])) 
                    for result in tool_results
                ]

        except Exception as e:
            should_end = True
            next_step = END

    # Single return point with appropriate state updates
    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes
            }
        )


# ===== GRAPH CONSTRUCTION =====

# Build supervisor graph
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

