"""MemoryAgent – Graph that extracts and stores user memories on a schedule."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)

# Initialize the language model (shared instance across all calls)
llm = init_chat_model()


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Extract the user's state from recent conversation messages and
    determine whether new memories should be created.

    Args:
        state (State): Current conversational state containing messages.
        runtime (Runtime[Context]): LangGraph runtime with context and store.

    Returns:
        dict: Updated state containing new LLM messages.
    """
    user_id: str = runtime.context.user_id
    model: str = runtime.context.model
    system_prompt: str = runtime.context.system_prompt

    # Retrieve last few user messages to use as query
    recent_messages: List[str] = [m.content for m in state.messages[-3:]]
    query: str = str(recent_messages)

    try:
        memories = await cast(BaseStore, runtime.store).asearch(
            ("memories", user_id), query=query, limit=10
        )
    except Exception as e:
        logger.error("Failed to retrieve memories for user %s: %s", user_id, e)
        memories = []

    # Format memories for inclusion in the system prompt
    formatted_memories: str = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted_memories:
        formatted_memories = f"<memories>\n{formatted_memories}\n</memories>"

    # Inject memories and current timestamp into the system prompt
    sys_prompt: str = system_prompt.format(
        user_info=formatted_memories, time=datetime.now().isoformat()
    )

    # Invoke the LLM with system + conversation context
    try:
        msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
            [{"role": "system", "content": sys_prompt}, *state.messages],
            context=utils.split_model_and_provider(model),
        )
    except Exception as e:
        logger.exception("LLM invocation failed for user %s: %s", user_id, e)
        return {"messages": []}

    return {"messages": [msg]}


async def store_memory(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    Execute memory storage operations requested by the LLM.

    Args:
        state (State): Current conversational state containing messages.
        runtime (Runtime[Context]): LangGraph runtime with context and store.

    Returns:
        dict: Tool response messages confirming stored memories.
    """
    tool_calls: List[dict] = getattr(state.messages[-1], "tool_calls", [])

    if not tool_calls:
        logger.info("No tool calls found; skipping memory storage.")
        return {"messages": []}

    try:
        saved_memories = await asyncio.gather(
            *(
                tools.upsert_memory(
                    **tc["args"],
                    user_id=runtime.context.user_id,
                    store=cast(BaseStore, runtime.store),
                )
                for tc in tool_calls
            )
        )
    except Exception as e:
        logger.exception("Memory storage failed for user %s: %s", runtime.context.user_id, e)
        saved_memories = []

    # Link tool responses back to the LLM
    results: List[Dict[str, Any]] = [
        {"role": "tool", "content": mem, "tool_call_id": tc["id"]}
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State) -> str:
    """
    Determine the next step in the workflow.

    Args:
        state (State): Current conversational state.

    Returns:
        str: Next node name or END marker.
    """
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        return "store_memory"
    return END


# Build the LangGraph workflow
builder = StateGraph(State, context_schema=Context)

# Define nodes
builder.add_node(call_model)
builder.add_node(store_memory)

# Define workflow edges
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")  # feedback loop

# Compile the graph
graph = builder.compile()
graph.name = "MemoryAgent"

__all__ = ["graph"]
