# LangGraph Reference Guide

**Last Updated:** 2026-02-12

---

## Quick Start

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Define a simple node
def my_node(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)

# Compile with checkpointer
graph = builder.compile(checkpointer=InMemorySaver())
```

---

## Core Concepts

### 1. Graph Structure

LangGraph uses a **directed graph** where:
- **Nodes** = Computational units (functions, agents, tools)
- **Edges** = Control flow between nodes
- **State** = Data flowing through the graph

```
    ┌─────────┐
    │  START   │
    └────┬────┘
         │
    ┌────▼────┐
    │   Node A  │
    └────┬────┘
         │
    ┌────▼────┐     ┌─────────┐
    │   Node B  ├────►│  Node C  │
    └────┬────┘     └────┬────┘
         │                  │
    ┌────▼────┐         │
    │   Node D  │         │
    └────┬────┘         │
         │                │
         └────►─────────┴────► END
```

### 2. State Management

State is the **single source of truth** for your workflow.

```python
from typing import Annotated, TypedDict
from operator import add
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    # Annotated with reducer function
    messages: Annotated[list[BaseMessage], add_messages]
    # Custom fields
    user_input: str
    context: str
    step_count: int
```

### State Update Pattern

```python
def node_a(state: GraphState) -> dict:
    return {
        "messages": [state["messages"][-1]],  # Reducer: add_messages appends
        "step_count": state.get("step_count", 0) + 1,  # Direct update
    }
```

---

## Graph Patterns

### Sequential Chain

```python
builder = StateGraph(GraphState)
builder.add_node("step1", process_step1)
builder.add_node("step2", process_step2)
builder.add_node("step3", process_step3)

builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
builder.add_edge("step3", END)
```

### Conditional Branching

```python
from langgraph.graph import StateGraph

def route_function(state: GraphState) -> str:
    if state["category"] == "A":
        return "process_a"
    else:
        return "process_b"

builder = StateGraph(GraphState)
builder.add_node("router", router_node)
builder.add_node("process_a", handle_a)
builder.add_node("process_b", handle_b)

builder.add_conditional_edges(
    "router",
    {
        "process_a": route_function,
        "process_b": route_function,
    }
)
```

### Parallel Execution

```python
from langgraph.graph import Send, invoke_branch

def parallel_branch(state: GraphState) -> list[Send]:
    return [
        Send("process_a", state),
        Send("process_b", state),
        Send("process_c", state),
    ]

builder.add_node("branch", parallel_branch)
builder.add_node("process_a", handle_a)
builder.add_node("process_b", handle_b)
builder.add_node("process_c", handle_c)

# All three execute concurrently
builder.add_conditional_edges("branch", [lambda s: [s for s in [Send("process_a"), Send("process_b"), Send("process_c")]])
```

### Cyclic Workflow (Loops)

```python
def should_continue(state: GraphState) -> str:
    return "continue" if state["count"] < 3 else "end"

builder = StateGraph(GraphState)
builder.add_node("process", process_step)
builder.add_node("check", check_completion)

builder.add_edge(START, "process")
builder.add_edge("process", "check")

# Loop back to process or go to END
builder.add_conditional_edges("check", {
    "continue": "process",
    "end": END,
})
```

---

## Checkpointing & Persistence

### Why Checkpointing Matters

| Benefit | Description |
|----------|-------------|
| **Time Travel** | Replay from any previous state |
| **Fault Tolerance** | Resume after failures |
| **Human-in-the-Loop** | Inspect/modify state mid-execution |
| **Long-Running** | State across hours/days |
| **Debugging** | Step-by-step state inspection |

### Checkpointer Types

```python
# In-memory (development)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# PostgreSQL (production - sync)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# PostgreSQL (production - async)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
checkpointer = await AsyncPostgresSaver.from_conn_string(DB_URI)

# SQLite (local persistence)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")
```

### Using Checkpoints

```python
# Compile with checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Config with thread_id for conversation tracking
config = {
    "configurable": {
        "thread_id": "conversation-123",
        "user_id": "user-abc"
    }
}

# Run with checkpointing
result = graph.invoke(initial_state, config)

# List checkpoints
checkpoints = list(checkpointer.list(config))

# Get specific checkpoint
checkpoint = checkpointer.get(config, checkpoint_id)

# Resume from checkpoint
graph.invoke(new_input, config)
```

### Time Travel

```python
from langgraph.checkpoint import Checkpoint

# Get state at specific checkpoint
state_before = checkpointer.get_tuple(config)

# Resume execution from that point
for event in graph.stream(input, config, stream_mode="values"):
    # Can inspect and modify state here
    pass
```

---

## Streaming

### Stream Modes

```python
# Stream values (state updates)
for chunk in graph.stream(input, config, stream_mode="values"):
    print(chunk["messages"])

# Stream updates (deltas)
for chunk in graph.stream(input, config, stream_mode="updates"):
    print(chunk)

# Stream tokens (LLM output)
async for token in graph.astream(input, config, stream_mode="tokens"):
    async for token in token:
        print(token, end="", flush=True)
```

---

## StateReducers

### Built-in Reducers

```python
from operator import add
from langgraph.graph import MessagesState

# Using MessagesState (has add_messages reducer)
class MyState(MessagesState):
    # add_messages automatically appends to list
    custom_field: str
```

### Custom Reducers

```python
def append_unique(value: list, new_item: str) -> list:
    if new_item not in value:
        return value + [new_item]
    return value

from typing import Annotated
class State(TypedDict):
    items: Annotated[list, append_unique]
```

---

## Multi-Agent Patterns

### Supervisor Pattern

```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    members=["coder", "tester", "reviewer"],
    model="claude-sonnet-4-5-20250929",
)

# Returns a graph with supervisor routing to member agents
graph = supervisor.graph()
```

### Handoff Pattern

```python
def should_handoff(state: GraphState) -> str:
    if "specialist_needed" in state["messages"][-1].content:
        return "specialist_agent"
    return "continue"

builder = StateGraph(GraphState)
builder.add_node("general_agent", general_handler)
builder.add_node("specialist_agent", specialist_handler)
builder.add_conditional_edges("general_agent", {
    "specialist_agent": should_handoff,
    "continue": END,
})
```

---

## Common Patterns

### 1. Retry Logic

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "unreliable_tool",
    tool_calling_function,
    retry_policy=RetryPolicy(
        max_attempts=3,
        handle_exceptions=True,
    )
)
```

### 2. Middleware Integration

```python
from langgraph.prebuilt import ToolNode

# Wrap tools as nodes
tools_node = ToolNode(tools=[my_tool])

builder = StateGraph(GraphState)
builder.add_node("tools", tools_node)
builder.add_edge(START, "tools")
```

### 3. Human-in-the-Loop

```python
from langgraph.types import interrupt, Command

def human_review(state: GraphState) -> Command:
    # Interrupt and wait for human input
    return interrupt(
        {"question": "Should we proceed?"},
        resume_value="approve"
    )

builder.add_node("review", human_review)
```

---

## Error Handling

```python
def safe_node(state: GraphState) -> dict:
    try:
        result = risky_operation()
        return {"result": result, "error": None}
    except Exception as e:
        return {
            "result": None,
            "error": str(e),
            "retry_needed": True
        }

def should_retry(state: GraphState) -> str:
    return "retry" if state.get("error") else END

builder = StateGraph(GraphState)
builder.add_node("safe_node", safe_node)
builder.add_conditional_edges("safe_node", {
    "retry": "safe_node",  # Loop back on error
    "end": END,
})
```

---

## Best Practices

| Area | Practice |
|--------|----------|
| State Design | Use TypedDict with Annotated fields |
| Checkpointing | Always enable in production |
| Streaming | Use `stream_mode="values"` for responsive UX |
| Error Handling | Use conditional edges for error recovery |
| Testing | Use `InMemorySaver` for tests |
| Async | Use async checkpointers for production |
| Naming | Descriptive node/edge names for debugging |

---

## External References

| Resource | URL |
|-----------|-----|
| Official Docs | https://docs.langchain.com/oss/python/langgraph |
| GitHub | https://github.com/langchain-ai/langgraph |
| Examples | https://github.com/langchain-ai/langgraph-supervisor-py |
| Persistence Guide | https://docs.langchain.com/oss/python/langgraph/persistence |
