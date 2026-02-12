# LangChain Ecosystem Overview

**Last Updated:** 2026-02-12

---

## The Three Layers

```
┌────────────────────────────────────────────────────────────────────────┐
│                    LangChain Ecosystem                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: High-Level Abstraction                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  LangChain (create_agent)                          │       │
│  │  - Pre-built architecture                           │       │
│  │  - < 10 lines to start                          │       │
│  │  - Built on LangGraph                          │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                              │
│  Layer 2: Orchestration Framework                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  LangGraph (StateGraph)                          │       │
│  │  - DAG-based workflows                             │       │
│  │  - State management                                 │       │
│  │  - Checkpointing (persistence)                     │       │
│  │  - Human-in-the-loop                               │       │
│  │  - Time travel                                     │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                              │
│  Layer 3: Agent Harnesses                            │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  DeepAgents SDK                                   │       │
│  │  - TodoListMiddleware (planning)                  │       │
│  │  - FilesystemMiddleware (context)                  │       │
│  │  - SubAgentMiddleware (delegation)                │       │
│  │  - DeepAgents CLI (coding tool)                   │       │
│  └───────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## LangChain v1.x (High-Level Framework)

### What It Is

- **Purpose**: Quick agent building with pre-built architecture
- **Target**: Developers who want to get started fast
- **Philosophy**: Opinionated, batteries-included

### Core Benefits

| Benefit | Description |
|----------|-------------|
| Standard Model Interface | Swap providers without code changes |
| Pre-built Agent Architecture | Common patterns built-in |
| Built on LangGraph | Durable execution inherited |
| LangSmith Integration | Debug and observability |

### Basic Usage

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
})
```

### Middleware Support

LangChain 1.0 introduced **middleware architecture**:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def inject_context(request: ModelRequest) -> str:
    # Modify state before model call
    last_query = request.state["messages"][-1].text
    return f"Context: {retrieved_context}\n\n{last_query}"

agent = create_agent(
    model="gpt-4",
    tools=[],
    middleware=[inject_context]  # <-- Composable middleware
)
```

---

## LangGraph (Orchestration Framework)

### Core Concepts

1. **Graph-Based**: Workflows as nodes and edges
2. **Stateful**: Central state object flows through graph
3. **Persistent**: Checkpointing enables time travel
4. **Cyclic**: Can loop back to previous nodes
5. **Parallel**: Independent nodes execute concurrently

### Key Components

| Component | Purpose |
|------------|-----------|
| State | Single source of truth for workflow data |
| Node | Computation unit (LLM call, tool use, decision) |
| Edge | Transition between nodes |
| Checkpointer | State persistence backend |
| START/END | Special entry/exit nodes |

### Supported Checkpointers

| Checkpointer | Use Case |
|--------------|-------------|
| `InMemorySaver` | Development, testing |
| `AsyncPostgresSaver` | Production async apps |
| `PostgresSaver` | Production persistent storage |
| `SQLiteSaver` | Local file persistence |
| `RedisSaver` | Distributed state |

---

## DeepAgents SDK

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              DeepAgents Middleware Architecture          │
├──────────────────────────────────────────────────────────────┤
│                                                        │
│  1. TodoListMiddleware                                │
│     - Task decomposition (write_todos)                    │
│     - Track multi-step objectives                             │
│     - Progress tracking                                    │
│                                                        │
│  2. FilesystemMiddleware                                  │
│     - ls, read_file, write_file, edit_file               │
│     - Pluggable backends (local, PostgreSQL, S3)        │
│     - Context isolation                                    │
│                                                        │
│  3. SubAgentMiddleware                                    │
│     - Spawn specialized sub-agents                          │
│     - Context isolation                                      │
│     - Independent model/tools per subagent                   │
│     - Agent-to-agent communication                        │
└──────────────────────────────────────────────────────────────┘
```

### Built-in Tools

| Tool | Description |
|--------|-------------|
| `write_todos` | Decompose task into sub-objectives |
| `ls` | List files in filesystem |
| `read_file` | Read file contents |
| `write_file` | Write to file |
| `edit_file` | Edit existing file |
| `glob` | Pattern matching in files |
| `grep` | Search file contents |

### SubAgent Pattern

```python
from deepagents.middleware.subagents import SubAgentMiddleware

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    middleware=[
        SubAgentMiddleware(
            default_model="claude-sonnet-4-5-20250929",
            subagents=[
                {
                    "name": "coder",
                    "description": "Writes and edits code",
                    "system_prompt": "You are an expert coder",
                    "tools": [write_file, edit_file],
                    "model": "claude-haiku-4-5",  # Cheaper for coding
                },
                {
                    "name": "tester",
                    "description": "Runs tests",
                    "tools": [run_tests],
                    "model": "gpt-4.1-mini",  # Fast for testing
                }
            ],
        )
    ],
)
```

---

## LangSmith (Observability)

### Features

- **Tracing**: Full execution path visualization
- **Debugging**: Step-by-step state inspection
- **Evaluation**: Compare agent runs
- **Deployment**: Host agents directly

### Setup

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="lsv_..."
```

---

## Version History

| Version | Year | Key Changes |
|----------|--------|--------------|
| LangChain 0.x | 2023-2024 | Original framework |
| LangGraph | 2024 | Split orchestration layer |
| DeepAgents SDK | 2025 (Oct) | Middleware architecture |
| LangChain 1.0 | 2025 (Dec) | Simplification, no Pydantic |

---

## External Resources

| Resource | URL |
|-----------|-----|
| Official Docs | https://python.langchain.com |
| LangGraph Docs | https://docs.langchain.com/oss/python/langgraph |
| DeepAgents GitHub | https://github.com/langchain-ai/deepagents |
| LangSmith | https://smith.langchain.com |
| Discord | https://discord.gg/langchain |
