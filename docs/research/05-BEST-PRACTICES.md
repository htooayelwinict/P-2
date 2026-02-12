# Production Best Practices for AI Agents

**Last Updated:** 2026-02-12

---

## Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│               Production Agent Development Lifecycle                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Design                                                   │
│     - Pattern selection                                         │
│     - Architecture planning                                     │
│     - Tool design                                            │
│                                                                │
│  2. Development                                              │
│     - State management                                       │
│     - Error handling                                         │
│     - Testing                                               │
│                                                                │
│  3. Deployment                                                │
│     - Checkpointing                                         │
│     - Monitoring                                            │
│     - Observability                                        │
│                                                                │
│  4. Operations                                                │
│     - Evaluation                                            │
│     - Cost optimization                                      │
│     - Continuous improvement                                 │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Architecture Design

### Start Simple

```
❌ Bad: Over-engineering from start
┌───────────────────────────────────────────────────────┐
│  Building full multi-agent system with         │
│  supervisor, 5 specialists, handoffs,          │
│  routing, memory backend, custom          │
│  middleware, monitoring...                       │
│                                                │
│  Result: 3 months to MVP                        │
└───────────────────────────────────────────────────────┘

✅ Good: Progressive complexity
┌───────────────────────────────────────────────────────┐
│  Phase 1: Single agent with tools             │
│  Phase 2: Add checkpointing                    │
│  Phase 3: Add subagent delegation            │
│  Phase 4: Add supervisor                     │
│                                                │
│  Result: MVP in 2-4 weeks                   │
└───────────────────────────────────────────────────────┘
```

### Pattern Selection Matrix

| Complexity | Pattern | Rationale |
|------------|-----------|------------|
| Simple tasks | ReAct | Minimal overhead |
| Multi-step known | Plan-and-Execute | Predictable cost |
| Unknown path | Reflection | Self-correcting |
| Multiple domains | Handoff | Specialist focus |
| Heterogeneous team | Supervisor | Central coordination |

---

## 2. State Management

### Explicit State Schemas

```python
from typing import Annotated, TypedDict
from operator import add
from langchain_core.messages import BaseMessage

# ✅ Good: Explicit schema with reducers
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    context: dict

# ❌ Bad: Untyped dictionaries
def bad_node(state: dict) -> dict:
    return {"messages": state["messages"] + [new_msg]}
```

### State Design Principles

| Principle | Description |
|------------|-------------|
| **Single Source of Truth** | State flows through graph, no hidden state |
| **Immutable Updates** | Never mutate in-place, return new state |
| **Reducer Functions** | Use `add_messages` for lists, custom reducers |
| **Typed Schemas** | Use TypedDict or Pydantic for validation |
| **Minimal State** | Only store what's needed for execution |

### Checkpointing Strategy

```python
# Development: In-memory
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# Production: Async PostgreSQL
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncio

async def get_checkpointer():
    DB_URI = "postgresql://user:pass@host/db"
    return await AsyncPostgresSaver.from_conn_string(DB_URI)

# Production: Sync PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(DB_URI)
```

| Environment | Checkpointer | Rationale |
|-------------|--------------|------------|
| Local dev | InMemorySaver | No setup, fast |
| Testing | InMemorySaver | Reproducible, isolated |
| Production | AsyncPostgresSaver | Async, persistent |
| Serverless | SqliteSaver | File-based, portable |
| Distributed | Redis/Postgres | Shared state across instances |

---

## 3. Error Handling

### Comprehensive Error Types

| Error Type | Handling Strategy |
|-------------|-------------------|
| LLM API failure | Retry with exponential backoff |
| Tool execution error | Fallback to alternative tool |
| Context overflow | Summarize old messages |
| Network timeout | Increase timeout, retry |
| Invalid output | Validation with retry |
| Rate limiting | Queue with backpressure |

### Implementation

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.types import RetryPolicy

# ✅ Retry policy for transient failures
builder.add_node(
    "unreliable_tool",
    tool_calling_function,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_delay=1.0,  # seconds
        max_delay=60.0,
        backoff_factor=2.0,
    )
)

# ✅ Explicit error handling
def safe_node(state: AgentState) -> dict:
    try:
        result = risky_operation()
        return {
            "success": True,
            "result": result,
            "error": None
        }
    except ValueError as e:
        return {
            "success": False,
            "result": None,
            "error": f"Validation: {e}",
            "retry": True  # Signal for retry
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": f"Unexpected: {e}",
            "fatal": True  # Don't retry
        }

# ✅ Conditional error recovery
def error_router(state: AgentState) -> str:
    if state.get("fatal"):
        return "end"  # Don't retry fatal errors
    elif state.get("retry"):
        return "retry"  # Loop back
    else:
        return "continue"  # Proceed
```

---

## 4. Testing Strategy

### Test Pyramid

```
                  ┌─────────────────────────────────┐
                  │  Integration Tests       │
                  │  (E2E scenarios)       │
                  └─────────────┬─────────────┘
                                │
                    ┌─────────────┴─────────────┐
                    │  Unit Tests               │
                    │  (Mock LLM/tools)         │
                    └─────────────┬─────────────┘
                                  │
                        ┌─────────────┴─────────────┐
                        │  Property Tests           │
                        │  (State, reducers)       │
                        └──────────────────────────────┘
```

### Testing Best Practices

| Area | Practice |
|-------|----------|
| **Mock LLM calls** | Deterministic unit tests |
| **Test state reducers** | Verify immutability |
| **Checkpoint tests** | Verify persistence |
| **Edge cases** | Empty state, large inputs |
| **Integration tests** | Real tool backends |
| **Evaluation dataset** | Golden Q&A pairs |

### Example Test Structure

```python
# tests/test_agent.py

def test_state_reducer():
    """Test state updates follow reducer logic."""
    state = AgentState(messages=[], count=0)
    updated = reducer(state, {"messages": [new_msg], "count": 1})
    assert updated["messages"] == [new_msg]
    assert updated["count"] == 1

def test_checkpoint_roundtrip():
    """Test state persists and recovers correctly."""
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test"}}
    result = graph.invoke(initial_state, config)

    # State should be checkpointed
    checkpoints = list(checkpointer.list(config))
    assert len(checkpoints) > 0

def test_error_recovery():
    """Test agent recovers from errors."""
    # Simulate failure then success
    with mock_tool_that_fails_once():
        result = graph.invoke(state_with_retry)
        assert result["success"] == True
```

---

## 5. Deployment

### Pre-Deployment Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│            Pre-Deployment Checklist                        │
├───────────────────────────────────────────────────────────────────┤
│                                                            │
│  Configuration                                           │
│  ├─ Environment variables configured                     │
│  ├─ API keys validated                                │
│  ├─ Database connections tested                          │
│  └─ Model access verified                              │
│                                                            │
│  Observability                                          │
│  ├─ LangSmith tracing enabled                         │
│  ├─ Logging configured                              │
│  └─ Metrics dashboard ready                          │
│                                                            │
│  Safety                                                 │
│  ├─ Tool filtering enforced                            │
│  ├─ Human approval gates                           │
│  └─ Input validation                                │
│                                                            │
│  Performance                                            │
│  ├─ Async operations only                           │
│  ├─ Connection pooling                              │
│  └─ Request batching                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Production Configuration

```python
# config/prod.py
import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

PROD_CONFIG = {
    # Model settings
    "model": {
        "default": "claude-sonnet-4-5-20250929",
        "fallback": "claude-haiku-4-5-20251001",
        "timeout": 120,
        "max_retries": 3,
    },

    # Checkpointing
    "checkpoint": {
        "async": True,
        "connection_string": os.getenv("DB_URI"),
        "max_checkpoints": 100,
    },

    # Observability
    "langsmith": {
        "enabled": os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        "project_name": "production-agents",
        "sample_rate": 1.0,  # 100% for debugging
    },

    # Safety
    "safety": {
        "max_iterations": 50,
        "human_approval_required": True,
        "allowed_tools": ["safe_tool_1", "safe_tool_2"],
    },
}
```

---

## 6. Observability

### LangSmith Integration

```python
# Environment setup
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # New tracing
os.environ["LANGCHAIN_API_KEY"] = "lsv_..."
os.environ["LANGSMITH_PROJECT"] = "my-agent"

# Automatic tracing with create_agent
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[my_tools],
    # Tracing is automatic with LANGCHAIN_API_KEY
)
```

### Custom Observability

| Metric | How to Track |
|---------|---------------|
| **Token usage** | Per agent, per tool, per session |
| **Latency** | LLM time, tool time, total time |
| **Success rate** | Tasks completed vs failed |
| **Tool usage** | Which tools, how often |
| **State size** | Message count, context bytes |
| **Cost** | Token cost × model price |

```python
# Custom instrumentation
import time, functools

def track_metrics(agent):
    """Decorator to track agent metrics."""
    @functools.wraps(agent)
    def wrapper(state, config=None):
        start_time = time.time()
        token_count_before = get_token_count()

        try:
            result = agent(state, config)

            # Collect metrics
            duration = time.time() - start_time
            tokens_used = get_token_count() - token_count_before

            log_metrics({
                "agent": agent.__name__,
                "duration": duration,
                "tokens": tokens_used,
                "success": result.get("success", True),
            })

            return result
        except Exception as e:
            log_metrics({
                "agent": agent.__name__,
                "error": str(e),
                "success": False,
            })
            raise

    return wrapper
```

---

## 7. Cost Optimization

### Strategies

| Strategy | Impact |
|----------|----------|
| **Model routing** | Use cheap models for simple tasks |
| **Caching** | Cache embeddings, tool results |
| **Context pruning** | Drop old messages, summarize |
| **Tool batching** | Combine multiple tool calls |
| **Parallel execution** | Run independent nodes concurrently |
| **Token counting** | Monitor and limit per request |

### Model Routing Example

```python
def route_model(state: AgentState) -> str:
    """Route to appropriate model based on task complexity."""
    last_message = state["messages"][-1].content.lower()

    # Simple queries → cheap model
    if any(word in last_message for word in ["what", "how", "list"]):
        return "haiku"  # ~$0.25/M tokens

    # Complex reasoning → expensive model
    if any(word in last_message for word in ["analyze", "compare", "design"]):
        return "sonnet"  # ~$3/M tokens

    # Default → mid-tier
    return "opus"  # ~$15/M tokens

# LangGraph conditional routing
builder.add_conditional_edges(
    "model_router",
    {
        "haiku": haiku_agent,
        "sonnet": sonnet_agent,
        "opus": opus_agent,
    }
)
```

### Caching Strategy

```python
from functools import lru_cache
from hashlib import sha256

def cache_key(func_call):
    """Generate cache key from function call."""
    import json
    return sha256(json.dumps({
        "name": func_call.name,
        "args": func_call.args,
        "kwargs": {}
    }).hexdigest())

@lru_cache(maxsize=1000)
def cached_tool(arg1, arg2):
    """Cached tool implementation."""
    result = expensive_operation(arg1, arg2)
    return result

# In agent graph
builder.add_node("tool", cached_tool)
```

---

## 8. Security

### Security Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                   Agent Security Layers                   │
├───────────────────────────────────────────────────────────────────┤
│                                                            │
│  Layer 1: Input                                        │
│  - Content validation                                     │
│  - Rate limiting                                        │
│  - Sanitization                                         │
│                                                            │
│  Layer 2: Agent                                        │
│  - Tool filtering                                       │
│  - Prompt injection prevention                           │
│  - Output validation                                    │
│                                                            │
│  Layer 3: Environment                                    │
│  - API key rotation                                     │
│  - Encrypted storage                                   │
│  - Audit logging                                         │
│                                                            │
│  Layer 4: Human                                       │
│  - Approval gates                                       │
│  - Override capabilities                                  │
│  - Emergency shutdown                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Tool Security

```python
from typing import Literal
from langchain.tools import tool

# ✅ Whitelist approach
ALLOWED_TOOLS = {
    "read_file": True,
    "write_file": True,
    "search": True,
    # DANGEROUS tools excluded
    # "delete_file": False,
    # "execute_code": False,
}

@tool
def safe_tool(path: str) -> str:
    """Tool with security check."""
    # Validate path
    if not path.startswith("/allowed/"):
        raise ValueError(f"Access denied: {path}")

    # Sanitize input
    clean_path = path.replace("..", "").replace("/", "")

    return read_allowed_file(clean_path)

# ✅ Human approval for dangerous operations
def dangerous_operation_confirmation() -> str:
    """Require human approval for destructive actions."""
    # Interrupt for human input
    return interrupt(
        {
            "action": "delete_files",
            "target": "/important/data"
        },
        resume_value="approve"
    )
```

---

## 9. Evaluation

### Evaluation Framework

| Dimension | Metrics |
|------------|----------|
| **Correctness** | Did agent achieve goal? |
| **Efficiency** | Tokens used, time taken |
| **Safety** | Policy violations, unsafe outputs |
| **User Satisfaction** | Manual rating, implicit feedback |
| **Robustness** | Error recovery rate |

### Golden Dataset Testing

```python
# tests/evaluation.py
EVALUATION_DATASET = [
    {
        "input": "What's the weather in SF?",
        "expected_tools": ["weather"],
        "expected_tool_args": {"city": "San Francisco"},
        "success_criteria": lambda r: "sunny" in r.lower(),
    },
    {
        "input": "Create a Python hello world",
        "expected_tools": ["write_file"],
        "success_criteria": lambda r: "hello" in r.lower() and ".py" in r,
    },
    # ... more test cases
]

def evaluate_agent(agent_func, dataset):
    """Run agent against evaluation dataset."""
    results = []
    for case in dataset:
        result = agent_func({"messages": [{"role": "user", "content": case["input"]}]})
        passed = case["success_criteria"](result["output"])
        results.append({
            "case": case["input"],
            "passed": passed,
            "tools_used": result.get("tool_calls", []),
            "tokens": result.get("tokens", 0),
        })
    return results
```

---

## Quick Reference

### Do's ✅

- Start simple, add complexity incrementally
- Use explicit state schemas with reducers
- Enable checkpointing in production
- Implement comprehensive error handling
- Add observability from day one
- Test with real tool backends
- Use model routing for cost optimization
- Implement human-in-the-loop for safety
- Monitor token usage and costs
- Evaluate on golden datasets

### Don'ts ❌

- Don't over-engineer initial architecture
- Don't skip error handling
- Don't deploy without monitoring
- Don't use same model for all tasks
- Don't allow unbounded iterations
- Don't store secrets in code
- Don't skip checkpointing in production
- Don't ignore edge cases
- Don't ship without tests

---

## External Resources

| Resource | URL |
|-----------|-----|
| LangSmith | https://smith.langchain.com |
| LangGraph Deployment | https://docs.langchain.com/oss/python/langgraph/deployment |
| OpenAI Agents Guide | https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf |
| Azure Agent Patterns | https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns |
