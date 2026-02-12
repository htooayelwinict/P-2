# Agent Architecture Patterns

**Last Updated:** 2026-02-12

---

## Pattern Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                 Agent Architecture Patterns (2025-2026)           │
├────────────────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Cognitive Architectures (Single Agent)                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ReAct (Reason + Act)                              │   │
│  │  Plan-and-Execute                                      │   │
│  │  Reflection                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                               │
│  2. Multi-Agent Patterns                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Sequential/Concurrent                                 │   │
│  │  Supervisor-Worker (Hierarchical)                       │   │
│  │  Handoff/Routing                                       │   │
│  │  Group Chat/Democratic                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                               │
│  3. Enterprise Patterns (Azure/OpenAI)                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Semantic Routing                                     │   │
│  │  Orchestration Layers                                  │   │
│  │  Multi-Agent Protocol (MCP)                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. ReAct (Reason + Act)

### Concept

Agent iterates through:
1. **Reason**: LLM decides what to do
2. **Act**: Execute tool/action
3. **Observe**: See result, update context
4. **Repeat**: Continue until done

### Diagram

```
┌──────┐
│  User │
│  Query│
└───┬───┘
    │
    ▼
┌───────────────────────────────────────────┐
│            ReAct Loop              │
│  ┌─────────────────────────────────┐   │
│  │  1. Thought (Reason)         │   │
│  │  "I need to check the weather" │   │
│  └──────────────┬─────────────┘   │
│                 │                │
│                 ▼                │
│  ┌─────────────┐           │   │
│  │  2. Action   │           │   │
│  │  call_tool  │           │   │
│  └──────┬──────┘           │   │
│         │                     │   │
│         ▼                     │   │
│  ┌──────────────┐          │   │
│  │  3. Observation│          │   │
│  │  "Weather: sunny"           │   │
│  └──────┬───────┘          │   │
│         │                     │   │
│         ▼                     │   │
│  ┌──────────────┐          │   │
│  │  4. Thought    │          │   │
│  │  "Now I can answer"         │   │
│  └─────────────────┘          │   │
│                                 │   │
│  ┌─────────────────────────────┐   │
│  │  Final Answer            │   │
│  │  "The weather in SF is sunny"│   │
│  └─────────────────────────────┘   │
└───────────────────────────────────────────┘
```

### When to Use

| Scenario | ReAct Score |
|-----------|---------------|
| Simple queries | ⭐⭐⭐ |
| Tool exploration | ⭐⭐⭐ |
| Unknown path ahead | ⭐⭐⭐ |
| Quick prototypes | ⭐⭐⭐ |
| Complex multi-step | ⭐ (consider Plan-and-Execute) |

### Implementation

```python
from langchain.agents import create_react_agent

agent = create_react_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[search, calculator, weather],
    max_iterations=10,  # Prevent infinite loops
)
```

---

## 2. Plan-and-Execute

### Concept

1. **Plan**: Generate complete plan before acting
2. **Execute**: Follow plan step-by-step
3. **Re-plan**: Adjust plan if needed

### Diagram

```
┌──────────┐
│  User    │
│  Request  │
└─────┬────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│         1. Planner Phase                │
│  ┌───────────────────────────────────┐    │
│  │  "To accomplish X, I will:     │    │
│  │  1. Search for info              │    │
│  │  2. Analyze results              │    │
│  │  3. Formulate answer             │    │
│  └───────────────────────────────────┘    │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐    │
│  │       2. Execute Phase      │    │
│  │  ┌─────────────────────────┐   │    │
│  │  │  Step 1: Search      │   │    │
│  │  └──────────┬──────────┘   │    │
│  │             │               │    │
│  │             ▼               │    │
│  │  ┌─────────────────────────┐   │    │
│  │  │  Step 2: Analyze  │   │    │
│  │  └──────────┬──────────┘   │    │
│  │             │               │    │
│  │             ▼               │    │
│  │  ┌─────────────────────────┐   │    │
│  │  │  Step 3: Answer   │    │    │
│  │  └─────────────────────────┘   │    │
│                 │               │    │
│                 ▼               │    │
│  ┌──────────────────────────────────┐    │
│  │         Final Answer        │    │    │
│  │  "Here is the answer..."    │    │
│  └──────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### When to Use

| Scenario | Plan-and-Execute Score |
|-----------|----------------------|
| Multi-step tasks | ⭐⭐⭐ |
| Need visibility | ⭐⭐⭐ |
| Cost optimization | ⭐⭐⭐ |
| Complex workflows | ⭐⭐⭐ |
| Unknown execution path | ⭐⭐⭐ |

### Implementation

```python
# LangChain built-in planning agents
from langchain.agents import create_plan_and_execute_agent

agent = create_plan_and_execute_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[search, read_file, write_file],
    planner_prompt="Break down the task into clear steps",
)
```

---

## 3. Reflection

### Concept

Agent critiques its own output before finalizing.

### Diagram

```
┌──────────┐
│  User    │
│  Question │
└─────┬────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│         1. Initial Response           │
│  ┌───────────────────────────────────┐  │
│  │  "Paris is capital of France"    │  │
│  │  Confidence: 85%              │  │
│  └───────────────────────────────────┘  │
│                 │                    │
│                 ▼                    │
│  ┌──────────────────────────────────┐  │
│  │      2. Reflection           │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │  "Wait, let me verify. │  │  │
│  │  │  Paris is capital, but... │  │  │
│  │  │  User might mean Paris, │  │  │
│  │  │  Texas. Let me clarify."│  │  │
│  └─────────────────────────────┘  │  │
│                 │                    │
│                 ▼                    │
│  ┌──────────────────────────────────┐  │
│  │    3. Clarification       │  │
│  │  "Did you mean Paris, France │  │
│  │  or Paris, Texas?"         │  │
│  └──────────────────────────────────┘  │
│                 │                    │
│                 ▼                    │
│  ┌──────────────────────────────────┐  │
│  │      4. Final Answer       │  │
│  │  "Clarified answer..."         │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### When to Use

| Scenario | Reflection Score |
|-----------|-----------------|
| High-stakes outputs | ⭐⭐⭐ |
| Factual accuracy critical | ⭐⭐⭐ |
| Self-verification needed | ⭐⭐⭐ |
| Learning from mistakes | ⭐⭐⭐ |

---

## 4. Supervisor-Worker (Hierarchical)

### Concept

Central supervisor delegates tasks to specialized workers.

### Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                     Supervisor Agent                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - Orchestrates workflow                       │   │
│  │  - Routes to specialists                       │   │
│  │  - Aggregates results                          │   │
│  │  - Ensures completion                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                           │
│         ┌─────────┬─────────┬─────────┬───────────────┐
│         │         │         │         │               │
│         ▼         ▼         ▼         │               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐
│  │  Coder    │ │ Tester   │ │ Reviewer  │ │ Deployer   │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘
└───────────────────────────────────────────────────────────────────────┘
```

### LangGraph Supervisor

```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    members=["coder", "tester", "reviewer"],
    model="claude-sonnet-4-5-20250929",
    output_mode="full"  # Returns full routing state
)

# Returns compiled graph
graph = supervisor.compile()
```

### Custom Supervisor

```python
def supervisor_node(state: SupervisorState) -> Command:
    messages = state["messages"]
    last_message = messages[-1]

    # Route based on intent
    if "code" in last_message.content.lower():
        return Send("coder", state)
    elif "test" in last_message.content.lower():
        return Send("tester", state)
    elif "review" in last_message.content.lower():
        return Send("reviewer", state)
    else:
        return END

builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("coder", coder_agent)
builder.add_node("tester", tester_agent)
builder.add_node("reviewer", reviewer_agent)
```

---

## 5. Handoff/Routing

### Concept

Agent hands off to specialized agent based on query/domain.

### Handoff Types

| Type | Trigger | Example |
|--------|-----------|----------|
| Semantic | Domain match ("coding" → coder agent) |
| Capability | Tool availability ("file_access" → file_agent) |
| Context | Conversation length (summary → summarizer) |
| Explicit | User handoff |

### Diagram

```
┌────────────────────────────────────────────────────────────┐
│               Generalist Agent                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  "I can help with many things, but I notice   │  │
│  │  you need help with coding. Let me connect   │  │
│  │  you to our specialist."                       │  │
│  └───────────────────────────────────────────────────┘  │
│                         │                        │
│                         ▼                        │
│              [HANDOFF]                        │
│                         │                        │
│                         │                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Coding Specialist                │  │
│  │  "I can help with Python, JS, TS..."      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# LangChain handoff pattern
from langchain.agents import create_agent, handoff

def route_to_specialist(state: AgentState, next_agent: str):
    return {
        "messages": [
            {"role": "assistant", "content": f"Connecting you to {next_agent}..."}
        ]
    }

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[...],
    handoffs=[
        handoff(
            name="coder",
            description="Expert Python developer",
            agent=coder_agent,
            condition=lambda s: "code" in s.lower()
        ),
        handoff(
            name="researcher",
            description="Expert at web research",
            agent=researcher_agent,
            condition=lambda s: "search" in s.lower()
        ),
    ],
)
```

---

## 6. Sequential vs Concurrent

### Sequential

```
Step 1 ──► Step 2 ──► Step 3 ──► End
```

### Concurrent (Group Chat)

```
          ┌─────────┐
          │  User    │
          │  Query   │
          └─────┬────┘
                │
        ┌───────┴───────┬───────────────┐
        │                   │               │
        ▼                   ▼               ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Agent A       │  │  Agent B       │  │  Agent C       │
│  "Let's think" │  │  "I'll check"  │  │  "I'll verify"│
└───────────────┘  └───────────────┘  └───────────────┘
        │                   │               │
        └───────────┬───────┴───────────────┘
                      │
                      ▼
              ┌──────────────────────────┐
              │      Aggregation        │
              │  "Combining all views" │
              └──────────────────────────┘
```

---

## 7. Multi-Agent Protocol (MCP)

### Concept

Standard for agent-to-agent communication.

### Components

| Component | Purpose |
|------------|-----------|
| Agent Cards | Discovery and capabilities |
| Resources | Shared resources |
| Messages | Communication protocol |
| Prompts | Template-based interactions |

### Benefits

- **Discovery**: Agents can find each other
- **Standardization**: Common message format
- **Interoperability**: Cross-framework communication
- **Scalability**: Dynamic agent addition

---

## Pattern Selection Guide

```
┌────────────────────────────────────────────────────────────────┐
│                 Which Pattern to Use?                 │
├────────────────────────────────────────────────────────────────┤
│                                                          │
│  Simple, exploratory ────────────────────────┐       │
│  Use: ReAct                                    │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                          │
│  Multi-step, need visibility ───────────────────────┐       │
│  Use: Plan-and-Execute                          │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                          │
│  High-stakes, accuracy critical ────────────────────────┐       │
│  Use: Reflection                                 │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                          │
│  Multiple specialists needed ────────────────────────┐       │
│  Use: Supervisor-Worker                          │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                          │
│  Domain-specific tasks ────────────────────────┐           │
│  Use: Handoff/Routing                          │           │
│  └───────────────────────────────────────────────────────┘           │
└────────────────────────────────────────────────────────────────────┘
```

---

## Industry Sources

| Source | URL |
|---------|-----|
| Azure Agent Patterns | https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns |
| Google Design Patterns | https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system |
| OpenAI Practical Guide | https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf |
| LangChain Blog | https://blog.langchain.com/choosing-the-right-multi-agent-architecture |
| ArXiv Multi-Agent Survey | https://arxiv.org/abs/2601.14351 |
| DEV Architecture Guide | https://dev.to/sohail-akbar/the-ultimate-guide-to-ai-agent-architectures-in-2025-2j1c |
