# DeepAgents SDK Reference

**Last Updated:** 2026-02-12

---

## Overview

DeepAgents is a **standalone agent harness** built on LangChain and LangGraph, inspired by Claude Code, Deep Research, and Manus.

```
┌────────────────────────────────────────────────────────────────────┐
│                    DeepAgents Architecture                      │
├────────────────────────────────────────────────────────────────────┤
│                                                              │
│  Core Philosophy                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Opinionated agent harness with batteries included     │    │
│  │  - Planning (write_todos)                        │    │
│  │  - Filesystem (read, write, edit)                │    │
│  │  - Subagents (spawn, delegate)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                              │
│  Built On                                                   │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  LangGraph (durable execution, streaming)       │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                              │
│  Two Components                                           │
│  ┌────────────────────────────────────────────────────────┐       │
│  │  SDK: Middleware for building custom agents    │       │
│  │  CLI: Interactive coding tool                │       │
│  └────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# SDK only
pip install deepagents

# With CLI
pip install "deepagents[cli]"

# From source
pip install -e "git+https://github.com/langchain-ai/deepagents.git"
```

---

## Middleware Architecture

DeepAgents uses **composable middleware** to add capabilities:

```
┌──────────────────────────────────────────────────────────────────┐
│                   create_deep_agent()                      │
├────────────────────────────────────────────────────────────────────┤
│                                                             │
│                   Middleware Stack                              │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐       │
│  │  1. TodoListMiddleware                              │       │
│  │     - Adds write_todos tool                       │       │
│  │     - Task decomposition                             │       │
│  │     - Progress tracking                               │       │
│  ├───────────────────────────────────────────────────────┤       │
│  │  2. FilesystemMiddleware                               │       │
│  │     - ls, read_file, write_file, edit_file       │       │
│  │     - grep, glob                                  │       │
│  │     - Pluggable backends                        │       │
│  ├───────────────────────────────────────────────────────┤       │
│  │  3. SubAgentMiddleware                                │       │
│  │     - Spawning subagents                          │       │
│  │     - Context isolation                              │       │
│  │     - Agent-to-agent communication                  │       │
│  └───────────────────────────────────────────────────────┘       │
│                                                             │
│  [!] All middleware is composable and optional              │
└───────────────────────────────────────────────────────────────────────┘
```

---

## TodoListMiddleware

### Purpose

Planning and task decomposition for multi-step workflows.

### Provided Tools

| Tool | Description |
|-------|-------------|
| `write_todos` | Create todo items from user goal |
| `update_todos` | Update existing todos |
| `complete_todos` | Mark todos as complete |

### Usage

```python
from deepagents.middleware.todos import TodoListMiddleware
from langchain.agents import create_deep_agent

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    middleware=[
        TodoListMiddleware(
            max_todos=10,
            prioritization="auto"
        )
    ],
)
```

---

## FilesystemMiddleware

### Purpose

Provides file system access for context management and long-term memory.

### Provided Tools

| Tool | Description | Example |
|-------|-------------|----------|
| `ls` | List files | `ls("/path")` |
| `read_file` | Read file contents | `read_file("/path/file.txt")` |
| `write_file` | Write new file | `write_file("/path/new.txt", "content")` |
| `edit_file` | Edit existing file | `edit_file("/path/file.txt", replacements)` |
| `glob` | Pattern matching | `glob("**/*.py")` |
| `grep` | Search contents | `grep("pattern", "/path")` |

### Built-in Backends

| Backend | Description |
|----------|-------------|
| `LocalBackend` | Local filesystem access |
| `PostgresBackend` | PostgreSQL database |
| `MemoryBackend` | In-memory storage |
| `S3Backend` | AWS S3 storage |
| `CompositeBackend` | Multiple backends combined |

### Usage

```python
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.postgres import PostgresBackend

# Create backend
backend = PostgresBackend(
    host="localhost",
    database="agent_files",
    user="postgres",
    password="..."
)

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    middleware=[
        FilesystemMiddleware(backend=backend)
    ],
)
```

### Custom Backend

```python
from deepagents.backends import BackendProtocol

class S3Backend(BackendProtocol):
    def __init__(self, bucket: str):
        self.s3 = boto3.client("s3")

    def ls_info(self, path: str) -> list[FileInfo]:
        # List objects in S3
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=path
        )
        return [
            FileInfo(
                path=obj["Key"],
                size=obj["Size"],
                modified_at=obj["LastModified"]
            )
            for obj in response["Contents"]
        ]

    # Implement other methods...
```

---

## SubAgentMiddleware

### Purpose

Delegate work to specialized agents with context isolation.

### Key Features

| Feature | Description |
|----------|-------------|
| Spawning | Create new subagent instances |
| Context Isolation | Subagents have separate state |
| Independent Models | Use different models per subagent |
| Custom Tools | Subagents have their own toolset |
| Delegation | Main agent can route to specialists |

### Subagent Schema

```python
from deepagents.middleware.subagents import SubAgentMiddleware

SubAgentMiddleware(
    default_model="claude-sonnet-4-5-20250929",
    default_tools=[],  # Tools for main agent
    subagents=[
        {
            # Required fields
            "name": "weather",  # Unique identifier
            "description": "Get weather for cities",  # For routing

            # Optional customization
            "system_prompt": "You are a weather specialist",
            "tools": [get_weather],
            "model": "gpt-4.1-mini",  # Different model
            "middleware": [],  # Subagent middleware

            # Advanced options
            "max_loops": 5,  # Limit iterations
            "timeout": 60,  # Seconds
            "handoff": True,  # Can handoff to other subagents
        }
    ],
)
```

### Subagent Communication

```python
# Main agent can delegate to subagent
def delegate_to_subagent(state: AgentState) -> str:
    return "weather_subagent"  # Routes to weather subagent

# Subagent can handoff back
def handback_to_main(state: AgentState) -> str:
    return "main_agent"
```

---

## create_deep_agent()

```python
from langchain.agents import create_deep_agent
from deepagents.middleware.todos import TodoListMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

agent = create_deep_agent(
    # Model configuration
    model="claude-sonnet-4-5-20250929",
    temperature=0.7,

    # Middleware stack
    middleware=[
        TodoListMiddleware(),
        FilesystemMiddleware(backend=LocalBackend()),
        SubAgentMiddleware(subagents=[...]),
    ],

    # Optional: custom tools for main agent
    tools=[custom_tool],

    # Optional: system prompt
    system_prompt="You are a helpful coding assistant",
)
```

---

## DeepAgents CLI

### Installation

```bash
# CLI is included with full install
pip install "deepagents[cli]"

# Or install separately
pip install deepagents-cli
```

### Usage

```bash
# Start interactive CLI
deepagents

# With custom agent
deepagents --agent my_agent.py

# With memory backend
deepagents --backend postgres://localhost/agent_db

# Resume from thread
deepagents --thread-id abc-123
```

### CLI Features

| Feature | Description |
|----------|-------------|
| Interactive Chat | Terminal-based agent interaction |
| Conversation Resume | Continue from previous session |
| Web Search | Built-in search capabilities |
| Remote Sandboxes | Modal, Runloop, Daytona support |
| Persistent Memory | Cross-session memory |
| Custom Skills | Extend with custom skills |
| Human Approval | Approve tool calls manually |

---

## Filesystem Backend Protocol

### Interface

```python
from deepagents.backends import BackendProtocol, FileInfo, GrepMatch

class CustomBackend(BackendProtocol):
    """Implement all required methods."""

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files at path."""

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Glob pattern matching."""

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None
    ) -> list[GrepMatch] | str:
        """Search file contents."""
```

### PostgresBackend Example

```sql
-- Schema for filesystem backend
CREATE TABLE files (
    path VARCHAR(512) PRIMARY KEY,
    content TEXT,
    size INTEGER,
    modified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_files_path ON files(path);
CREATE INDEX idx_files_glob ON files USING gin(path gin_trgm_ops);
```

---

## Patterns

### 1. Specialist Subagents

```python
SubAgentMiddleware(subagents=[
    {
        "name": "coder",
        "description": "Expert Python developer",
        "model": "claude-sonnet-4-5-20250929",
        "tools": [write_file, edit_file],
    },
    {
        "name": "tester",
        "description": "Runs pytest and interprets results",
        "model": "gpt-4.1-mini",
        "tools": [run_tests, read_file],
    },
    {
        "name": "reviewer",
        "description": "Code review expert",
        "model": "claude-3-5-sonnet-20241022",
        "tools": [read_file],
    },
])
```

### 2. Hierarchical Planning

```python
# Main agent uses write_todos to plan
# Then delegates to subagents for execution
# Subagents update todos as they complete tasks
```

### 3. Context Isolation

```python
# Main agent context: user goal, overall progress
# Subagent context: specific task details, file workspace
# No shared state - communication via explicit delegation
```

---

## Best Practices

| Area | Recommendation |
|--------|----------------|
| Model Selection | Use cheaper models for subagents (Haiku, mini) |
| Filesystem | Use PostgresBackend for production persistence |
| Subagents | Limit to 3-5 specialists per agent |
| Memory | Use FilesystemMiddleware for long-term context |
| Testing | Test each subagent independently |
| Monitoring | Use LangSmith tracing across subagents |

---

## External Resources

| Resource | URL |
|-----------|-----|
| GitHub | https://github.com/langchain-ai/deepagents |
| GitHub (CLI) | https://github.com/langchain-ai/deep-agents-from-scratch |
| Documentation | https://docs.langchain.com/oss/python/deepagents |
| Blog (Multi-Agent) | https://blog.langchain.com/building-multi-agent-applications-with-deep-agents |
| CLI UI | https://github.com/langchain-ai/deep-agents-ui |
