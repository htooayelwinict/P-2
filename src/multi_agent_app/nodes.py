from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
import uuid

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .classifier import LLMTaskClassifier, TaskClassifier
from .states import GlobalState, UnsafeState


class BridgeAccessError(PermissionError):
    """Raised when a non-supervisor caller tries to invoke the bridge."""


def normalize_scope_prefixes(input_text: str, scope_name: str) -> str:
    """Map '/<scope>/...' references onto scoped root '/' for virtual backends."""
    if not input_text:
        return input_text
    scoped = re.sub(rf"(?<!\\w)/{re.escape(scope_name)}/", "/", input_text)
    scoped = re.sub(rf"(?<!\\w)/{re.escape(scope_name)}(?=\\s|$|[.,;:!?])", "/", scoped)
    return scoped


@dataclass
class DeepAgentsScopedNode:
    scope_name: str
    backend: FilesystemBackend
    model: BaseChatModel
    agent: Any

    @classmethod
    def create(
        cls,
        scope_name: str,
        scope_root: Path,
        model: BaseChatModel,
        system_prompt: str,
    ) -> "DeepAgentsScopedNode":
        backend = FilesystemBackend(root_dir=scope_root, virtual_mode=True)
        agent = create_deep_agent(
            model=model,
            backend=backend,
            system_prompt=system_prompt,
            checkpointer=False,
        )
        return cls(scope_name=scope_name, backend=backend, model=model, agent=agent)

    def respond(self, input_text: str) -> str:
        normalized_input = normalize_scope_prefixes(input_text, self.scope_name)
        result = self.agent.invoke({"messages": [{"role": "user", "content": normalized_input}]})
        messages = result.get("messages", [])
        if not messages:
            return ""
        last = messages[-1]
        content = getattr(last, "content", "")
        return content if isinstance(content, str) else str(content)


@dataclass
class ReadOnlyScopedNode:
    """DeepAgents node with write operations blocked via interrupt_on.

    Only allows read_file and ls operations. Write, edit, glob, and grep
    are automatically rejected without human intervention.
    """

    scope_name: str
    backend: FilesystemBackend
    model: BaseChatModel
    agent: Any
    checkpointer: MemorySaver = field(default_factory=MemorySaver)

    # Tools to auto-reject (read-only enforcement)
    BLOCKED_TOOLS: tuple[str, ...] = ("write_file", "edit_file", "glob", "grep")

    @classmethod
    def create(
        cls,
        scope_name: str,
        scope_root: Path,
        model: BaseChatModel,
        system_prompt: str,
    ) -> "ReadOnlyScopedNode":
        backend = FilesystemBackend(root_dir=scope_root, virtual_mode=True)
        checkpointer = MemorySaver()

        # Configure interrupt_on to block write operations
        interrupt_config = {
            tool: {"allowed_decisions": ["reject"]}
            for tool in cls.BLOCKED_TOOLS
        }

        agent = create_deep_agent(
            model=model,
            backend=backend,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            interrupt_on=interrupt_config,
        )
        return cls(
            scope_name=scope_name,
            backend=backend,
            model=model,
            agent=agent,
            checkpointer=checkpointer,
        )

    def respond(self, input_text: str) -> str:
        """Invoke agent with auto-reject loop for blocked tools."""
        normalized_input = normalize_scope_prefixes(input_text, self.scope_name)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": normalized_input}]},
            config=config,
        )

        # Auto-reject loop: if agent hit an interrupt, reject and continue
        while result.get("__interrupt__"):
            # Resume with auto-reject (decisions is a list with type field)
            result = self.agent.invoke(
                Command(
                    resume={
                        "decisions": [
                            {
                                "type": "reject",
                                "message": "SECURITY ALERT: file creation is not allowed. Terminate this attempt and do not retry with different paths.",
                            }
                        ]
                    }
                ),
                config=config,
            )

        messages = result.get("messages", [])
        if not messages:
            return ""
        last = messages[-1]
        content = getattr(last, "content", "")
        return content if isinstance(content, str) else str(content)


@dataclass
class SupervisorAgentNode:
    classifier: TaskClassifier
    worker: DeepAgentsScopedNode

    @classmethod
    def create(
        cls,
        base_dir: Path,
        worker_model: BaseChatModel,
        classifier_model: BaseChatModel,
        classifier: TaskClassifier | None = None,
    ) -> "SupervisorAgentNode":
        worker = DeepAgentsScopedNode.create(
            scope_name="admin",
            scope_root=base_dir / "admin",
            model=worker_model,
            system_prompt=(
                "You are the supervisor agent for administrator workflows. "
                "Use built-in filesystem tools for file operations."
            ),
        )
        route_classifier = classifier or LLMTaskClassifier(model=classifier_model)
        return cls(classifier=route_classifier, worker=worker)

    def invoke(self, state: GlobalState) -> GlobalState:
        admin_input = state.get("admin_input", "").strip()
        if not admin_input:
            return {
                **state,
                "origin": "supervisor",
                "route": "respond_admin",
                "supervisor_response": "Empty admin input.",
            }

        route = self.classifier.classify(admin_input)
        if route == "route_bridge":
            return {
                **state,
                "origin": "supervisor",
                "route": "route_bridge",
            }

        response = self.worker.respond(admin_input)
        return {
            **state,
            "origin": "supervisor",
            "route": "respond_admin",
            "supervisor_response": response,
        }


@dataclass
class BridgeNode:
    """Security gate that projects GlobalState to UnsafeState.

    Enforces two critical security properties:
    1. Access Control: Only accepts state with origin="supervisor"
    2. Secret Stripping: Forwards ONLY admin_input as bridge_input

    Returns UnsafeState which cannot hold secret_context or secret_key_ref.
    """

    def invoke(self, state: GlobalState) -> UnsafeState:
        if state.get("origin") != "supervisor":
            raise BridgeAccessError("Bridge accepts input only from supervisor.")

        return {
            "origin": "bridge",
            "bridge_input": state.get("admin_input", ""),
        }


@dataclass
class CustomerServiceAgentNode:
    """Customer-facing agent with read-only filesystem access.

    Uses ReadOnlyScopedNode to block write, edit, glob, and grep operations.
    Only read_file and ls are permitted.
    """

    worker: ReadOnlyScopedNode

    @classmethod
    def create(cls, base_dir: Path, worker_model: BaseChatModel) -> "CustomerServiceAgentNode":
        worker = ReadOnlyScopedNode.create(
            scope_name="docs",
            scope_root=base_dir / "docs",
            model=worker_model,
            system_prompt=(
                "You are a customer-facing documentation assistant. "
                "Use built-in filesystem tools for file operations. "
                "You can only read files and list directories."
            ),
        )
        return cls(worker=worker)

    def invoke(self, state: UnsafeState) -> UnsafeState:
        input_text = (state.get("user_input") or state.get("bridge_input") or "").strip()
        if not input_text:
            return {**state, "response": "Empty user input."}
        return {**state, "response": self.worker.respond(input_text)}
