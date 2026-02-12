from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.language_models.chat_models import BaseChatModel

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
    worker: DeepAgentsScopedNode

    @classmethod
    def create(cls, base_dir: Path, worker_model: BaseChatModel) -> "CustomerServiceAgentNode":
        worker = DeepAgentsScopedNode.create(
            scope_name="docs",
            scope_root=base_dir / "docs",
            model=worker_model,
            system_prompt=(
                "You are a customer-facing documentation assistant. "
                "Use built-in filesystem tools for file operations."
            ),
        )
        return cls(worker=worker)

    def invoke(self, state: UnsafeState) -> UnsafeState:
        input_text = (state.get("user_input") or state.get("bridge_input") or "").strip()
        if not input_text:
            return {**state, "response": "Empty user input."}
        return {**state, "response": self.worker.respond(input_text)}
