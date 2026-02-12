from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from .config import ModelMode, OpenRouterSettings, load_dotenv


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    chunks.append(maybe_text)
            elif isinstance(item, str):
                chunks.append(item)
        return " ".join(chunks)
    return str(content)


class DeterministicToolChatModel(BaseChatModel):
    """Offline model used for local deterministic behavior."""

    agent_role: str = "agent"

    @property
    def _llm_type(self) -> str:
        return "deterministic-tool-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        # DeepAgents binds tool schemas to models; this local model does not need
        # schema-aware generation, but must support the call.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_text = _message_text(messages[-1].content if messages else "")
        lowered = last_text.lower()

        if "return exactly one label" in lowered and "task:" in lowered:
            task_text = lowered.split("task:", 1)[1]
            bridge_keywords = (
                "docs",
                "documentation",
                "customer",
                "user document",
                "knowledge base",
            )
            label = (
                "route_bridge"
                if any(keyword in task_text for keyword in bridge_keywords)
                else "respond_admin"
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=label))])

        content = f"{self.agent_role} handled: {last_text}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


@dataclass
class RuntimeModels:
    supervisor: BaseChatModel
    customer: BaseChatModel
    classifier: BaseChatModel
    source: str


def _openrouter_chat_model(
    *,
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> BaseChatModel:
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def resolve_runtime_models(base_dir: Path, mode: ModelMode = "auto") -> RuntimeModels:
    load_dotenv(base_dir / ".env")
    settings = OpenRouterSettings.from_env()

    if mode in {"auto", "openrouter"} and settings.enabled:
        supervisor = _openrouter_chat_model(
            model_name=settings.supervisor_model or "",
            api_key=settings.api_key or "",
            base_url=settings.base_url,
            temperature=settings.temperature,
        )
        customer = _openrouter_chat_model(
            model_name=settings.customer_model or "",
            api_key=settings.api_key or "",
            base_url=settings.base_url,
            temperature=settings.temperature,
        )
        classifier = _openrouter_chat_model(
            model_name=settings.classifier_model or settings.supervisor_model or "",
            api_key=settings.api_key or "",
            base_url=settings.base_url,
            temperature=0.0,
        )
        return RuntimeModels(
            supervisor=supervisor,
            customer=customer,
            classifier=classifier,
            source="openrouter",
        )

    if mode == "openrouter":
        raise ValueError(
            "OpenRouter model mode requested but required env vars are missing. "
            "Expected at least OPENROUTER_API_KEY and OPENROUTER_MODEL."
        )

    return RuntimeModels(
        supervisor=DeterministicToolChatModel(agent_role="supervisor"),
        customer=DeterministicToolChatModel(agent_role="customer-service"),
        classifier=DeterministicToolChatModel(agent_role="classifier"),
        source="offline",
    )
