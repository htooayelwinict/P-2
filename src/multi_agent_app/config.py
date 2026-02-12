from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ModelMode = Literal["auto", "openrouter", "offline"]


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        value = value[1:-1]
    return value


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = _parse_env_value(value)


@dataclass
class OpenRouterSettings:
    api_key: str | None
    base_url: str
    supervisor_model: str | None
    customer_model: str | None
    classifier_model: str | None
    temperature: float

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.supervisor_model and self.customer_model)

    @classmethod
    def from_env(cls) -> "OpenRouterSettings":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = (
            os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        default_model = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL")
        supervisor_model = os.getenv("OPENROUTER_SUPERVISOR_MODEL") or default_model
        customer_model = os.getenv("OPENROUTER_CUSTOMER_MODEL") or default_model
        classifier_model = os.getenv("OPENROUTER_CLASSIFIER_MODEL") or supervisor_model
        temperature_raw = os.getenv("OPENROUTER_TEMPERATURE", "0")
        try:
            temperature = float(temperature_raw)
        except ValueError:
            temperature = 0.0
        return cls(
            api_key=api_key,
            base_url=base_url,
            supervisor_model=supervisor_model,
            customer_model=customer_model,
            classifier_model=classifier_model,
            temperature=temperature,
        )

