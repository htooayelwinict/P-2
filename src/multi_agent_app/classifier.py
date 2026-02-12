from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel


RouteDecision = Literal["respond_admin", "route_bridge"]


class TaskClassifier(Protocol):
    def classify(self, admin_input: str) -> RouteDecision:
        ...


@dataclass
class LLMTaskClassifier:
    """Classifier that routes solely by LLM output label.

    The LLM is prompted with detailed criteria and examples to classify
    requests as either 'respond_admin' or 'route_bridge'.

    Args:
        model: Chat model used for classification.
    """

    model: BaseChatModel

    def classify(self, admin_input: str) -> RouteDecision:
        prompt = f"""You are a task router for an admin assistant system. Analyze the user's request and classify it into one of two routes.

ROUTES:
1. respond_admin - Handle directly as administrative/system tasks (coding, debugging, system operations, data analysis)
2. route_bridge - Route to specialized knowledge base for customer-facing documentation and support content

DECISION CRITERIA:
- Use 'route_bridge' if the request involves:
  * Reading, writing, or updating customer-facing documentation
  * Searching knowledge bases or help articles
  * Answering questions based on user manuals or guides
  * Creating content for end-users or customers
  * Handling support documentation queries

- Use 'respond_admin' if the request involves:
  * Writing or debugging code
  * System administration tasks
  * Data processing or analysis
  * Internal tooling or automation
  * Development-related queries
  * General conversation or questions not requiring specialized docs

EXAMPLES:
- "Fix the authentication bug in login.py" → respond_admin
- "Update the API reference docs for customers" → route_bridge
- "Search our knowledge base for password reset instructions" → route_bridge
- "Analyze this CSV file and generate a report" → respond_admin
- "Write a help article about how users can export data" → route_bridge

USER REQUEST: {admin_input}

OUTPUT FORMAT: Return exactly one word - either 'route_bridge' or 'respond_admin'. No explanation needed."""
        response = self.model.invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
        lowered = text.lower()

        if "route_bridge" in lowered:
            return "route_bridge"
        return "respond_admin"
