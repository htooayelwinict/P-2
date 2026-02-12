from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .config import ModelMode
from .classifier import TaskClassifier
from .models import resolve_runtime_models
from .nodes import BridgeNode, CustomerServiceAgentNode, SupervisorAgentNode
from .states import GlobalState, UnsafeState


@dataclass
class MultiAgentRuntime:
    base_dir: Path
    model_source: str
    supervisor: SupervisorAgentNode
    bridge: BridgeNode
    customer: CustomerServiceAgentNode
    admin_graph: Any
    user_graph: Any

    @classmethod
    def create(
        cls,
        base_dir: Path | None = None,
        model_mode: ModelMode = "auto",
        classifier: TaskClassifier | None = None,
    ) -> "MultiAgentRuntime":
        root = base_dir or Path(__file__).resolve().parents[2]
        models = resolve_runtime_models(root, mode=model_mode)
        supervisor = SupervisorAgentNode.create(
            base_dir=root,
            worker_model=models.supervisor,
            classifier_model=models.classifier,
            classifier=classifier,
        )
        customer = CustomerServiceAgentNode.create(base_dir=root, worker_model=models.customer)
        runtime = cls(
            base_dir=root,
            model_source=models.source,
            supervisor=supervisor,
            bridge=BridgeNode(),
            customer=customer,
            admin_graph=None,
            user_graph=None,
        )
        runtime.admin_graph = runtime._build_admin_graph()
        runtime.user_graph = runtime._build_user_graph()
        return runtime

    def _route_after_supervisor(self, state: GlobalState) -> str:
        return state.get("route", "respond_admin")

    def _bridge_node(self, state: GlobalState) -> GlobalState:
        """Project GlobalState to UnsafeState boundary.

        This is the security checkpoint where:
        1. BridgeNode.invoke() validates origin == "supervisor"
        2. Only admin_input is forwarded (secrets dropped)
        3. Returns GlobalState with bridge_admin_input for next node

        The returned dict is merged into admin graph's GlobalState.
        """
        payload: UnsafeState = self.bridge.invoke(state)
        return {
            "origin": "bridge",
            "bridge_admin_input": payload.get("bridge_input", ""),
        }

    def _customer_from_bridge_node(self, state: GlobalState) -> GlobalState:
        """Invoke customer service with sanitized state.

        Extracts bridge_admin_input from GlobalState and constructs a fresh
        UnsafeState for the customer node. The customer never sees secrets.
        """
        unsafe_state: UnsafeState = {
            "origin": "bridge",
            "bridge_input": state.get("bridge_admin_input", ""),
        }
        result = self.customer.invoke(unsafe_state)
        return {
            "customer_response": result.get("response", ""),
        }

    def _build_admin_graph(self):
        builder = StateGraph(GlobalState)
        builder.add_node("supervisor", self.supervisor.invoke)
        builder.add_node("bridge", self._bridge_node)
        builder.add_node("customer_from_bridge", self._customer_from_bridge_node)
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {"respond_admin": END, "route_bridge": "bridge"},
        )
        builder.add_edge("bridge", "customer_from_bridge")
        builder.add_edge("customer_from_bridge", END)
        return builder.compile()

    def _build_user_graph(self):
        builder = StateGraph(UnsafeState)
        builder.add_node("customer", self.customer.invoke)
        builder.add_edge(START, "customer")
        builder.add_edge("customer", END)
        return builder.compile()

    def run_admin_turn(self, admin_input: str) -> str:
        state: GlobalState = {"origin": "admin_cli", "admin_input": admin_input}
        result = self.admin_graph.invoke(state)
        if result.get("route") == "route_bridge":
            return result.get("customer_response", "")
        return result.get("supervisor_response", "")

    def run_user_turn(self, user_input: str) -> str:
        state: UnsafeState = {"origin": "user_cli", "user_input": user_input}
        result = self.user_graph.invoke(state)
        return result.get("response", "")
