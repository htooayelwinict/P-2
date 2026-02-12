from typing import Literal, TypedDict


class GlobalState(TypedDict, total=False):
    """State used by supervisor and bridge."""

    origin: Literal["admin_cli", "supervisor", "bridge"]
    admin_input: str
    route: Literal["respond_admin", "route_bridge"]
    supervisor_response: str
    customer_response: str
    bridge_admin_input: str

    # Admin-only fields should never cross into UnsafeState.
    secret_context: str
    secret_key_ref: str


class UnsafeState(TypedDict, total=False):
    """State used by customer-facing node."""

    origin: Literal["user_cli", "bridge"]
    user_input: str
    bridge_input: str
    response: str
