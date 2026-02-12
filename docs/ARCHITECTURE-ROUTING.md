# Architecture Routing and State Flow

## Purpose

This document explains how routing and state passing work in the current implementation.
The system uses:

- LangGraph for orchestration and state flow control
- DeepAgents for node behavior and built-in filesystem tools
- Scoped filesystem backends for admin and customer isolation

## Source of Truth in Code

- Runtime graph assembly: `src/multi_agent_app/runtime.py`
- State schemas: `src/multi_agent_app/states.py`
- Node behavior: `src/multi_agent_app/nodes.py`
- Classifier behavior: `src/multi_agent_app/classifier.py`

## High-Level Graph Topology

```text
ADMIN GRAPH (GlobalState)
=========================

START
  |
  v
[supervisor]
  |-- route = "respond_admin" --> END
  |
  |-- route = "route_bridge" --> [bridge] --> [customer_from_bridge] --> END


USER GRAPH (UnsafeState)
========================

START --> [customer] --> END
```

## State Schemas

### GlobalState (admin graph)

```text
origin: "admin_cli" | "supervisor" | "bridge"
admin_input: str
route: "respond_admin" | "route_bridge"
supervisor_response: str
customer_response: str
bridge_admin_input: str
secret_context: str
secret_key_ref: str
```

### UnsafeState (customer graph)

```text
origin: "user_cli" | "bridge"
user_input: str
bridge_input: str
response: str
```

Important boundary:
- `admin_input` is used in `GlobalState`
- `bridge_input` is used in `UnsafeState`

## Routing Logic

Supervisor node behavior:

1. Read `admin_input`.
2. If empty, return `route="respond_admin"` with empty-input response.
3. Otherwise call classifier.
4. If classifier returns `route_bridge`, branch to bridge path.
5. Else stay in supervisor path and respond directly.

Classifier behavior:

1. Prompts model to return one label:
   - `route_bridge`
   - `respond_admin`
2. Applies keyword guard:
   - route to bridge only when docs/customer intent keywords are present
   - otherwise force `respond_admin`
3. Uses deterministic fallback to the same guard if model output is unexpected.

## Bridge Contract

The bridge accepts only supervisor-originated payloads.

Input contract (from `GlobalState`):

```text
origin must be "supervisor"
admin_input is the only payload forwarded
```

Bridge output contract (`UnsafeState`):

```text
{
  "origin": "bridge",
  "bridge_input": <admin_input>
}
```

If `origin != "supervisor"`, bridge raises `BridgeAccessError`.

## State Passing Sequence

```text
Admin CLI input:
  { origin="admin_cli", admin_input="..." }

  -> supervisor.invoke(...)
     returns route and response fields on GlobalState

  if route == respond_admin:
     END
     return supervisor_response

  if route == route_bridge:
     -> bridge.invoke(GlobalState)
        => UnsafeState { origin="bridge", bridge_input=admin_input }

     -> runtime bridge wrapper maps to GlobalState field:
        bridge_admin_input = bridge_input

     -> customer_from_bridge builds UnsafeState:
        { origin="bridge", bridge_input=bridge_admin_input }

     -> customer.invoke(...)
        returns UnsafeState.response

     -> END
     return customer_response
```

## DeepAgents Scope Model

Each node has its own DeepAgents instance with a scoped backend:

- Supervisor worker:
  - `FilesystemBackend(root_dir=<base>/admin, virtual_mode=True)`
- Customer worker:
  - `FilesystemBackend(root_dir=<base>/docs, virtual_mode=True)`

`virtual_mode=True` is required so root_dir is an actual boundary.

### Path Prefix Normalization

Before sending user text to a scoped DeepAgents node, input is normalized:

- `/admin/...` becomes `/...` inside admin scope
- `/docs/...` becomes `/...` inside docs scope

This keeps user-facing path phrasing compatible with virtual scoped roots.

## ASCII Infographic: End-to-End

```text
                +--------------------------------------+
                |          ADMIN MODE (CLI)            |
                +------------------+-------------------+
                                   |
                                   v
                      GlobalState {origin=admin_cli,
                                   admin_input=...}
                                   |
                                   v
                       +-----------------------+
                       |   supervisor node     |
                       | classify + respond    |
                       +----------+------------+
                                  |
                     +------------+-------------+
                     |                          |
     route=respond_admin                 route=route_bridge
                     |                          |
                     v                          v
                 END (admin)              +-------------+
                 return                   | bridge node |
          supervisor_response             +------+------+ 
                                                 |
                                                 v
                                  UnsafeState {origin=bridge,
                                               bridge_input=...}
                                                 |
                                                 v
                                   +--------------------------+
                                   | customer_from_bridge node|
                                   +------------+-------------+
                                                |
                                                v
                                        customer node
                                                |
                                                v
                                           END (admin)
                                           return
                                        customer_response


                +--------------------------------------+
                |           USER MODE (CLI)            |
                +------------------+-------------------+
                                   |
                                   v
                      UnsafeState {origin=user_cli,
                                   user_input=...}
                                   |
                                   v
                             customer node
                                   |
                                   v
                               END (user)
                              return response
```

## Current Invariants

1. Admin and user entrypoints use separate graphs.
2. Bridge only accepts supervisor-originated state.
3. Bridge payload into customer side uses `bridge_input` (not `admin_input`).
4. DeepAgents filesystem access is scoped by `virtual_mode=True`.
5. Routing control is in LangGraph, execution behavior is in DeepAgents nodes.
