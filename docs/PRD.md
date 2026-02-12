# PRD: Multi-Agent Supervisor-Bridge-Customer Architecture

## Status
- Drafted for implementation
- Date: 2026-02-12

## Objective
Implement a multi-agent architecture with strict state boundaries:
- `Supervisor Agent Node` with `GlobalState` for administrator workflows
- `Bridge Node` with `GlobalState` that accepts only supervisor-originated traffic
- `Customer Service Agent Node` with `UnsafeState` for user-facing workflows

## Primary Actors
- Administrator (admin CLI mode)
- End User (user CLI mode)

## Scope
### In Scope
- Two separate CLI modes:
  - `admin` mode routes through `Supervisor Agent Node`
  - `user` mode routes directly to `Customer Service Agent Node`
- Supervisor supports admin tasks and can use `read_file` on `/admin/**`
- Supervisor classifies `admin_input: str` and conditionally routes to bridge
- Bridge only accepts requests from supervisor
- Bridge forwards only selected state:
  - `admin_input: str`
- Customer Service Agent supports user CLI and can use `read_file` on `/docs/**`
- No sanitization logic in this phase

### Out of Scope
- Input/output sanitization and redaction
- External authentication/authorization services
- Remote model hosting and deployment infrastructure

## Functional Requirements
1. `Supervisor Agent Node`
- Input: `admin_input: str`
- Tools: `read_file` scoped to `/admin/**`
- Behavior:
  - classify task intent
  - either answer admin directly
  - or route to bridge via conditional edge

2. `Bridge Node`
- Input source: supervisor only
- Validation:
  - reject non-supervisor origin
- Output payload:
  - include only `admin_input: str`
  - drop all other fields

3. `Customer Service Agent Node`
- State model: `UnsafeState`
- Entry points:
  - user CLI mode
  - bridge payload from supervisor route
- Tools: `read_file` scoped to `/docs/**`

4. CLI
- Single executable entry with mode switch:
  - `--mode admin`
  - `--mode user`
- Interactive loop with `exit|quit` termination

## State Contracts
1. `GlobalState`
- Used by supervisor and bridge
- Contains admin-facing fields

2. `UnsafeState`
- Used by customer node
- Must not include supervisor-only secret fields
- Bridge may populate only `admin_input`

## Non-Functional Requirements
- Deterministic file access constraints by path allowlist
- Clear, testable routing between nodes
- Minimal dependencies and local execution

## Acceptance Criteria
1. Admin CLI can read files under `/admin/**`.
2. User CLI cannot read files under `/admin/**`.
3. Customer node can read files under `/docs/**`.
4. Supervisor conditional routing can send admin task to bridge.
5. Bridge rejects non-supervisor callers.
6. Bridge forwards only `admin_input` field.
7. System runs locally via CLI in both modes.

