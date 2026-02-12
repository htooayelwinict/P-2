# LangChain DeepAgents & LangGraph - Research Summary

**Research Date:** 2026-02-12
**Sources:** Official docs, Context7 MCP, Gemini MCP, Web Search, Academic Papers

---

## Quick Reference

| Topic | Document |
|--------|----------|
| Overview & Ecosystem | [01-LANGCHAIN-ECOSYSTEM.md](01-LANGCHAIN-ECOSYSTEM.md) |
| LangGraph Framework | [02-LANGGRAPH-REFERENCE.md](02-LANGGRAPH-REFERENCE.md) |
| DeepAgents SDK | [03-DEEPAGENTS-SDK.md](03-DEEPAGENTS-SDK.md) |
| Agent Architecture Patterns | [04-AGENT-PATTERNS.md](04-AGENT-PATTERNS.md) |
| Production Best Practices | [05-BEST-PRACTICES.md](05-BEST-PRACTICES.md) |

---

## Framework Selection Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                    When to use what?                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                             │
│  Quick prototyping          │     │
│  Simple tasks              │     │  Use LangChain create_agent
│  <10 lines of code        │     │
│                           │     │
│  Complex workflows          ───────┼─────────┐
│  State management          │           │           │
│  Custom orchestration     │           │  Use LangGraph StateGraph
│  Production deployment     │           │           │
│                           │           │           │
│  Multi-step planning       ───────────┼───────────┤
│  Task decomposition       │              │           │  Use DeepAgents SDK
│  File system access       │              │           │
│  Subagent spawning       │              │           │
│  Persistent memory        │              │           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Insights (2025-2026)

1. **LangChain v1.0** = Major simplification, dropping Pydantic models
2. **DeepAgents** = New standard for complex, multi-step agents
3. **LangGraph** = Foundation for all agent runtimes
4. **Middleware Architecture** = Default pattern for production systems
5. **Supervisor-Worker** = Dominant multi-agent pattern

---

## External Sources

- **Official:** https://python.langchain.com, https://docs.langchain.com
- **GitHub:** https://github.com/langchain-ai/deepagents
- **Context7:** /websites/langchain, /langchain_oss_python_langgraph
- **Papers:** arXiv:2601.14351 (Multi-Agent Systems)
