---
'@composio/core': minor
'@composio/slim': minor
'@composio/experimental': minor
---

Adds an experimental agent kit to `@composio/core/experimental`. `LLMDelegate` sends sub-tasks to OpenAI-compatible providers such as Groq and OpenRouter with your own API key, with per-agent call and token budgets, a model allow-list, retries, and the key redacted from errors. `MemoryStore` and `ScopedMemory` give several agents one shared memory with `personal`, `project:*` and `team:*` spaces, per-agent read and write policies, provenance, a quarantine for entries that come from external content, and refusal of secrets. `createDelegateTool` and `createMemoryTools` expose both as local tools (`LOCAL_DELEGATE_LLM`, `LOCAL_MEMORY_SEARCH`, `LOCAL_MEMORY_WRITE`, `LOCAL_MEMORY_FORGET`) for `composio.create(userId, { experimental: { localTools } })`. `@composio/experimental` adds `createFileMemoryStorage`, a lock-protected file storage that lets several processes share one memory file on Node.
