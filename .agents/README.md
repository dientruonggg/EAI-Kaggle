# Agent Control Layer (`.agents/`)

<agent_control_layer version="1.0">

<purpose>
  Universal, project-agnostic agent control rules. These files define HOW the
  agent thinks and behaves — not what it builds or how tasks are tracked.

  Copy this directory verbatim into any repository. No project-specific edits required.
</purpose>

<boundary>
  `.agents/` owns: mode declaration, session startup protocol, retry budgets,
  escalation triggers, completion gate, and code/naming conventions.

  `.agents/` NEVER owns: task workflow (→ `process/`), artifact chain
  (→ `process/_seeds/`), toolchain commands (→ `AGENTS.md`), or project context
  (→ `process/context/`).
</boundary>

## File Catalog

| File | Purpose |
|------|---------|
| [`behavior.md`](behavior.md) | Mode declaration, session startup, persistent state reload |
| [`guardrails.md`](guardrails.md) | Retry budget, escalation triggers, stop conditions, completion gate |
| [`conventions/naming.md`](conventions/naming.md) | Universal naming and code hygiene rules |

</agent_control_layer>
