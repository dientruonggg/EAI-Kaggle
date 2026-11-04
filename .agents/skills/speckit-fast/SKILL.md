---
name: "speckit-fast"
description: "Executes a streamlined 4-step fast track: specify → plan → tasks → implement without review gates for rapid prototyping."
compatibility: "Requires spec-kit project structure with .specify/ directory"
metadata:
  author: "EAI-2026 Team"
  source: "workflows/fast-sdd"
---

## User Input

```text
$ARGUMENTS
```

## Purpose & Execution Sequence

When rapid implementation is preferred without manual review gates or multi-pass audits:

1. **Phase 1: `/speckit-specify`**
   - Directly draft baseline specification in `specs/<id>/spec.md`.
2. **Phase 2: `/speckit-plan`**
   - Formulate technical plan in `specs/<id>/plan.md`.
3. **Phase 3: `/speckit-tasks`**
   - Breakdown immediate tasks in `specs/<id>/tasks.md`.
4. **Phase 4: `/speckit-implement`**
   - Immediately execute tasks sequentially.

## Execution Directive

When invoked as `/speckit-fast <arguments>`, produce the spec, plan, and tasks in a rapid pass, then begin implementation immediately.
