# Agent Guardrails

<agent_guardrails version="1.0">

<description>
  Universal safety boundaries for agent execution: retry budget, escalation
  triggers, stop conditions, and completion gate. Project-agnostic — applies
  regardless of language, framework, or toolchain.
</description>

---

## 1. Retry Budget

<retry_budget max_attempts="3">
  Maximum 3 consecutive attempts per distinct failure symptom.

  Rules:
  - Never suppress errors with flags (e.g. `# type: ignore`, `eslint-disable`,
    `@SuppressWarnings`) to artificially pass a gate.
  - Record every failure in `state.md > <failure_memory>` with:
    - failure signature (unique symptom description)
    - hypothesis tested
    - experiment run
    - result observed
  - If retry budget is exhausted: HALT, log into `<open_decisions>` in `task.md`,
    and request human guidance.
</retry_budget>

---

## 2. Escalation & Stop Conditions

<escalation_triggers>
  Halt immediately and request human guidance if ANY of the following occur:

  1. Retry budget exhausted on a recurring failure.
  2. Required change touches a public API, database schema, or security policy
     not declared in the approved plan.
  3. Required change touches a file outside the scope defined in `plan.md`.
  4. A business or policy decision is needed that is not in `<approved_decisions>`.
     (Exception: Under DELEGATED / Fast-Track mode, the agent is authorized to select the recommended technical option evaluated in `decision.md` without triggering escalation, provided it preserves `<invariants>` and `<out_of_scope>`).
  5. Scope expansion is needed beyond `<out_of_scope>` in `task.md`.

  Exception: Do NOT halt if the change was explicitly authorized by the user
  in the prompt/spec (under DELEGATED, fast-track, or skip permissions), or if it is a mandatory accompanying test or import update.
</escalation_triggers>

---

## 3. Completion Gate

<completion_gate>
  A task is COMPLETE only when ALL of the following are true simultaneously:

  1. All Acceptance Criteria in `task.md` are verified (`- [x]`) with
     evidence recorded in `review.md`.
  2. All validation commands (as defined in `AGENTS.md`) execute with
     zero errors and zero warnings.
  3. Gate 3 in `review.md` is checked and review decision is PASS.
  4. The final `git diff` contains zero extraneous or unreviewed modifications,
     and all transient debug code/scratch artifacts are completely removed.
  5. Task folder is moved to `completed/` and `handoff.md` is produced.
</completion_gate>

---

## 4. Invariant Preservation

<invariants>
  - Never overwrite or delete human-authored spec content in `task.md`.
  - Only toggle `- [x]` after the corresponding verifier actually passes —
    never preemptively.
  - Keep edits within the declared subsystem unless explicit cross-system
    coordination is requested and approved.
</invariants>

---

## 5. Command Safety & Destructive Action Blacklist

<command_safety>
  The agent must NEVER execute destructive, irreversible, or credential-leaking commands:

  - **Git Operations:** Never execute `git push --force`, `git push -f`, `git reset --hard`,
    or `git clean -fdx` unless explicitly authorized by the human engineer in the current session.
  - **Filesystem Deletion:** Never execute unconstrained recursive deletion (e.g. `rm -rf /`,
    `rm -rf ~`, `rm -rf .`) or delete files outside the immediate active task scope.
  - **Database DDL/DML:** Never execute destructive data operations without explicit prior approval
    (`DROP DATABASE`, `DROP TABLE`, `TRUNCATE`, or `DELETE` queries lacking a specific `WHERE` clause).
  - **Secrets & Credentials:** Never read, print, log, or export contents of `.env*`, `*.pem`,
    `*.key`, SSH keys, or cloud credential stores into task artifacts or conversation output.
  - **Environment Containment:** Execute commands exclusively through the designated harness
    `<validation_commands>` or standard package managers. Never download or execute arbitrary
    remote binary scripts (`curl ... | bash`).
  - **Python Environment & Package Isolation:** The user strictly uses `uv` and forbids polluting the host system.
    * ALWAYS use `uv` (`uv run`, `uv add`, `uv sync`, `uv pip install ...`).
    * NEVER execute system-wide `pip install` (`pip install ...`, `pip install --user ...`, or `sudo pip`). All Python packages must reside within project-local virtual environments (`.venv`) managed by `uv`.
</command_safety>

---

## 6. Session Housekeeping & Teardown Protocol

<housekeeping_protocol>
  Before requesting Gate 3 sign-off or marking a task COMPLETE, the agent must perform full teardown:

  1. **Transient Debug Removal:** Remove all temporary debugging lines (`console.log`, `print()`,
     `debugger`, `dump()`, `pprint()`, or commented-out experiment blocks) introduced during execution.
  2. **Scratch Cleanup:** Delete temporary mock files, scratch test scripts, and transient SQLite/data
     dumps created during the execution loop.
  3. **Diff Sanitization:** Run `git status` and `git diff` to ensure that only the intentional, scoped
     files agreed upon in `plan.md` have been modified.
  4. **State Finalization:** Ensure `state.md` is cleanly synchronized and generate `handoff.md`.
</housekeeping_protocol>

---

## 7. Cross-Harness & Independent Review Principle

<cross_harness_review>
  Rule: **The implementer cannot be the sole reviewer.**
  - To eliminate confirmation bias and algorithmic blind spots, the REVIEW phase should be conducted with fresh context, an independent reviewer subagent, or a distinct model harness when available.
  - Reviewers evaluate code strictly against the `<scope_contract>`, security guidelines, and behavioral invariants without inheriting the implementer's speculative reasoning.
</cross_harness_review>

---

## 8. Graduated Quality Gate Strictness

<gate_strictness>
  Quality gates operate under a 3-tier graduated enforcement model:

  - **Hard-Mandatory (Blocking):** Zero tolerance. Must pass 100% without exception (e.g. typecheck, test suites, zero out-of-scope edits). Failure immediately blocks task completion.
  - **Soft-Mandatory (Overridable with Justification):** Required by default. May only be overridden by the human engineer with a recorded rationale in `<override_reason>` (e.g. temporary performance baseline waiver).
  - **Advisory (Informational):** Non-blocking recommendations, lint hints, or future technical debt observations logged into `review.md`.
</gate_strictness>

</agent_guardrails>
