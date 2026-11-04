# Agent Behavior Protocol

<behavior_protocol version="1.0">

<description>
  Universal rules governing HOW the agent declares its operating mode,
  reloads state at session start, and navigates context. Project-agnostic.
</description>

---

## 1. Mode Declaration

Every response that advances a task MUST open with a mode declaration on the
first line. The declared mode must match the current phase recorded in `task.md`.

```
[MODE: RESEARCH]   — read-only, no source changes
[MODE: INNOVATE]   — generating options, no source changes
[MODE: PLAN]       — writing plan artifacts, no source changes
[MODE: EXECUTE]    — making scoped source changes
[MODE: REVIEW]     — reviewing, no code fixes
```

Omit the mode declaration only for pure conversational exchanges that do not
advance a task (e.g., answering a factual question, clarifying scope).

In continuous autonomous execution (DELEGATED / Fast-Track), the agent opens with the starting phase's mode. When transitioning across phases within a single turn, emit an explicit transition marker:
`>>> [PHASE TRANSITION: <OLD_PHASE> -> <NEW_PHASE>]` and proceed under the new mode immediately without pausing.

---

## 2. Working Modes & Phase Transition Protocol

<working_modes_protocol>
  Execution posture is determined by `<working_mode>` in `task.md` or user prompt instruction:

  ### PAIR Mode (Default — Step-by-Step Collaboration):
  - The agent works on one phase at a time.
  - Upon completing a phase, it updates the corresponding artifact and **HALTS** to let the human engineer inspect, discuss, and sign off the Gate (G1, G2, G3).
  - Waits for user prompt (e.g., "Start next phase") before updating `<current_phase>` in `task.md` and continuing.

  ### DELEGATED Mode (Autonomous / Fast-Track / Skip Permissions):
  - **Activation:** `<working_mode>DELEGATED</working_mode>` in `task.md` OR explicit prompt directive ("fast-track", "skip permissions", "run automatically", "auto-advance", "autonomous").
  - **Core Rule:** **DO NOT HALT AFTER EACH PHASE TO WAIT FOR USER PROMPT "NEXT".**
  - **Continuous Transition Workflow:**
    1. When the current phase meets its exit criteria/checklist, the agent checks `- [x]`.
    2. Populates auto-approval into the artifact: `approved_by: [AUTO: DELEGATED]` with timestamp and technical rationale (in INNOVATE: adopts the optimal Recommendation; in PLAN: locks the scope contract).
    3. Immediately updates `<current_phase>` in `task.md` to the next phase (`RESEARCH` → `INNOVATE` → `PLAN` → `EXECUTE` → `REVIEW`).
    4. Instantiates the next phase seed artifact (Copy-On-Demand) and **CONTINUES EXECUTION IMMEDIATELY** within the same session/turn.
  - **Sole Stop Conditions in DELEGATED:**
    - Task is 100% COMPLETE (Gate 3 PASS, housekeeping cleaned, `handoff.md` generated, moved to `completed/`).
    - OR a true Escalation Trigger is tripped (retry budget exhausted after 3 attempts, destructive command, or unresolvable invariant conflict).

  ### MANUAL & DIAGNOSE-ONLY Modes:
  - `MANUAL`: Human leads command-by-command; agent provides scoped assistance.
  - `DIAGNOSE-ONLY`: Runs Research & Review for root-cause audit without mutating source code.
</working_modes_protocol>

---

## 3. Prompt-Driven Task Initialization Protocol

<prompt_task_initialization>
  When the user requests a new task via prompt (without an existing task folder or `task.md` in `active/`), the Agent **MUST NOT** ask the user to run terminal commands or manually copy seed files. Instead, the Agent automatically recognizes prompt keywords, routes the directory, and initializes `task.md`:

  ### Keyword & Pseudo-Slash Command Routing Table:

  Users may specify tasks using natural language keywords OR handy prefix commands (**Pseudo-Slash Commands**) at the start of the prompt:

  | Target Directory | Slash Command (Recommended) | Natural Keywords | Scope & Working Mode |
  |---|---|---|---|
  | **`process/features/active/{task-slug}/`** | `/feature`<br>`/big-task`<br>`/epic` | `big task`, `feature`, `big changes`, `epic` | Major features, domain subsystems, architectural refactors (≥ 5 files, multiple phases). Default mode: **PAIR**. |
  | **`process/general-plans/active/{task-slug}/`** | `/task`<br>`/small-task`<br>`/bug`<br>`/quick-fix` | `small task`, `general changes`, `small changes`, `bug fix`, `quick fix` | Standalone tasks, quick bug fixes, general adjustments (< 5 files). Default mode: **PAIR**. |
  | **`process/general-plans/active/{task-slug}/`** | `/hotfix` | `hotfix`, `emergency fix`, `production bug` | Emergency production fix. Auto-activates **DELEGATED / Fast-Track** mode (auto-certifies G0–G2, pauses only at G3). |
  | *(Scope-derived)* | `/fast-track`<br>`/delegated`<br>`/auto` | `fast-track`, `autonomous`, `auto-advance`, `skip permissions` | Full autonomous execution. Sets `<working_mode>DELEGATED</working_mode>` and runs continuously through all phases. |
  | *(Framework Maintenance)* | `/update`<br>`/upgrade` | `update instructions`, `upgrade framework`, `check updates` | Maintenance protocol: queries remote registry for a newer framework version. If a newer version exists, updates the instructions framework while preserving user configs (`preserveUserFiles`) and updates `instruction-version.json`. If already on the latest version, outputs `"nothing changed"`. |

  *Fallback Rule:* If no explicit keyword or slash command is found in the prompt, infer from scope (< 5 files or localized fix $\rightarrow$ `general-plans/`; new capability or multi-module impact $\rightarrow$ `features/`).

  ### Framework Update Protocol (`/update`):
  When the prompt begins with `/update` or `/upgrade` (or user asks to update instructions/framework):
  1. **Inspect Local Version:** Read `instruction-version.json` at workspace root to inspect current `version`, `language`, and `updateStrategy`.
  2. **Query Latest Version:** Query the registry (e.g. `npm view @damphuquy/agent-init version` or execute `npx @damphuquy/agent-init update`).
  3. **Conditional Update:**
     - **If newer version found:** Execute framework update (e.g. `npx @damphuquy/agent-init update` or `npx @damphuquy/agent-init@latest . --force`), preserving project-specific files defined in `preserveUserFiles` (`AGENTS.md#validation_commands`, active tasks). Update `instruction-version.json` timestamp and version. Report upgrade details.
     - **If no newer version (already up to date):** Respond directly with `"nothing changed"` (e.g. `nothing changed: instruction framework is already at the latest version vX.Y.Z`) and make no file changes.

  ### Automated Scaffolding & Activation Steps:
  1. **Derive Slug:** Generate a concise, kebab-case `{task-slug}` from the prompt (e.g., `CHG-001-change-password`, `AUTH-002-rate-limiting`, or `{TICKET-ID}-{slug}`).
  2. **Create Directory:** Create `process/features/active/{task-slug}` or `process/general-plans/active/{task-slug}`.
  3. **Instantiate Seed:** Copy `process/_seeds/task-template.md.seed` to `{task-dir}/task.md`.
  4. **Hydrate Specification:**
     - Extract task goal into `<goal>`.
     - Convert user criteria and requirements into actionable markdown checkboxes `- [ ]` under `<acceptance_criteria>`.
     - Set `<working_mode>` (`PAIR` by default, or `DELEGATED` if prompt indicates fast-track / autonomous execution).
     - Set `<status>ACTIVE</status>` and `<current_phase>RESEARCH</current_phase>`.
  5. **Declare Mode & Execute Immediately:** Output `[MODE: RESEARCH]`, announce the initialized `task.md` path, and immediately proceed with the RESEARCH phase without requiring manual user setup.
</prompt_task_initialization>

---

## 4. Session Startup Protocol

Before continuing any in-progress task, reload persistent state in this order:

<startup_sequence>
  1. Read `task.md` — confirm current phase and open gates.
  2. Read `research.md` if Research phase is complete.
  3. Read `decision.md` if Innovate phase is complete — confirm approved decisions.
  4. Read `plan.md` — confirm current slice index and scope contract.
  5. Read `state.md` — confirm completed slices, failure memory, retry budget, next action.
  6. Re-read any source files that changed since last context load.
</startup_sequence>

**Do NOT rely on conversation memory alone.** Always verify against the
file-based artifacts listed above.

---

## 5. Context Navigation Rules

<context_rules>
  <rule id="minimum_context">
    Gather minimum sufficient context only. Never scan the full repository or
    perform drive-by refactoring outside the active task scope.
  </rule>

  <rule id="information_priority">
    Load context in this priority order:
    1. Task spec and acceptance criteria (`task.md`)
    2. Research and decision artifacts (`research.md`, `decision.md`)
    3. Relevant test suites
    4. Domain models and port interfaces
    5. Configuration and dependency injection setup
    6. Concrete infrastructure implementations
  </rule>

  <rule id="no_stale_context">
    Re-read relevant files after any repository change. Never act on stale
    in-memory snapshots.
  </rule>
</context_rules>

</behavior_protocol>
