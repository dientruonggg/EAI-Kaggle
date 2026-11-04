---
name: "speckit-pipeline"
description: "Chains all 9 Spec Kit skills: constitution → specify → clarify → checklist → plan → tasks → analyze → implement → converge into one complete guided SDD run."
compatibility: "Requires spec-kit project structure with .specify/ directory"
metadata:
  author: "EAI-2026 Team"
  source: "workflows/full-pipeline"
---

## User Input

```text
$ARGUMENTS
```

## Purpose & Execution Sequence

When invoked with a feature description (e.g. `/speckit-pipeline <mô tả tính năng>`), the agent executes the full 9-skill Spec-Driven Development cycle in order:

1. **Step 1: `/speckit-constitution`** (Hiến chương & Ràng buộc)
   - Synchronizes non-negotiable constraints, target limits (ESP32-S3, <200k params, INT8 PTQ).
2. **Step 2: `/speckit-specify`** (Đặc tả yêu cầu)
   - Generates user scenarios (P1, P2, P3) and Given/When/Then acceptance criteria in `specs/<id>/spec.md`.
3. **Step 3: `/speckit-clarify`** (Làm rõ rủi ro)
   - Detects ambiguities in `spec.md` and prompts for clarification before architecture locking.
4. **Step 4: `/speckit-checklist`** (Checklist chất lượng)
   - Generates requirements & testing checklists in `specs/<id>/checklists.md`.
5. **Step 5: `/speckit-plan`** (Kiến trúc & Giải pháp)
   - Defines technical approach, model constraints, and dependencies in `specs/<id>/plan.md`.
6. **Step 6: `/speckit-tasks`** (Phân rã tác vụ)
   - Generates dependency-ordered, actionable work items in `specs/<id>/tasks.md`.
7. **Step 7: `/speckit-analyze`** (Phân tích chéo tính nhất quán)
   - Verifies consistency across `spec.md`, `plan.md`, and `tasks.md` before coding begins.
8. **Step 8: `/speckit-implement`** (Thực thi mã nguồn)
   - Implements code and unit tests task by task.
9. **Step 9: `/speckit-converge`** (Hội tụ & Nghiệm thu)
   - Audits the resulting codebase against specification, appending any remaining gap tasks until `Converged`.

## Execution Directive

If triggered directly in chat, execute the current pending phase for `$ARGUMENTS`, present the artifact diff to the user, and guide the user through the next skill in the chain.
