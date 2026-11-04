# Task 05 — Safe Agent API Observability

## Status

READY. Remote branch được tạo sẵn: `feat/safe-observability`, base `origin/last-mile-base`.

## Mission

Cho team quan sát được một agent turn chạy đến đâu, tool nào được gọi và mất bao lâu mà không rò dữ liệu học tập hay secret.

## Ownership

- Branch: `feat/safe-observability`.
- Chỉ sửa `apps/agent-api/**`.
- Không sửa extension, `packages/**`, root package files hoặc `.agents/**`.
- Không cài dependency mới; dùng logger/console adapter nhỏ hiện có thể test.

## Module và Safe log contract

Tạo `apps/agent-api/src/observability/agent-observer.ts`. Không export qua shared packages.

```ts
export type SafeAgentEvent =
  | { event: "agent.turn.started"; turnId: string }
  | {
      event: "agent.tool.completed";
      turnId: string;
      step: 1 | 2 | 3;
      toolName: AgentToolName;
      status: "SUCCESS";
    }
  | {
      event: "agent.tool.completed";
      turnId: string;
      step: 1 | 2 | 3;
      toolName: AgentToolName;
      status: "ERROR";
      errorCode: AppErrorCode;
    }
  | {
      event: "agent.turn.completed";
      turnId: string;
      durationMs: number;
      grounding: GroundingKind;
      toolCount: number;
      modelName: string;
    }
  | {
      event: "agent.turn.failed";
      turnId: string;
      durationMs: number;
      errorCode: AppErrorCode;
      retryable: boolean;
    };

export interface AgentObserver {
  emit(event: SafeAgentEvent): void;
}
```

Mỗi event phải là JSON metadata có tên ổn định:

```text
agent.turn.started   turnId
agent.tool.completed turnId step toolName status errorCode?
agent.turn.completed turnId durationMs grounding toolCount modelName
agent.turn.failed    turnId durationMs errorCode retryable
```

Tuyệt đối không log raw question, visible text/code, memory note/question/answer, full URL/query, messages, system prompt, tool result, headers, API key hoặc stack trace.

## Semantics bắt buộc

Không đổi HTTP/shared contracts. Tạo observer nhỏ inject được:

- `runAgentTurn` nhận observer optional trong dependencies.
- Tool loop emit sau mỗi tool, chỉ kèm metadata an toàn.
- Route đo tổng duration và emit completed/failed.
- Default observer ghi JSON một dòng ra stdout.
- Tests inject fake observer, không monkey-patch console nếu tránh được.
- `started` chỉ emit sau khi request parse thành công và có `turnId` hợp lệ.
- `step` là model step hiện tại, chỉ nhận 1, 2 hoặc 3.
- `toolCount` là số event `agent.tool.completed` của turn, gồm cả SUCCESS và ERROR.
- SUCCESS không có `errorCode`; ERROR bắt buộc có `errorCode`.
- `modelName` lấy từ `AgentTurnResponse.model.name`; không log base URL/provider headers.
- Mỗi started turn phải kết thúc bằng đúng một terminal event: completed hoặc failed, không cả hai.

## Tests bắt buộc

- successful turn có started/tool/completed;
- provider error có started/failed và đúng error code;
- tool failure có `agent.tool.completed` status ERROR;
- tool SUCCESS không có errorCode và tool ERROR luôn có errorCode;
- duration không âm và toolCount đúng;
- serialized captured events không chứa fixture secret/raw question/viewport/memory/prompt;
- HTTP response không chứa stack trace.

## Acceptance Criteria

```text
npm run typecheck --workspace @vlc/agent-api
npm test --workspace @vlc/agent-api
npm run build --workspace @vlc/agent-api
```

Sau đó chạy một Qwen turn thật. Handoff chỉ paste safe log sample, model, tool names, duration và commit hash.
