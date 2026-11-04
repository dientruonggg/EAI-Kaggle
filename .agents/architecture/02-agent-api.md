# 02 — Kiến trúc con: Agent API

## 1. Trách nhiệm

Agent API nhận context đã được extension lọc, chạy một agent loop nhỏ và trả lời có grounding. Backend sở hữu:

- HTTP validation;
- system prompt;
- tool registry chỉ đọc;
- vòng lặp model/tool tối đa 3 bước;
- `LlmProvider` và OpenAI-compatible adapter;
- error mapping và safe logging.

Backend không đọc browser, không giữ memory và không có quyền ghi Firefox storage.

## 2. Cấu trúc đề xuất

```text
apps/agent-api/
├── package.json
├── src/
│   ├── server.ts
│   ├── app.ts
│   ├── config/env.ts
│   ├── routes/
│   │   ├── health.ts
│   │   └── agent-turn.ts
│   ├── agent/
│   │   ├── run-agent-turn.ts
│   │   ├── system-prompt.ts
│   │   └── tool-registry.ts
│   └── providers/
│       ├── llm-provider.ts
│       └── openai-compatible-provider.ts
└── tests/
```

## 3. HTTP surface

### `GET /health`

Response `200`:

```json
{
  "status": "ok",
  "service": "viewport-learning-agent-api",
  "contractVersion": "0.1"
}
```

### `POST /v1/agent/turn`

- Parse bằng `AgentTurnRequestSchema` từ `@vlc/contracts`.
- Chạy `runAgentTurn()`.
- Parse output bằng `AgentTurnResponseSchema` trước khi trả.
- Không log `visibleText`, question, note hoặc answer thô.

HTTP mapping:

| Case | Status |
|---|---:|
| Success | `200` |
| Invalid schema | `400` |
| Context empty/stale | `422` |
| Provider unavailable/timeout | `503` |
| Unexpected internal error | `500` |

Error body vẫn dùng `Result`/`AppError`, không trả stack trace.

## 4. Agent loop tối thiểu

```text
validate request
  -> create immutable TurnToolContext
  -> buildSystemPrompt
  -> createAgentToolRegistry(context)
  -> provider.generate(messages, tool definitions)
      -> final text: normalize response, return
      -> tool calls: validate input, execute pure tools, append results
  -> repeat, maximum 3 model steps
  -> if still no final answer: AGENT_MAX_STEPS
```

Mỗi tool execute phải trả `Result<T>`. Một tool fail không làm server crash; kết quả lỗi được đưa lại cho model để model trả lời trung thực.

## 5. Tool registry phía Agent

| Tool name | Đọc gì | Side effect |
|---|---|---|
| `get_viewport_context` | `ViewportContext` của request | Không |
| `search_memory` | Tối đa 5 `MemorySummary` đã preload | Không |
| `read_memory` | Một memory trong tập preload theo ID | Không |
| `propose_marker` | Context hiện tại và status model đề xuất | Chỉ tạo proposal trong response |
| `search_web` | Nguồn ngoài khi permission bật | Tùy chọn; không thuộc golden E2E |

`save_marker`, `update_understanding` và `forget_memory` không nằm trong server tool registry vì storage thuộc browser và các lệnh này cần user confirmation.

## 6. Quy tắc system prompt

`buildSystemPrompt()` phải cố định các nguyên tắc:

1. Chỉ trả lời dựa trên viewport, memory được cung cấp hoặc web result thật.
2. Không nói người dùng “đã hiểu” chỉ vì đã nhìn thấy/đã hỏi.
3. Khi câu hỏi phụ thuộc vị trí hiện tại, gọi `get_viewport_context`.
4. Khi câu hỏi nhắc lịch sử hoặc có related memory, gọi `search_memory` trước khi khẳng định.
5. Match score thấp phải diễn đạt như gợi ý, không phải ký ức chắc chắn.
6. Không tự nhận đã lưu. `propose_marker` chỉ là đề xuất chờ xác nhận.
7. Nếu context thiếu, nói thiếu gì; không bịa.
8. Không lặp raw viewport dài trong câu trả lời.
9. Trả lời ngắn, phù hợp sidebar.

## 7. Public functions bắt buộc

```ts
buildApp(deps: AppDependencies): FastifyInstance;

buildSystemPrompt(): string;

createAgentToolRegistry(
  context: TurnToolContext,
  deps: AgentToolDependencies,
): ReadonlyMap<AgentToolName, AgentTool>;

runAgentTurn(
  input: AgentTurnRequest,
  deps: AgentDependencies,
): Promise<Result<AgentTurnResponse>>;

createOpenAICompatibleProvider(
  input: OpenAICompatibleProviderOptions,
): LlmProvider;
```

## 8. `LlmProvider` boundary

```ts
interface LlmProvider {
  generate(input: LlmGenerateInput): Promise<Result<LlmGenerateOutput>>;
}
```

`runAgentTurn()` không import OpenAI client trực tiếp. Unit tests inject fake provider với script:

1. model yêu cầu `get_viewport_context`;
2. harness thực thi tool;
3. model trả final answer;
4. test kiểm tra `toolTrace` và grounding.

## 9. Response normalization

Backend phải tự tạo các trường hệ thống, không tin model tự điền:

- `contractVersion`;
- `turnId`;
- `toolTrace`;
- `model.provider` và `model.name`;
- timestamps nếu có.

Model chỉ cung cấp nội dung answer, grounding intent và proposal. Proposal phải được validate; proposal invalid bị bỏ, không làm mất answer.

## 10. Logging và secrets

- `OPENAI_API_KEY` chỉ ở process environment. `OPENAI_BASE_URL` chọn OpenRouter hoặc endpoint tương thích.
- `.env` không commit; `.env.example` không chứa secret thật.
- Log được: request ID, turn ID, duration, tool names, error code.
- Không log: URL query nhạy cảm, raw viewport, raw memory note, raw prompt, API key.
- `safeUrl` phải loại fragment và các query param trước khi log.

## 11. Acceptance Criteria của API

- `GET /health` trả đúng schema.
- Invalid request trả `400` và `VALIDATION_ERROR`.
- Fake-provider test chứng minh ít nhất một vòng model -> tool -> model.
- Loop dừng tối đa 3 model steps.
- Unknown tool và invalid tool input trở thành tool error có kiểm soát.
- Related memory không thuộc request không thể đọc bằng ID đoán mò.
- Response luôn parse được bằng shared schema.
- Provider timeout/error map sang `MODEL_ERROR` hoặc `AGENT_UNAVAILABLE`.
- Test hoặc log assertion chứng minh raw viewport không bị log.
