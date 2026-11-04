# Task Agent 2 — Agent API

> Status 2026-09-12: implementation đã được nối vào `main`; tests/build/runtime health pass. Ollama local và tunnel đã pass inference/tool-call smoke. Safe logging theo AC còn thiếu; xem task `05`.

## Mission

Xây Fastify Agent API trong `apps/agent-api/**`, gồm validation, system prompt, pure tool registry, agent loop tối đa 3 bước, OpenAI-compatible adapter và fake-provider tests.

## Execution harness

- Dùng npm và TypeScript strict; không dùng pnpm/yarn/Python.
- Bắt đầu bằng baseline commands; không thay root dependency/lockfile.
- Thay TODO bằng implementation thật và tests; không chỉ làm health route.
- Sau mỗi work package, chạy test/typecheck của workspace.
- API runtime phải chạy JavaScript `dist/server.js` do tsup build, không chạy `.ts` ở production path.
- Tự xử lý helper nội bộ; chỉ hỏi team khi cần đổi contract/dependency/scope.

## Branch và ownership

- Branch: `feat/agent-api`
- Chỉ được sửa: `apps/agent-api/**`
- Không sửa: root manifests/lockfile, `.agents/**`, extension, packages dùng chung, code cũ.

Nếu cần dependency mới hoặc đổi contract, ghi proposal trong handoff.

## Đọc trước khi code

1. `.agents/architecture/00-system-overview.md`
2. `.agents/architecture/02-agent-api.md`
3. `.agents/architecture/03-memory-tools-skills.md`
4. `.agents/architecture/04-shared-contracts.md`
5. `.agents/architecture/05-integration-e2e.md`

## Public functions phải implement đúng tên

```text
buildApp
buildSystemPrompt
createAgentToolRegistry
runAgentTurn
createOpenAICompatibleProvider
```

Endpoints bắt buộc:

```text
GET  /health
POST /v1/agent/turn
```

## File map bắt buộc

| File | Code chính | Được gọi bởi |
|---|---|---|
| `src/server.ts` | load env, create dependencies, listen | `npm run dev/start` |
| `src/app.ts` | `buildApp` và register routes | server/tests |
| `src/config/env.ts` | parse env OpenAI-compatible/CORS | server |
| `src/routes/health.ts` | `GET /health` | Fastify |
| `src/routes/agent-turn.ts` | parse schema, call `runAgentTurn`, map HTTP | Fastify |
| `src/agent/system-prompt.ts` | `buildSystemPrompt` | agent loop |
| `src/agent/tool-registry.ts` | `createAgentToolRegistry` | agent loop |
| `src/agent/run-agent-turn.ts` | `runAgentTurn` max 3 steps | route |
| `src/providers/llm-provider.ts` | provider interface/re-export | agent loop/providers |
| `src/providers/openai-compatible-provider.ts` | `createOpenAICompatibleProvider` | server composition |

Không để route gọi OpenAI SDK trực tiếp; mọi inference phải đi qua `runAgentTurn -> LlmProvider`.

## Work packages

### A. Fastify surface

- App factory nhận dependencies để test không mở port.
- Route parse request/response bằng schema từ `@vlc/contracts`.
- CORS chỉ cho extension/local demo origins được config.
- HTTP error mapping đúng tài liệu.

### B. Prompt

- Grounded on viewport/memory/web result thật.
- Không suy ra UNDERSTOOD từ view/question.
- Recall trước claim phụ thuộc lịch sử.
- `propose_marker` không nói đã lưu.
- Thiếu dữ kiện thì trả `INSUFFICIENT`.

### C. Tool registry

- `get_viewport_context`
- `search_memory`
- `read_memory`
- `propose_marker`
- `search_web` optional và permission-gated.

Tools chỉ đọc immutable turn context hoặc tạo proposal. Không thêm server-side memory DB.

### D. Agent loop/provider

- Tối đa 3 model steps.
- Validate tên và arguments của từng tool call.
- Append tool result rồi gọi model tiếp.
- Unknown/failed tool thành controlled error.
- `runAgentTurn` chỉ phụ thuộc `LlmProvider`, không phụ thuộc trực tiếp OpenAI SDK.
- OpenAI-compatible key/base URL/model đọc từ env; timeout được map thành error.
- Chạy được OpenRouter bằng `OPENAI_BASE_URL=https://openrouter.ai/api/v1` mà không đổi code.
- Dùng OpenAI Chat Completions-compatible `tools`/`tool_calls` để OpenRouter chạy được; model cấu hình phải hỗ trợ tool calling.

### E. Safety/logging

- Log request/turn ID, duration, tool name, error code.
- Không log raw question, viewport, memory note, prompt hoặc key.
- Không trả stack trace.

## Tests bắt buộc

- Health route schema.
- Invalid request -> 400/VALIDATION_ERROR.
- Fake provider flow: model -> get_viewport_context -> model final.
- Fake provider flow có memory search.
- Max-step termination.
- Unknown tool và invalid args.
- Read memory ID ngoài preload set bị từ chối.
- Web search permission false.
- Provider error/timeout mapping.
- Response parse shared schema và raw data không xuất hiện trong captured logs.

## Acceptance Criteria

- `npm run typecheck --workspace @vlc/agent-api` pass.
- `npm test --workspace @vlc/agent-api` pass.
- `npm run build --workspace @vlc/agent-api` pass.
- Với env hợp lệ, `npm run dev:api` phục vụ cả hai endpoint.
- Tool trace chứng minh ít nhất một loop thật trong test/demo.
- Không sửa file ngoài ownership.

## Tự do triển khai

Được tự thiết kế helper/parser/provider internals. Không thêm Cloudflare Durable Objects, database, scheduler, workflow hoặc multi-agent. `search_web` có thể để disabled adapter nếu thiếu thời gian; phải trả lỗi permission/capability rõ và không làm hỏng golden E2E.

## Handoff

Dùng template ở `.agents/architecture/05-integration-e2e.md`, kèm sample curl health/turn nhưng không kèm secret hay raw private context.
