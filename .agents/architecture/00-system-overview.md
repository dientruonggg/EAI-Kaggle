# 00 — Kiến trúc tổng

## 1. Hệ thống làm gì?

Viewport Learning Companion là Firefox extension được người học **chủ động mở khi cần**. Extension biết phần HTML đang hiện trong viewport, gợi lại dấu mốc học tập liên quan, cho phép hỏi dựa trên đúng đoạn đang xem và lưu trạng thái `Đã hiểu`, `Chưa hiểu` hoặc `Xem lại sau`.

Đây không phải công cụ theo dõi nền. Việc đã cuộn qua một đoạn không được coi là đã đọc hoặc đã hiểu.

## 2. Ba khối triển khai độc lập

```text
┌──────────────────────────────────────┐
│ apps/firefox-extension               │
│ policy -> capture -> sidebar -> HTTP │
└───────────────┬──────────────────────┘
                │ AgentTurnRequest / AgentTurnResponse
                v
┌──────────────────────────────────────┐
│ apps/agent-api                       │
│ validate -> agent loop -> LLM        │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ packages/contracts + packages/memory │
│ shared schemas + browser repository  │
└──────────────────────────────────────┘
          ^ dùng bởi extension và API
```

Không có kết nối trực tiếp từ backend vào Firefox storage. Extension preload tối đa 5 memory liên quan vào request. Backend không giữ dữ liệu sau khi trả response.

## 3. Cấu trúc npm workspace mục tiêu

```text
Hackathon-T9-2026/
├── package.json
├── package-lock.json
├── tsconfig.base.json
├── .env.example
├── apps/
│   ├── firefox-extension/
│   │   ├── package.json
│   │   ├── manifest.json
│   │   ├── src/
│   │   └── tests/
│   └── agent-api/
│       ├── package.json
│       ├── src/
│       └── tests/
├── packages/
│   ├── contracts/
│   │   ├── package.json
│   │   └── src/
│   └── memory/
│       ├── package.json
│       ├── src/
│       └── tests/
└── .agents/
    ├── architecture/
    └── tasks/
```

Tên package:

| Folder | Package name | Vai trò |
|---|---|---|
| `apps/firefox-extension` | `@vlc/firefox-extension` | UI, policy, DOM capture, orchestration phía browser |
| `apps/agent-api` | `@vlc/agent-api` | HTTP API, agent loop, LLM adapter |
| `packages/contracts` | `@vlc/contracts` | Type và Zod schema dùng chung |
| `packages/memory` | `@vlc/memory` | Repository trên `browser.storage.local` và matcher |

## 4. Luồng mở sidebar

```text
user click icon
  -> checkSourcePolicy
      -> BLOCK: hiện lý do, dừng; không đọc DOM, không gọi API
      -> ALLOW:
          -> captureCurrentViewport
          -> memoryRepository.searchMemory
          -> createShortSession
          -> render context + memory gợi lại
```

Mở sidebar chưa tự gọi model. Điều này giảm chi phí, tăng riêng tư và giúp demo ổn định.

## 5. Luồng hỏi Agent

```text
user submits question
  -> runAskAgentPipeline
  -> refresh viewport nếu context đã stale
  -> buildAgentTurnRequest
  -> POST /v1/agent/turn
  -> runAgentTurn (tối đa 3 model steps)
      -> model có thể gọi pure tools trên request context
      -> tool result quay lại model
      -> model trả answer + grounding + proposal
  -> sidebar render response
  -> update short session
```

Agent tools phía server chỉ đọc context đã được gửi kèm request hoặc tạo proposal. Chúng không tự sửa Firefox storage.

## 6. Luồng ghi memory có xác nhận

```text
user presses one of three marker buttons
  -> executeConfirmedMemoryCommand
  -> MemoryRepository.saveMarker/updateUnderstanding
  -> storage returns success
  -> UI mới hiển thị “Đã lưu”
```

Nếu model gọi `propose_marker`, response chỉ tạo `PendingUserAction`. Chỉ click xác nhận của người dùng mới biến proposal thành lệnh ghi.

## 7. Nơi ra quyết định

| Quyết định | Chủ sở hữu | Lý do |
|---|---|---|
| Trang có được đọc không | Extension policy | Phải quyết định trước DOM/network |
| Nội dung viewport nào được gửi | Extension capture | Browser mới thấy DOM và viewport |
| Memory nào liên quan | `@vlc/memory` | Dữ liệu nằm cục bộ và matcher phải deterministic |
| Có gọi tool nào để trả lời | Agent loop | Đây là reasoning/runtime concern |
| Có ghi/xóa memory không | Người dùng + extension | Model không có quyền side effect trực tiếp |
| API key và model | Backend | Không lộ secret trong extension |

## 8. Stack chốt

| Phần | Công nghệ |
|---|---|
| Runtime | Node.js 22+ |
| Package manager | npm 10+ workspaces |
| Language | TypeScript strict |
| Extension | Firefox WebExtension Manifest V2 |
| UI | Vanilla TS + HTML/CSS |
| API | Fastify |
| Validation | Zod |
| LLM adapter | package `openai`, OpenAI-compatible API sau `LlmProvider` |
| Unit/integration tests | Vitest |
| Extension development | `web-ext` hoặc `about:debugging` |
| Extension bundler | Vite + `vite-plugin-web-extension` |
| Durable local data | `browser.storage.local` |

Manifest phải khai báo một `browser_specific_settings.gecko.id` cố định. Temporary Add-on bị unload khi Firefox restart; demo chính chỉ cam kết reopen sidebar/reload extension với cùng profile và ID. Không quảng bá đây là cloud sync hoặc memory đa thiết bị.

TypeScript không chạy trực tiếp trong Firefox/Node production. Build artifacts bắt buộc:

- `@vlc/firefox-extension`: Vite tạo `apps/firefox-extension/dist/manifest.json` và JavaScript bundles;
- `@vlc/agent-api`: tsup tạo `apps/agent-api/dist/server.js`;
- `@vlc/contracts` và `@vlc/memory`: `tsc` tạo `dist/*.js` và `dist/*.d.ts`.

## 9. Những gì học từ một pipeline agent lớn hơn

MVP giữ các pattern tổng quát sau:

- agent shell mỏng, capability nằm trong module riêng;
- tool có description, input schema và execute rõ ràng;
- tool registry nhận dependency qua factory để test được;
- recall trước khi trả lời câu hỏi phụ thuộc lịch sử;
- record phải tự đủ nghĩa khi đọc lại ngoài cuộc hội thoại;
- chỉ xác nhận đã lưu sau khi storage thành công;
- E2E phải chứng minh write rồi recall, không chỉ mock UI.

MVP không mang theo các phần không cần thiết: vector search, cloud object storage, durable state, scheduling, workflow, channel adapter, skill loader động hoặc self-modifying prompt.

## 10. Non-goals

- Không đọc PDF trong bản demo.
- Không đọc YouTube, social feed, email, banking, health portal hoặc private chat.
- Không hỗ trợ ChatGPT như nguồn demo mặc định.
- Không tracking tab/scroll nền.
- Không chứng minh “người dùng thật sự hiểu” nếu họ chưa tự xác nhận.
- Không dùng embeddings; vì vậy không tuyên bố semantic recall.
- Không auth, production deployment hoặc Firefox Add-ons publishing trong ngày thi.
