# 07 — Integration Status

Snapshot: 2026-09-12, local `main` tại commit tích hợp `6f4c768`.

## Kết luận hiện tại

Ba phần đã nối với nhau bằng contract thật:

```text
Firefox sidebar
  -> source policy
  -> content-script viewport capture
  -> @vlc/memory trên browser.storage.local
  -> POST /v1/agent/turn
  -> Fastify agent loop
  -> OpenAI-compatible provider
  -> Ollama qwen3:8b hoặc OpenRouter
```

Core path không còn business placeholder. Phần chưa hoàn tất gồm Privacy/Memory UI, safe logs và Golden E2E với thao tác Firefox thật. `search_web` vẫn là permission-gated stub và không thuộc demo bắt buộc.

## Đã triển khai

### Firefox extension

- Source policy chạy trước DOM capture và HTTP.
- Capture lấy text từ DOM block giao với viewport và áp giới hạn text/code/anchor. Không tuyên bố cắt chính xác từng ký tự nếu một block dài vượt mép viewport.
- Sidebar mở không gọi model.
- Open pipeline capture và recall memory.
- Ask pipeline capture lại viewport ở mỗi thao tác hỏi, build request và gọi API local.
- Ba marker buttons chỉ ghi sau thao tác người dùng.
- Resume theo text quote, heading, rồi scroll ratio.
- Runtime boundary guards không bundle Zod vào add-on; request vẫn được kiểm bằng shared Zod schema trong tests/API.

### Agent API

- Fastify listen ở `127.0.0.1:8787`.
- Health, CORS, request validation và error mapping đã có.
- Agent loop tối đa 3 model steps với viewport/memory/propose-marker tools.
- Provider dùng OpenAI Chat Completions-compatible `tools/tool_calls`.
- Artifact `dist/server.js` bundle shared contracts và chạy được bằng Node thuần.
- Timeout/kết nối model map thành `AGENT_UNAVAILABLE`, `retryable:true`.

### Contracts và memory

- Shared TypeScript/Zod contracts dùng chung cho extension và API.
- `ShortSession.tabId` chỉ ở browser session, không gửi lên model API.
- Optional fields đã thống nhất với `exactOptionalPropertyTypes`.
- Memory upsert theo canonical URL + exact fingerprint; revision tăng khi ghi lại.
- Storage keys đúng kiến trúc:
  - `vlc:markers:v1`
  - `vlc:policy:v1`
  - `vlc:settings:v1`
- Matcher deterministic: `1.00`, `0.85`, `0.60`; không dùng vector/embedding.

## Bằng chứng kiểm tra

Lần kiểm tra gần nhất:

```text
TypeScript: pass trên 4 workspaces
Tests: 61 pass
Agent API build: pass, dist/server.js chạy bằng Node
Firefox extension build: pass
web-ext lint: 0 errors, 0 notices, 0 warnings
web-ext run: temporary add-on được cài thành công
GET /health: 200, CORS trả đúng moz-extension origin
POST /v1/agent/turn khi Ollama tắt: 503 AGENT_UNAVAILABLE, retryable=true
Ollama local qwen3:8b inference: pass, 100% GPU
Ollama local tool calling: pass, gọi đúng get_viewport_context
Tunnel /v1/models và /v1/chat/completions: pass
Cloudflared Windows Service: Running, Automatic
```

`web-ext run` chứng minh Firefox nhận manifest và cài add-on. Chưa tuyên bố golden E2E pass vì môi trường automation không nhìn/click được cửa sổ Firefox.

## Trạng thái model ngày 2026-09-12

- Ollama CLI `0.32.5` hoạt động tại `127.0.0.1:11434`.
- `qwen3:8b` và `bge-m3:latest` đã có sẵn.
- POST local có tool definition trả `finish_reason=tool_calls` và gọi đúng `get_viewport_context`.
- `https://qwen.luongduytoan.io.vn/v1/models` trả model list.
- POST qua tunnel trả đúng nội dung smoke `TUNNEL_OK`.

Model/tunnel không còn là blocker. Demo trên cùng máy vẫn ưu tiên localhost để giảm phụ thuộc mạng.

## Memory thực tế

- Short-term nằm trong biến `session` ở RAM của sidebar: context gần nhất, tối đa 10 chat messages, tối đa 5 memory summaries và pending action. Đóng/reload sidebar có thể mất session này.
- Long-term nằm trong `browser.storage.local` của Firefox profile, không nằm trong folder repository và backend không giữ bản sao.
- Keys: `vlc:markers:v1`, `vlc:policy:v1`, `vlc:settings:v1` (key settings mới dành chỗ, chưa có UI sử dụng).
- Marker chỉ được lưu sau click của người dùng. Nó giữ source/anchor/status, text quote tối đa 240 ký tự, câu hỏi/answer summary tối đa 500 ký tự, evidence, timestamps và revision; không giữ raw viewport 4000 ký tự.
- Manifest đã có permission `storage`. Không cần và không nên xin quyền đọc folder dự án; Firefox tự quản lý file profile.
- `browser.storage.local` không được ứng dụng mã hóa thêm. Khi người dùng bấm Ask, viewport và tối đa 5 memory summaries liên quan được truyền tới Agent API/provider để xử lý; backend không chủ động lưu chúng vào database.

## Tools, prompt và phần chưa nối UI

Agent tools có code thật tại `apps/agent-api/src/agent/tool-registry.ts`:

- `get_viewport_context`
- `search_memory`
- `read_memory`
- `propose_marker`
- `search_web`

`search_web` hiện trả danh sách rỗng nếu được phép và extension đang gửi `allowWebSearch=false`; không quảng bá như search thật.

System prompt thật nằm tại `apps/agent-api/src/agent/system-prompt.ts`.

Repository đã có `searchMemory`, `saveMarker`, `updateUnderstanding`, `forgetMemory`, `getSourcePolicySettings` và `updateSourcePolicy`. Sidebar hiện mới nối recall/save/resume; chưa có UI denylist, list nhiều marker hoặc delete marker.

Agent API hiện chỉ log lúc server start. Task gốc yêu cầu log `turnId`, duration, tool name và error code nhưng không log raw data; phần này còn thiếu.

## Checklist kế tiếp trước demo

1. Task `04`: Privacy/Memory UI (denylist, marker list, forget).
2. Task `05`: safe observability cho Agent API.
3. Task `06`: harness chạy, inspect storage, Firefox Temporary Add-on/signed package và Golden E2E.
4. Merge `04`/`05`, sau đó chạy root verify và Golden E2E từ `06`.
5. Quay video demo chỉ sau khi ask, save, reopen, recall, resume và block/unblock đều được quan sát thật.

Giữ `API_PORT=8787`; extension hiện cố định URL và host permission cho port này.

## Fallback

- Ollama/tunnel lỗi trở lại: đổi `.env` sang OpenRouter; không sửa source.
- Model quá chậm: warm model bằng một request trước demo và giữ `OPENAI_TIMEOUT_MS=60000`.
- Agent API lỗi: marker local vẫn hoạt động; nhưng video E2E chính phải có ít nhất một lượt hỏi thành công.
- Firefox không inject content script vào tab cũ: reload article sau khi cài Temporary Add-on.
