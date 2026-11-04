# 08 — Last-mile Plan and Frozen Seams

## Mục tiêu

Ba người hoàn thiện song song mà không sửa cùng file. Không mở rộng sang web search thật, cloud memory, PDF/video, tracking nền hoặc dashboard.

## Workstream A — Privacy và Memory UI

Owner: `apps/firefox-extension/**`.

Luồng phải hoàn thiện:

```text
Sidebar mở
-> load source policy từ MemoryRepository
-> load tối đa 5 marker liên quan trang/section hiện tại
-> render denylist + marker list

User bấm Block current domain
-> updateSourcePolicy(BLOCK)
-> session BLOCKED
-> không capture/không gọi API ở lần mở tiếp theo

User bấm Forget marker
-> forgetMemory(ONE)
-> reload danh sách
```

Không xin quyền truy cập folder dự án. Manifest đã có WebExtension permission `storage`; long-term memory nằm trong `browser.storage.local` của Firefox profile. User consent là nút bấm rõ ràng trước mỗi ghi/xóa/block.

## Workstream B — Safe observability

Owner: `apps/agent-api/**`.

API log metadata để cả team biết agent đang chạy:

```text
agent.turn.started
agent.tool.completed
agent.turn.completed
agent.turn.failed
```

Chỉ log `turnId`, `step`, `toolName`, status/error code, model, duration. Không log raw question, viewport, memory, prompt, URL query hoặc secret.

## Workstream C — Demo/release harness

Owner: `scripts/**`, root `README.md`, `.env.example`, `.agents/**`, và root `package.json`/`package-lock.json` chỉ nếu cần đăng ký script `demo:preflight`.

- Chuẩn hóa lệnh kiểm tra Ollama, API, build và Firefox Temporary Add-on.
- Hướng dẫn inspect `browser.storage.local` bằng Extension Debugger.
- Chạy Golden E2E sau khi merge A/B.
- Tách rõ Temporary Add-on cho ngày thi và signed `.xpi`/AMO cho cài lâu dài.

## Frozen seams giữa ba workstream

Không sửa shared contracts. Workstream A phải gọi đúng interface đang export:

```ts
memoryRepository.getSourcePolicySettings()
memoryRepository.updateSourcePolicy({
  action: "BLOCK" | "ALLOW" | "RESET",
  domain: string,
  userConfirmed: true,
})
memoryRepository.searchMemory({ source, anchor, limit: 5 })
memoryRepository.forgetMemory({
  scope: "ONE",
  memoryId: string,
  userConfirmed: true,
})
```

Workstream B không đổi HTTP contract:

```text
GET  /health
POST /v1/agent/turn
```

Workstream C chỉ gọi public npm scripts/endpoints/storage keys:

```text
npm run verify
npm run demo:preflight
npm run dev:api
npm run start:firefox --workspace @vlc/firefox-extension
http://127.0.0.1:11434/v1
http://127.0.0.1:8787
vlc:markers:v1
vlc:policy:v1
```

## Merge rule

1. Cả ba branch remote được tạo từ `origin/last-mile-base` ghi trong `../tasks/README.md`.
2. C có thể viết harness/docs đồng thời nhưng chưa tuyên bố Golden E2E pass.
3. Merge A/B theo thứ tự bất kỳ vì ownership không giao nhau.
4. Rebase/merge C sau A/B, chạy root `npm run verify`.
5. Chỉ sửa bug integration trên commit riêng có reproduction rõ.

## Privacy boundary cần nói đúng

- Marker và policy được lưu local trong `browser.storage.local`; storage này không được ứng dụng mã hóa thêm.
- Khi người dùng bấm Ask, viewport hiện tại và tối đa 5 `MemorySummary` liên quan được gửi tới Agent API rồi provider/model đang cấu hình để xử lý lượt đó.
- Backend MVP không chủ động ghi database, nhưng chữ “local memory” không có nghĩa dữ liệu summary chưa từng rời trình duyệt trong lúc inference.
- Demo mặc định dùng Ollama localhost. Nếu đổi sang OpenRouter/tunnel khác, team phải hiểu provider đó nhận context của lượt hỏi.
- Safe logging task phải chứng minh raw viewport, memory summary và prompt không bị ghi log.
