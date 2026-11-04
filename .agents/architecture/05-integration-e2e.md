# 05 — Integration, Merge và E2E Plan

## 1. Mục tiêu tích hợp

Ba nhánh có thể code cùng lúc mà không cùng sửa một file. Điểm nối duy nhất:

1. Type/schema qua `@vlc/contracts`.
2. Local persistence qua interface `MemoryRepository` của `@vlc/memory`.
3. Network qua `POST /v1/agent/turn`.

Không import từ `apps/agent-api/src` sang extension và không import từ `apps/firefox-extension/src` sang API.

## 2. Ownership matrix

| Agent | Branch | Được sửa | Không được sửa |
|---|---|---|---|
| 1 — Firefox | `feat/firefox-extension` | `apps/firefox-extension/**` | root manifests, API, packages dùng chung |
| 2 — Agent API | `feat/agent-api` | `apps/agent-api/**` | root manifests, extension, packages dùng chung |
| 3 — Contracts/Memory | `feat/contracts-memory` | `packages/contracts/**`, `packages/memory/**`, root npm/config files | hai app |

`firefox_extension/` cũ nằm ngoài kiến trúc mới và không agent nào được sửa nếu team chưa quyết định migration riêng.

## 3. Cách bắt đầu song song

- Cả ba đọc contract trong `04-shared-contracts.md`.
- Agent 1 và 2 có thể code ngay theo type names, nhưng không tạo bản copy type riêng trong app.
- Agent 3 ưu tiên làm `@vlc/contracts` trước và gửi commit hash sớm.
- Sau khi commit contracts có sẵn, Agent 1/2 merge commit đó vào branch mình rồi typecheck.
- Nếu contract package chưa build, dùng workspace import; không dùng relative path xuyên package.

## 4. Thứ tự merge

```text
1. feat/contracts-memory
2. feat/agent-api
3. feat/firefox-extension
4. integration fixes (chỉ sửa lỗi nối, không thêm scope)
5. npm run verify
6. manual Firefox golden E2E
```

Merge Agent 3 trước vì root lockfile và shared packages là nền. Agent 2 trước Agent 1 để extension có API thật khi chạy manual test.

## 5. Root commands đã thống nhất

```bash
npm install
npm run typecheck
npm test
npm run build
npm run verify
npm run dev:api
npm run dev:extension
```

Chỉ Agent 3 được chạy lệnh cài thêm package làm thay đổi root `package-lock.json` trong thời gian chia nhánh. Nếu Agent 1/2 cần thư viện mới, ghi tên/version/lý do trong handoff để Agent 3 thêm bằng một commit dependency riêng.

## 6. Contract tests tại boundary

### Extension -> API

- Extension tạo request bằng `buildAgentTurnRequest()`.
- Test parse request bằng `AgentTurnRequestSchema`.
- Mock server response cũng phải parse bằng `AgentTurnResponseSchema`.

### Extension -> Memory

- Pipeline chỉ gọi interface `MemoryRepository`.
- Fake repository dùng cùng `Result<T>`.
- Browser implementation được test riêng bằng fake `StorageAreaLike`.

### API -> Provider

- Agent loop chỉ gọi `LlmProvider`.
- Fake provider phải chứng minh tool loop.
- OpenAI-compatible adapter có smoke test config, không gọi model thật trong unit suite.

## 7. Golden E2E

### Chuẩn bị

1. Chạy `npm install`.
2. Copy `.env.example` thành `.env`, điền model key cục bộ.
3. Chạy `npm run dev:api`.
4. Chạy `npm run build --workspace @vlc/firefox-extension`.
5. Firefox -> `about:debugging` -> This Firefox -> Load Temporary Add-on.
6. Chọn manifest trong output extension.

### Kịch bản

1. Mở một article HTML đã chọn trước.
2. Cuộn tới heading H và mở sidebar.
3. Xác nhận sidebar chỉ hiện text trong viewport.
4. Hỏi “Đoạn này đang giải thích gì?”.
5. Xác nhận answer và grounding label hiển thị trong sidebar.
6. Click `Chưa hiểu`; thao tác bấm là xác nhận ghi local memory.
7. Đóng rồi mở lại sidebar tại cùng heading.
8. Xác nhận card “Bạn từng ở đây” hiển thị `Chưa hiểu`.
9. Click resume/highlight.
10. Đổi thành `Đã hiểu`; xác nhận cùng marker ID có revision tăng.

### Nhánh từ chối

1. Mở PDF hoặc domain hard-block.
2. Mở sidebar.
3. UI hiện lý do bị chặn.
4. Network log chứng minh không gọi `/v1/agent/turn`.
5. Unit test `no capture/API when blocked` là bằng chứng tự động cho nhánh không đọc DOM.

## 8. Definition of Done chung

- `npm run verify` pass từ root.
- Không có secret trong Git/status/output.
- Không có contract duplicate trong app.
- Không có import xuyên app.
- Không có raw viewport trong log hoặc long-term memory.
- Unit tests pass và golden E2E được chạy thật trên Firefox.
- Demo trong 3 phút, không cần PDF/YouTube/ChatGPT.
- README chạy local đủ để thành viên khác làm lại từ checkout sạch.

## 9. Handoff bắt buộc của mỗi agent

```text
Branch:
Commit:
Owned files changed:
Commands run:
Tests passed:
Manual checks:
Known limitations:
Dependency request (if any):
Contract change proposal (if any, do not implement):
```

## 10. Xử lý conflict

- Conflict trong root/lockfile: giữ version từ Agent 3 rồi cài dependency bằng commit mới của Agent 3.
- Conflict contract: dừng merge, so với `04-shared-contracts.md`; không chọn “ours/theirs” mù.
- API response lệch schema: sửa implementation, không nới schema để che lỗi.
- Extension cần field mới: dùng fallback hiện có hoặc gửi contract proposal.
- E2E fail do model: dùng fake provider để cô lập; sau đó kiểm tra adapter/prompt, không sửa memory/UI trước.
