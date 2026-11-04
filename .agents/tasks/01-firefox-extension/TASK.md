# Task Agent 1 — Firefox Extension

> Status 2026-09-12: core implementation đã được nối vào `main`; automated AC pass. Còn Privacy/Memory UI và manual Golden E2E; xem task `04` và `06`.

## Mission

Xây phần Firefox Temporary Add-on hoàn chỉnh trong `apps/firefox-extension/**`: source policy, viewport capture, sidebar, pipeline gọi API, marker buttons và resume/highlight. Tối ưu để demo trực tiếp trong ngày hackathon.

## Execution harness

- Dùng npm và TypeScript strict; không dùng pnpm/yarn/Python.
- Bắt đầu bằng `npm install` và baseline commands; không thay root dependency.
- Thay TODO bằng implementation thật, không dừng ở scaffold/build placeholder.
- Sau mỗi work package, chạy test/typecheck của workspace.
- Chỉ báo done khi AC và Firefox manual check đều đạt; build pass với zero tests chưa phải done.
- Tự quyết định helper nội bộ; chỉ hỏi team khi cần đổi contract hoặc mở rộng quyền dữ liệu.

## Branch và ownership

- Branch: `feat/firefox-extension`
- Chỉ được sửa: `apps/firefox-extension/**`
- Không sửa: root manifests/lockfile, `.agents/**`, `apps/agent-api/**`, `packages/**`, `src/`, `firefox_extension/` cũ.

Nếu cần dependency mới hoặc đổi contract, ghi proposal trong handoff; không tự sửa vùng của Agent 3.

## Đọc trước khi code

1. `.agents/architecture/00-system-overview.md`
2. `.agents/architecture/01-firefox-extension.md`
3. `.agents/architecture/03-memory-tools-skills.md`
4. `.agents/architecture/04-shared-contracts.md`
5. `.agents/architecture/05-integration-e2e.md`

## Public functions phải implement đúng tên

```text
checkSourcePolicy
captureCurrentViewport
buildAgentTurnRequest
runOpenSidebarPipeline
runAskAgentPipeline
executeConfirmedMemoryCommand
requestAgentTurn
highlightOrResume
```

Không tạo local copies của shared interfaces; import từ `@vlc/contracts` và `@vlc/memory`.

## File map bắt buộc

| File | Code chính | Được gọi bởi |
|---|---|---|
| `manifest.json` | source entrypoints, permissions, sidebar, Gecko ID | Vite plugin/Firefox |
| `vite.config.ts` | build TypeScript + manifest thành `dist/` | npm scripts |
| `src/background/index.ts` | browser action, active tab, sidebar/message coordination | Firefox events |
| `src/content/index.ts` | đăng ký message handlers, không tự đọc DOM | background/sidebar |
| `src/content/capture-current-viewport.ts` | `captureCurrentViewport` | content message handler |
| `src/content/highlight-or-resume.ts` | `highlightOrResume` | content message handler |
| `src/policy/check-source-policy.ts` | `checkSourcePolicy` | open pipeline trước content message |
| `src/pipeline/open-sidebar-pipeline.ts` | `runOpenSidebarPipeline` | sidebar startup |
| `src/pipeline/ask-agent-pipeline.ts` | `buildAgentTurnRequest`, `runAskAgentPipeline` | sidebar submit |
| `src/pipeline/confirmed-memory-command.ts` | `executeConfirmedMemoryCommand` | marker button confirmation |
| `src/services/agent-api-client.ts` | `requestAgentTurn` | ask pipeline |
| `src/sidebar/main.ts` | UI state/render/event handlers | `index.html` |

Nếu thêm file helper thì vẫn phải giữ exports ở các file trên để Agent khác và tests tìm đúng seam.

## Work packages

### A. Manifest/build

- Manifest V2 Firefox desktop 142+.
- Gecko ID cố định.
- Giữ `data_collection_permissions` khớp dữ liệu thực sự gửi: website activity/content; không đổi thành `none` khi còn gọi model ngoài extension.
- Quyền tối thiểu: storage, active tab/host cần thiết, sidebar/browser action.
- Build output load được bằng `about:debugging`.
- Không hardcode API key.
- Source TypeScript phải được Vite build thành JavaScript trong `dist/`; Firefox chỉ load `dist/manifest.json`.

### B. Policy

- Policy chạy trước content message.
- Block PDF, video, social, private chat, webmail, sensitive portals và protocol ngoài HTTP(S).
- User allow/block settings đi qua repository.
- Blocked source không capture và không HTTP.

### C. Capture

- Chỉ visible viewport HTML.
- Bỏ hidden/off-screen/nav/footer/form values.
- Áp dụng đúng capture limits.
- Tạo source/anchor/fingerprint deterministic.
- Không giữ raw full-page content.

### D. Sidebar/pipelines

- State machine đúng kiến trúc.
- Mở sidebar chỉ policy/capture/recall; chưa gọi LLM.
- Ask pipeline refresh context khi stale rồi gọi API.
- Render answer, grounding, related memory và error.
- Ba status buttons và confirmation path.

### E. Memory/resume

- Gọi `MemoryRepository`; không tự thao tác storage bằng logic trùng lặp.
- Chỉ hiển thị saved sau `ok:true`.
- Resume theo text quote -> heading -> scroll ratio.

## Tests bắt buộc

- Policy allow/block/unknown và “no capture/API when blocked”.
- Visible DOM filtering và length limits.
- Request builder parse được bằng shared schema.
- Open pipeline không gọi model.
- Ask pipeline happy/error/stale cases.
- Confirmed command không chạy khi thiếu literal `userConfirmed:true`.
- Reopen với fake repository recall được marker.

## Acceptance Criteria

- `npm run typecheck --workspace @vlc/firefox-extension` pass.
- `npm test --workspace @vlc/firefox-extension` pass.
- `npm run build --workspace @vlc/firefox-extension` tạo add-on load được.
- `npm run lint:extension --workspace @vlc/firefox-extension` không có manifest error.
- Chạy được các bước extension trong golden E2E.
- Không sửa file ngoài ownership.

## Tự do triển khai

Được tự chia component/helper/CSS và dùng browser polyfill nếu dependency đã có. Ưu tiên UI rõ và ổn định hơn animation. Không thêm React, authentication hoặc feature ngoài scope.

## Handoff

Dùng template ở `.agents/architecture/05-integration-e2e.md`, kèm đường dẫn manifest build output và các bước load thủ công.
