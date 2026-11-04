# Architecture Index — Viewport Learning Companion

Thư mục này là nguồn kiến trúc chuẩn để ba nhánh code cùng lúc. Nếu tài liệu cũ ở nơi khác mâu thuẫn với thư mục này, dùng nội dung trong `.agents/architecture`.

## Đọc theo thứ tự

1. [`../../.ask/Viewport_Learning_Companion_MVP_Spec.md`](../../.ask/Viewport_Learning_Companion_MVP_Spec.md) — sản phẩm phục vụ ai, làm gì và demo E2E nào.
2. [`../../.ask/listtools-skills.md`](../../.ask/listtools-skills.md) — danh sách skills, tools và khái niệm memory.
3. [`00-system-overview.md`](00-system-overview.md) — kiến trúc tổng và request flow.
4. [`01-firefox-extension.md`](01-firefox-extension.md) — kiến trúc con phía Firefox.
5. [`02-agent-api.md`](02-agent-api.md) — kiến trúc con phía Agent API và agent loop.
6. [`03-memory-tools-skills.md`](03-memory-tools-skills.md) — abstract chi tiết của skill/tool và memory pipeline.
7. [`04-shared-contracts.md`](04-shared-contracts.md) — type, function, endpoint và error code đã khóa.
8. [`05-integration-e2e.md`](05-integration-e2e.md) — cách ghép ba nhánh và tiêu chí demo.
9. [`06-one-day-plan.md`](06-one-day-plan.md) — timebox và fallback cho ngày thi.
10. [`07-integration-status.md`](07-integration-status.md) — trạng thái code/verify hiện tại và checklist trước demo.
11. [`08-last-mile-plan.md`](08-last-mile-plan.md) — ba workstream cuối, seam giao tiếp và thứ tự merge.
12. [`../tasks/README.md`](../tasks/README.md) — ownership, nhánh và thứ tự merge.

## Quyết định đã khóa

- Monorepo TypeScript strict dùng npm workspaces.
- Node.js 22+, npm 10+.
- Chỉ dùng npm; không tạo `pnpm-lock.yaml`, `yarn.lock` hoặc đổi package manager.
- Source ứng dụng/package dùng TypeScript; build phải tạo JavaScript chạy được trong `dist/`.
- Firefox desktop 142+ Temporary Add-on, Manifest V2 cho prototype trong ngày thi.
- Sidebar dùng Vanilla TypeScript + HTML/CSS; không cần React.
- Agent API dùng Fastify; backend stateless.
- LLM đi qua `LlmProvider`; adapter đầu tiên dùng package `openai` theo OpenAI-compatible API.
- Short-term memory ở state của sidebar.
- Long-term memory ở `browser.storage.local` qua `@vlc/memory`.
- Source policy chạy trước DOM capture và trước mọi network request.
- Model chỉ được đề xuất ghi/xóa/cập nhật; ứng dụng chỉ thực thi sau thao tác người dùng.
- Không dùng vector DB, cloud memory, scheduler, workflow engine, multi-agent hoặc background tracking trong MVP.
- E2E vàng: `mở bài -> đọc viewport -> hỏi -> đánh dấu -> đóng/mở sidebar -> nhớ -> cập nhật`.

## Quy tắc chống conflict

1. Mỗi agent chỉ sửa folder mình sở hữu trong [`../tasks/README.md`](../tasks/README.md).
2. Chỉ Agent 3 sửa root `package.json`, `package-lock.json`, `tsconfig.base.json` và hai package dùng chung.
3. Public contract trong `04-shared-contracts.md` là bất biến trong lúc ba nhánh chạy.
4. Tên tool dùng `snake_case`; tên hàm TypeScript dùng `camelCase` theo bảng mapping đã khóa.
5. Mọi thay đổi contract phải được ghi thành đề xuất trong handoff, không tự sửa contract giữa chừng.
6. Không import source file xuyên app/package. Chỉ import từ public package export hoặc gọi HTTP.

## Tự do triển khai

Mỗi agent được tự chọn helper, class, cách chia file con, CSS và cấu trúc test bên trong vùng sở hữu. Điều không được tự đổi là public type, public function, endpoint, folder ownership, security rule và Acceptance Criteria.
