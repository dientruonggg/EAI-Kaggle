# Task Agent 3 — Shared Contracts, Memory và Workspace

> Status 2026-09-12: implementation đã được nối vào `main`; contracts/memory tests pass và storage keys đã chuẩn hóa. Repository đã có search/save/update/forget/policy; đưa các operation ra UI thuộc task `04`.

## Mission

Xây nền tảng dùng chung: npm workspaces, shared TypeScript/Zod contracts, browser-local memory repository, deterministic matcher và tests. Đây là nhánh merge đầu tiên.

## Execution harness

- Dùng npm và TypeScript strict; không dùng pnpm/yarn/Python.
- Chạy baseline trước khi sửa; giữ lockfile duy nhất là `package-lock.json`.
- Ưu tiên contract commit trong checkpoint 2 giờ; sau đó mới hoàn tất repository.
- Thay TODO bằng exports/schema/implementation/tests thật.
- `tsc` phải emit JavaScript và declarations vào `dist/` cho hai packages.
- Tự quyết định helper nội bộ; chỉ hỏi team khi frozen contract thật sự không thể triển khai.

## Branch và ownership

- Branch: `feat/contracts-memory`
- Được sửa:
  - `package.json`
  - `package-lock.json`
  - `tsconfig.base.json`
  - `.env.example`
  - `.gitignore` chỉ để bổ sung Node/env/build ignores
  - `packages/contracts/**`
  - `packages/memory/**`
- Không sửa: `.agents/**`, `apps/firefox-extension/**`, `apps/agent-api/**`, code cũ.

## Đọc trước khi code

1. `.agents/architecture/00-system-overview.md`
2. `.agents/architecture/03-memory-tools-skills.md`
3. `.agents/architecture/04-shared-contracts.md`
4. `.agents/architecture/05-integration-e2e.md`

## Public exports bắt buộc

`@vlc/contracts` export toàn bộ types, constants và Zod schemas được liệt kê trong `04-shared-contracts.md`.

`@vlc/memory` export:

```text
MemoryRepository
StorageAreaLike
Clock
IdGenerator
createBrowserStorageMemoryRepository
matchMemories
```

## File map bắt buộc

| File | Code chính | Consumer |
|---|---|---|
| `packages/contracts/src/common.ts` | version, `Result`, errors, shared primitives | tất cả packages |
| `packages/contracts/src/source.ts` | source policy/settings schemas | extension/memory |
| `packages/contracts/src/context.ts` | source ref, anchor, viewport schemas | extension/API/memory |
| `packages/contracts/src/memory.ts` | marker/search/command schemas | extension/API/memory |
| `packages/contracts/src/agent-api.ts` | session, request/response/health schemas | extension/API |
| `packages/contracts/src/tools.ts` | agent tool I/O schemas | API |
| `packages/contracts/src/index.ts` | explicit public exports | mọi consumer |
| `packages/memory/src/memory-repository.ts` | repository/storage interfaces | extension/adapter |
| `packages/memory/src/browser-storage-memory-repository.ts` | local storage implementation | extension |
| `packages/memory/src/memory-matcher.ts` | `matchMemories` | repository/tests |
| `packages/memory/src/index.ts` | public exports | extension |

Không để `@vlc/contracts` import `@vlc/memory`; dependency chỉ đi một chiều `memory -> contracts`.

## Work packages

### A. Workspace

- npm workspaces cho `apps/*` và `packages/*`.
- Root scripts: build, test, typecheck, verify, dev:api, dev:extension.
- TypeScript strict và Vitest.
- `.env.example`: port, allowed origins, OpenAI-compatible key/base URL/model, timeout và optional OpenRouter headers.
- Chỉ nhánh này sở hữu lockfile.

### B. Contracts

- Tách file theo common/source/context/memory/agent/tools.
- Mỗi public type có Zod schema tương ứng.
- Schema composition không circular.
- Export qua một `src/index.ts` rõ ràng.
- Contract version literal `0.1`.

### C. Repository

- Inject `StorageAreaLike`, clock và ID generator.
- Storage keys đúng kiến trúc.
- CRUD dùng `Result<T>`, map storage exception thành `MEMORY_WRITE_FAILED`.
- `saveMarker` upsert theo exact fingerprint.
- `updateUnderstanding` giữ createdAt, tăng revision, append evidence.
- `forgetMemory` xử lý đúng ONE/SOURCE/ALL.
- Policy update không cho user override hard-block ở layer policy; repository chỉ lưu settings hợp lệ.

### D. Matcher

- Scope canonical URL trước.
- Score: fingerprint 1.00, URL+heading 0.85, URL 0.60.
- Sort score giảm, tie-break updatedAt mới hơn.
- Optional status filter; limit max 5.
- Không embedding, không fuzzy/semantic claim.

## Tests bắt buộc

- Tất cả exported schemas accept valid fixture và reject invalid boundary.
- Save new marker và upsert same fingerprint không duplicate.
- Update tăng revision, giữ createdAt và status mới.
- Matching score/order/reason chính xác.
- Cross-source records không leak.
- Storage get/set/remove error mapping.
- Forget ONE/SOURCE/ALL.
- Policy settings round-trip.
- Record không chứa raw viewport hoặc secret fields.

## Acceptance Criteria

- `npm install` từ root thành công.
- `npm run typecheck --workspace @vlc/contracts` pass.
- `npm run typecheck --workspace @vlc/memory` pass.
- `npm test --workspace @vlc/contracts` pass.
- `npm test --workspace @vlc/memory` pass.
- Gửi commit contracts sớm để Agent 1/2 merge.
- Cuối task, `npm run verify` pass với những workspace đã hiện diện.
- Không sửa file ngoài ownership.

## Tự do triển khai

Được chọn cách chia schema/helper và transaction-in-memory nội bộ. Không dùng SQLite, IndexedDB wrapper, vector DB, cloud storage hoặc model. Không nới contract chỉ để test dễ hơn.

## Handoff

Dùng template ở `.agents/architecture/05-integration-e2e.md`. Bắt buộc ghi commit hash chứa contracts ổn định và danh sách dependency đã khóa trong `package-lock.json`.
