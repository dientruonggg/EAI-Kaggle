# Three-Agent Work Plan

## Trạng thái tích hợp

Ba task đã được hợp nhất vào `main` local. Không tiếp tục code trên ba nhánh cũ trừ khi cần truy vết commit. Trạng thái verify và việc còn lại nằm tại [`../architecture/07-integration-status.md`](../architecture/07-integration-status.md).

Các task `01`–`03` bên dưới được giữ làm acceptance criteria và lịch sử phân công. Phase last-mile mới dùng task `04`–`06`.

## Phase last-mile — ba người làm song song

Ba remote branch được chủ dự án tạo sẵn từ cùng một base. Mỗi người chỉ fetch và switch đúng branch của mình; không tự tạo lại từ SHA khác.

```text
BASE_REF: origin/last-mile-base
```

Trước khi code, chạy `git fetch origin --prune`, switch remote branch được giao và xác nhận `git merge-base --is-ancestor origin/last-mile-base HEAD` trả exit code `0`.

| Agent | Folder task | Branch đề xuất | Ownership không giao nhau |
|---|---|---|---|
| A | [`04-privacy-memory-ui/TASK.md`](04-privacy-memory-ui/TASK.md) | `feat/privacy-memory-ui` | `apps/firefox-extension/**` |
| B | [`05-safe-observability/TASK.md`](05-safe-observability/TASK.md) | `feat/safe-observability` | `apps/agent-api/**` |
| C — bạn + Codex | [`06-demo-release-harness/TASK.md`](06-demo-release-harness/TASK.md) | `feat/demo-release-harness` | root scripts/docs và `.agents/**`; không sửa business logic app |

Ba task map nhau qua contracts hiện có; không agent nào được tự sửa `packages/contracts/**` hoặc `packages/memory/**` trong phase này. Merge `04` và `05` trước theo thứ tự bất kỳ, `06` merge cuối và chạy Golden E2E trên `main` đã hợp nhất.

## Phase core — lịch sử đã hoàn thành

Ba task dưới đây là prompt cũ được giữ làm lịch sử. Không dùng thứ tự merge này cho phase last-mile.

Có thể copy nguyên nội dung `TASK.md` tương ứng cho coding agent. Task đã chứa mission, file map, public contracts, tests, AC và handoff; không cần viết lại thành prompt ngắn hơn.

| Thứ tự merge | Folder task | Branch | Ownership |
|---:|---|---|---|
| 1 | [`03-contracts-memory/TASK.md`](03-contracts-memory/TASK.md) | `feat/contracts-memory` | root npm/config, `packages/contracts/**`, `packages/memory/**` |
| 2 | [`02-agent-api/TASK.md`](02-agent-api/TASK.md) | `feat/agent-api` | `apps/agent-api/**` |
| 3 | [`01-firefox-extension/TASK.md`](01-firefox-extension/TASK.md) | `feat/firefox-extension` | `apps/firefox-extension/**` |

| Người | Task | Vì sao |
|---|---|---|
| Bạn + Codex | Agent 1 — Firefox Extension | Phức tạp nhất: manifest/Vite, DOM, privacy, UI và E2E hội tụ |
| Thành viên 2 | Agent 2 — Agent API | Boundary rõ: Fastify, tool loop, OpenAI-compatible API |
| Thành viên 3 | Agent 3 — Contracts + Memory | Deterministic, testable, cần giao contract commit sớm |

Agent 3 merge đầu tiên, nhưng cả ba có thể bắt đầu cùng lúc dựa trên frozen contract.

## Quy tắc chung

1. Đọc toàn bộ `.agents/architecture/README.md` và các file được task chỉ định.
2. Không sửa file ngoài ownership.
3. Không tự thay public contract.
4. Được tự do triển khai bên trong ownership miễn đạt AC.
5. Không commit secret, `.env`, raw browsing content hoặc dữ liệu cá nhân.
6. Không mở rộng sang PDF, YouTube, ChatGPT, tracking nền, vector DB hoặc cloud memory.
7. Bàn giao theo template trong `05-integration-e2e.md`.
8. Chỉ dùng npm và TypeScript; không thêm Python, pnpm hoặc yarn. JavaScript trong `dist/` là output build, không phải source viết tay.

## Dependency rule của phase last-mile

- Task 04 và 05 không cài dependency, không sửa root `package.json` hoặc `package-lock.json`.
- Chỉ task 06 được phép sửa root package/lockfile nếu script harness thật sự cần dependency; phải ghi rõ lý do trong handoff.
- Không task nào sửa `packages/contracts/**` hoặc `packages/memory/**`.

## Dependency rule cũ của phase core

Bootstrap dependency và root lockfile đã được chuẩn bị trước khi chia nhánh. Sau đó chỉ Agent 3 sửa `package-lock.json`. Agent 1/2 không tự chạy `npm install <new-package>` nếu việc đó sửa root lockfile; hãy ghi dependency request trong handoff.
