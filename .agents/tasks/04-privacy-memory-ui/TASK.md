# Task 04 — Firefox Privacy and Memory UI

## Status

READY. Remote branch được tạo sẵn: `feat/privacy-memory-ui`, base `origin/last-mile-base`.

## Mission

Cho người dùng nhìn thấy và điều khiển hai thứ đã có ở repository nhưng chưa có UI đầy đủ: website denylist và marker bộ nhớ của trang hiện tại.

## Ownership

- Branch: `feat/privacy-memory-ui`.
- Chỉ sửa `apps/firefox-extension/**`.
- Không sửa `apps/agent-api/**`, `packages/**`, root package files hoặc `.agents/**`.
- Không cài dependency mới.

## Quyết định về quyền và nơi lưu

- Manifest đã có permission `storage`; không thêm popup xin quyền ổ đĩa và không đọc folder dự án.
- Dùng `browser.storage.local` thông qua `memoryRepository`, không tạo localStorage/IndexedDB/file JSON thứ hai.
- Không tuyên bố storage được mã hóa. Khi Ask, tối đa 5 memory summaries hiện vẫn được đưa vào request Agent API theo contract hiện có.
- Mỗi operation có side effect chỉ chạy sau click rõ ràng của người dùng với `userConfirmed:true`.
- Temporary Firefox profile có storage riêng và có thể mất khi profile bị reset/gỡ add-on.

## UI phải có

Một disclosure/section nhỏ trong sidebar, không biến MVP thành dashboard:

1. `Website không được đọc`
   - hiện domain tab hiện tại;
   - nút `Không đọc website này`;
   - list `blockedDomains` do user tạo;
   - nút `Gỡ` cho từng domain;
   - empty state `Bạn chưa chặn website nào`.
2. `Dấu mốc trên trang này`
   - render tối đa 5 `relatedMemories`;
   - status, heading, revision và thời gian cập nhật;
   - nút Resume đúng marker;
   - nút `Xóa` đúng marker, có confirmation UI rõ ràng.

Hard-block built-in như PDF/YouTube/social/chat/webmail/tài chính/y tế chỉ hiển thị policy message; không đưa vào list có thể gỡ.

## Hàm chính và seam bắt buộc

Tạo đúng module `apps/firefox-extension/src/pipeline/privacy-memory-controls.ts` và export:

```ts
export interface PrivacyMemoryState {
  currentDomain: string;
  isCurrentDomainUserBlocked: boolean;
  blockedDomains: string[];
  relatedMemories: MemorySummary[];
}

export interface PrivacyMemoryDependencies {
  memoryRepository: MemoryRepository;
}

export function loadPrivacyMemoryState(
  input: { currentDomain: string; relatedMemories: MemorySummary[] },
  deps: PrivacyMemoryDependencies,
): Promise<Result<PrivacyMemoryState>>;

export function blockCurrentDomain(
  input: { domain: string; userConfirmed: true },
  deps: PrivacyMemoryDependencies,
): Promise<Result<SourcePolicySettings>>;

export function unblockDomain(
  input: { domain: string; userConfirmed: true },
  deps: PrivacyMemoryDependencies,
): Promise<Result<SourcePolicySettings>>;

export function forgetConfirmedMarker(
  input: { memoryId: string; userConfirmed: true },
  deps: PrivacyMemoryDependencies,
): Promise<Result<ForgetMemoryOutput>>;
```

UI loading/error state tiếp tục dùng `busy`, `renderShellStatus` và `Result<T>` hiện có; không thêm các trạng thái đó vào shared contract. Hard-block vẫn lấy từ `session.policy`, không đưa vào `PrivacyMemoryState`.

Mapping bắt buộc:

```ts
blockCurrentDomain  -> updateSourcePolicy({ action: "BLOCK", ... })
unblockDomain       -> updateSourcePolicy({ action: "RESET", ... })
forgetConfirmedMarker -> forgetMemory({ scope: "ONE", ... })
```

Implement bằng các method đã frozen trong `08-last-mile-plan.md`. Không đổi interface `MemoryRepository`.

## Behavior bắt buộc

- Normalize domain: lowercase, bỏ `www.` và dấu chấm cuối.
- Block thành công: chỉ báo success sau `ok:true`, chuyển UI hiện tại sang blocked/disable Ask và marker.
- Lần mở sidebar sau: policy phải dừng trước capture và Agent API.
- Unblock dùng action `RESET`, reload list sau thành công; không tự nói đã capture lại nếu chưa chạy pipeline mới.
- Delete marker dùng `forgetMemory({scope:"ONE"})`; chỉ remove UI sau `deletedCount===1`.
- Không tự lưu mọi viewport. Marker vẫn chỉ được tạo bởi ba nút trạng thái hiện có.

## Tests bắt buộc

- empty/list state của denylist;
- block/unblock gọi đúng repository input;
- block dừng capture/API;
- hard-block không bị allowlist override;
- render tối đa 5 marker;
- delete đúng memory ID và xử lý deletedCount 0;
- storage error không render success;
- serialized UI/diagnostic logs không chứa raw viewport hoặc memory summary;
- existing open/ask/marker/resume tests vẫn pass.

## Acceptance Criteria

```text
npm run typecheck --workspace @vlc/firefox-extension
npm test --workspace @vlc/firefox-extension
npm run build --workspace @vlc/firefox-extension
npm run lint:extension --workspace @vlc/firefox-extension
```

Tất cả pass; lint có 0 errors/warnings/notices. Handoff ghi commit hash, file list và manual observation block -> reopen -> unblock, save -> reopen -> delete.
