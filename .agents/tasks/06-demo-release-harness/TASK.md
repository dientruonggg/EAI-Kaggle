# Task 06 — Demo, Storage Inspection and Firefox Release Harness

## Status

READY cho phần scripts/docs. Remote branch được tạo sẵn: `feat/demo-release-harness`, base `origin/last-mile-base`. Golden E2E chỉ được đánh PASS sau khi merge task 04 và 05. Owner: bạn + Codex.

## Mission

Tạo đường chạy lặp lại được cho cả team và chuẩn bị cách đưa extension vào Firefox. Không sửa business logic extension/API trong khi hai agent khác đang làm.

## Ownership

- Branch: `feat/demo-release-harness`.
- Được sửa:
  - `README.md`, `.env.example`;
  - root `package.json`/`package-lock.json` chỉ khi thật sự cần script orchestration;
  - `scripts/**`;
  - `.agents/**` và tài liệu `.ask` liên quan trạng thái.
- Không sửa `apps/firefox-extension/src/**`, `apps/agent-api/src/**` hoặc `packages/**`.

## File đầu ra bắt buộc

- `scripts/demo-preflight.ts` — read-only checks, exit `0` khi các check bắt buộc pass và khác `0` khi fail.
- `.agents/architecture/09-firefox-demo-runbook.md` — lệnh chạy và hướng dẫn Temporary Add-on/storage.
- `.agents/architecture/10-firefox-signing-release.md` — signed `.xpi`, self-distribution và AMO; không chứa credential.
- `.agents/evidence/demo-e2e-result.md` — bảng PASS/FAIL/evidence không chứa raw browsing content.
- Root npm script `demo:preflight` gọi file TypeScript trên; task 06 được phép thêm `tsx` root devDependency nếu hoisting hiện tại không đủ.

## Deliverables

1. Lệnh preflight kiểm tra:
   - Node/npm;
   - Ollama process, model `qwen3:8b`, port `11434`;
   - local tool-call smoke;
   - optional tunnel `/v1/models` và inference smoke nếu `TUNNEL_BASE_URL` được đặt;
   - Agent API health port `8787`;
   - build/lint extension.
2. Lệnh mở Firefox Temporary Add-on bằng `web-ext run`.
3. Hướng dẫn mở article HTML sau khi add-on được cài.
4. Hướng dẫn inspect storage qua `about:debugging` -> This Firefox -> Inspect -> Storage:
   - `vlc:markers:v1`;
   - `vlc:policy:v1`.
5. Checklist Golden E2E có cột PASS/FAIL/evidence.
6. Tài liệu riêng cho signed `.xpi`/AMO dựa trên hướng dẫn Mozilla hiện hành.

Không nhúng key vào command/file. Tunnel check không phải blocker của localhost demo; kết quả phải ghi rõ REQUIRED hoặc OPTIONAL.

## Baseline hướng dẫn Firefox đã kiểm chứng

Manual Temporary Add-on cho repo này:

1. Chạy `npm run build --workspace @vlc/firefox-extension`.
2. Mở `about:debugging` -> `This Firefox` -> `Load Temporary Add-on...`.
3. Chọn `apps/firefox-extension/dist/manifest.json` (không chọn source `manifest.json` và không chọn folder root).
4. Mở/reload một article HTTP(S) bình thường sau khi cài add-on, rồi bấm icon để mở sidebar.
5. Sau mỗi thay đổi source TypeScript/HTML/CSS, build lại để cập nhật `dist`, sau đó bấm `Reload` trong `about:debugging` nếu không chạy watcher.

Automated development path hiện có:

```text
npm run build --workspace @vlc/firefox-extension
npm run start:firefox --workspace @vlc/firefox-extension
```

`web-ext run` tự mở Firefox và cài add-on tạm từ `dist`. Vì `source-dir` là `dist`, source edit chỉ có hiệu lực sau khi Vite tạo lại build output. Task 06 có thể bổ sung build watch orchestration nhưng không được sửa business logic.

Không dùng `about:*`, Reader View, Firefox PDF viewer hoặc `addons.mozilla.org` làm article test vì content script không được inject vào privileged/restricted pages.

Nguồn chuẩn cần trích trong runbook:

- <https://extensionworkshop.com/documentation/develop/temporary-installation-in-firefox/>
- <https://extensionworkshop.com/documentation/develop/getting-started-with-web-ext/>
- <https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Content_scripts>

## Nguyên tắc memory

- Không tìm hoặc sửa file profile vật lý bằng tay.
- Không tuyên bố Temporary Add-on storage là vĩnh viễn qua profile reset/gỡ add-on.
- Không seed dữ liệu cá nhân; smoke fixture phải là article công khai/giả lập.

## Golden E2E sau merge

```text
article -> sidebar -> viewport đúng -> ask -> Qwen answer
-> safe tool log -> save marker -> inspect vlc:markers:v1
-> close/open sidebar -> recall -> resume (revision không đổi)
-> bấm lưu lại cùng marker -> revision tăng
-> block domain -> reopen không capture/API -> unblock
-> xóa marker -> marker biến mất sau reload
```

## Acceptance Criteria

- Người mới clone repo có thể chạy theo README mà không hỏi lệnh ẩn.
- `npm run verify` pass trên main đã merge.
- Có bằng chứng Qwen local. Chỉ bắt buộc bằng chứng tunnel khi `TUNNEL_BASE_URL` được cấu hình; nếu không, ghi `SKIPPED_OPTIONAL`.
- Golden E2E chỉ ghi PASS cho bước đã quan sát thật.
- Không commit `.env`, AMO credentials, tunnel token hoặc raw browsing content.
- Không tự push/publish add-on khi chưa có yêu cầu rõ của chủ dự án.
