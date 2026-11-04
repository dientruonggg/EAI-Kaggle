# 01 — Kiến trúc con: Firefox Extension

## 1. Trách nhiệm

Extension là trust boundary quan trọng nhất. Nó sở hữu:

- browser action và sidebar UI;
- source policy;
- content script đọc viewport;
- short session;
- long-term memory client;
- HTTP client gọi Agent API;
- user-confirmed commands;
- highlight/resume trên trang.

Extension không chứa API key và không tự chạy model.

## 2. Cấu trúc đề xuất

```text
apps/firefox-extension/
├── package.json
├── manifest.json
├── src/
│   ├── background/index.ts
│   ├── content/
│   │   ├── capture-current-viewport.ts
│   │   └── highlight-or-resume.ts
│   ├── policy/check-source-policy.ts
│   ├── pipeline/
│   │   ├── open-sidebar-pipeline.ts
│   │   ├── ask-agent-pipeline.ts
│   │   └── confirmed-memory-command.ts
│   ├── services/agent-api-client.ts
│   └── sidebar/
│       ├── index.html
│       ├── main.ts
│       └── styles.css
└── tests/
```

Agent 1 được đổi tên helper bên trong, nhưng các public functions trong phần 7 phải giữ nguyên.

## 3. Source policy

`checkSourcePolicy()` phải chạy bằng URL/title/content type có sẵn từ tab metadata, trước khi gọi content script.

Thứ tự:

1. Chặn protocol khác `http:`/`https:`.
2. Áp dụng user blocked domains.
3. Áp dụng user allowed domains nếu có override rõ ràng và không thuộc hard block.
4. Hard block PDF, video platform, social feed, webmail, banking, health portal, private chat và trang nội bộ nhạy cảm.
5. Cho phép tài liệu kỹ thuật, blog và article HTML thông thường.
6. `UNKNOWN` phải hỏi/hiện không hỗ trợ, không tự capture.

`BLOCK` hoặc `ASK` đều không được gọi Agent API. MVP có thể render `ASK` như unsupported thay vì xây permission dialog.

## 4. Viewport capture

Chỉ lấy node có giao với viewport (`getBoundingClientRect`) và có text hiển thị. Bỏ qua:

- `script`, `style`, `noscript`, `nav`, `footer`, modal ẩn;
- `display:none`, `visibility:hidden`, `aria-hidden=true`;
- nội dung ngoài viewport;
- input value, cookie, token và metadata nhạy cảm;
- toàn bộ page source.

Giới hạn:

- `visibleText`: tối đa 4.000 ký tự sau normalize whitespace;
- `visibleCodeBlocks`: tối đa 3 block, mỗi block tối đa 1.000 ký tự;
- `anchor.textQuote`: tối đa 240 ký tự;
- `heading`: heading gần viewport nhất hoặc title fallback;
- `scrollRatio`: số từ `0` đến `1`;
- `fingerprint`: hash deterministic từ canonical URL + heading + normalized text quote.

Không lưu raw viewport vào long-term memory.

## 5. Sidebar state machine

```text
IDLE
  -> CHECKING_SOURCE
  -> BLOCKED | CAPTURING
  -> READY
  -> ASKING
  -> READY_WITH_ANSWER
  -> SAVING
  -> READY_WITH_MEMORY
  -> ERROR
```

UI tối thiểu:

- source badge và heading hiện tại;
- card “Bạn từng ở đây” nếu có match;
- input câu hỏi và nút gửi;
- answer + grounding label;
- ba nút `Đã hiểu`, `Chưa hiểu`, `Xem lại sau`;
- nút “Quay lại vị trí” khi có anchor;
- lỗi retry được và lý do trang bị chặn.

## 6. Browser messaging

Sidebar/background không đọc DOM trực tiếp. Chúng gửi message tới content script:

```ts
type ExtensionMessage =
  | { type: "CAPTURE_CURRENT_VIEWPORT"; payload: CaptureViewportInput }
  | { type: "HIGHLIGHT_OR_RESUME"; payload: HighlightOrResumeInput };
```

Response luôn là `Result<T>` từ `@vlc/contracts`; không throw raw object qua message boundary.

## 7. Public functions bắt buộc

```ts
checkSourcePolicy(
  input: SourcePolicyInput,
): Promise<Result<SourcePolicyResult>>;

captureCurrentViewport(
  input: CaptureViewportInput,
): Promise<Result<ViewportContext>>;

buildAgentTurnRequest(
  input: BuildAgentTurnRequestInput,
): AgentTurnRequest;

runOpenSidebarPipeline(
  input: OpenSidebarPipelineInput,
  deps: OpenSidebarPipelineDependencies,
): Promise<Result<OpenSidebarPipelineOutput>>;

runAskAgentPipeline(
  input: AskAgentPipelineInput,
  deps: AskAgentPipelineDependencies,
): Promise<Result<AgentTurnResponse>>;

executeConfirmedMemoryCommand(
  input: ConfirmedMemoryCommandInput,
  deps: ConfirmedMemoryCommandDependencies,
): Promise<Result<MemoryMarker>>;

requestAgentTurn(
  input: AgentTurnRequest,
): Promise<Result<AgentTurnResponse>>;

highlightOrResume(
  input: HighlightOrResumeInput,
): Promise<Result<HighlightOrResumeOutput>>;
```

Type chi tiết nằm trong [`04-shared-contracts.md`](04-shared-contracts.md).

## 8. Dependencies được inject

Pipeline không import singleton ngầm. Dependencies gồm:

- `sourcePolicy`;
- `viewportCapture`;
- `memoryRepository`;
- `agentApiClient`;
- `clock`;
- `idGenerator`.

Điều này cho phép Vitest dùng fake DOM/message/storage/API mà không cần Firefox thật cho unit test.

## 9. Error behavior

| Tình huống | Error code | UI |
|---|---|---|
| Trang bị chặn | `SOURCE_BLOCKED` | Lý do, không nút hỏi |
| PDF/video/private app | `SOURCE_UNSUPPORTED` | Nói rõ ngoài MVP |
| Không có text visible | `NO_READABLE_CONTENT` | Gợi ý cuộn tới đoạn cần hỏi |
| Tab đổi sau capture | `CONTEXT_STALE` | Capture lại một lần |
| API tắt | `AGENT_UNAVAILABLE` | Cho retry; marker local vẫn dùng được |
| Storage lỗi | `MEMORY_WRITE_FAILED` | Không hiển thị “Đã lưu” |

## 10. Acceptance Criteria của extension

- Policy unit test chứng minh blocked source không gọi capture và API.
- Viewport unit test chứng minh text ngoài viewport/hidden không được đưa vào context.
- Mở sidebar trên article HTML render heading và memory match mà chưa gọi model.
- Ask pipeline gửi request đúng schema và render response/error.
- Ba marker button ghi qua `MemoryRepository` sau click người dùng.
- Reopen sidebar cùng URL/heading hiển thị marker đã lưu.
- Highlight/resume có fallback theo `scrollRatio` nếu text quote không còn tồn tại.
- Manifest có `storage`, host permission tối thiểu, sidebar/browser action và Gecko ID cố định.
- Manifest khai báo Firefox data collection trung thực: website URL/context và viewport content được gửi tới API/model chỉ sau user activation.
