# 03 — Skills, Tools và Memory Contracts

## 1. Cách đọc tài liệu này

- **Skill** là năng lực hoàn chỉnh mà người dùng nhìn thấy.
- **Tool** là primitive có `name`, `description`, `input`, `output`, quyền và failure behavior.
- **Command** là tool có side effect; MVP chỉ chạy command sau user confirmation.
- **Memory** là record có cấu trúc trong storage, không phải “AI tự nhớ trong đầu”.

Một tool trong code không chỉ là input/output. Để ba module nối an toàn, nó cần đủ:

```text
name + owner + input schema + output schema
+ side effect + permission + error codes + test contract
```

## 2. Danh sách skills nhanh

| Skill | Giá trị người dùng | Tools chính |
|---|---|---|
| Source Safety | Chỉ đọc nguồn được phép | `check_source_policy` |
| Viewport Awareness | Biết đúng phần HTML đang hiện | `capture_current_viewport`, `get_viewport_context` |
| Contextual Q&A | Hỏi “đoạn này” không copy-paste | `get_viewport_context` |
| Memory Recall | Biết từng tương tác tại section | `search_memory`, `read_memory` |
| Learning Marker | Lưu 3 trạng thái học | `propose_marker`, `save_marker` |
| Understanding Tracking | Phân biệt user xác nhận và AI suy đoán | `update_understanding` |
| Resume Learning | Quay lại anchor đã lưu | `highlight_or_resume` |
| Controlled Research | Tìm thêm khi user bật quyền | `search_web` — optional |
| Privacy Control | Chặn nguồn/xóa memory | `update_source_policy`, `forget_memory` |

## 3. Skill abstracts

### 3.1 Source Safety

```yaml
goal: Không đọc hoặc gửi dữ liệu từ nguồn bị chặn.
trigger: User mở extension hoặc tab hiện tại thay đổi.
input: SourcePolicyInput
output: SourcePolicyResult
tools: [check_source_policy]
side_effect: none
deny_rule: BLOCK hoặc ASK dừng pipeline trước DOM và HTTP.
acceptance: blocked source có 0 lần capture và 0 lần gọi API.
```

### 3.2 Viewport Awareness

```yaml
goal: Tạo context nhỏ, đúng phần đang hiện, đủ để hỏi.
trigger: Source được ALLOW và cần mở/refresh context.
input: CaptureViewportInput
output: ViewportContext
tools: [capture_current_viewport, get_viewport_context]
side_effect: short session only
limits: text 4000 chars; 3 code blocks; anchor quote 240 chars.
acceptance: hidden/off-screen text không xuất hiện trong output.
```

### 3.3 Contextual Q&A

```yaml
goal: Trả lời câu hỏi dựa trên section hiện tại.
trigger: User submit câu hỏi.
input: AgentTurnRequest
output: AgentTurnResponse
tools: [get_viewport_context, search_memory?, search_web?]
side_effect: update short session only
failure: thiếu context thì nói thiếu dữ kiện, không bịa.
acceptance: response có grounding ref tới viewport hoặc báo INSUFFICIENT.
```

### 3.4 Memory Recall

```yaml
goal: Gợi lại dấu mốc liên quan mà không nhớ nhầm.
trigger: Sau capture hoặc khi câu hỏi phụ thuộc lịch sử.
input: SearchMemoryInput
output: MemorySummary[] (max 5)
tools: [search_memory, read_memory]
side_effect: optional touch lastVisitedAt; không đổi understanding status.
confidence: mọi hit có matchScore và matchReason.
acceptance: khác source scope không bao giờ bị trả về.
```

### 3.5 Learning Marker

```yaml
goal: Lưu UNDERSTOOD, NOT_UNDERSTOOD hoặc REVIEW_LATER tại đúng anchor.
trigger: User click marker; hoặc xác nhận proposal của Agent.
input: SaveMarkerInput với userConfirmed=true
output: MemoryMarker đã persisted
tools: [propose_marker, save_marker]
side_effect: write browser.storage.local
failure: chỉ hiện “Đã lưu” sau storage success.
acceptance: reopen sidebar tìm lại được marker.
```

### 3.6 Understanding Tracking

```yaml
goal: Theo dõi trạng thái học bằng bằng chứng trung thực.
trigger: User đổi trạng thái hoặc trả lời checkpoint.
input: UpdateUnderstandingInput với userConfirmed=true
output: MemoryMarker revision mới
tools: [update_understanding]
side_effect: update một marker; giữ createdAt, tăng revision.
rule: việc đã xem/đã hỏi không tự biến thành UNDERSTOOD.
acceptance: status chỉ đổi sau thao tác rõ ràng của user.
```

### 3.7 Resume Learning

```yaml
goal: Đưa user về gần đúng vị trí đã lưu.
trigger: User click “Quay lại vị trí”.
input: HighlightOrResumeInput
output: HighlightOrResumeOutput
tools: [highlight_or_resume]
side_effect: scroll/highlight DOM tạm thời
fallback: textQuote -> heading -> scrollRatio.
acceptance: không tìm thấy exact quote vẫn có fallback và báo strategy.
```

### 3.8 Controlled Research — optional

```yaml
goal: Bổ sung nguồn ngoài viewport khi user yêu cầu.
trigger: User hỏi tìm thêm và allowWebSearch=true.
input: SearchWebToolInput
output: SearchWebToolOutput
tools: [search_web]
side_effect: network request
deny_rule: permission=false thì PERMISSION_DENIED.
acceptance: không làm hoặc tắt tool không ảnh hưởng golden E2E.
```

### 3.9 Privacy Control

```yaml
goal: User kiểm soát domain và dữ liệu đã lưu.
trigger: User block/unblock domain hoặc xóa memory.
input: UpdateSourcePolicyInput | ForgetMemoryInput
output: SourcePolicySettings | ForgetMemoryOutput
tools: [update_source_policy, forget_memory]
side_effect: update browser.storage.local
permission: luôn cần thao tác user.
acceptance: xóa xong read/search không còn trả record đó.
```

## 4. Tool registry và mapping tên

| Protocol tool name | TypeScript function | Owner | Model gọi? | Side effect |
|---|---|---|---:|---:|
| `check_source_policy` | `checkSourcePolicy` | Extension | Không | Không |
| `capture_current_viewport` | `captureCurrentViewport` | Extension | Không | Không |
| `get_viewport_context` | registry closure | API | Có | Không |
| `search_memory` | `MemoryRepository.searchMemory` | Memory/API closure | Có | Không |
| `read_memory` | `MemoryRepository.readMemory` | Memory/API closure | Có | Không |
| `propose_marker` | registry closure | API | Có | Không; chỉ proposal |
| `save_marker` | `MemoryRepository.saveMarker` | Extension + Memory | Không | Có |
| `update_understanding` | `MemoryRepository.updateUnderstanding` | Extension + Memory | Không | Có |
| `highlight_or_resume` | `highlightOrResume` | Extension | Không | DOM tạm thời |
| `forget_memory` | `MemoryRepository.forgetMemory` | Extension + Memory | Không | Có |
| `update_source_policy` | `MemoryRepository.updateSourcePolicy` | Extension + Memory | Không | Có |
| `search_web` | optional server adapter | API | Có nếu bật | Network |

Trong API, `search_memory` và `read_memory` chỉ thao tác trên tối đa 5 summaries nằm trong immutable `TurnToolContext`. Chúng không gọi ngược về browser.

## 5. Abstract của từng tool

### `check_source_policy`

- Input: URL, title, content type và settings.
- Output: `ALLOW | BLOCK | ASK`, source type, reason code, user message.
- Error: `VALIDATION_ERROR`.
- Invariant: phải hoàn tất trước capture.

### `capture_current_viewport`

- Input: tab ID/context ID và capture limits.
- Output: sanitized `ViewportContext`.
- Error: `SOURCE_BLOCKED`, `NO_READABLE_CONTENT`, `CONTEXT_STALE`.
- Invariant: không trả raw full page hoặc form values.

### `get_viewport_context`

- Input: `{}`.
- Output: context của turn hiện tại.
- Error: `NO_READABLE_CONTENT`.
- Invariant: pure; không recapture browser.

### `search_memory`

- Input: query, optional status, limit `1..5`.
- Output: summaries sắp giảm theo `matchScore`.
- Error: `VALIDATION_ERROR`.
- Invariant: chỉ cùng `sourceScope`; hit nào cũng có score/reason.

### `read_memory`

- Input: `memoryId`.
- Output: một summary hoặc `null`.
- Error: `MEMORY_NOT_FOUND` nếu API tool đọc ID ngoài preload set.
- Invariant: không đoán/scan ID toàn storage.

### `propose_marker`

- Input: status, optional note/reason.
- Output: `PendingUserAction`.
- Error: `VALIDATION_ERROR`.
- Invariant: không ghi storage và response phải nói cần user xác nhận.

### `save_marker`

- Input: source, anchor, status, optional note/question/answer summary, `userConfirmed:true`.
- Output: persisted `MemoryMarker` revision 1.
- Error: `MEMORY_WRITE_FAILED`, `PERMISSION_DENIED`.
- Invariant: stored record tự đủ nghĩa và không chứa raw viewport.

### `update_understanding`

- Input: memory ID, next status, evidence, `userConfirmed:true`.
- Output: persisted marker tăng revision.
- Error: `MEMORY_NOT_FOUND`, `MEMORY_WRITE_FAILED`, `PERMISSION_DENIED`.
- Invariant: giữ audit evidence; không đổi source scope.

### `highlight_or_resume`

- Input: anchor đã lưu.
- Output: found, strategy và optional scroll ratio.
- Error: `CONTEXT_STALE` nếu tab khác nguồn.
- Invariant: highlight có timeout và không sửa nội dung thật của trang.

### `forget_memory`

- Input: scope `ONE | SOURCE | ALL`, selector tương ứng, `userConfirmed:true`.
- Output: số record đã xóa.
- Error: `PERMISSION_DENIED`, `MEMORY_WRITE_FAILED`.
- Invariant: scope xóa phải explicit; không mặc định `ALL`.

### `update_source_policy`

- Input: domain, action `ALLOW | BLOCK | RESET`, `userConfirmed:true`.
- Output: settings mới.
- Error: `VALIDATION_ERROR`, `MEMORY_WRITE_FAILED`.
- Invariant: user override không vượt hard-block nhạy cảm trong MVP.

### `search_web` — optional

- Input: query tự đủ nghĩa.
- Output: tối đa 5 title/URL/snippet.
- Error: `PERMISSION_DENIED`, `NETWORK_ERROR`.
- Invariant: chỉ chạy khi `allowWebSearch=true`; source phải hiện trong answer.

## 6. Long-term memory model

Một marker lưu:

- schema version và ID;
- source scope gồm canonical URL + hostname;
- title và heading;
- anchor nhỏ: text quote + scroll ratio + fingerprint;
- learning status;
- note/question/answer summary tùy chọn;
- evidence do user xác nhận;
- created/updated/last visited timestamps;
- revision.

Không lưu:

- raw full viewport;
- full chat transcript;
- model prompt;
- API key;
- password/form input;
- embedding/vector.

Storage keys:

```text
vlc:markers:v1       -> Record<memoryId, MemoryMarker>
vlc:policy:v1        -> SourcePolicySettings
vlc:settings:v1      -> ExtensionSettings
```

## 7. Deterministic memory matching

Không gọi đây là semantic search. Matcher dùng thứ tự:

| Match | Score | Reason |
|---|---:|---|
| Cùng fingerprint | `1.00` | `EXACT_FINGERPRINT` |
| Cùng canonical URL + normalized heading | `0.85` | `SAME_PAGE_HEADING` |
| Cùng canonical URL | `0.60` | `SAME_PAGE` |
| Khác source scope | không trả | — |

Nếu bằng điểm: `updatedAt` mới hơn đứng trước. Trả tối đa 5 record. `SAME_PAGE` chỉ hiển thị như “Có thể liên quan trên trang này”, không nói chắc “Bạn từng học đoạn này”.

## 8. Update thay vì nhớ chồng

Khi cùng fingerprint đã có marker:

1. đọc record hiện tại;
2. cập nhật status/note/evidence;
3. giữ `createdAt`;
4. đặt `updatedAt=now`;
5. tăng `revision`;
6. write thành công rồi mới trả success.

MVP không cần archive/supersede record phức tạp, nhưng revision giúp tránh mất lịch sử logic và dễ debug demo.

## 9. Short-term memory

Short session chỉ sống trong sidebar state:

- context hiện tại;
- tối đa 5 related memories;
- tối đa 10 chat messages;
- pending user action;
- current UI status và last error.

Đóng sidebar có thể làm mất transcript ngắn hạn; long-term marker vẫn còn. Đây là behavior chấp nhận được của MVP.

## 10. Memory E2E cần chứng minh

1. Lưu marker tại article A, heading H.
2. Đóng/mở sidebar hoặc reload extension cùng profile/Gecko ID.
3. Capture lại article A, heading H.
4. Search trả marker với `SAME_PAGE_HEADING` hoặc `EXACT_FINGERPRINT`.
5. UI hiển thị status cũ.
6. User đổi status.
7. Repository trả cùng ID, revision tăng, status mới.

Không dùng fake vector hay claim paraphrase recall trong demo.

