# 06 — Kế hoạch triển khai trong một ngày

## 1. Mục tiêu ngày thi

Ưu tiên một golden E2E chạy thật hơn nhiều feature dở dang. Sau mỗi checkpoint, main branch phải còn build được.

## 2. Timeline đề xuất

| Thời gian | Agent 1 — Firefox | Agent 2 — API | Agent 3 — Contracts/Memory |
|---|---|---|---|
| 0:00–0:30 | Đọc task, load scaffold Firefox | Đọc task, chạy health skeleton | Khóa schemas/common types |
| 0:30–2:00 | Policy + viewport capture tests | Fake-provider tool loop + routes | Contracts + schemas, commit sớm |
| Checkpoint 2:00 | Merge contracts commit vào branch | Merge contracts commit vào branch | Bàn giao contract hash |
| 2:00–4:00 | Sidebar + open/ask pipelines | OpenAI-compatible adapter + prompt | Repository + matcher + tests |
| Checkpoint 4:00 | Request parse shared schema | API response parse shared schema | Merge full Agent 3 branch |
| 4:00–5:30 | Nối API + marker/resume | Chạy API thật với Ollama; OpenRouter dự phòng | Hỗ trợ contract-only integration |
| 5:30–6:30 | Golden E2E và blocked-source E2E | Theo dõi safe logs/tool trace | Kiểm tra persisted marker/revision |
| 6:30–7:30 | UI polish và demo reset | Error UX/model fallback | Fix boundary tests |
| 7:30–8:00 | Freeze code, rehearsal 3 phút | Freeze code | `npm run verify`, freeze lockfile |

## 3. Checkpoint contracts

Đến giờ 2, Agent 3 phải giao tối thiểu:

- `Result`, errors, source/context/memory types;
- Agent request/response/tool types;
- Zod schemas;
- stable `src/index.ts` exports.

Agent 1/2 không chờ toàn bộ repository mới bắt đầu; họ code bằng frozen docs và merge contract commit tại checkpoint.

## 4. Model preflight

Trước integration:

1. Ưu tiên Ollama `qwen3:8b` cùng máy qua `http://127.0.0.1:11434/v1`.
2. Chạy `/api/tags` và `/v1/models` trước khi điền `.env`.
3. Gọi một request nhỏ để xác nhận base URL, model và `tools/tool_calls`.
4. Giới hạn agent loop 3 steps và context size theo contract.
5. Chuẩn bị OpenRouter làm dự phòng; chỉ đổi env, không đổi code.

Agent 2 dùng `openai.chat.completions.create()` hoặc API tương thích có cùng `tools`/`tool_calls`. Không dùng riêng một SDK/provider làm OpenRouter không chạy được.

## 5. Demo data preflight

- Chọn sẵn một article HTML tĩnh có heading rõ và đoạn text ngắn.
- Chọn sẵn một PDF để chứng minh policy block.
- Xóa/reset marker demo trước rehearsal nếu cần.
- Không dùng ChatGPT, email, banking hoặc trang cá nhân làm demo source.
- Giữ tab article ở đúng heading để không mất thời gian.

## 6. Fallback theo rủi ro

| Rủi ro | Fallback được phép |
|---|---|
| Model chính lỗi | Đổi `OPENAI_MODEL` sang model dự phòng có tool calling |
| Ollama/tunnel timeout hoặc 502 | Dùng direct localhost; nếu vẫn lỗi thì đổi env sang OpenRouter |
| Tool calling không ổn định | Siết system prompt/tool schema; không tăng quá 3 steps |
| Dynamic site capture nhiễu | Demo bằng article HTML tĩnh đã test |
| Exact quote không tìm lại | Resume theo heading rồi scroll ratio |
| Integration type lệch | Parse schema tại boundary, sửa implementation không nới contract |
| Thiếu thời gian | Bỏ `search_web`, checkpoint hiểu, options UI; giữ golden E2E |

Không được fallback bằng hardcode câu trả lời demo, giả memory success hoặc bỏ policy.

## 7. Freeze rule

Sau checkpoint 6:30:

- không thêm dependency;
- không đổi contract;
- không refactor lớn;
- chỉ sửa bug chặn E2E;
- chạy `npm run verify` sau mỗi integration fix;
- rehearsal bằng checkout/build hiện tại, không bằng dev state chưa commit.
