# 04 — Shared Contracts (đã khóa cho ba nhánh)

## 1. Quy tắc chung

- `packages/contracts` export cả TypeScript type và Zod schema tương ứng.
- Tất cả object qua HTTP, browser message hoặc storage phải parse được bằng schema.
- JSON fields dùng `camelCase`; tool protocol names dùng `snake_case`.
- Timestamp là ISO-8601 UTC string.
- URL lưu/log phải là `safeUrl`: bỏ fragment và query params không cần thiết.
- Public contract version của MVP là literal `"0.1"`.

```ts
export const CONTRACT_VERSION = "0.1" as const;
export type ContractVersion = typeof CONTRACT_VERSION;
export type IsoTimestamp = string;
```

## 2. Common result và error

```ts
export type AppErrorCode =
  | "VALIDATION_ERROR"
  | "SOURCE_BLOCKED"
  | "SOURCE_UNSUPPORTED"
  | "NO_READABLE_CONTENT"
  | "CONTEXT_STALE"
  | "MEMORY_NOT_FOUND"
  | "MEMORY_WRITE_FAILED"
  | "PERMISSION_DENIED"
  | "AGENT_UNAVAILABLE"
  | "AGENT_MAX_STEPS"
  | "MODEL_ERROR"
  | "NETWORK_ERROR"
  | "TOOL_NOT_FOUND"
  | "TOOL_EXECUTION_ERROR"
  | "INTERNAL_ERROR";

export interface AppError {
  code: AppErrorCode;
  message: string;
  retryable: boolean;
  details?: Record<string, string | number | boolean | null>;
}

export type Result<T> =
  | { ok: true; data: T }
  | { ok: false; error: AppError };
```

Không đặt raw viewport, raw prompt, API key hoặc stack trace trong `details`.

## 3. Source policy contracts

```ts
export type SourceType =
  | "HTML_ARTICLE"
  | "DOCUMENTATION"
  | "BLOG"
  | "SEARCH_RESULTS"
  | "PDF"
  | "VIDEO_PLATFORM"
  | "SOCIAL_FEED"
  | "PRIVATE_CHAT"
  | "WEBMAIL"
  | "SENSITIVE_PORTAL"
  | "UNKNOWN";

export type SourcePolicyDecision = "ALLOW" | "BLOCK" | "ASK";

export type SourcePolicyReasonCode =
  | "SUPPORTED_HTML"
  | "USER_ALLOWED_DOMAIN"
  | "USER_BLOCKED_DOMAIN"
  | "NON_HTTP_PROTOCOL"
  | "PDF_NOT_SUPPORTED"
  | "VIDEO_NOT_SUPPORTED"
  | "SOCIAL_NOT_SUPPORTED"
  | "PRIVATE_CONTENT_BLOCKED"
  | "SENSITIVE_CONTENT_BLOCKED"
  | "UNKNOWN_SOURCE";

export interface SourcePolicySettings {
  allowedDomains: string[];
  blockedDomains: string[];
  updatedAt: IsoTimestamp;
}

export interface SourcePolicyInput {
  url: string;
  title?: string;
  contentType?: string;
  settings: SourcePolicySettings;
}

export interface SourcePolicyResult {
  decision: SourcePolicyDecision;
  sourceType: SourceType;
  reasonCode: SourcePolicyReasonCode;
  safeUrl: string;
  hostname?: string;
  userMessage: string;
}

export interface UpdateSourcePolicyInput {
  domain: string;
  action: "ALLOW" | "BLOCK" | "RESET";
  userConfirmed: true;
}
```

## 4. Viewport contracts

```ts
export interface CaptureLimits {
  maxTextChars: 4000;
  maxCodeBlocks: 3;
  maxCodeCharsPerBlock: 1000;
  maxAnchorQuoteChars: 240;
}

export interface CaptureViewportInput {
  tabId: number;
  expectedUrl: string;
  limits?: CaptureLimits;
}

export interface SourceRef {
  canonicalUrl: string;
  safeUrl: string;
  hostname: string;
  title: string;
}

export interface ViewportAnchor {
  heading: string;
  textQuote: string;
  scrollRatio: number;
  fingerprint: string;
}

export interface ViewportContext {
  contextId: string;
  source: SourceRef;
  anchor: ViewportAnchor;
  visibleText: string;
  visibleCodeBlocks: string[];
  capturedAt: IsoTimestamp;
}

export interface HighlightOrResumeInput {
  tabId: number;
  expectedCanonicalUrl: string;
  anchor: ViewportAnchor;
}

export interface HighlightOrResumeOutput {
  found: boolean;
  strategy: "TEXT_QUOTE" | "HEADING" | "SCROLL_RATIO" | "NONE";
  appliedScrollRatio?: number;
}
```

`scrollRatio` phải nằm trong `[0, 1]`; string fields được trim; empty `visibleText` là error.

## 5. Memory contracts

```ts
export type MemoryStatus =
  | "UNDERSTOOD"
  | "NOT_UNDERSTOOD"
  | "REVIEW_LATER";

export type MemoryEvidenceKind =
  | "USER_MARK"
  | "USER_CHECKPOINT";

export interface MemoryEvidence {
  kind: MemoryEvidenceKind;
  summary: string;
  createdAt: IsoTimestamp;
}

export interface MemoryMarker {
  schemaVersion: 1;
  id: string;
  source: SourceRef;
  anchor: ViewportAnchor;
  status: MemoryStatus;
  note?: string;
  question?: string;
  answerSummary?: string;
  evidence: MemoryEvidence[];
  createdAt: IsoTimestamp;
  updatedAt: IsoTimestamp;
  lastVisitedAt: IsoTimestamp;
  revision: number;
}

export type MemoryMatchReason =
  | "EXACT_FINGERPRINT"
  | "SAME_PAGE_HEADING"
  | "SAME_PAGE";

export interface MemorySummary {
  id: string;
  source: SourceRef;
  anchor: ViewportAnchor;
  status: MemoryStatus;
  note?: string;
  question?: string;
  answerSummary?: string;
  updatedAt: IsoTimestamp;
  revision: number;
  matchScore: number;
  matchReason: MemoryMatchReason;
}

export interface SaveMarkerInput {
  source: SourceRef;
  anchor: ViewportAnchor;
  status: MemoryStatus;
  note?: string;
  question?: string;
  answerSummary?: string;
  evidence: MemoryEvidence;
  userConfirmed: true;
}

export interface SearchMemoryInput {
  source: SourceRef;
  anchor: ViewportAnchor;
  query?: string;
  status?: MemoryStatus;
  limit?: number;
}

export interface UpdateUnderstandingInput {
  memoryId: string;
  nextStatus: MemoryStatus;
  note?: string;
  evidence: MemoryEvidence;
  userConfirmed: true;
}

export type ForgetMemoryInput =
  | { scope: "ONE"; memoryId: string; userConfirmed: true }
  | { scope: "SOURCE"; canonicalUrl: string; userConfirmed: true }
  | { scope: "ALL"; userConfirmed: true };

export interface ForgetMemoryOutput {
  deletedCount: number;
}
```

Validation limits:

- `note`, `question`, `answerSummary`, evidence summary: tối đa 500 ký tự mỗi field;
- `evidence`: tối đa 20 entries, khi vượt thì giữ 20 mới nhất;
- `revision`: integer `>= 1`;
- `matchScore`: `[0,1]`;
- search limit mặc định 5 và tối đa 5.

## 6. Memory repository boundary

Các type sau thuộc public export của `@vlc/memory`:

```ts
export interface StorageAreaLike {
  get(keys?: string | string[] | null): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
  remove(keys: string | string[]): Promise<void>;
}

export interface Clock {
  now(): Date;
}

export interface IdGenerator {
  createId(): string;
}

export interface MemoryRepository {
  saveMarker(input: SaveMarkerInput): Promise<Result<MemoryMarker>>;
  searchMemory(input: SearchMemoryInput): Promise<Result<MemorySummary[]>>;
  readMemory(memoryId: string): Promise<Result<MemoryMarker | null>>;
  updateUnderstanding(
    input: UpdateUnderstandingInput,
  ): Promise<Result<MemoryMarker>>;
  forgetMemory(
    input: ForgetMemoryInput,
  ): Promise<Result<ForgetMemoryOutput>>;
  getSourcePolicySettings(): Promise<Result<SourcePolicySettings>>;
  updateSourcePolicy(
    input: UpdateSourcePolicyInput,
  ): Promise<Result<SourcePolicySettings>>;
}

export function createBrowserStorageMemoryRepository(
  storage: StorageAreaLike,
  clock: Clock,
  idGenerator: IdGenerator,
): MemoryRepository;

export function matchMemories(
  markers: readonly MemoryMarker[],
  input: SearchMemoryInput,
): MemorySummary[];
```

`saveMarker()` phải update record hiện có khi fingerprint trùng; không tạo duplicate.

## 7. Short session contracts

```ts
export type SidebarStatus =
  | "IDLE"
  | "CHECKING_SOURCE"
  | "BLOCKED"
  | "CAPTURING"
  | "READY"
  | "ASKING"
  | "READY_WITH_ANSWER"
  | "SAVING"
  | "READY_WITH_MEMORY"
  | "ERROR";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface PendingUserAction {
  id: string;
  type: "CONFIRM_MARKER";
  payload: {
    status: MemoryStatus;
    note?: string;
  };
  confirmationText: string;
}

export interface ShortSession {
  sessionId: string;
  status: SidebarStatus;
  policy?: SourcePolicyResult;
  context?: ViewportContext;
  relatedMemories: MemorySummary[];
  messages: ChatMessage[];
  pendingAction?: PendingUserAction;
  lastError?: AppError;
}
```

Messages tối đa 10; mỗi message tối đa 2.000 ký tự trong request.

## 8. Agent API contracts

```ts
export interface AgentPermissions {
  allowWebSearch: boolean;
}

export interface AgentTurnRequest {
  contractVersion: "0.1";
  turnId: string;
  question: string;
  context: ViewportContext;
  relatedMemories: MemorySummary[];
  history: ChatMessage[];
  permissions: AgentPermissions;
}

export type GroundingKind =
  | "VIEWPORT"
  | "MEMORY"
  | "WEB"
  | "INSUFFICIENT";

export interface GroundingRef {
  kind: Exclude<GroundingKind, "INSUFFICIENT">;
  refId: string;
  label: string;
  url?: string;
}

export interface ToolTraceEntry {
  step: number;
  toolName: AgentToolName;
  status: "SUCCESS" | "ERROR";
  errorCode?: AppErrorCode;
}

export interface AgentTurnResponse {
  contractVersion: "0.1";
  turnId: string;
  answer: string;
  grounding: GroundingKind;
  groundingRefs: GroundingRef[];
  suggestedActions: PendingUserAction[];
  toolTrace: ToolTraceEntry[];
  model: {
    provider: string;
    name: string;
  };
}

export interface HealthResponse {
  status: "ok";
  service: "viewport-learning-agent-api";
  contractVersion: "0.1";
}
```

Limits: question 1..2.000 chars, related memory tối đa 5, history tối đa 10, answer tối đa 4.000, suggested actions tối đa 3, tool trace tối đa 10.

## 9. Agent tool input/output contracts

```ts
export type AgentToolName =
  | "get_viewport_context"
  | "search_memory"
  | "read_memory"
  | "propose_marker"
  | "search_web";

export interface GetViewportContextToolInput {}
export type GetViewportContextToolOutput = ViewportContext;

export interface SearchMemoryToolInput {
  query: string;
  status?: MemoryStatus;
  limit?: number;
}
export type SearchMemoryToolOutput = MemorySummary[];

export interface ReadMemoryToolInput {
  memoryId: string;
}
export type ReadMemoryToolOutput = MemorySummary | null;

export interface ProposeMarkerToolInput {
  status: MemoryStatus;
  note?: string;
  reason: string;
}
export type ProposeMarkerToolOutput = PendingUserAction;

export interface SearchWebToolInput {
  query: string;
}
export interface SearchWebResultItem {
  title: string;
  url: string;
  snippet: string;
}
export interface SearchWebToolOutput {
  results: SearchWebResultItem[];
}
```

## 10. Extension pipeline contracts

Các dependency interfaces nằm trong app extension, nhưng signature bị khóa:

```ts
export interface OpenSidebarPipelineInput {
  tabId: number;
  url: string;
  title?: string;
  contentType?: string;
}

export interface OpenSidebarPipelineOutput {
  session: ShortSession;
}

export interface OpenSidebarPipelineDependencies {
  getPolicySettings(): Promise<Result<SourcePolicySettings>>;
  checkPolicy(input: SourcePolicyInput): Promise<Result<SourcePolicyResult>>;
  capture(input: CaptureViewportInput): Promise<Result<ViewportContext>>;
  memoryRepository: MemoryRepository;
  clock: Clock;
  idGenerator: IdGenerator;
}

export interface AskAgentPipelineInput {
  question: string;
  session: ShortSession;
}

export interface AskAgentPipelineDependencies {
  capture(input: CaptureViewportInput): Promise<Result<ViewportContext>>;
  requestAgentTurn(input: AgentTurnRequest): Promise<Result<AgentTurnResponse>>;
  clock: Clock;
  idGenerator: IdGenerator;
}

export interface BuildAgentTurnRequestInput {
  turnId: string;
  question: string;
  session: ShortSession;
  permissions: AgentPermissions;
}

export interface ConfirmedMemoryCommandInput {
  status: MemoryStatus;
  session: ShortSession;
  note?: string;
  userConfirmed: true;
}

export interface ConfirmedMemoryCommandDependencies {
  memoryRepository: MemoryRepository;
  clock: Clock;
}
```

Public signatures:

```ts
export function checkSourcePolicy(
  input: SourcePolicyInput,
): Promise<Result<SourcePolicyResult>>;

export function captureCurrentViewport(
  input: CaptureViewportInput,
): Promise<Result<ViewportContext>>;

export function buildAgentTurnRequest(
  input: BuildAgentTurnRequestInput,
): AgentTurnRequest;

export function runOpenSidebarPipeline(
  input: OpenSidebarPipelineInput,
  deps: OpenSidebarPipelineDependencies,
): Promise<Result<OpenSidebarPipelineOutput>>;

export function runAskAgentPipeline(
  input: AskAgentPipelineInput,
  deps: AskAgentPipelineDependencies,
): Promise<Result<AgentTurnResponse>>;

export function executeConfirmedMemoryCommand(
  input: ConfirmedMemoryCommandInput,
  deps: ConfirmedMemoryCommandDependencies,
): Promise<Result<MemoryMarker>>;

export function requestAgentTurn(
  input: AgentTurnRequest,
): Promise<Result<AgentTurnResponse>>;

export function highlightOrResume(
  input: HighlightOrResumeInput,
): Promise<Result<HighlightOrResumeOutput>>;
```

## 11. Agent runtime contracts

Các interfaces sau nằm trong app API:

```ts
export interface TurnToolContext {
  viewport: ViewportContext;
  relatedMemories: readonly MemorySummary[];
  permissions: AgentPermissions;
}

export interface AgentTool<I = unknown, O = unknown> {
  name: AgentToolName;
  description: string;
  inputSchema: z.ZodType<I>;
  execute(input: I): Promise<Result<O>>;
}

export interface LlmToolCall {
  id: string;
  name: string;
  arguments: unknown;
}

export interface LlmGenerateInput {
  systemPrompt: string;
  messages: Array<{
    role: "user" | "assistant" | "tool";
    content: string;
    toolCallId?: string;
  }>;
  tools: ReadonlyArray<{
    name: AgentToolName;
    description: string;
    inputJsonSchema: Record<string, unknown>;
  }>;
}

export interface LlmGenerateOutput {
  text?: string;
  toolCalls: LlmToolCall[];
  modelName: string;
}

export interface LlmProvider {
  readonly providerName: string;
  generate(input: LlmGenerateInput): Promise<Result<LlmGenerateOutput>>;
}

export interface AgentToolDependencies {
  idGenerator: IdGenerator;
  searchWeb?: (
    input: SearchWebToolInput,
  ) => Promise<Result<SearchWebToolOutput>>;
}

export interface AgentDependencies extends AgentToolDependencies {
  provider: LlmProvider;
}

export interface OpenAICompatibleProviderOptions {
  apiKey: string;
  baseUrl: string;
  modelName: string;
  timeoutMs: number;
  defaultHeaders?: Record<string, string>;
}
```

Public signatures:

```ts
export function buildSystemPrompt(): string;

export function createAgentToolRegistry(
  context: TurnToolContext,
  deps: AgentToolDependencies,
): ReadonlyMap<AgentToolName, AgentTool>;

export function runAgentTurn(
  input: AgentTurnRequest,
  deps: AgentDependencies,
): Promise<Result<AgentTurnResponse>>;

export function createOpenAICompatibleProvider(
  input: OpenAICompatibleProviderOptions,
): LlmProvider;
```

## 12. Browser message contract

```ts
export type ExtensionMessage =
  | {
      type: "CAPTURE_CURRENT_VIEWPORT";
      payload: CaptureViewportInput;
    }
  | {
      type: "HIGHLIGHT_OR_RESUME";
      payload: HighlightOrResumeInput;
    };
```

## 13. Ví dụ HTTP

Request rút gọn:

```json
{
  "contractVersion": "0.1",
  "turnId": "turn-123",
  "question": "Đoạn này giải thích gì?",
  "context": {
    "contextId": "ctx-123",
    "source": {
      "canonicalUrl": "https://example.com/agent-loop",
      "safeUrl": "https://example.com/agent-loop",
      "hostname": "example.com",
      "title": "Agent Loop"
    },
    "anchor": {
      "heading": "Tool execution",
      "textQuote": "The loop validates and executes a tool call...",
      "scrollRatio": 0.42,
      "fingerprint": "sha256:abc"
    },
    "visibleText": "The loop validates and executes a tool call...",
    "visibleCodeBlocks": [],
    "capturedAt": "2026-09-12T08:00:00.000Z"
  },
  "relatedMemories": [],
  "history": [],
  "permissions": { "allowWebSearch": false }
}
```

Response rút gọn:

```json
{
  "contractVersion": "0.1",
  "turnId": "turn-123",
  "answer": "Đoạn này mô tả vòng lặp kiểm tra lời gọi tool trước khi thực thi.",
  "grounding": "VIEWPORT",
  "groundingRefs": [
    { "kind": "VIEWPORT", "refId": "ctx-123", "label": "Tool execution" }
  ],
  "suggestedActions": [],
  "toolTrace": [
    { "step": 1, "toolName": "get_viewport_context", "status": "SUCCESS" }
  ],
  "model": { "provider": "openai-compatible", "name": "configured-model" }
}
```

## 14. Contract change process

Nếu implementation phát hiện contract chưa đủ:

1. Không tự sửa public contract trên nhánh feature.
2. Ghi `Contract change proposal` trong handoff: vấn đề, field/signature đề xuất, ảnh hưởng Agent 1/2/3.
3. Integrator quyết định và sửa một commit contract riêng.
4. Ba nhánh rebase/merge commit contract đó rồi mới tiếp tục.
