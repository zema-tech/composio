/**
 * @fileoverview Delegates sub-tasks to OpenAI-compatible LLM providers (Groq, OpenRouter, ...)
 * using the caller's own API key. **Experimental — shape may change in future releases.**
 *
 * - Platform-neutral: uses the global `fetch` and never reads the environment. Pass the key in.
 * - The key never appears in errors, logs or results.
 * - Exposes chat completions only (no batch, fine-tuning or key management).
 * - Per-agent budgets and an optional model allow-list stop a looping agent from burning the key.
 *
 * @example
 * ```typescript
 * const groq = new LLMDelegate({
 *   provider: 'groq',
 *   apiKey: process.env.GROQ_API_KEY ?? '',
 *   budget: { maxCalls: 50, maxTotalTokens: 200_000 },
 * });
 * const { text } = await groq.chat({
 *   agentId: 'claude',
 *   messages: [{ role: 'user', content: 'Classify this email: ...' }],
 * });
 * ```
 */
import { z } from 'zod/v3';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface DelegateBudget {
  /** Maximum number of calls per agent. A call counts once it is sent, even if it later fails. */
  maxCalls?: number;
  /** Soft cap on total tokens per agent: calls stop once the consumed total reaches it. */
  maxTotalTokens?: number;
}

export type DelegateProvider = 'groq' | 'openrouter' | 'custom';

export interface LLMDelegateOptions {
  provider: DelegateProvider;
  /** The caller's API key for the provider. Required. */
  apiKey: string;
  /** Required for the `custom` provider, for example `https://host/v1`. */
  baseUrl?: string;
  /** Groq defaults to `llama-3.3-70b-versatile`. OpenRouter has no default: pass a model. */
  defaultModel?: string;
  /** When set, only these models can be used. */
  allowedModels?: string[];
  budget?: DelegateBudget;
  /** Retries on 429 and 5xx responses and network errors. Defaults to 2. */
  maxRetries?: number;
  /** Per-request timeout. Defaults to 30 seconds. */
  timeoutMs?: number;
  /** Upper bound on output tokens for a single call. Defaults to 2048. */
  maxOutputTokens?: number;
  /** Override `fetch`, for tests or custom transports. */
  fetch?: typeof fetch;
}

export interface ChatRequest {
  /** Identifies the calling agent for budgeting. Defaults to `default`. */
  agentId?: string;
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
  /** Ask for a JSON object response (`response_format: json_object`). */
  json?: boolean;
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface UsageSnapshot extends TokenUsage {
  calls: number;
}

export interface ChatResult {
  text: string;
  model: string;
  provider: DelegateProvider;
  usage: TokenUsage & { estimated: boolean };
}

export type LLMDelegateErrorCode =
  | 'missing_api_key'
  | 'missing_base_url'
  | 'model_not_allowed'
  | 'no_model'
  | 'invalid_request'
  | 'budget_exceeded'
  | 'http_error'
  | 'timeout'
  | 'bad_response';

export class LLMDelegateError extends Error {
  readonly code: LLMDelegateErrorCode;
  readonly status?: number;

  constructor(code: LLMDelegateErrorCode, message: string, status?: number) {
    super(message);
    this.name = 'LLMDelegateError';
    this.code = code;
    this.status = status;
  }
}

const ChatCompletionResponseSchema = z.object({
  model: z.string().optional(),
  choices: z.array(z.object({ message: z.object({ content: z.string().nullable() }) })).min(1),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
    })
    .optional(),
});

const PRESETS: Record<'groq' | 'openrouter', { baseUrl: string; defaultModel?: string }> = {
  groq: { baseUrl: 'https://api.groq.com/openai/v1', defaultModel: 'llama-3.3-70b-versatile' },
  openrouter: { baseUrl: 'https://openrouter.ai/api/v1' },
};

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

const emptyUsage = (): UsageSnapshot => ({
  calls: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
});

export class LLMDelegate {
  readonly provider: DelegateProvider;
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly defaultModel?: string;
  private readonly allowedModels?: string[];
  private readonly budget: DelegateBudget;
  private readonly maxRetries: number;
  private readonly timeoutMs: number;
  private readonly maxOutputTokens: number;
  private readonly fetchImpl: typeof fetch;
  private readonly usageByAgent = new Map<string, UsageSnapshot>();

  constructor(options: LLMDelegateOptions) {
    this.provider = options.provider;
    const apiKey = options.apiKey.trim();
    if (!apiKey) {
      throw new LLMDelegateError(
        'missing_api_key',
        `An API key is required for the "${options.provider}" provider.`
      );
    }
    this.apiKey = apiKey;

    const preset = options.provider === 'custom' ? undefined : PRESETS[options.provider];
    const baseUrl = options.baseUrl ?? preset?.baseUrl;
    if (!baseUrl) {
      throw new LLMDelegateError('missing_base_url', 'The "custom" provider requires a baseUrl.');
    }
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.defaultModel = options.defaultModel ?? preset?.defaultModel;
    this.allowedModels = options.allowedModels;
    this.budget = options.budget ?? {};
    this.maxRetries = options.maxRetries ?? 2;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.maxOutputTokens = options.maxOutputTokens ?? 2048;
    this.fetchImpl = options.fetch ?? fetch;
  }

  /** Recorded consumption for one agent, or summed over all agents when omitted. */
  usage(agentId?: string): UsageSnapshot {
    if (agentId) return { ...(this.usageByAgent.get(agentId) ?? emptyUsage()) };
    const total = emptyUsage();
    for (const u of this.usageByAgent.values()) {
      total.calls += u.calls;
      total.promptTokens += u.promptTokens;
      total.completionTokens += u.completionTokens;
      total.totalTokens += u.totalTokens;
    }
    return total;
  }

  async chat(request: ChatRequest): Promise<ChatResult> {
    const agentId = request.agentId ?? 'default';
    const model = request.model ?? this.defaultModel;
    if (!model) {
      throw new LLMDelegateError('no_model', 'No model given and no defaultModel configured.');
    }
    if (this.allowedModels && !this.allowedModels.includes(model)) {
      throw new LLMDelegateError(
        'model_not_allowed',
        `Model "${model}" is not allowed. Allowed: ${this.allowedModels.join(', ')}`
      );
    }
    if (request.messages.length === 0) {
      throw new LLMDelegateError('invalid_request', 'messages must not be empty.');
    }

    const used = this.usageByAgent.get(agentId) ?? emptyUsage();
    if (this.budget.maxCalls !== undefined && used.calls >= this.budget.maxCalls) {
      throw new LLMDelegateError(
        'budget_exceeded',
        `Call budget exhausted for "${agentId}" (${this.budget.maxCalls}).`
      );
    }
    if (
      this.budget.maxTotalTokens !== undefined &&
      used.totalTokens >= this.budget.maxTotalTokens
    ) {
      throw new LLMDelegateError(
        'budget_exceeded',
        `Token budget exhausted for "${agentId}" (${this.budget.maxTotalTokens}).`
      );
    }
    used.calls += 1;
    this.usageByAgent.set(agentId, used);

    const body: Record<string, unknown> = {
      model,
      messages: request.messages,
      max_tokens: Math.min(request.maxTokens ?? this.maxOutputTokens, this.maxOutputTokens),
    };
    if (request.temperature !== undefined) body.temperature = request.temperature;
    if (request.json) body.response_format = { type: 'json_object' };

    const parsed = ChatCompletionResponseSchema.safeParse(
      await this.post('/chat/completions', body)
    );
    const text = parsed.success ? parsed.data.choices[0]?.message.content : undefined;
    if (!parsed.success || text === null || text === undefined) {
      throw new LLMDelegateError(
        'bad_response',
        'The provider response has no choices[0].message.content.'
      );
    }

    const reportedPrompt = parsed.data.usage?.prompt_tokens;
    const reportedCompletion = parsed.data.usage?.completion_tokens;
    const estimated = reportedPrompt === undefined || reportedCompletion === undefined;
    const promptTokens =
      reportedPrompt ??
      Math.ceil(request.messages.reduce((sum, m) => sum + m.content.length, 0) / 4);
    const completionTokens = reportedCompletion ?? Math.ceil(text.length / 4);
    used.promptTokens += promptTokens;
    used.completionTokens += completionTokens;
    used.totalTokens += promptTokens + completionTokens;

    return {
      text,
      model: parsed.data.model ?? model,
      provider: this.provider,
      usage: {
        promptTokens,
        completionTokens,
        totalTokens: promptTokens + completionTokens,
        estimated,
      },
    };
  }

  private redact(text: string): string {
    return text.split(this.apiKey).join('[REDACTED]');
  }

  private async post(path: string, body: unknown): Promise<unknown> {
    let lastError: LLMDelegateError | undefined;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const res = await this.fetchImpl(`${this.baseUrl}${path}`, {
          method: 'POST',
          headers: { 'content-type': 'application/json', authorization: `Bearer ${this.apiKey}` },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (res.ok) {
          try {
            return await res.json();
          } catch {
            throw new LLMDelegateError('bad_response', 'The provider returned invalid JSON.');
          }
        }

        const detail = this.redact((await res.text().catch(() => '')).slice(0, 300));
        lastError = new LLMDelegateError(
          'http_error',
          `The provider responded ${res.status}. ${detail}`.trim(),
          res.status
        );
        const retryable = res.status === 429 || res.status >= 500;
        if (!retryable || attempt === this.maxRetries) throw lastError;
        const retryAfter = Number(res.headers.get('retry-after'));
        await sleep(
          Number.isFinite(retryAfter) && retryAfter > 0
            ? Math.min(retryAfter * 1000, 10_000)
            : 400 * 2 ** attempt
        );
      } catch (error) {
        if (error instanceof LLMDelegateError) throw error;
        const aborted = error instanceof Error && error.name === 'AbortError';
        lastError = aborted
          ? new LLMDelegateError('timeout', `Timed out after ${this.timeoutMs} ms.`)
          : new LLMDelegateError(
              'http_error',
              this.redact(error instanceof Error ? error.message : String(error))
            );
        if (attempt === this.maxRetries) throw lastError;
        await sleep(400 * 2 ** attempt);
      } finally {
        clearTimeout(timer);
      }
    }
    throw lastError ?? new LLMDelegateError('http_error', 'Unknown error.');
  }
}
