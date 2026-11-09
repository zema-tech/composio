import { describe, it, expect } from 'vitest';
import { LLMDelegate, LLMDelegateError } from '../../../src/experimental/agent-kit/delegate';

const KEY = 'gsk_TESTKEY1234567890abcdefghij';
const userMessage = [{ role: 'user' as const, content: 'ciao' }];

const okResponse = (
  content = 'ciao',
  usage: Record<string, number> | null = { prompt_tokens: 10, completion_tokens: 5 }
) =>
  new Response(
    JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      choices: [{ message: { content } }],
      ...(usage ? { usage } : {}),
    }),
    { status: 200, headers: { 'content-type': 'application/json' } }
  );

const rejectsWithCode = async (promise: Promise<unknown>, code: string) => {
  const error = await promise.then(
    () => undefined,
    (e: unknown) => e
  );
  expect(error).toBeInstanceOf(LLMDelegateError);
  expect(error).toMatchObject({ code });
};

describe('LLMDelegate', () => {
  it('sends the request to the provider with the right URL, header and body', async () => {
    let seenUrl = '';
    let seenInit: RequestInit | undefined;
    const delegate = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      fetch: async (url, init) => {
        seenUrl = String(url);
        seenInit = init;
        return okResponse();
      },
    });

    const result = await delegate.chat({ agentId: 'claude', messages: userMessage, json: true });

    expect(result.text).toBe('ciao');
    expect(result.usage).toEqual({
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
      estimated: false,
    });
    expect(seenUrl).toBe('https://api.groq.com/openai/v1/chat/completions');
    expect(seenInit?.headers).toMatchObject({ authorization: `Bearer ${KEY}` });
    expect(JSON.parse(String(seenInit?.body))).toMatchObject({
      model: 'llama-3.3-70b-versatile',
      response_format: { type: 'json_object' },
    });
    expect(delegate.usage('claude')).toEqual({
      calls: 1,
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    });
  });

  it('estimates usage when the provider does not report it', async () => {
    const delegate = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      fetch: async () => okResponse('abcdefgh', null),
    });
    const result = await delegate.chat({ messages: [{ role: 'user', content: 'abcd' }] });
    expect(result.usage).toEqual({
      promptTokens: 1,
      completionTokens: 2,
      totalTokens: 3,
      estimated: true,
    });
  });

  it('enforces the call budget per agent', async () => {
    const delegate = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      budget: { maxCalls: 2 },
      fetch: async () => okResponse(),
    });
    await delegate.chat({ agentId: 'a', messages: userMessage });
    await delegate.chat({ agentId: 'a', messages: userMessage });
    await rejectsWithCode(
      delegate.chat({ agentId: 'a', messages: userMessage }),
      'budget_exceeded'
    );
    await expect(delegate.chat({ agentId: 'b', messages: userMessage })).resolves.toBeDefined();
  });

  it('enforces the token budget', async () => {
    const delegate = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      budget: { maxTotalTokens: 15 },
      fetch: async () => okResponse(),
    });
    await delegate.chat({ messages: userMessage });
    await rejectsWithCode(delegate.chat({ messages: userMessage }), 'budget_exceeded');
  });

  it('requires a model on OpenRouter and honors the allow-list', async () => {
    const openrouter = new LLMDelegate({
      provider: 'openrouter',
      apiKey: 'sk-or-v1-abcdefghijklmnopqrstuv',
      fetch: async () => okResponse(),
    });
    await rejectsWithCode(openrouter.chat({ messages: userMessage }), 'no_model');

    const groq = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      allowedModels: ['llama-3.3-70b-versatile'],
      fetch: async () => okResponse(),
    });
    await rejectsWithCode(
      groq.chat({ model: 'other', messages: userMessage }),
      'model_not_allowed'
    );
  });

  it('retries on 429 and then succeeds', async () => {
    let calls = 0;
    const delegate = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      fetch: async () => {
        calls += 1;
        return calls === 1
          ? new Response('rate limited', { status: 429, headers: { 'retry-after': '0' } })
          : okResponse();
      },
    });
    await expect(delegate.chat({ messages: userMessage })).resolves.toMatchObject({ text: 'ciao' });
    expect(calls).toBe(2);
  });

  it('never leaks the API key in errors', async () => {
    const delegate = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      maxRetries: 0,
      fetch: async () => new Response(`invalid key ${KEY}`, { status: 401 }),
    });
    const error = await delegate.chat({ messages: userMessage }).then(
      () => undefined,
      (e: unknown) => e
    );
    expect(error).toBeInstanceOf(LLMDelegateError);
    expect(error).toMatchObject({ code: 'http_error', status: 401 });
    expect(String((error as Error).message)).not.toContain(KEY);
    expect(String((error as Error).message)).toContain('[REDACTED]');
  });

  it('rejects malformed provider responses', async () => {
    const delegate = new LLMDelegate({
      provider: 'groq',
      apiKey: KEY,
      fetch: async () => new Response(JSON.stringify({ choices: [] }), { status: 200 }),
    });
    await rejectsWithCode(delegate.chat({ messages: userMessage }), 'bad_response');
  });

  it('fails fast without an API key or, for custom, without a baseUrl', () => {
    expect(() => new LLMDelegate({ provider: 'groq', apiKey: '  ' })).toThrow(/API key/);
    expect(() => new LLMDelegate({ provider: 'custom', apiKey: KEY })).toThrow(/baseUrl/);
  });
});
