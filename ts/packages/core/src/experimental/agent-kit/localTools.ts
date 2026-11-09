/**
 * @fileoverview Local tools that expose the LLM delegate and the shared memory to agents running
 * on a Composio session. **Experimental — shape may change in future releases.**
 *
 * The tools run in your process, so they use your own provider key and your own memory storage.
 *
 * @example
 * ```typescript
 * const tools = [
 *   createDelegateTool(new LLMDelegate({ provider: 'groq', apiKey: groqKey }), { agentId: 'claude' }),
 *   ...createMemoryTools(new ScopedMemory(store, { agent: 'claude', read: ['*'], write: ['*'] })),
 * ];
 * const session = await composio.create('user_123', { experimental: { localTools: tools } });
 * ```
 */
import { z } from 'zod/v3';
import { createCustomTool } from '../../models/CustomTool';
import type { LocalTool } from '../../types/customTool.types';
import type { LLMDelegate } from './delegate';
import { formatMemoryForPrompt, type ScopedMemory } from './memory';

export interface DelegateToolOptions {
  /** Identifies the calling agent for the delegate's budget. Defaults to `composio`. */
  agentId?: string;
}

/** `LOCAL_DELEGATE_LLM`: hands a cheap, fast sub-task to another model through the user's key. */
export function createDelegateTool(
  delegate: LLMDelegate,
  options: DelegateToolOptions = {}
): LocalTool {
  const agentId = options.agentId ?? 'composio';
  return createCustomTool('DELEGATE_LLM', {
    name: 'Delegate to another LLM',
    description:
      'Hand a quick, low-cost sub-task (summary, classification, extraction, translation) to ' +
      "another model through the user's own API key. The text leaves for an external provider, " +
      'so do not send secrets or personal data that is not needed. Usage is budget-limited.',
    inputParams: z.object({
      prompt: z.string().min(1).describe('The task to perform.'),
      system: z.string().optional().describe('Optional system instructions.'),
      model: z.string().optional().describe('Model to use. Defaults to the configured model.'),
      maxTokens: z.number().int().min(1).max(4096).optional(),
      json: z.boolean().optional().describe('Ask for a JSON response.'),
    }),
    execute: async input => {
      const result = await delegate.chat({
        agentId,
        model: input.model,
        messages: [
          ...(input.system ? [{ role: 'system' as const, content: input.system }] : []),
          { role: 'user' as const, content: input.prompt },
        ],
        maxTokens: input.maxTokens,
        json: input.json,
      });
      return {
        text: result.text,
        model: result.model,
        provider: result.provider,
        usage: result.usage,
      };
    },
  });
}

const SOURCE_TO_ORIGIN = {
  user_said: 'user',
  own_reasoning: 'agent',
  external_content: 'external',
} as const;

/** `LOCAL_MEMORY_SEARCH`, `LOCAL_MEMORY_WRITE` and `LOCAL_MEMORY_FORGET` over a scoped memory. */
export function createMemoryTools(memory: ScopedMemory): LocalTool[] {
  const search = createCustomTool('MEMORY_SEARCH', {
    name: 'Search shared memory',
    description:
      'Search the shared memory (preferences, decisions, facts about projects and people) written ' +
      'by you or by other agents.',
    inputParams: z.object({
      query: z.string().min(1),
      spaces: z.array(z.string()).optional().describe('For example ["personal","project:acme"].'),
      tags: z.array(z.string()).optional(),
      limit: z.number().int().min(1).max(20).optional(),
    }),
    execute: async input => {
      const hits = await memory.search(input);
      return {
        summary: formatMemoryForPrompt(hits),
        results: hits.map(({ entry, score }) => ({
          id: entry.id,
          space: entry.space,
          text: entry.text,
          tags: entry.tags,
          author: entry.author,
          updatedAt: entry.updatedAt,
          score: Number(score.toFixed(3)),
        })),
      };
    },
  });

  const write = createCustomTool('MEMORY_WRITE', {
    name: 'Write to shared memory',
    description:
      'Save a DURABLE fact to the shared memory (not temporary data). Declare the source: ' +
      '"user_said" if the user told you, "own_reasoning" if it is your own conclusion, ' +
      '"external_content" if it comes from an email, web page or tool output (it stays ' +
      'quarantined until the user verifies it). Never save API keys, passwords or tokens.',
    inputParams: z.object({
      text: z.string().min(1).max(4000),
      space: z.string().default('personal'),
      tags: z.array(z.string()).max(10).optional(),
      source: z.enum(['user_said', 'own_reasoning', 'external_content']),
      ttlDays: z.number().int().min(1).optional().describe('Optional expiry.'),
    }),
    execute: async input => {
      const entry = await memory.write({
        space: input.space,
        text: input.text,
        tags: input.tags,
        origin: SOURCE_TO_ORIGIN[input.source],
        ttlDays: input.ttlDays,
      });
      return { id: entry.id, space: entry.space, verified: entry.verified };
    },
  });

  const forget = createCustomTool('MEMORY_FORGET', {
    name: 'Forget a shared memory',
    description: 'Delete an entry from the shared memory by id, in spaces you can write to.',
    inputParams: z.object({ id: z.string().min(1) }),
    execute: async input => ({ deleted: await memory.forget(input.id) }),
  });

  return [search, write, forget];
}
