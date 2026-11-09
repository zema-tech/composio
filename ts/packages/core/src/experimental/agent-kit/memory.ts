/**
 * @fileoverview Shared memory for several agents (Claude, Grok, Codex, ...).
 * **Experimental — shape may change in future releases.**
 *
 * - Spaces: `personal`, `project:<name>`, `team:<name>`, with per-agent read/write policies.
 * - Provenance: every entry records which agent wrote it and from what kind of source.
 * - Quarantine: entries derived from external content (emails, web pages, tool results) stay
 *   unverified and are never returned to agents until a human verifies them.
 * - Secrets (API keys, tokens, private keys) are refused.
 * - Platform-neutral: persistence goes through the small {@link MemoryStorage} interface, so it
 *   works on Node and on workerd. The default storage is in memory.
 * - Search is lexical and dependency-free. Replace `search` for semantic retrieval.
 */
import { z } from 'zod/v3';

export const MemoryOriginSchema = z.enum(['user', 'agent', 'external']);
export type MemoryOrigin = z.infer<typeof MemoryOriginSchema>;

export const MemoryEntrySchema = z.object({
  id: z.string(),
  space: z.string(),
  text: z.string(),
  tags: z.array(z.string()),
  /** Name of the agent that wrote the entry. */
  author: z.string(),
  origin: MemoryOriginSchema,
  verified: z.boolean(),
  createdAt: z.string(),
  updatedAt: z.string(),
  expiresAt: z.string().optional(),
});
export type MemoryEntry = z.infer<typeof MemoryEntrySchema>;

const MemoryFileSchema = z.object({ version: z.literal(1), entries: z.array(MemoryEntrySchema) });

export interface MemoryWriteInput {
  space: string;
  text: string;
  tags?: string[];
  author: string;
  origin?: MemoryOrigin;
  ttlDays?: number;
}

export interface MemorySearchQuery {
  query: string;
  spaces?: string[];
  spaceFilter?: (space: string) => boolean;
  tags?: string[];
  limit?: number;
  includeUnverified?: boolean;
}

export interface MemorySearchHit {
  entry: MemoryEntry;
  score: number;
}

export type MemoryErrorCode =
  'invalid_input' | 'secret_detected' | 'access_denied' | 'not_found' | 'corrupt_storage';

export class MemoryError extends Error {
  readonly code: MemoryErrorCode;

  constructor(code: MemoryErrorCode, message: string) {
    super(message);
    this.name = 'MemoryError';
    this.code = code;
  }
}

/**
 * Where the serialized memory lives. Implement it over a file, a database or a KV store.
 * `withLock` is optional: provide it when several processes share the same storage.
 */
export interface MemoryStorage {
  /** The raw serialized state, or `undefined` when nothing has been stored yet. */
  read(): Promise<string | undefined>;
  write(serialized: string): Promise<void>;
  /** Runs `fn` while holding an exclusive lock across processes. */
  withLock?<T>(fn: () => Promise<T>): Promise<T>;
}

export class InMemoryStorage implements MemoryStorage {
  private value: string | undefined;

  async read(): Promise<string | undefined> {
    return this.value;
  }

  async write(serialized: string): Promise<void> {
    this.value = serialized;
  }
}

const SPACE_PATTERN = /^[a-z0-9][a-z0-9:_\-./]{0,63}$/i;
const MAX_TEXT_LENGTH = 4000;
const DAY_MS = 86_400_000;

const SECRET_PATTERNS: RegExp[] = [
  /gsk_[A-Za-z0-9]{20,}/, // Groq
  /sk-or-v1-[A-Za-z0-9]{20,}/, // OpenRouter
  /sk-[A-Za-z0-9_-]{20,}/, // OpenAI, Anthropic and similar
  /gh[pousr]_[A-Za-z0-9]{30,}/, // GitHub
  /\bAKIA[0-9A-Z]{16}\b/, // AWS access key id
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
  /\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*/,
];

export function looksLikeSecret(text: string): boolean {
  return SECRET_PATTERNS.some(pattern => pattern.test(text));
}

const normalize = (s: string) =>
  s
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();

const tokenize = (s: string) =>
  normalize(s)
    .split(/[^a-z0-9]+/)
    .filter(token => token.length >= 2);

const matchSpace = (pattern: string, space: string) => {
  if (pattern === '*') return true;
  if (pattern.endsWith('*')) return space.startsWith(pattern.slice(0, -1));
  return pattern === space;
};

const cleanTags = (tags: string[]) => [
  ...new Set(tags.map(tag => tag.trim().toLowerCase()).filter(Boolean)),
];

export interface MemoryStoreOptions {
  /** Defaults to an in-memory storage. */
  storage?: MemoryStorage;
  /** For tests. */
  now?: () => Date;
}

export class MemoryStore {
  private readonly storage: MemoryStorage;
  private readonly now: () => Date;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(options: MemoryStoreOptions = {}) {
    this.storage = options.storage ?? new InMemoryStorage();
    this.now = options.now ?? (() => new Date());
  }

  async write(input: MemoryWriteInput): Promise<MemoryEntry> {
    const text = input.text.trim();
    const space = input.space.trim();
    const author = input.author.trim();
    if (!text) throw new MemoryError('invalid_input', 'text is empty.');
    if (text.length > MAX_TEXT_LENGTH) {
      throw new MemoryError('invalid_input', `text exceeds ${MAX_TEXT_LENGTH} characters.`);
    }
    if (!SPACE_PATTERN.test(space)) {
      throw new MemoryError('invalid_input', `Invalid space: "${space}".`);
    }
    if (!author) throw new MemoryError('invalid_input', 'author is required.');
    const tags = cleanTags(input.tags ?? []);
    if (tags.length > 10 || tags.some(tag => tag.length > 32)) {
      throw new MemoryError('invalid_input', 'At most 10 tags of at most 32 characters each.');
    }
    if (looksLikeSecret(text) || tags.some(looksLikeSecret)) {
      throw new MemoryError(
        'secret_detected',
        'The text looks like it contains a secret (API key, token): not saved.'
      );
    }
    const origin = input.origin ?? 'agent';
    const nowIso = this.now().toISOString();

    return this.mutate(entries => {
      const key = normalize(text);
      const duplicate = entries.find(e => e.space === space && normalize(e.text) === key);
      if (duplicate) {
        duplicate.tags = [...new Set([...duplicate.tags, ...tags])];
        duplicate.updatedAt = nowIso;
        return { ...duplicate };
      }
      const entry: MemoryEntry = {
        id: globalThis.crypto.randomUUID(),
        space,
        text,
        tags,
        author,
        origin,
        verified: origin !== 'external',
        createdAt: nowIso,
        updatedAt: nowIso,
        ...(input.ttlDays && input.ttlDays > 0
          ? { expiresAt: new Date(this.now().getTime() + input.ttlDays * DAY_MS).toISOString() }
          : {}),
      };
      entries.push(entry);
      return { ...entry };
    });
  }

  async search(query: MemorySearchQuery): Promise<MemorySearchHit[]> {
    const queryTokens = [...new Set(tokenize(query.query))];
    if (queryTokens.length === 0) return [];
    const wantedTags = (query.tags ?? []).map(tag => tag.toLowerCase());
    const hits: MemorySearchHit[] = [];

    for (const entry of await this.loadLive()) {
      if (!this.inScope(entry, query)) continue;
      if (!entry.verified && !query.includeUnverified) continue;
      if (wantedTags.length > 0 && !wantedTags.every(tag => entry.tags.includes(tag))) continue;

      const entryTokens = tokenize(`${entry.text} ${entry.tags.join(' ')}`);
      const counts = new Map<string, number>();
      for (const token of entryTokens) counts.set(token, (counts.get(token) ?? 0) + 1);

      let matched = 0;
      for (const token of queryTokens) {
        const count = counts.get(token);
        if (count) matched += 1 + Math.log(1 + count) * 0.25;
        else if (token.length >= 4 && entryTokens.some(t => t.startsWith(token))) matched += 0.5;
      }
      if (matched === 0) continue;
      let score = matched / queryTokens.length;
      for (const tag of entry.tags) if (queryTokens.includes(tag)) score += 0.25;
      hits.push({ entry: { ...entry }, score });
    }

    hits.sort((a, b) => b.score - a.score || b.entry.updatedAt.localeCompare(a.entry.updatedAt));
    return hits.slice(0, Math.max(1, Math.min(query.limit ?? 5, 50)));
  }

  async list(options: Omit<MemorySearchQuery, 'query'> = {}): Promise<MemoryEntry[]> {
    return (await this.loadLive())
      .filter(e => this.inScope(e, options) && (e.verified || options.includeUnverified))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
      .slice(0, Math.max(1, Math.min(options.limit ?? 20, 200)))
      .map(e => ({ ...e }));
  }

  async get(id: string): Promise<MemoryEntry | undefined> {
    const entry = (await this.loadLive()).find(e => e.id === id);
    return entry ? { ...entry } : undefined;
  }

  async update(id: string, patch: { text?: string; tags?: string[] }): Promise<MemoryEntry> {
    const text = patch.text?.trim();
    if (patch.text !== undefined && (!text || text.length > MAX_TEXT_LENGTH)) {
      throw new MemoryError('invalid_input', 'Invalid text.');
    }
    if (text && looksLikeSecret(text)) {
      throw new MemoryError('secret_detected', 'The text looks like it contains a secret.');
    }
    return this.mutate(entries => {
      const entry = entries.find(e => e.id === id);
      if (!entry) throw new MemoryError('not_found', `Entry ${id} not found.`);
      if (text) entry.text = text;
      if (patch.tags) entry.tags = cleanTags(patch.tags).slice(0, 10);
      entry.updatedAt = this.now().toISOString();
      return { ...entry };
    });
  }

  /** A human action: makes a quarantined entry visible to agents. */
  async verify(id: string): Promise<MemoryEntry> {
    return this.mutate(entries => {
      const entry = entries.find(e => e.id === id);
      if (!entry) throw new MemoryError('not_found', `Entry ${id} not found.`);
      entry.verified = true;
      entry.updatedAt = this.now().toISOString();
      return { ...entry };
    });
  }

  async forget(id: string): Promise<boolean> {
    return this.mutate(entries => {
      const index = entries.findIndex(e => e.id === id);
      if (index < 0) return false;
      entries.splice(index, 1);
      return true;
    });
  }

  private inScope(
    entry: MemoryEntry,
    scope: { spaces?: string[]; spaceFilter?: (space: string) => boolean }
  ): boolean {
    if (scope.spaces && scope.spaces.length > 0 && !scope.spaces.includes(entry.space))
      return false;
    if (scope.spaceFilter && !scope.spaceFilter(entry.space)) return false;
    return true;
  }

  private isLive(entry: MemoryEntry): boolean {
    return !entry.expiresAt || new Date(entry.expiresAt).getTime() > this.now().getTime();
  }

  /** Loads the stored entries. A corrupt file is an error: it is never silently reset. */
  private async load(): Promise<MemoryEntry[]> {
    const raw = await this.storage.read();
    if (raw === undefined || raw.trim() === '') return [];
    let json: unknown;
    try {
      json = JSON.parse(raw);
    } catch {
      throw new MemoryError('corrupt_storage', 'The stored memory is not valid JSON.');
    }
    const parsed = MemoryFileSchema.safeParse(json);
    if (!parsed.success) {
      throw new MemoryError(
        'corrupt_storage',
        'The stored memory does not match the expected format.'
      );
    }
    return parsed.data.entries;
  }

  private async loadLive(): Promise<MemoryEntry[]> {
    return (await this.load()).filter(e => this.isLive(e));
  }

  /** Read-modify-write, serialized in this process and locked across processes when supported. */
  private mutate<T>(fn: (entries: MemoryEntry[]) => T): Promise<T> {
    const run = async (): Promise<T> => {
      const entries = (await this.load()).filter(e => this.isLive(e));
      const result = fn(entries);
      await this.storage.write(JSON.stringify({ version: 1, entries }, null, 2));
      return result;
    };
    const locked = () => (this.storage.withLock ? this.storage.withLock(run) : run());
    const next = this.queue.then(locked);
    this.queue = next.catch(() => undefined);
    return next;
  }
}

// ---------- per-agent access ----------

export interface MemoryAccessPolicy {
  /** The agent name, recorded as `author` on every entry it writes. */
  agent: string;
  /** Readable spaces: `personal`, `project:*`, `*`. */
  read: string[];
  /** Writable spaces. Entries can only be forgotten in spaces the agent can write. */
  write: string[];
}

/** A view of the memory with the permissions of a single agent. */
export class ScopedMemory {
  readonly agent: string;
  private readonly store: MemoryStore;
  private readonly policy: MemoryAccessPolicy;

  constructor(store: MemoryStore, policy: MemoryAccessPolicy) {
    this.store = store;
    this.policy = policy;
    this.agent = policy.agent;
  }

  canRead(space: string): boolean {
    return this.policy.read.some(pattern => matchSpace(pattern, space));
  }

  canWrite(space: string): boolean {
    return this.policy.write.some(pattern => matchSpace(pattern, space));
  }

  async write(input: Omit<MemoryWriteInput, 'author'>): Promise<MemoryEntry> {
    if (!this.canWrite(input.space)) {
      throw new MemoryError(
        'access_denied',
        `${this.agent} cannot write to the space "${input.space}".`
      );
    }
    return this.store.write({ ...input, author: this.agent });
  }

  async search(query: {
    query: string;
    spaces?: string[];
    tags?: string[];
    limit?: number;
  }): Promise<MemorySearchHit[]> {
    for (const space of query.spaces ?? []) {
      if (!this.canRead(space)) {
        throw new MemoryError('access_denied', `${this.agent} cannot read the space "${space}".`);
      }
    }
    // Agents never see quarantined entries.
    return this.store.search({
      ...query,
      spaceFilter: space => this.canRead(space),
      includeUnverified: false,
    });
  }

  async forget(id: string): Promise<boolean> {
    const entry = await this.store.get(id);
    if (!entry) return false;
    if (!this.canWrite(entry.space)) {
      throw new MemoryError(
        'access_denied',
        `${this.agent} cannot delete entries in the space "${entry.space}".`
      );
    }
    return this.store.forget(id);
  }
}

/** Formats search hits for insertion into an agent prompt. */
export function formatMemoryForPrompt(hits: MemorySearchHit[]): string {
  if (hits.length === 0) return '(no memories found)';
  return hits
    .map(({ entry: e }) => {
      const flag = e.origin === 'external' ? ' [from external content]' : '';
      return `- [${e.space}] ${e.text} (id ${e.id}, by ${e.author}, ${e.updatedAt.slice(0, 10)})${flag}`;
    })
    .join('\n');
}
