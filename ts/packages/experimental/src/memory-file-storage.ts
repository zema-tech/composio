/**
 * File-backed storage for the shared memory in `@composio/core/experimental` (`MemoryStore`).
 *
 * Several processes can share one file: every mutation takes a lock file, and writes go through a
 * temporary file and an atomic rename. Node only; `@composio/core` stays platform-neutral.
 *
 * @example
 * ```typescript
 * import { MemoryStore } from '@composio/core/experimental';
 * import { createFileMemoryStorage } from '@composio/experimental';
 *
 * const store = new MemoryStore({ storage: createFileMemoryStorage('./.agent-memory.json') });
 * ```
 */
import { mkdir, open, readFile, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

/** Structurally compatible with `MemoryStorage` from `@composio/core/experimental`. */
export interface FileMemoryStorage {
  read(): Promise<string | undefined>;
  write(serialized: string): Promise<void>;
  withLock<T>(fn: () => Promise<T>): Promise<T>;
}

export interface FileMemoryStorageOptions {
  /** How long to wait for the lock before failing. Defaults to 3000 ms. */
  lockTimeoutMs?: number;
  /** A lock file older than this is treated as orphaned and removed. Defaults to 10000 ms. */
  staleLockMs?: number;
}

const isMissing = (error: unknown) =>
  error instanceof Error && 'code' in error && error.code === 'ENOENT';
const isTaken = (error: unknown) =>
  error instanceof Error && 'code' in error && error.code === 'EEXIST';

export function createFileMemoryStorage(
  filePath: string,
  options: FileMemoryStorageOptions = {}
): FileMemoryStorage {
  const lockTimeoutMs = options.lockTimeoutMs ?? 3000;
  const staleLockMs = options.staleLockMs ?? 10_000;
  const lockPath = `${filePath}.lock`;

  const acquire = async (): Promise<() => Promise<void>> => {
    await mkdir(dirname(filePath), { recursive: true });
    const deadline = Date.now() + lockTimeoutMs;
    for (;;) {
      try {
        const handle = await open(lockPath, 'wx');
        await handle.close();
        return async () => {
          await unlink(lockPath).catch(() => undefined);
        };
      } catch (error) {
        if (!isTaken(error)) throw error;
        const info = await stat(lockPath).catch(() => undefined);
        if (info && Date.now() - info.mtimeMs > staleLockMs) {
          await unlink(lockPath).catch(() => undefined);
          continue;
        }
        if (Date.now() > deadline) {
          throw new Error(`Could not acquire the memory lock at ${lockPath}.`);
        }
        await new Promise(resolve => setTimeout(resolve, 25 + Math.random() * 50));
      }
    }
  };

  return {
    async read() {
      try {
        return await readFile(filePath, 'utf8');
      } catch (error) {
        if (isMissing(error)) return undefined;
        throw error;
      }
    },
    async write(serialized) {
      await mkdir(dirname(filePath), { recursive: true });
      const temporary = `${filePath}.${process.pid}.${globalThis.crypto.randomUUID()}.tmp`;
      await writeFile(temporary, serialized, 'utf8');
      await rename(temporary, filePath);
    },
    async withLock(fn) {
      const release = await acquire();
      try {
        return await fn();
      } finally {
        await release();
      }
    },
  };
}
