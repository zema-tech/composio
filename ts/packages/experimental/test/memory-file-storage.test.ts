import { describe, expect, it } from 'vitest';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createFileMemoryStorage } from '../src/memory-file-storage';

const withTempDir = async (run: (dir: string) => Promise<void>) => {
  const dir = await mkdtemp(join(tmpdir(), 'composio-memory-'));
  try {
    await run(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
};

describe('createFileMemoryStorage', () => {
  it('returns undefined before the first write and round-trips afterwards', async () => {
    await withTempDir(async dir => {
      const storage = createFileMemoryStorage(join(dir, 'nested', 'memory.json'));
      expect(await storage.read()).toBeUndefined();
      await storage.write('{"hello":"world"}');
      expect(await storage.read()).toBe('{"hello":"world"}');
    });
  });

  it('keeps read-modify-write cycles from two storages on the same file from losing updates', async () => {
    await withTempDir(async dir => {
      const file = join(dir, 'counter.json');
      const a = createFileMemoryStorage(file);
      const b = createFileMemoryStorage(file);
      const increment = (storage: typeof a) =>
        storage.withLock(async () => {
          const current = Number((await storage.read()) ?? '0');
          await storage.write(String(current + 1));
        });

      await Promise.all(Array.from({ length: 20 }, (_, i) => increment(i % 2 === 0 ? a : b)));

      expect(await a.read()).toBe('20');
      expect(
        (await readdir(dir)).filter(name => name.endsWith('.lock') || name.endsWith('.tmp'))
      ).toEqual([]);
    });
  });

  it('removes a stale lock left by a crashed process', async () => {
    await withTempDir(async dir => {
      const file = join(dir, 'memory.json');
      const storage = createFileMemoryStorage(file, { staleLockMs: 0 });
      await storage.write('seed');
      const { writeFile } = await import('node:fs/promises');
      await writeFile(`${file}.lock`, '');
      await new Promise(resolve => setTimeout(resolve, 5));

      await expect(storage.withLock(async () => 'ok')).resolves.toBe('ok');
    });
  });

  it('releases the lock when the callback throws', async () => {
    await withTempDir(async dir => {
      const storage = createFileMemoryStorage(join(dir, 'memory.json'));
      await expect(
        storage.withLock(async () => {
          throw new Error('boom');
        })
      ).rejects.toThrow('boom');
      await expect(storage.withLock(async () => 'again')).resolves.toBe('again');
    });
  });
});
