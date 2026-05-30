/**
 * kernel-host-shims.ts
 *
 * MUST be the first import of the trimmed WASM kernel entry (kernel-entry.ts).
 * Makes the Lorcana kernel deterministic and QuickJS-safe by replacing host
 * nondeterminism (clock, RNG-based IDs) with values driven by the search host,
 * and by polyfilling the few globals QuickJS lacks.
 *
 * Determinism contract: the host (search worker) calls `__hostTick(n)` before
 * each command so that clock/ID outputs are a pure function of (seed, tick).
 * This keeps replay, snapshot hashing, and transposition fingerprints stable.
 *
 * NOTE: this addresses ENVELOPE nondeterminism only. The two game-logic leaks
 * found in Phase 0 (move-cards-from-under-effect.ts, zone-operations.ts) are
 * fixed by source patches, not here — see the Phase 0 report §3.3.
 */

declare const globalThis: Record<string, unknown>;

// ---------------------------------------------------------------------------
// Deterministic logical clock + seeded counter, driven by the host.
// ---------------------------------------------------------------------------
let __seed = "lorcana";
let __tick = 0; // monotonically advanced by the host, once per command
let __counter = 0; // strictly-increasing within a tick

export function __hostSetSeed(seed: string): void {
  __seed = seed;
}
export function __hostTick(tick: number): void {
  __tick = tick;
  __counter = 0;
}

/** Deterministic replacement for Date.now() / new Date().getTime(). */
function deterministicNow(): number {
  // logical milliseconds: stable across runs, monotonic within a game
  return __tick * 1000 + __counter;
}

/** Deterministic, collision-free ID source (replaces `${Date.now()}-${Math.random()}`). */
export function deterministicId(prefix = "id"): string {
  __counter += 1;
  return `${prefix}-${__seed}-${__tick}-${__counter}`;
}

// ---------------------------------------------------------------------------
// Install shims onto the global object (QuickJS host).
// Guarded so this is a no-op under Node/Bun/browser during normal dev/test.
// ---------------------------------------------------------------------------
export function installKernelHostShims(): void {
  const g = globalThis;

  // --- clock ---
  if (typeof g.Date === "undefined" || (g.__forceDeterministicClock as boolean)) {
    const RealDate = (g.Date as DateConstructor) ?? undefined;
    const D = function (this: unknown, ...args: unknown[]) {
      if (args.length === 0) {
        // new Date()  -> deterministic instant
        return RealDate ? new RealDate(deterministicNow()) : ({ getTime: deterministicNow } as unknown);
      }
      // explicit-arg construction is deterministic already; pass through
      return RealDate ? new (RealDate as any)(...args) : ({} as unknown);
    } as unknown as DateConstructor;
    (D as unknown as { now: () => number }).now = deterministicNow;
    g.Date = D;
  } else {
    (g.Date as unknown as { now: () => number }).now = deterministicNow;
  }

  // --- performance.now (perf cache in state-scoped-value-cache.ts) ---
  if (typeof g.performance === "undefined") {
    g.performance = { now: () => deterministicNow() };
  } else {
    (g.performance as { now: () => number }).now = () => deterministicNow();
  }

  // --- timers: kernel uses setTimeout(fn,0)-style scheduling only ---
  if (typeof g.setTimeout === "undefined") {
    g.setTimeout = ((fn: () => void) => {
      Promise.resolve().then(fn);
      return 0 as unknown;
    }) as unknown;
    g.clearTimeout = (() => {}) as unknown;
    g.setInterval = (() => 0) as unknown;
    g.clearInterval = (() => {}) as unknown;
  }
  if (typeof g.queueMicrotask === "undefined") {
    g.queueMicrotask = ((fn: () => void) => {
      Promise.resolve().then(fn);
    }) as unknown;
  }

  // --- structuredClone (used in one non-hot path) ---
  if (typeof g.structuredClone === "undefined") {
    g.structuredClone = (<T>(v: T): T => JSON.parse(JSON.stringify(v))) as unknown;
  }

  // --- TextEncoder/Decoder + Buffer (serialization helpers) ---
  if (typeof g.TextEncoder === "undefined") {
    // minimal UTF-8 encoder; replace with a vetted polyfill in production
    g.TextEncoder = class {
      encode(s: string): Uint8Array {
        const out: number[] = [];
        for (const ch of s) {
          const c = ch.codePointAt(0)!;
          if (c < 0x80) out.push(c);
          else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
          else out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
        }
        return Uint8Array.from(out);
      }
    } as unknown;
  }

  // --- console -> no-op (logtape is separately routed to a null sink) ---
  if (typeof g.console === "undefined") {
    const noop = () => {};
    g.console = { log: noop, info: noop, warn: noop, error: noop, debug: noop } as unknown;
  }

  // --- process.env: kernel reads it via `globalThis as {process?...}` guards ---
  if (typeof g.process === "undefined") {
    g.process = { env: {} } as unknown;
  }
}

/**
 * Configure @logtape/logtape with a null sink so its 11 top-level imports
 * resolve and run without touching console/process. Call once at boot.
 * (Pseudo-wiring; match to the installed logtape major version.)
 *
 *   import { configure } from "@logtape/logtape";
 *   await configure({ sinks: { null: () => {} }, loggers: [{ category: [], sinks: ["null"], lowestLevel: "fatal" }] });
 */
