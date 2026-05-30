/**
 * Phase 0 conformance harness — DETERMINISM + REPLAY.
 *
 * Proves the property the entire ML stack depends on:
 *   same seed  =>  bit-identical game-state trajectory (incl. hidden deck order)
 *   diff seed  =>  different trajectory (randomness is actually seed-driven)
 *
 * Drives full automated games via the engine's own AI strategy, which exercises
 * the pre-game mulligan shuffle (the Phase 0 patch site) and in-game card-effect
 * shuffles. We compare a CANONICAL game-state fingerprint that excludes envelope
 * metadata (command IDs / timestamps), since those are host-clock/ID concerns
 * handled by kernel-host-shims at the WASM boundary, not game logic.
 */
import { describe, expect, it } from "bun:test";
import { createPlayerId } from "#core";
import {
  LorcanaMultiplayerTestEngine,
  CANONICAL_PLAYER_ONE,
  CANONICAL_PLAYER_TWO,
} from "./testing";
import { bestDeckAwareOracleLoreRaceAutomatedActionStrategy as STRATEGY } from "./automation";

const P1 = createPlayerId(CANONICAL_PLAYER_ONE);
const P2 = createPlayerId(CANONICAL_PLAYER_TWO);

type AnyState = any;

/** Canonical, replay-relevant fingerprint of authoritative state. */
function canonical(state: AnyState): string {
  const ctx = state.ctx ?? {};
  const zones = ctx.zones?.private?.zoneCards ?? {};
  // zone contents IN ORDER (deck order is the thing the shuffle patch governs)
  const zoneSnapshot: Record<string, string[]> = {};
  for (const key of Object.keys(zones).sort()) {
    zoneSnapshot[key] = [...(zones[key] ?? [])].map(String);
  }
  const G = state.G ?? {};
  return JSON.stringify({
    zones: zoneSnapshot,
    lore: G.lore ?? G.players ?? null,
    turn: ctx.status?.turn ?? null,
    phase: ctx.status?.phase ?? null,
    step: ctx.status?.step ?? null,
    priority: ctx.priority?.holder ?? null,
    seed: ctx.random?.seed ?? null,
    draws: ctx.random?.draws ?? null,
  });
}

function isOver(state: AnyState): boolean {
  const s = state.ctx?.status ?? {};
  return Boolean(s.winner || s.gameOver || s.ended || s.phase === "gameOver" || s.phase === "ended");
}

/** Play a full automated game; return canonical final fingerprint + telemetry. */
function playGame(seed: string, stepCap = 45) {
  const engine = LorcanaMultiplayerTestEngine.createWithFixture(
    { hand: 7, deck: 60 },
    { hand: 7, deck: 60 },
    { skipPreGame: false, seed },
  );
  let steps = 0;
  let lastFp = canonical(engine.getAuthoritativeState());
  let stalls = 0;
  try {
    while (steps < stepCap) {
      const state = engine.getAuthoritativeState();
      if (isOver(state)) break;
      let res: any;
      try {
        res = engine.asServer().takeAutomatedActionForCurrentActor({ strategy: STRATEGY });
      } catch {
        break; // no legal automated action available => treat as terminal for harness
      }
      if (!res || res.finalResult?.success === false) {
        stalls += 1;
        if (stalls > 3) break;
      } else {
        stalls = 0;
      }
      const fp = canonical(engine.getAuthoritativeState());
      if (fp === lastFp) {
        stalls += 1;
        if (stalls > 3) break; // no state progress => terminal/stuck
      } else {
        lastFp = fp;
      }
      steps += 1;
    }
  } finally {
    // dispose if available
    (engine as any).dispose?.();
  }
  return { fp: lastFp, steps };
}

describe("Phase 0 — determinism conformance", () => {
  const seeds = ["alpha-7", "bravo-42"];

  it("same seed => bit-identical trajectory (two independent engine instances)", () => {
    for (const seed of seeds) {
      const a = playGame(seed);
      const b = playGame(seed);
      expect(a.steps).toBe(b.steps);
      expect(a.fp).toBe(b.fp);
    }
  }, 120000);

  it("different seeds => at least one divergent trajectory (seed actually drives RNG)", () => {
    const fps = seeds.map((s) => playGame(s).fp);
    const unique = new Set(fps);
    expect(unique.size).toBeGreaterThan(1);
  }, 120000);

  it("live full-mulligan deck order is seed-reproducible (alterHand move path)", () => {
    // The live alterHand move shuffles via the seeded framework RandomAPI, so this
    // proves the production mulligan path is reproducible. (The exported
    // performMulligan utility is covered separately in phase0-mulligan-helper.test.ts.)
    const postMulliganDeckOrder = (seed: string) => {
      const e = LorcanaMultiplayerTestEngine.createWithFixture(
        { hand: 7, deck: 60 },
        { hand: 7, deck: 60 },
        { skipPreGame: false, seed },
      );
      e.asLorcanaPlayerOne().chooseFirstPlayer(P1);
      const p1Hand = e.getCardInstanceIdsInZone("hand", P1);
      const p2Hand = e.getCardInstanceIdsInZone("hand", P2);
      const r1 = e.asLorcanaPlayerOne().mulligan(p1Hand); // full mulligan -> shuffle
      const r2 = e.asLorcanaPlayerTwo().mulligan(p2Hand); // full mulligan -> shuffle
      expect(r1.success).toBe(true);
      expect(r2.success).toBe(true);
      const deck = [
        ...e.getCardInstanceIdsInZone("deck", P1).map(String),
        "|",
        ...e.getCardInstanceIdsInZone("deck", P2).map(String),
      ].join(",");
      (e as any).dispose?.();
      return deck;
    };
    // Same seed, two independent engine instances => identical post-shuffle deck order.
    expect(postMulliganDeckOrder("mull-seed")).toBe(postMulliganDeckOrder("mull-seed"));
    // Sanity: a different seed produces a different shuffle.
    expect(postMulliganDeckOrder("mull-seed")).not.toBe(postMulliganDeckOrder("other-seed"));
  }, 60000);
});
