/**
 * Phase 0 — direct determinism test for the patched performMulligan helper
 * (zone-operations.ts performMulligan / the `?? Math.random` default).
 *
 * This exported utility is NOT on the live alterHand path (which already shuffles
 * via the seeded framework), but it is exported for tooling. Pre-patch it shuffled
 * with global Math.random, so two same-seed drafts diverged. Post-patch it wires
 * the per-draft seeded RandomAPI, so identical seed => identical post-mulligan deck.
 */
import { describe, expect, it } from "bun:test";
import { create } from "mutative";
import { createPlayerId } from "#core";
import { performMulligan } from "./core/runtime/zone-operations";
import { createInitialTCGCtx } from "./core/runtime/types";
import type { ZoneRuntimeDef } from "./core/runtime/types";

const zoneDefs: Record<string, ZoneRuntimeDef> = {
  deck: { id: "deck", name: "Deck", visibility: "secret", ordered: true, ownerScoped: true },
  hand: { id: "hand", name: "Hand", visibility: "private", ordered: false, ownerScoped: true },
  play: { id: "play", name: "Play", visibility: "public", ordered: true, ownerScoped: false },
  discard: { id: "discard", name: "Discard", visibility: "public", ordered: true, ownerScoped: false },
};

function freshState(seed: string) {
  const ctx = createInitialTCGCtx({ matchID: "m", gameID: "lorcana", rulesetHash: "r" }) as any;
  ctx.random = { seed, draws: 0 };
  ctx.zones.public.zoneSummaries = {
    deck: { revision: 0, count: 0 },
    hand: { revision: 0, count: 0 },
    play: { revision: 0, count: 0 },
    discard: { revision: 0, count: 0 },
  };
  ctx.zones.private.zoneCards = { deck: [], hand: [], play: [], discard: [] };
  const p1 = createPlayerId("p1");
  for (let i = 1; i <= 30; i++) {
    const id = `card-${i}`;
    ctx.zones.private.cardIndex[id] = { zoneKey: "deck", ownerID: p1, controllerID: p1 };
    if (i <= 7) {
      ctx.zones.private.zoneCards.hand.push(id);
      ctx.zones.private.cardIndex[id].zoneKey = "hand";
    } else {
      ctx.zones.private.zoneCards.deck.push(id);
    }
  }
  ctx.zones.public.zoneSummaries.hand.count = 7;
  ctx.zones.public.zoneSummaries.deck.count = 23;
  return { G: { turn: 1 } as any, ctx };
}

function deckOrderAfterMulligan(seed: string): string {
  const out = create(freshState(seed), (draft: any) => {
    performMulligan(draft, zoneDefs, "p1", "hand", "deck", 7, 1);
  });
  return (out.ctx.zones.private.zoneCards.deck as string[]).join(",");
}

describe("Phase 0 — performMulligan helper determinism (patched utility)", () => {
  it("same seed => identical post-mulligan deck order", () => {
    expect(deckOrderAfterMulligan("seed-A")).toBe(deckOrderAfterMulligan("seed-A"));
  });
  it("different seed => different post-mulligan deck order", () => {
    expect(deckOrderAfterMulligan("seed-A")).not.toBe(deckOrderAfterMulligan("seed-B"));
  });
});
