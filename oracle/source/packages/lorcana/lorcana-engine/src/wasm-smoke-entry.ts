// WASM smoke: runs REAL patched engine determinism code (performMulligan) under
// QuickJS-WASM. Proves the kernel logic executes as WASM and stays deterministic.
// Minimal QuickJS host shims (QuickJS already has Date/Math/JSON/Proxy).
const g: any = globalThis as any;
if (typeof g.structuredClone === "undefined") g.structuredClone = (v: any) => JSON.parse(JSON.stringify(v));
if (typeof g.performance === "undefined") g.performance = { now: () => 0 };
if (typeof g.setTimeout === "undefined") g.setTimeout = (fn: any) => { Promise.resolve().then(fn); return 0; };
if (typeof g.queueMicrotask === "undefined") g.queueMicrotask = (fn: any) => { Promise.resolve().then(fn); };

import { create } from "mutative";
import { createPlayerId } from "#core";
import { performMulligan } from "./core/runtime/zone-operations";
import { createInitialTCGCtx } from "./core/runtime/types";

function freshState(seed: string) {
  const ctx: any = createInitialTCGCtx({ matchID: "m", gameID: "lorcana", rulesetHash: "r" });
  ctx.random = { seed, draws: 0 };
  ctx.zones.public.zoneSummaries = {
    deck: { revision: 0, count: 0 }, hand: { revision: 0, count: 0 },
    play: { revision: 0, count: 0 }, discard: { revision: 0, count: 0 },
  };
  ctx.zones.private.zoneCards = { deck: [], hand: [], play: [], discard: [] };
  const p1 = createPlayerId("p1");
  for (let i = 1; i <= 30; i++) {
    const id = `card-${i}`;
    ctx.zones.private.cardIndex[id] = { zoneKey: "deck", ownerID: p1, controllerID: p1 };
    if (i <= 7) { ctx.zones.private.zoneCards.hand.push(id); ctx.zones.private.cardIndex[id].zoneKey = "hand"; }
    else { ctx.zones.private.zoneCards.deck.push(id); }
  }
  ctx.zones.public.zoneSummaries.hand.count = 7;
  ctx.zones.public.zoneSummaries.deck.count = 23;
  return { G: { turn: 1 } as any, ctx };
}
const zoneDefs: any = {
  deck: { id: "deck", name: "Deck", visibility: "secret", ordered: true, ownerScoped: true },
  hand: { id: "hand", name: "Hand", visibility: "private", ordered: false, ownerScoped: true },
  play: { id: "play", name: "Play", visibility: "public", ordered: true, ownerScoped: false },
  discard: { id: "discard", name: "Discard", visibility: "public", ordered: true, ownerScoped: false },
};
function deckOrder(seed: string): string {
  const out: any = create(freshState(seed), (draft: any) => { performMulligan(draft, zoneDefs, "p1", "hand", "deck", 7, 1); });
  return (out.ctx.zones.private.zoneCards.deck as string[]).join(",");
}
try {
  const a = deckOrder("seed-A"), b = deckOrder("seed-A"), c = deckOrder("seed-B");
  console.log("WASM_RUN_OK");
  console.log("same_seed_identical=" + String(a === b));
  console.log("diff_seed_differs=" + String(a !== c));
  console.log("deck_seedA=" + a);
} catch (e: any) {
  console.log("WASM_RUN_ERROR: " + (e && e.message ? e.message : String(e)));
}
