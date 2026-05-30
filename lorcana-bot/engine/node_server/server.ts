/**
 * Phase 1A bridge — Node/Bun server hosting the headless Lorcana kernel.
 *
 * Transport: newline-delimited JSON over stdin/stdout (one request per line,
 * one response per line). Coarse-grained — one round-trip per *game step*, not
 * per MCTS node. For Phase 1 self-play / bootstrap this is the recommended
 * Option B from CLAUDE.md; the in-process WASM path (Option A) can be promoted
 * later behind the same Python interface (bridge.py).
 *
 * Action model: we expose the engine's automation candidate enumeration as the
 * legal-action set. Each `AutomatedActionCandidate` carries a `family`
 * (-> policy category), source/target card ids (-> policy source/target), and a
 * stable key. Executing a *chosen* action is done by handing the planner a
 * one-shot strategy that orders the chosen candidate first (the planner executes
 * the first candidate that succeeds). A synthetic "passTurn" action maps to the
 * planner's pass fallback (a strategy returning zero candidates).
 *
 * Run from the simulator workspace root so the engine package resolves:
 *   bun run <this file>
 */

import {
  LorcanaMultiplayerTestEngine,
  CANONICAL_PLAYER_ONE,
  CANONICAL_PLAYER_TWO,
} from "../../../lorcana-simulator/packages/lorcana/lorcana-engine/src/testing/index.ts";
import {
  bestDeckAwareOracleLoreRaceAutomatedActionStrategy,
  bestDeckAwareLoreRaceAutomatedActionStrategy,
  deckAwareLoreRaceAutomatedActionStrategy,
} from "../../../lorcana-simulator/packages/lorcana/lorcana-engine/src/automation/index.ts";
import type {
  AutomatedActionCandidate,
  AutomatedActionStrategy,
  AutomatedActionCandidateSummary,
} from "../../../lorcana-simulator/packages/lorcana/lorcana-engine/src/automation/index.ts";

type AnyEngine = any;

const STRATEGIES: Record<string, AutomatedActionStrategy> = {
  best: bestDeckAwareOracleLoreRaceAutomatedActionStrategy,
  bestFair: bestDeckAwareLoreRaceAutomatedActionStrategy,
  deckAware: deckAwareLoreRaceAutomatedActionStrategy,
};

const FORCED_FAMILIES = new Set([
  "chooseWhoGoesFirst",
  "alterHand",
  "resolveBag",
  "resolveEffect",
]);

const PASS_KEY = "passTurn";

/** Exact replica of the engine's internal getCandidateKey (planner.ts). */
function candidateKey(candidate: AutomatedActionCandidate): string {
  switch (candidate.family) {
    case "chooseWhoGoesFirst":
      return `chooseWhoGoesFirst:${candidate.firstPlayerId}`;
    case "alterHand":
      return `alterHand:${candidate.plan}:${candidate.cardsToMulligan.join(",")}`;
    case "resolveBag":
      return `resolveBag:${candidate.bagId}:${candidate.choiceIndex ?? ""}:${candidate.namedCard ?? ""}:${candidate.resolveOptional ?? ""}:${candidate.targets?.join(",") ?? ""}:${candidate.destinations?.map((d) => `${d.zone}:${d.cards.join(",")}`).join("|") ?? ""}`;
    case "resolveEffect":
      return `resolveEffect:${candidate.effectId}:${candidate.choiceIndex ?? ""}:${candidate.namedCard ?? ""}:${candidate.resolveOptional ?? ""}:${candidate.targets?.join(",") ?? ""}:${candidate.destinations?.map((d) => `${d.zone}:${d.cards.join(",")}`).join("|") ?? ""}`;
    case "putCardIntoInkwell":
      return `putCardIntoInkwell:${candidate.cardId}`;
    case "playCard":
      return `playCard:${candidate.cardId}:${typeof candidate.cost === "object" ? JSON.stringify(candidate.cost) : candidate.cost}:${candidate.choiceIndex ?? ""}:${candidate.resolveOptional ?? ""}:${candidate.targets?.join(",") ?? ""}`;
    case "activateAbility":
      return `activateAbility:${candidate.cardId}:${candidate.abilityIndex}:${candidate.choiceIndex ?? ""}:${candidate.targets?.join(",") ?? ""}:${candidate.costs?.banishCharacters?.join(",") ?? ""}:${candidate.costs?.banishItems?.join(",") ?? ""}:${candidate.costs?.discardCards?.join(",") ?? ""}:${candidate.costs?.exertCharacters?.join(",") ?? ""}`;
    case "quest":
      return `quest:${candidate.cardId}`;
    case "challenge":
      return `challenge:${candidate.attackerId}:${candidate.defenderId}`;
    case "moveCharacterToLocation":
      return `moveCharacterToLocation:${candidate.characterId}:${candidate.locationId}`;
    default:
      return JSON.stringify(candidate);
  }
}

/** Pull the (source, target) card ids out of a candidate for the policy head. */
function candidateRefs(c: AutomatedActionCandidate): { src: string | null; tgt: string | null } {
  switch (c.family) {
    case "putCardIntoInkwell":
      return { src: c.cardId, tgt: null };
    case "playCard":
      return { src: c.cardId, tgt: c.targets?.[0] ?? null };
    case "activateAbility":
      return { src: c.cardId, tgt: c.targets?.[0] ?? null };
    case "quest":
      return { src: c.cardId, tgt: null };
    case "challenge":
      return { src: c.attackerId, tgt: c.defenderId };
    case "moveCharacterToLocation":
      return { src: c.characterId, tgt: c.locationId };
    case "resolveBag":
      return { src: null, tgt: c.targets?.[0] ?? null };
    case "resolveEffect":
      return { src: null, tgt: c.targets?.[0] ?? null };
    default:
      return { src: null, tgt: null };
  }
}

/** Strategy that orders the chosen candidate first; rest follow (keeps progress). */
function chooseKeyStrategy(targetKey: string): AutomatedActionStrategy {
  return {
    name: `choose:${targetKey}`,
    summarizeCandidates(_ctx, candidates): AutomatedActionCandidateSummary[] {
      const ordered = [...candidates].sort((a, b) => {
        const ak = candidateKey(a) === targetKey ? 0 : 1;
        const bk = candidateKey(b) === targetKey ? 0 : 1;
        return ak - bk;
      });
      return ordered.map((candidate) => ({
        candidate,
        family: candidate.family,
        heuristics: [],
        stableKey: candidateKey(candidate),
      }));
    },
  };
}

/** Strategy that yields no candidates -> planner falls back to passTurn. */
const PASS_STRATEGY: AutomatedActionStrategy = {
  name: "pass",
  summarizeCandidates() {
    return [];
  },
};

function viewFor(actorId: string | undefined): "playerOne" | "playerTwo" {
  return actorId === CANONICAL_PLAYER_TWO ? "playerTwo" : "playerOne";
}

class Session {
  engine: AnyEngine;
  seed: string;
  steps = 0;
  private snapshots = new Map<number, any>();
  private snapCounter = 0;

  constructor(seed: string) {
    this.seed = seed;
    this.engine = LorcanaMultiplayerTestEngine.createWithFixture(
      { hand: 7, deck: 60 },
      { hand: 7, deck: 60 },
      { skipPreGame: false, seed },
    );
  }

  /** Clone the authoritative state and stash it; return a handle id. */
  snapshot(): number {
    const id = ++this.snapCounter;
    const state = structuredClone(this.engine.getAuthoritativeState());
    this.snapshots.set(id, state);
    return id;
  }

  /** Restore a previously-taken snapshot (clone so it stays reusable). */
  restore(id: number): boolean {
    const stored = this.snapshots.get(id);
    if (!stored) return false;
    this.engine.loadState(structuredClone(stored));
    return true;
  }

  dropSnapshot(id: number): void {
    this.snapshots.delete(id);
  }

  private currentActor(): string | undefined {
    const enumr = this.engine.asServer().enumerateAutomatedActionsForCurrentActor();
    return enumr.actorId;
  }

  observe() {
    const server = this.engine.asServer();
    const enumr = server.enumerateAutomatedActionsForCurrentActor();
    const actorId: string | undefined = enumr.actorId;
    const view = viewFor(actorId);
    const board: any = this.engine.getBoard(view);

    const selfId = actorId ?? CANONICAL_PLAYER_ONE;
    const oppId = selfId === CANONICAL_PLAYER_ONE ? CANONICAL_PLAYER_TWO : CANONICAL_PLAYER_ONE;

    const families = new Set(enumr.candidates.map((c: AutomatedActionCandidate) => c.family));
    const forced = [...families].some((f) => FORCED_FAMILIES.has(f as string));

    const legal: any[] = enumr.candidates.map((c: AutomatedActionCandidate, idx: number) => {
      const refs = candidateRefs(c);
      return { idx, stableKey: candidateKey(c), family: c.family, src: refs.src, tgt: refs.tgt };
    });
    if (!forced && board.status === "playing") {
      legal.push({ idx: legal.length, stableKey: PASS_KEY, family: "passTurn", src: null, tgt: null });
    }

    const mkPlayer = (pid: string) => {
      const pb = board.players[pid] ?? {};
      return {
        lore: pb.lore ?? 0,
        handCount: pb.handCount ?? 0,
        deckCount: pb.deckCount ?? 0,
        inkwell: (pb.inkwell ?? []).length,
        discard: (pb.discard ?? []).length,
        play: (pb.play ?? []).length,
      };
    };

    const cards: any[] = [];
    for (const c of Object.values(board.cards) as any[]) {
      cards.push({
        id: c.id,
        owner: c.ownerId === selfId ? 0 : 1,
        zone: c.zone,
        cardType: c.cardType ?? "unknown",
        cost: c.playCost ?? c.cost ?? 0,
        strength: c.strength ?? 0,
        willpower: c.willpower ?? 0,
        lore: c.lore ?? 0,
        damage: c.damage ?? 0,
        exerted: Boolean(c.exerted),
        drying: Boolean(c.drying),
        ready: !c.exerted,
        keywords: c.keywords ?? [],
        definitionId: c.definitionId ?? null,
        hidden: Boolean(c.hidden),
      });
    }

    const done = board.status === "finished";
    return {
      turn: board.turnNumber ?? 0,
      phase: board.phase ?? null,
      step: board.step ?? null,
      status: board.status,
      done,
      winner: board.winner ?? null,
      actor: actorId ?? null,
      self: selfId,
      opp: oppId,
      players: { self: mkPlayer(selfId), opp: mkPlayer(oppId) },
      cards,
      legal,
      forced,
    };
  }

  step(stableKey: string) {
    const server = this.engine.asServer();
    let executed: string | null = null;
    let success = false;
    if (stableKey === PASS_KEY) {
      const res = server.takeAutomatedActionForCurrentActor({ strategy: PASS_STRATEGY });
      success = Boolean(res.finalResult?.success) || res.fallbackTaken === "passTurn";
      executed = res.fallbackTaken === "passTurn" ? PASS_KEY : (res.selectedCandidate ? candidateKey(res.selectedCandidate) : null);
    } else {
      const res = server.takeAutomatedActionForCurrentActor({ strategy: chooseKeyStrategy(stableKey) });
      success = Boolean(res.finalResult?.success);
      executed = res.selectedCandidate ? candidateKey(res.selectedCandidate) : (res.fallbackTaken ?? null);
    }
    this.steps += 1;
    const obs = this.observe();
    return { obs, executed, requested: stableKey, matched: executed === stableKey, success };
  }

  stepAuto(strategyName: string) {
    const strategy = STRATEGIES[strategyName] ?? STRATEGIES.best;
    const server = this.engine.asServer();
    const res = server.takeAutomatedActionForCurrentActor({ strategy });
    const executed = res.selectedCandidate ? candidateKey(res.selectedCandidate) : (res.fallbackTaken ?? null);
    const family = res.selectedCandidate?.family ?? (res.fallbackTaken ?? null);
    this.steps += 1;
    const obs = this.observe();
    return { obs, executed, family, success: Boolean(res.finalResult?.success) };
  }
}

// ---- stdin/stdout JSON-line loop --------------------------------------------

let session: Session | null = null;

function handle(req: any): any {
  switch (req.op) {
    case "ping":
      return { ok: true, pong: true };
    case "reset": {
      session = new Session(String(req.seed ?? "seed-0"));
      return { ok: true, obs: session.observe() };
    }
    case "observe": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, obs: session.observe() };
    }
    case "step": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, ...session.step(String(req.stableKey)) };
    }
    case "step_auto": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, ...session.stepAuto(String(req.strategy ?? "best")) };
    }
    case "snapshot": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, id: session.snapshot() };
    }
    case "restore": {
      if (!session) return { ok: false, error: "no session" };
      const ok = session.restore(Number(req.id));
      if (!ok) return { ok: false, error: `no snapshot ${req.id}` };
      return { ok: true, obs: session.observe() };
    }
    case "drop_snapshot": {
      if (!session) return { ok: false, error: "no session" };
      session.dropSnapshot(Number(req.id));
      return { ok: true };
    }
    case "close":
      return { ok: true, bye: true };
    default:
      return { ok: false, error: `unknown op ${req.op}` };
  }
}

const decoder = new TextDecoder();
let buffer = "";

async function main() {
  process.stdout.write(JSON.stringify({ ok: true, ready: true }) + "\n");
  for await (const chunk of Bun.stdin.stream()) {
    buffer += decoder.decode(chunk);
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      let resp: any;
      try {
        const req = JSON.parse(line);
        resp = handle(req);
        if (req.op === "close") {
          process.stdout.write(JSON.stringify(resp) + "\n");
          return;
        }
      } catch (err: any) {
        resp = { ok: false, error: String(err?.stack ?? err?.message ?? err) };
      }
      process.stdout.write(JSON.stringify(resp) + "\n");
    }
  }
}

main();
