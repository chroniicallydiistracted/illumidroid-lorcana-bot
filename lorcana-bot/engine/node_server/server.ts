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
  defaultLoreRaceAutomatedActionStrategy,
  boardControlLoreRaceAutomatedActionStrategy,
  aggressiveBoardControlLoreRaceAutomatedActionStrategy,
} from "../../../lorcana-simulator/packages/lorcana/lorcana-engine/src/automation/index.ts";
import type {
  AutomatedActionCandidate,
  AutomatedActionStrategy,
  AutomatedActionCandidateSummary,
} from "../../../lorcana-simulator/packages/lorcana/lorcana-engine/src/automation/index.ts";
import { readdirSync, readFileSync } from "node:fs";
import { all001Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/001/index.ts";
import { all002Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/002/index.ts";
import { all003Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/003/index.ts";
import { all004Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/004/index.ts";
import { all005Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/005/index.ts";
import { all006Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/006/index.ts";
import { all007Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/007/index.ts";
import { all008Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/008/index.ts";
import { all009Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/009/index.ts";
import { all010Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/010/index.ts";
import { all011Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/011/index.ts";
import { all012Cards } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/cards/012/index.ts";
import { resolveLorcanaDeckListTextFromPool } from "../../../lorcana-simulator/packages/lorcana/lorcana-cards/src/utils/deck-list-resolver.ts";

type AnyEngine = any;

// --- real tournament decks (resolved against the full card catalog) ----------
const CARD_POOL = [
  all001Cards, all002Cards, all003Cards, all004Cards, all005Cards, all006Cards,
  all007Cards, all008Cards, all009Cards, all010Cards, all011Cards, all012Cards,
].flat().filter((c: any) => c?.name != null);

const DECKS_DIR = `${import.meta.dir}/../../decks`;

type DeckEntry = { id: string; name: string; text: string; size: number };
const DECK_REGISTRY: DeckEntry[] = [];
const DECK_BY_ID: Record<string, DeckEntry> = {};
const _resolvedCache: Record<string, any[]> = {};

function loadDecks(): void {
  let files: string[] = [];
  try {
    files = readdirSync(DECKS_DIR).filter((f: string) => f.endsWith(".json")).sort();
  } catch {
    return; // no decks dir -> placeholder fallback remains available
  }
  for (const f of files) {
    try {
      const j = JSON.parse(readFileSync(`${DECKS_DIR}/${f}`, "utf-8"));
      if (!j?.cards) continue;
      const id = f.replace(/\.json$/, "");
      const entry: DeckEntry = { id, name: j.name ?? id, text: j.cards, size: 0 };
      DECK_REGISTRY.push(entry);
      DECK_BY_ID[id] = entry;
    } catch {
      // skip malformed deck files
    }
  }
}
loadDecks();

function resolveDeck(id: string): any[] {
  if (_resolvedCache[id]) return _resolvedCache[id];
  const entry = DECK_BY_ID[id];
  if (!entry) throw new Error(`unknown deck "${id}"`);
  const { cards } = resolveLorcanaDeckListTextFromPool(entry.text, CARD_POOL) as any;
  entry.size = cards.length;
  _resolvedCache[id] = cards;
  return cards;
}

/** Deterministic deck pick from a seed string (stable across processes). */
function pickDeck(seed: string, salt: string): string {
  let h = 2166136261 >>> 0;
  const s = seed + ":" + salt;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return DECK_REGISTRY[h % DECK_REGISTRY.length].id;
}

// The planner DEFAULTS an unset informationPolicy to "oracle" (full-deck
// visibility into the opponent's hidden cards). Oracle play must NEVER generate
// training targets — it would teach the net to clone decisions made with info
// it can't see. `fair()` forces public-zones-only knowledge; the registry below
// is fair-by-default and isolates oracle to explicit, eval-only use.
function fair(base: AutomatedActionStrategy, name?: string): AutomatedActionStrategy {
  return { ...base, name: name ?? `fair:${base.name}`, informationPolicy: "fair" };
}

/** Effective policy a strategy resolves to (mirrors planner's `?? "oracle"`). */
function effectivePolicy(s: AutomatedActionStrategy): "fair" | "oracle" {
  return (s.informationPolicy ?? "oracle") as "fair" | "oracle";
}

const STRATEGIES: Record<string, AutomatedActionStrategy> = {
  // --- FAIR (training-safe): public-zones knowledge only ---
  best: bestDeckAwareLoreRaceAutomatedActionStrategy,             // already fair
  fairBest: bestDeckAwareLoreRaceAutomatedActionStrategy,
  fairDefault: fair(defaultLoreRaceAutomatedActionStrategy),
  fairControl: fair(boardControlLoreRaceAutomatedActionStrategy),
  fairAggro: fair(aggressiveBoardControlLoreRaceAutomatedActionStrategy),
  deckAware: fair(deckAwareLoreRaceAutomatedActionStrategy),      // forced fair
  // --- ORACLE (EVAL ONLY — never use for training data/targets) ---
  oracle: bestDeckAwareOracleLoreRaceAutomatedActionStrategy,
  oracleDeckAware: deckAwareLoreRaceAutomatedActionStrategy,
};

// Widen the automation enumeration beyond its defaults (targetPool 8,
// targetCombinationsPerFamily 16, choiceIndices 8, singerCombinations 16) so the
// legal-action set the bot learns/searches over covers far more of the true
// action space. Used for BOTH enumeration (the legal list) and execution (so a
// chosen key is always re-enumerable). Full getAvailableMoves grammar is a larger
// project; this closes most of the gap cheaply. Override via LORCANA_SEARCH_CAPS.
const SEARCH_CAPS = {
  targetPool: 24,
  targetCombinationsPerFamily: 48,
  choiceIndices: 16,
  singerCombinations: 32,
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

/** The FULL candidate descriptor the neural policy needs to rank actions.
 *  The stable key already encodes all of this (planner.ts) — the old policy only
 *  saw {src,tgt,tgt2,choice}, so candidates sharing family/src/first-target but
 *  differing in singer / shift target / 2nd-target / named card / discard-exert-
 *  banish cost / destination / optional / choice-index collapsed to identical
 *  features. This surfaces every distinguishing field (legacy tgt/tgt2 kept for
 *  back-compat with the current head during migration). */
type CandidateDescriptor = {
  src: string | null;
  targets: string[];          // all target card ids (first == tgt)
  singers: string[];          // sing / sing-together exert-cost characters
  shiftTarget: string | null; // the character being shifted onto
  banish: string[];           // banish-cost cards (characters + items)
  discard: string[];          // discard-cost cards (incl. shift discard)
  exert: string[];            // exert-cost characters
  costType: string;           // standard | free | shift | sing | none
  inkCost: number;            // numeric amount if known, else -1
  choice: number;             // choiceIndex (modal "or"), else -1
  resolveOptional: boolean;   // optional "may" resolution
  namedCard: string | null;   // named-card choice
  abilityIndex: number;       // activated-ability index, else -1
  destinationZones: string[]; // scry/destination zones
  tgt: string | null;         // legacy: targets[0]
  tgt2: string | null;        // legacy: targets[1]
};

function asArr(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function emptyDescriptor(): CandidateDescriptor {
  return { src: null, targets: [], singers: [], shiftTarget: null, banish: [], discard: [],
           exert: [], costType: "none", inkCost: -1, choice: -1, resolveOptional: false,
           namedCard: null, abilityIndex: -1, destinationZones: [], tgt: null, tgt2: null };
}

function candidateFeatures(c: AutomatedActionCandidate): CandidateDescriptor {
  const any = c as any;
  const costs = any.costs ?? {};
  const ci = any.choiceIndex;
  const d = emptyDescriptor();
  d.banish = [...asArr(costs.banishCharacters), ...asArr(costs.banishItems)];
  d.discard = asArr(costs.discardCards);
  d.exert = asArr(costs.exertCharacters);
  d.choice = typeof ci === "number" ? ci : -1;
  d.resolveOptional = any.resolveOptional === true;
  d.namedCard = typeof any.namedCard === "string" ? any.namedCard : null;
  d.abilityIndex = typeof any.abilityIndex === "number" ? any.abilityIndex : -1;
  d.destinationZones = Array.isArray(any.destinations)
    ? any.destinations.map((x: any) => String(x.zone)) : [];

  let targets = asArr(any.targets);
  switch (c.family) {
    case "putCardIntoInkwell":
    case "quest":
    case "activateAbility":
      d.src = c.cardId;
      break;
    case "playCard": {
      d.src = c.cardId;
      const cost = any.cost;
      if (typeof cost === "string") {
        d.costType = cost;
      } else if (cost && typeof cost === "object") {
        d.costType = String(cost.cost ?? "standard");
        if (typeof cost.amount === "number") d.inkCost = cost.amount;
        if (cost.shiftTarget) d.shiftTarget = String(cost.shiftTarget);
        if (cost.singer) d.singers = [String(cost.singer)];
        if (Array.isArray(cost.singerIds)) d.singers = asArr(cost.singerIds);
        if (Array.isArray(cost.discardCards)) d.discard = [...d.discard, ...asArr(cost.discardCards)];
        if (Array.isArray(cost.targets)) targets = [...targets, ...asArr(cost.targets)];
      }
      break;
    }
    case "challenge":
      d.src = c.attackerId;
      targets = [c.defenderId, ...targets];
      break;
    case "moveCharacterToLocation":
      d.src = c.characterId;
      targets = [c.locationId, ...targets];
      break;
    default:
      break;  // resolveBag / resolveEffect / setup: targets already pulled
  }
  if (d.singers.length === 0) d.singers = asArr(any.cardPlayed?.singerIds);
  d.targets = targets.filter((x) => typeof x === "string");
  d.tgt = d.targets[0] ?? null;
  d.tgt2 = d.targets[1] ?? null;
  return d;
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

/** Shallow-copy top level + G + ctx so loadState's in-place mutation
 *  (invalidateStaticEffects bumps G.staticEffectsVersion) can't corrupt the
 *  canonical immutable snapshot. Cheap (a few key spreads), not a deep clone. */
function shallowProtect(s: any): any {
  return { ...s, G: { ...s.G }, ctx: { ...s.ctx } };
}

class Session {
  engine: AnyEngine;
  seed: string;
  steps = 0;
  deckP1: string | null;
  deckP2: string | null;
  private snapshots = new Map<number, any>();
  private snapCounter = 0;

  constructor(seed: string, deckP1?: string | null, deckP2?: string | null) {
    this.seed = seed;
    this.deckP1 = deckP1 ?? null;
    this.deckP2 = deckP2 ?? null;
    if (this.deckP1 && this.deckP2) {
      // real tournament decks
      this.engine = LorcanaMultiplayerTestEngine.createWithFixture(
        { deck: resolveDeck(this.deckP1) },
        { deck: resolveDeck(this.deckP2) },
        { skipPreGame: false, seed },
      );
    } else {
      // placeholder fallback (no decks available / explicitly requested)
      this.engine = LorcanaMultiplayerTestEngine.createWithFixture(
        { hand: 7, deck: 60 },
        { hand: 7, deck: 60 },
        { skipPreGame: false, seed },
      );
    }
  }

  /** Opponent's true hidden zones (authoritative) — belief-training targets.
   *  Returned for the *current actor's* opponent. Used ONLY for supervision /
   *  determinization on the Python side, never fed into the policy/value trunk. */
  private oppHidden(selfId: string) {
    const oppId = selfId === CANONICAL_PLAYER_ONE ? CANONICAL_PLAYER_TWO : CANONICAL_PLAYER_ONE;
    const st: any = this.engine.getAuthoritativeState();
    const zc = st.ctx.zones.private.zoneCards;
    const grab = (base: string) =>
      (zc[`${base}:${oppId}`] ?? []).map((id: string) => ({
        id,
        def: this.engine.getCardDefinitionId?.(id) ?? null,
      }));
    return { hand: grab("hand"), deck: grab("deck"), inkwell: grab("inkwell") };
  }

  /** Repartition the opponent's hidden cards into hand/deck per a belief sample,
   *  then loadState — produces a determinized world the search runs on. Operates
   *  on the *current* authoritative state (caller restores the true root first). */
  determinize(selfId: string, handInstanceIds: string[], seed?: string): boolean {
    const oppId = selfId === CANONICAL_PLAYER_ONE ? CANONICAL_PLAYER_TWO : CANONICAL_PLAYER_ONE;
    const clone: any = structuredClone(this.engine.getAuthoritativeState());
    const zc = clone.ctx.zones.private.zoneCards;
    const sum = clone.ctx.zones.public.zoneSummaries;
    const handKey = `hand:${oppId}`;
    const deckKey = `deck:${oppId}`;
    const pool: string[] = [...(zc[handKey] ?? []), ...(zc[deckKey] ?? [])];
    const wantHand = handInstanceIds.filter((id) => pool.includes(id));
    const handSet = new Set(wantHand);
    let rest = pool.filter((id) => !handSet.has(id));
    if (seed) {
      // deterministic shuffle of the residual deck order from the seed
      let h = 0;
      for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
      const rng = () => ((h = (h * 1103515245 + 12345) >>> 0) / 0xffffffff);
      for (let i = rest.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [rest[i], rest[j]] = [rest[j], rest[i]];
      }
    }
    zc[handKey] = wantHand;
    zc[deckKey] = rest;
    if (sum[handKey]) sum[handKey] = { ...sum[handKey], count: wantHand.length, revision: (sum[handKey].revision || 0) + 1 };
    if (sum[deckKey]) sum[deckKey] = { ...sum[deckKey], count: rest.length, revision: (sum[deckKey].revision || 0) + 1 };
    this.engine.loadState(clone);
    return true;
  }

  /** O(1) snapshot: the engine state is immutable (Mutative structural sharing),
   *  so we just hold the reference — no deep clone. Every transition reassigns a
   *  NEW state object, leaving snapshots intact. */
  snapshot(): number {
    const id = ++this.snapCounter;
    this.snapshots.set(id, this.engine.getAuthoritativeState());
    return id;
  }

  /** O(1) restore: re-point the runtime at the snapshot. We shallow-copy the top
   *  level + G + ctx so `loadState`'s in-place `invalidateStaticEffects`
   *  (G.staticEffectsVersion++) lands on the copy, never the canonical snapshot.
   *  The deep state is structurally shared and immutable — safe to alias. */
  restore(id: number): boolean {
    const stored = this.snapshots.get(id);
    if (!stored) return false;
    this.engine.loadState(shallowProtect(stored));
    return true;
  }

  dropSnapshot(id: number): void {
    this.snapshots.delete(id);
  }

  /** Execute one chosen action key from the current state (no observe). */
  private executeKey(key: string): boolean {
    const server = this.engine.asServer();
    if (key === PASS_KEY) {
      const res = server.takeAutomatedActionForCurrentActor({ strategy: PASS_STRATEGY, searchCaps: SEARCH_CAPS });
      return Boolean(res.finalResult?.success) || res.fallbackTaken === "passTurn";
    }
    const res = server.takeAutomatedActionForCurrentActor({ strategy: chooseKeyStrategy(key), searchCaps: SEARCH_CAPS });
    return Boolean(res.finalResult?.success);
  }

  /** Execute a batch of descent paths IN-PROCESS from a root snapshot, returning
   *  each path's leaf observation. One IPC replaces (paths × pathLen) per-step
   *  round-trips — the core of the in-process search (architecture doc §4.2). */
  runPaths(rootId: number, paths: string[][]): any[] {
    const out: any[] = [];
    for (const keys of paths) {
      this.restore(rootId);
      for (const key of keys) {
        if (this.observe0Done()) break;   // stop early if a path hit terminal
        this.executeKey(key);
      }
      out.push(this.observe());
    }
    return out;
  }

  private observe0Done(): boolean {
    const b: any = this.engine.getBoard("playerOne");
    return b.status === "finished";
  }

  private currentActor(): string | undefined {
    const enumr = this.engine.asServer().enumerateAutomatedActionsForCurrentActor({ searchCaps: SEARCH_CAPS });
    return enumr.actorId;
  }

  observe() {
    const server = this.engine.asServer();
    const enumr = server.enumerateAutomatedActionsForCurrentActor({ searchCaps: SEARCH_CAPS });
    const actorId: string | undefined = enumr.actorId;
    const view = viewFor(actorId);
    const board: any = this.engine.getBoard(view);

    const selfId = actorId ?? CANONICAL_PLAYER_ONE;
    const oppId = selfId === CANONICAL_PLAYER_ONE ? CANONICAL_PLAYER_TWO : CANONICAL_PLAYER_ONE;

    const families = new Set(enumr.candidates.map((c: AutomatedActionCandidate) => c.family));
    const forced = [...families].some((f) => FORCED_FAMILIES.has(f as string));

    const legal: any[] = enumr.candidates.map((c: AutomatedActionCandidate, idx: number) => {
      return { idx, stableKey: candidateKey(c), family: c.family, ...candidateFeatures(c) };
    });
    if (!forced && board.status === "playing") {
      legal.push({ idx: legal.length, stableKey: PASS_KEY, family: "passTurn",
                   ...emptyDescriptor() });
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
      hidden: this.oppHidden(selfId),
    };
  }

  step(stableKey: string) {
    const server = this.engine.asServer();
    let executed: string | null = null;
    let success = false;
    if (stableKey === PASS_KEY) {
      const res = server.takeAutomatedActionForCurrentActor({ strategy: PASS_STRATEGY, searchCaps: SEARCH_CAPS });
      success = Boolean(res.finalResult?.success) || res.fallbackTaken === "passTurn";
      executed = res.fallbackTaken === "passTurn" ? PASS_KEY : (res.selectedCandidate ? candidateKey(res.selectedCandidate) : null);
    } else {
      const res = server.takeAutomatedActionForCurrentActor({ strategy: chooseKeyStrategy(stableKey), searchCaps: SEARCH_CAPS });
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
    const res = server.takeAutomatedActionForCurrentActor({ strategy, searchCaps: SEARCH_CAPS });
    const executed = res.selectedCandidate ? candidateKey(res.selectedCandidate) : (res.fallbackTaken ?? null);
    const family = res.selectedCandidate?.family ?? (res.fallbackTaken ?? null);
    this.steps += 1;
    const obs = this.observe();
    // `policy` lets the Python side assert no oracle play taints training data.
    return { obs, executed, family, success: Boolean(res.finalResult?.success),
             policy: effectivePolicy(strategy) };
  }
}

// ---- stdin/stdout JSON-line loop --------------------------------------------

let session: Session | null = null;

function handle(req: any): any {
  switch (req.op) {
    case "ping":
      return { ok: true, pong: true };
    case "reset": {
      const seed = String(req.seed ?? "seed-0");
      // deck selection: explicit ids, "placeholder" to force the fallback, or
      // deterministic pick from the seed when decks are available.
      let p1: string | null = null;
      let p2: string | null = null;
      const usePlaceholder = req.deckP1 === "placeholder" || req.deckP2 === "placeholder"
        || DECK_REGISTRY.length === 0;
      if (!usePlaceholder) {
        p1 = req.deckP1 ? String(req.deckP1) : pickDeck(seed, "p1");
        p2 = req.deckP2 ? String(req.deckP2) : pickDeck(seed, "p2");
        if (!DECK_BY_ID[p1] || !DECK_BY_ID[p2]) {
          return { ok: false, error: `unknown deck(s): ${p1}, ${p2}` };
        }
      }
      session = new Session(seed, p1, p2);
      return { ok: true, obs: session.observe(),
               decks: { p1: session.deckP1, p2: session.deckP2 } };
    }
    case "list_decks": {
      return { ok: true, decks: DECK_REGISTRY.map((d) => ({ id: d.id, name: d.name })) };
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
    case "determinize": {
      if (!session) return { ok: false, error: "no session" };
      const self = String(req.self);
      const handIds: string[] = Array.isArray(req.handInstanceIds) ? req.handInstanceIds : [];
      session.determinize(self, handIds, req.seed ? String(req.seed) : undefined);
      return { ok: true, obs: session.observe() };
    }
    case "run_paths": {
      if (!session) return { ok: false, error: "no session" };
      const paths: string[][] = Array.isArray(req.paths) ? req.paths : [];
      return { ok: true, obs: session.runPaths(Number(req.root), paths) };
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
