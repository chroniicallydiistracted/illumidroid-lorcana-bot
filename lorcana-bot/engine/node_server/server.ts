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

// effectively-uncapped caps, for the grammar-gap proof (capped vs uncapped vs raw).
const BIG_CAPS = {
  targetPool: 9999,
  targetCombinationsPerFamily: 9999,
  choiceIndices: 9999,
  singerCombinations: 9999,
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

/** EXACT strategy: offers ONLY the requested candidate (no fallback list). Used
 *  by search execution so a failed key can NEVER silently execute a different
 *  candidate — that would make MCTS evaluate a leaf for a line the tree did not
 *  choose, corrupting visit policies / leaf values (the training targets). */
function exactKeyStrategy(targetKey: string): AutomatedActionStrategy {
  return {
    name: `exact:${targetKey}`,
    summarizeCandidates(_ctx, candidates): AutomatedActionCandidateSummary[] {
      return candidates
        .filter((c) => candidateKey(c) === targetKey)
        .map((candidate) => ({ candidate, family: candidate.family, heuristics: [],
                               stableKey: candidateKey(candidate) }));
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

const HIST_MAX = 48;   // rolling window of recent PUBLIC events (§2.3)
const CANON = [CANONICAL_PLAYER_ONE, CANONICAL_PLAYER_TWO];

// Finding #4 — benign per-card meta a genuinely hidden hand/deck/inkwell card may carry
// (observed across real games: state/damage/publicFaceState/isDrying/revealed). Any key
// OUTSIDE this set on a card being MOVED into a hidden zone is play-context state (location,
// shift stacks, temporary grants, …) and is rejected fail-closed rather than normalized away.
const HIDDEN_ZONE_META_ALLOW = new Set<string>([
  "state", "damage", "isDrying", "publicFaceState", "revealed",
]);

// SCHEMA for the protected-facts ledger: runtime-state `G` paths that store card ids as
// COUNT-ONLY metrics — `turnMetadata` "<verb>ThisTurn" arrays consumed solely via `.length`
// (e.g. cardsPutIntoInkwellThisTurn @ condition-evaluator.ts:482, inkedThisTurn @
// turn-action-ink.ts:111 / derived-state.ts:560). For these the stored IDENTITY is irrelevant —
// the id still exists after a determinization permutes the hidden pool, so the count is invariant
// — so a hidden id appearing here is NOT a fact the actor knows and must NOT pin (it would
// condition the posterior on hidden truth). Any G path NOT in this set defaults to
// identity-bearing (fail-closed: an unrecognised private reference quarantines the state).
const COUNT_ONLY_G_PATHS = new Set<string>([
  "G.turnMetadata.cardsPlayedThisTurn",
  "G.turnMetadata.charactersQuesting",
  "G.turnMetadata.inkedThisTurn",
  "G.turnMetadata.cardsPutIntoInkwellThisTurn",
  "G.turnMetadata.shiftPlayedThisTurn",
  "G.turnMetadata.challengedCharactersThisTurn",
  "G.turnMetadata.banishedCharactersThisTurn",
]);

class Session {
  engine: AnyEngine;
  seed: string;
  steps = 0;
  deckP1: string | null;
  deckP2: string | null;
  private snapshots = new Map<number, any>();
  private snapCounter = 0;
  // per-game public action history (§2.3). Mutated ONLY by real step/stepAuto —
  // NOT by runPaths/executeKey — so search leaves share the root's history.
  private history: any[] = [];

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

  /** DIAGNOSTIC-ONLY (Tier-A) — HAND-ONLY determinization, INVALID FOR CLEAN-LABEL
   *  TRAINING. Replaced by full-`World` `determinizeWorld` in Tier-A Phase 3. It takes
   *  ONLY `handInstanceIds` and RANDOMIZES the remaining hidden zones internally, so a
   *  sampled World's opponent inkwell/deck and self deck ORDER are discarded (only the
   *  partition is leak-free, not honored). Use the gated diagnostic path only
   *  (`run_pimc_diagnostic` / `EngineSimulator(..., hand_only_diagnostic=true)`).
   *
   *  From the acting player's information set, ALL hidden zones are randomized so search
   *  cannot read privileged info (real inked cards, real future draws):
   *    - opponent HAND     = the belief sample (`handInstanceIds`)
   *    - opponent INKWELL  = sampled (count-consistent) from the remaining hidden pool
   *    - opponent DECK     = the rest, shuffled
   *    - SELF DECK         = shuffled (the actor knows its composition, not its draw order)
   *  Self hand + self inkwell stay as-is (the actor legitimately knows those).
   *  Operates on the *current* authoritative state (caller restores the true root). */
  determinize(selfId: string, handInstanceIds: string[], seed?: string): boolean {
    const oppId = selfId === CANONICAL_PLAYER_ONE ? CANONICAL_PLAYER_TWO : CANONICAL_PLAYER_ONE;
    const clone: any = structuredClone(this.engine.getAuthoritativeState());
    const zc = clone.ctx.zones.private.zoneCards;
    const sum = clone.ctx.zones.public.zoneSummaries;

    let h = 0;
    if (seed) for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
    const rng = seed ? () => ((h = (h * 1103515245 + 12345) >>> 0) / 0x100000000) : Math.random;
    const shuffle = (arr: string[]) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    };
    const bump = (key: string, count: number) => {
      if (sum[key]) sum[key] = { ...sum[key], count, revision: (sum[key].revision || 0) + 1 };
    };

    // ---- opponent: every hidden zone randomized (hand from belief, ink + deck resampled) ----
    const oHand = `hand:${oppId}`, oDeck = `deck:${oppId}`, oInk = `inkwell:${oppId}`;
    const inkCount = (zc[oInk] ?? []).length;          // public count; identities hidden
    const pool: string[] = [...(zc[oHand] ?? []), ...(zc[oDeck] ?? []), ...(zc[oInk] ?? [])];
    const handSet = new Set(handInstanceIds.filter((id) => pool.includes(id)));
    const wantHand = [...handSet];
    const remaining = shuffle(pool.filter((id) => !handSet.has(id)));
    const newInk = remaining.slice(0, inkCount);
    const newDeck = remaining.slice(inkCount);
    zc[oHand] = wantHand; zc[oInk] = newInk; zc[oDeck] = newDeck;
    bump(oHand, wantHand.length); bump(oInk, newInk.length); bump(oDeck, newDeck.length);

    // ---- searching player: own draw order is unknown to them -> shuffle own deck ----
    const sDeck = `deck:${selfId}`;
    if (zc[sDeck]) { zc[sDeck] = shuffle([...zc[sDeck]]); bump(sDeck, zc[sDeck].length); }

    this.engine.loadState(clone);
    return true;
  }

  /** Deterministic Fisher-Yates from a string seed (NO ambient randomness), PINNING the
   *  positions whose knowledge is protected: a pinned index keeps its current card, and only
   *  the remaining FREE positions are permuted among themselves. Used for the self-deck order
   *  when no explicit order is supplied — so a scryed / statically-revealed / live-referenced
   *  self-deck card keeps its known position instead of being shuffled away. */
  private seededShufflePinned(arr: string[], seed: string, pinned: Set<number>): string[] {
    let h = 0;
    for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
    const rng = () => ((h = (h * 1103515245 + 12345) >>> 0) / 0x100000000);
    const out = [...arr];
    const free: number[] = [];
    for (let i = 0; i < arr.length; i++) if (!pinned.has(i)) free.push(i);
    // Fisher-Yates over the FREE slots only (their cards), leaving pinned slots untouched.
    for (let k = free.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      const a = free[k], b = free[j];
      [out[a], out[b]] = [out[b], out[a]];
    }
    return out;
  }

  /** Observer-aware PROTECTED-FACTS LEDGER. Actor knowledge and engine references are computed
   *  and represented SEPARATELY (no longer unioned into one pin map), because they mean different
   *  things and must be ENFORCED differently:
   *
   *    (a) `actorVisiblePins` — EXACT-ID POSITIONAL pins, the ONLY identity the determinization is
   *        constrained to reproduce. Sourced SOLELY from the actor's own fog projection
   *        (`getBoard(view)`), the engine's visibility-correct source of truth (native
   *        `projectPlayerBoard` applies fog + reveal `visibleTo` via `isCardVisibleViaReveal` +
   *        static top-deck via `isTopDeckCardVisibleViaStaticEffect`). We do NOT raw-scan
   *        `ctx.zones.reveals` (that would pin opponent-PRIVATE reveals — Finding #2). The
   *        projection is REQUIRED: its absence FAILS CLOSED (Finding #1). For decks this preserves
   *        positional (top/bottom/scry) knowledge.
   *
   *    (b) `quarantineIds` — engine references to a repartitioned HIDDEN card the actor CANNOT see.
   *        Such a reference must NOT pin the hidden identity (that conditions the posterior on
   *        hidden truth). It is SCHEMA-CLASSIFIED by its `G` path: a COUNT-ONLY metric
   *        (`COUNT_ONLY_G_PATHS`, consumed via `.length`) is permutation-invariant and IGNORED;
   *        any other (identity-bearing) private reference cannot be safely pinned OR remapped, so
   *        the determinization is QUARANTINED as an unsupported state (fail closed, without biasing
   *        toward a world). The scan inspects VALUES and MAP KEYS (Finding #3). References to
   *        actor-visible or non-repartitioned cards are no-ops (the card does not move / is known).
   *
   *  This realises the audit's required split: exact-IDs only for actor-visible facts; identity
   *  ignored for count-only metrics; unsupported identity-bearing private references quarantined. */
  private collectProtectedFacts(clone: any, selfId: string, zoneKeys: string[]): {
    actorVisiblePins: Record<string, Map<number, string>>;
    quarantineIds: Set<string>;
    countOnlyRefIds: Set<string>;
  } {
    const idUniverse = new Set<string>(Object.keys(clone.ctx.zones.private.cardIndex ?? {}));
    const zc = clone.ctx.zones.private.zoneCards;
    const meta = clone.ctx.zones.private.cardMeta ?? {};

    // The set of ids the determinization may PERMUTE (the four repartitioned hidden zones). A
    // reference to a card OUTSIDE these zones (in play / discard) is irrelevant — that card never
    // moves, so its references stay valid regardless.
    const repartitioned = new Set<string>();
    for (const zk of zoneKeys) for (const id of (zc[zk] ?? [])) repartitioned.add(id);

    // (a) ACTOR-VISIBLE identities — the ONLY exact-id positional pins. Cards non-hidden in the
    // actor's own fog projection (the engine's visibility-correct source of truth: reveal
    // `visibleTo`, scry, static top-deck). REQUIRED — a projection failure FAILS CLOSED
    // (Finding #1); we never proceed without actor pins.
    const actorVisible = new Set<string>();
    const board: any = this.engine.getBoard(viewFor(selfId));
    if (!board || typeof board !== "object" || !board.cards) {
      throw new Error("determinize_world: actor fog projection unavailable (cannot compute actor-visible pins)");
    }
    for (const c of Object.values(board.cards) as any[]) {
      if (c && c.hidden === false && typeof c.id === "string" && idUniverse.has(c.id)) actorVisible.add(c.id);
    }
    // The board projection only surfaces top/bottom-of-deck cards individually, so a reveal the
    // actor is CURRENTLY resolving over mid-deck cards (e.g. a scry the actor is choosing from) is
    // genuine actor knowledge that `board.cards` does not expose. Add reveal-window cards using
    // the EXACT native `isCardVisibleViaReveal` `visibleTo` rule — `"all"` or a window naming the
    // actor — so scry/look knowledge is captured WHILE opponent-PRIVATE reveals stay excluded
    // (Finding #2). This is visibility-CORRECT, unlike the removed blind scan.
    for (const rv of (clone.ctx.zones.reveals?.active ?? [])) {
      const vt = rv?.visibleTo;
      const visibleToActor = vt === "all" || (Array.isArray(vt) && vt.includes(selfId));
      if (!visibleToActor) continue;
      for (const cid of (rv.cardIDs ?? [])) if (idUniverse.has(cid)) actorVisible.add(cid);
    }

    // (b) ENGINE references, SCHEMA-CLASSIFIED and kept DISTINCT from actor knowledge. A live
    // reference to a repartitioned HIDDEN card (one the actor cannot see) must NOT pin its exact
    // identity — that would condition the actor's posterior on hidden truth. The reference is
    // classified by WHERE it lives:
    //   * COUNT-ONLY metric path (turn-metadata "<verb>ThisTurn" arrays, consumed only via
    //     `.length` — verified for cardsPutIntoInkwellThisTurn @ condition-evaluator.ts:482 and
    //     inkedThisTurn @ turn-action-ink.ts:111 / derived-state.ts:560): identity is irrelevant
    //     (the id still exists after the permutation, so the count is invariant) -> IGNORE.
    //   * actor-visible: handled by (a) -> IGNORE here (the card does not move, ref stays valid).
    //   * anything else (identity-bearing private reference): we can neither pin (leak) nor safely
    //     remap it -> QUARANTINE the whole determinization as an UNSUPPORTED state (fail closed,
    //     without biasing toward any particular world). Unknown paths default to identity-bearing.
    const quarantineIds = new Set<string>();
    const countOnlyRefIds = new Set<string>();   // hidden + count-only-referenced -> deliberately NOT pinned
    const classify = (id: string, countOnly: boolean) => {
      if (!repartitioned.has(id)) return;     // never moves -> reference stays valid
      if (actorVisible.has(id)) return;       // actor knows it -> pinned by (a), does not move
      if (countOnly) { countOnlyRefIds.add(id); return; }  // count-only metric -> identity irrelevant
      quarantineIds.add(id);                  // identity-bearing private reference -> unsupported
    };
    const scan = (v: any, path: string, countOnly: boolean) => {
      if (v == null) return;
      if (typeof v === "string") { if (idUniverse.has(v)) classify(v, countOnly); return; }
      if (Array.isArray(v)) { for (const x of v) scan(x, path, countOnly); return; }
      if (typeof v === "object") {
        for (const k of Object.keys(v)) {
          const childPath = path ? `${path}.${k}` : k;
          const childCountOnly = countOnly || COUNT_ONLY_G_PATHS.has(childPath);
          if (idUniverse.has(k)) classify(k, childCountOnly);   // card id stored as a MAP KEY (Finding #3)
          scan(v[k], childPath, childCountOnly);
        }
      }
    };
    scan(clone.G, "G", false);
    // cardMeta cross-references (cardsUnder / stackParentId / atLocationId) are identity-bearing
    // and point at IN-PLAY cards (never repartitioned), so they classify to a no-op — but they are
    // scanned for completeness / fail-closed coverage.
    for (const m of Object.values(meta) as any[]) scan(m, "cardMeta", false);

    // a card referenced in BOTH a count-only and an identity-bearing path is quarantined (the
    // identity-bearing use wins); it is not a "free" count-only ref.
    for (const id of quarantineIds) countOnlyRefIds.delete(id);

    const actorVisiblePins: Record<string, Map<number, string>> = {};
    for (const zk of zoneKeys) {
      const m = new Map<number, string>();
      const cur: string[] = zc[zk] ?? [];
      for (let i = 0; i < cur.length; i++) if (actorVisible.has(cur[i])) m.set(i, cur[i]);
      actorVisiblePins[zk] = m;
    }
    return { actorVisiblePins, quarantineIds, countOnlyRefIds };
  }

  /** Expose the protected-facts ledger for the CURRENT actor (diagnostic / test / future sampler
   *  use). Returns the ACTOR-VISIBLE positional pins per repartitioned zone (`pins`) + their flat
   *  union (`pinnedIds`) — the ONLY identity the determinization constrains — kept DISTINCT from
   *  `quarantineIds`, the repartitioned hidden cards carrying an identity-bearing private engine
   *  reference (which make the state unsupported, not pinned). The sampler proposes worlds that
   *  reproduce `pins` and avoid `quarantineIds`. */
  protectedFacts(): {
    self: string | null; pins: Record<string, { index: number; id: string }[]>;
    pinnedIds: string[]; quarantineIds: string[]; countOnlyRefIds: string[];
  } {
    const actor = this.currentActor();
    if (!actor) return { self: null, pins: {}, pinnedIds: [], quarantineIds: [], countOnlyRefIds: [] };
    const oppId = actor === CANONICAL_PLAYER_ONE ? CANONICAL_PLAYER_TWO : CANONICAL_PLAYER_ONE;
    const clone: any = structuredClone(this.engine.getAuthoritativeState());
    const zoneKeys = [`hand:${oppId}`, `inkwell:${oppId}`, `deck:${oppId}`, `deck:${actor}`];
    const { actorVisiblePins, quarantineIds, countOnlyRefIds } = this.collectProtectedFacts(clone, actor, zoneKeys);
    const pins: Record<string, { index: number; id: string }[]> = {};
    const union = new Set<string>();
    for (const zk of zoneKeys) {
      pins[zk] = [...actorVisiblePins[zk].entries()].map(([index, id]) => ({ index, id }));
      for (const { id } of pins[zk]) union.add(id);
    }
    return {
      self: actor, pins, pinnedIds: [...union],
      quarantineIds: [...quarantineIds], countOnlyRefIds: [...countOnlyRefIds],
    };
  }

  /** Validate that a REWRITTEN clone is internally consistent BEFORE it is committed via
   *  `loadState` (Finding #6 — staged validation). Recomputes the agreement invariants the
   *  ENGINE ACTUALLY MAINTAINS: every card listed in a `zoneCards` lane has a `cardIndex` entry
   *  pointing to THAT lane with a string owner/controller, and no instance appears in two lanes.
   *  (The engine does NOT keep `cardIndex.index` position-synced — a clean baseline already has
   *  ~100 positional offsets — so positional index is intentionally NOT asserted here; the
   *  repartitioned lanes are re-indexed positionally by `reindexZone` regardless.) Throws on any
   *  violation; the caller never loads a failed clone, so the live engine state is preserved. */
  private validateCloneConsistency(clone: any): void {
    const zc = clone.ctx.zones.private.zoneCards;
    const idx = clone.ctx.zones.private.cardIndex;
    const seen = new Set<string>();
    for (const [zoneKey, ids] of Object.entries(zc) as [string, string[]][]) {
      for (const cardId of ids) {
        if (seen.has(cardId)) throw new Error(`determinize_world: clone has ${cardId} in two zones`);
        seen.add(cardId);
        const e = idx[cardId];
        if (!e || e.zoneKey !== zoneKey) {
          throw new Error(`determinize_world: clone cardIndex disagrees for ${cardId} (in ${zoneKey}, indexed at ${e?.zoneKey ?? "<none>"})`);
        }
        if (typeof e.ownerID !== "string" || typeof e.controllerID !== "string") {
          throw new Error(`determinize_world: clone cardIndex missing owner/controller for ${cardId}`);
        }
      }
    }
    // Finding #6: REVERSE-INDEX orphans — every `cardIndex` entry must point at a lane that
    // actually contains it (a clean baseline has zero). A repartition that left a stale forward
    // index would surface here even if the lane→index direction above happened to pass.
    for (const [cardId, e] of Object.entries(idx) as [string, any][]) {
      const lane = zc[e.zoneKey];
      if (!lane || !lane.includes(cardId)) {
        throw new Error(`determinize_world: clone reverse-index orphan ${cardId} (cardIndex -> ${e?.zoneKey ?? "<none>"} which does not contain it)`);
      }
    }
  }

  /** Reconcile the THREE parallel engine structures after a raw hidden-zone repartition,
   *  exactly as the native fixture loader maintains them together (`applyFixtureState`): the
   *  ordered `zoneCards`, the per-card location `cardIndex` (zoneKey + positional index +
   *  owner/controller), and the per-card `cardMeta`. Without this, `moveCard()` resolves a
   *  card via its STALE `cardIndex.zoneKey` and corrupts state (an instance in two zones).
   *
   *  STRICT-ADMISSION + ZONE-TRANSITION METADATA CONTRACT (fail-closed; no guessing):
   *   - Finding #5: every reindexed card MUST already have a `cardIndex` entry carrying a
   *     string `ownerID` AND `controllerID`. A missing entry/owner/controller is malformed
   *     authoritative state and THROWS (no `?? {}`, no fabricated fallback ownership).
   *     owner/controller are PRESERVED from the card's own entry (native zone-operations.ts:212
   *     keeps them across movement), never derived from the destination zone.
   *   - Finding #4: a card that MOVES into this hidden zone (its current `cardIndex.zoneKey`
   *     differs) may only carry the benign hidden-zone meta allowlist; any play-context special
   *     meta (`atLocationId`, `cardsUnder`, `stackParentId`, `temporaryKeywords`, a face-UP
   *     overlay, …) THROWS rather than being silently normalized. A moved card is rebuilt with
   *     clean hidden-zone meta (inkwell: slot-wise public ready/exerted + faceDown; hand/deck:
   *     ready). A card that STAYS in its zone keeps its own meta unchanged (index only).
   *  Does NOT touch public zone summaries (Finding #5 / F5): a hidden-only repartition
   *  preserves the public counts, so the summary (count + revision) must stay byte-identical. */
  private reindexZone(clone: any, zoneKey: string, newIds: string[], inkSlotStates?: (string | undefined)[]): void {
    const zc = clone.ctx.zones.private.zoneCards;
    const idx = clone.ctx.zones.private.cardIndex;
    const meta = clone.ctx.zones.private.cardMeta;
    const isInk = zoneKey.startsWith("inkwell:");
    zc[zoneKey] = [...newIds];
    for (let i = 0; i < newIds.length; i++) {
      const cardId = newIds[i];
      const prev = idx[cardId];
      // Finding #5: strict admission — reject malformed authoritative state, do not invent it.
      if (!prev || typeof prev.ownerID !== "string" || typeof prev.controllerID !== "string") {
        throw new Error(`determinize_world: card ${cardId} has no/invalid cardIndex owner/controller (malformed state)`);
      }
      const moving = prev.zoneKey !== zoneKey;
      idx[cardId] = { zoneKey, index: i, ownerID: prev.ownerID, controllerID: prev.controllerID };
      if (!moving) continue;                                // staying card: keep its meta, index already set
      // Findings #4 + #7: zone-transition metadata contract for a MOVED card — every meta key/value
      // gets EXPLICIT semantics (reject OR documented clear), never a silent discard.
      //   * unknown key                  -> REJECT (play-context state we don't model as hidden).
      //   * damage != 0                  -> REJECT (a hidden-pool card is undamaged; a damaged
      //                                     card is an in-play instance that must not be relocated).
      //   * isDrying                     -> REJECT (drying is summoning-sickness, play-only).
      //   * publicFaceState === "faceUp" -> REJECT (face-up == publicly visible, not hidden).
      //   * revealed (truthy)            -> CLEAR (explicit): a card reassigned to a NEW hidden
      //                                     position is freshly hidden there, so its stale reveal
      //                                     flag is dropped. This must NOT reject — an
      //                                     opponent-PRIVATE reveal is not actor knowledge
      //                                     (Finding #2), so the belief stays free to vary it.
      //   * state / damage(==0)          -> normalized by the clean rebuild below.
      const own = meta[cardId] ?? {};
      for (const k of Object.keys(own)) {
        if (!HIDDEN_ZONE_META_ALLOW.has(k)) {
          throw new Error(`determinize_world: card ${cardId} carries unsupported meta '${k}' for hidden-zone transition into ${zoneKey}`);
        }
      }
      if ((own.damage ?? 0) !== 0) {
        throw new Error(`determinize_world: card ${cardId} has non-zero damage (${own.damage}) and cannot be moved into hidden ${zoneKey}`);
      }
      if (own.isDrying) {
        throw new Error(`determinize_world: card ${cardId} is drying and cannot be moved into hidden ${zoneKey}`);
      }
      if (own.publicFaceState === "faceUp") {
        throw new Error(`determinize_world: card ${cardId} is face-up (known) and cannot be moved into hidden ${zoneKey}`);
      }
      // rebuild clean hidden-zone meta (a freshly placed hidden card has no damage / drying /
      // reveal state); inkwell preserves the public ready/exerted COUNT slot-wise + faceDown.
      meta[cardId] = isInk
        ? { state: (inkSlotStates && inkSlotStates[i]) ?? "ready", publicFaceState: "faceDown" }
        : { state: "ready" };
    }
  }

  /** Tier-A Phase 3 — full-`World` determinization (CLEAN-LABEL capable). Unlike the
   *  diagnostic hand-only `determinize`, this HONORS the sampled World's EXACT opponent
   *  hand / inkwell / deck partition and (when supplied) the EXACT self-deck order, with NO
   *  ambient randomness. It validates the spec against the authoritative hidden pool and
   *  FAILS CLOSED (throws) on any violation rather than silently repairing:
   *    - `selfId` must be one of the two canonical seats;
   *    - `world.seed` must be a non-empty (trimmed) string — for EVERY call;
   *    - every id must be a non-empty string;
   *    - each requested opponent zone count must equal the public zone size;
   *    - no id may repeat across the three opponent zones;
   *    - the requested opponent partition must conserve the authoritative hidden pool exactly;
   *    - a supplied selfDeckIds must be a count-correct, duplicate-free permutation of the
   *      authoritative self deck (conservation);
   *    - the world must reproduce every ACTOR-VISIBLE protected fact (`collectProtectedFacts`):
   *      each position whose current card is actor-known (reveal `visibleTo` / scry / static
   *      top-deck, from the engine's own fog projection) keeps that exact id at that exact index
   *      across ALL four repartitioned zones (decks positionally). This closes the "revealed deck
   *      card slides off top" and "ignores the acting player's self deck" gaps WITHOUT pinning any
   *      hidden identity the actor cannot see (count-only private metrics are ignored);
   *    - the state must carry NO identity-bearing private engine reference to a repartitioned
   *      HIDDEN card the actor cannot see — such a state is QUARANTINED (unsupported), because
   *      pinning would condition the posterior on hidden truth and remapping is unsafe.
   *  Validation runs ENTIRELY before any mutation, and the rewritten clone is re-validated
   *  (`validateCloneConsistency`) before `loadState`, so a rejected request leaves the live
   *  engine state unchanged. After repartition it reconciles `cardIndex` + `cardMeta`
   *  (`reindexZone`). When selfDeckIds is omitted the self deck order is shuffled
   *  DETERMINISTICALLY from `world.seed`, PINNING protected positions in place. Returns the
   *  REALIZED post-load partition. */
  determinizeWorld(selfId: string, world: any): {
    opponentHandIds: string[]; opponentInkwellIds: string[];
    opponentDeckIds: string[]; selfDeckIds: string[];
  } {
    // --- fail-closed admission checks (before cloning or mutating anything) ---
    if (selfId !== CANONICAL_PLAYER_ONE && selfId !== CANONICAL_PLAYER_TWO) {
      throw new Error(`determinize_world: invalid seat ${JSON.stringify(selfId)}`);
    }
    // F6: a determinization is only valid from the CURRENT actor's information set; reject any
    // other (even otherwise-canonical) seat so a caller can't mutate the wrong perspective.
    const actor = this.currentActor();
    if (selfId !== actor) {
      throw new Error(`determinize_world: selfId ${JSON.stringify(selfId)} is not the current actor ${JSON.stringify(actor)}`);
    }
    if (typeof world.seed !== "string" || world.seed.trim().length === 0) {
      throw new Error("determinize_world: world.seed must be a non-empty string");
    }
    const asIds = (v: unknown, label: string): string[] => {
      if (!Array.isArray(v)) throw new Error(`determinize_world: ${label} must be an array`);
      return v.map((x) => {
        if (typeof x !== "string" || x.length === 0) {
          throw new Error(`determinize_world: ${label} has a non-string/empty id ${JSON.stringify(x)}`);
        }
        return x;
      });
    };

    const oppId = selfId === CANONICAL_PLAYER_ONE ? CANONICAL_PLAYER_TWO : CANONICAL_PLAYER_ONE;
    const clone: any = structuredClone(this.engine.getAuthoritativeState());
    const zc = clone.ctx.zones.private.zoneCards;
    const meta = clone.ctx.zones.private.cardMeta;

    const oHand = `hand:${oppId}`, oInk = `inkwell:${oppId}`, oDeck = `deck:${oppId}`;
    const curHand: string[] = zc[oHand] ?? [], curInk: string[] = zc[oInk] ?? [], curDeck: string[] = zc[oDeck] ?? [];
    const reqHand = asIds(world.opponentHandIds, "opponentHandIds");
    const reqInk = asIds(world.opponentInkwellIds, "opponentInkwellIds");
    const reqDeck = asIds(world.opponentDeckIds, "opponentDeckIds");

    // (1) per-zone counts must match the public zone sizes (the actor knows the counts)
    if (reqHand.length !== curHand.length) throw new Error(`determinize_world: opponent hand count ${reqHand.length} != ${curHand.length}`);
    if (reqInk.length !== curInk.length) throw new Error(`determinize_world: opponent inkwell count ${reqInk.length} != ${curInk.length}`);
    if (reqDeck.length !== curDeck.length) throw new Error(`determinize_world: opponent deck count ${reqDeck.length} != ${curDeck.length}`);
    // (2) no duplicate id across the three requested opponent zones
    const all = [...reqHand, ...reqInk, ...reqDeck];
    if (new Set(all).size !== all.length) throw new Error("determinize_world: duplicate id across opponent hand/inkwell/deck");
    // (3) the requested partition must conserve the authoritative hidden pool EXACTLY
    const pool = new Set<string>([...curHand, ...curInk, ...curDeck]);
    if (all.length !== pool.size || all.some((id) => !pool.has(id))) {
      throw new Error("determinize_world: requested opponent partition is not the authoritative hidden pool");
    }

    const sDeck = `deck:${selfId}`;
    const curSelf: string[] = zc[sDeck] ?? [];

    // ---- observer-aware protected-facts ledger (actor-visible pins SEPARATE from engine refs) ----
    // `actorVisiblePins` = positions whose current card the ACTOR legally sees (the only exact-id
    // constraint). `quarantineIds` = repartitioned hidden cards carrying an identity-bearing
    // private engine reference: such a state is UNSUPPORTED (pinning would leak hidden truth,
    // remapping is unsafe) -> reject the whole determinization fail-closed, before any mutation.
    const { actorVisiblePins, quarantineIds } = this.collectProtectedFacts(clone, selfId, [oHand, oInk, oDeck, sDeck]);
    if (quarantineIds.size > 0) {
      throw new Error(`determinize_world: unsupported state — identity-bearing private engine reference to hidden card(s) ${[...quarantineIds].join(", ")} (cannot pin without leaking, cannot safely remap)`);
    }
    const enforcePins = (zoneKey: string, req: string[]) => {
      for (const [i, id] of actorVisiblePins[zoneKey]) {
        if (req[i] !== id) {
          throw new Error(`determinize_world: world violates an actor-visible fact in ${zoneKey} at slot ${i} (expected ${id}, got ${req[i] ?? "<none>"})`);
        }
      }
    };
    enforcePins(oHand, reqHand);
    enforcePins(oInk, reqInk);
    enforcePins(oDeck, reqDeck);

    // ---- self deck: honor a supplied order (pins enforced), else seeded shuffle pinning the
    //      actor-visible (scryed / statically-revealed) positions in place ----
    let newSelf: string[];
    if (world.selfDeckIds !== null && world.selfDeckIds !== undefined) {
      const reqSelf = asIds(world.selfDeckIds, "selfDeckIds");
      if (reqSelf.length !== curSelf.length) throw new Error(`determinize_world: self deck count ${reqSelf.length} != ${curSelf.length}`);
      if (new Set(reqSelf).size !== reqSelf.length) throw new Error("determinize_world: duplicate id in selfDeckIds");
      const selfPool = new Set<string>(curSelf);
      if (reqSelf.some((id) => !selfPool.has(id))) {
        throw new Error("determinize_world: selfDeckIds is not a permutation of the authoritative self deck (conservation failure)");
      }
      enforcePins(sDeck, reqSelf);
      newSelf = [...reqSelf];
    } else {
      newSelf = this.seededShufflePinned([...curSelf], world.seed, new Set(actorVisiblePins[sDeck].keys()));
    }

    // ---- ALL admission checks passed: now mutate (cardIndex + cardMeta reconciled per zone) ----
    // capture ONLY the public inkwell ready/exerted state per slot (an allowlisted scalar) so
    // the public available-ink count is preserved without transferring any identity's meta.
    const oldInkStates = curInk.map((id) => meta[id]?.state as string | undefined);
    this.reindexZone(clone, oHand, reqHand);
    this.reindexZone(clone, oInk, reqInk, oldInkStates);
    this.reindexZone(clone, oDeck, reqDeck);
    this.reindexZone(clone, sDeck, newSelf);

    // Finding #6: validate the rewritten clone BEFORE committing it. On any inconsistency we
    // throw here, having never called loadState — so the live engine state is left intact.
    this.validateCloneConsistency(clone);
    this.engine.loadState(clone);
    // realized post-load partition (strongest verification — what the engine actually holds)
    const after: any = this.engine.getAuthoritativeState();
    const z2 = after.ctx.zones.private.zoneCards;
    return {
      opponentHandIds: [...(z2[oHand] ?? [])],
      opponentInkwellIds: [...(z2[oInk] ?? [])],
      opponentDeckIds: [...(z2[oDeck] ?? [])],
      selfDeckIds: [...(z2[sDeck] ?? [])],
    };
  }

  /** Diagnostic state-integrity check (Phase 3 tests): proves the three parallel zone
   *  structures agree after a determinization — every card in a `zoneCards` list has a
   *  `cardIndex` entry pointing to THAT zone at THAT index with intact owner/controller (no
   *  stale location, no corruption) and no instance appears in two zones (no duplication). Also
   *  reports per-inkwell-zone READY counts (public available-ink preservation), the public zone
   *  summaries (count+revision — must be unchanged by a hidden repartition), the reverse-index
   *  orphan count, and the structured reveal windows (id + zone + index) so a test can assert
   *  positional reveal facts survive. Finding #7: `badOwner` now also flags a corrupted
   *  `controllerID`, and `revealedIds` carries each card's current zone + index. */
  checkConsistency(): {
    indexMismatches: number; multiZone: number; total: number; posMismatches: number;
    badOwner: number; orphanIndex: number; inkReady: Record<string, number>;
    summaries: Record<string, { count: number; revision: number }>;
    revealedIds: { id: string; zoneKey: string | null; index: number | null }[];
  } {
    const st: any = this.engine.getAuthoritativeState();
    const zc = st.ctx.zones.private.zoneCards;
    const idx = st.ctx.zones.private.cardIndex;
    const meta = st.ctx.zones.private.cardMeta;
    const sum = st.ctx.zones.public.zoneSummaries ?? {};
    const seen = new Set<string>();
    let indexMismatches = 0, multiZone = 0, total = 0, posMismatches = 0, badOwner = 0;
    const inkReady: Record<string, number> = {};
    for (const [zoneKey, ids] of Object.entries(zc) as [string, string[]][]) {
      const ownerScoped = /^(hand|deck|inkwell):/.test(zoneKey);
      const zonePlayer = zoneKey.slice(zoneKey.indexOf(":") + 1);
      let ready = 0;
      for (let i = 0; i < ids.length; i++) {
        const cardId = ids[i];
        total++;
        if (seen.has(cardId)) multiZone++;
        else seen.add(cardId);
        const e = idx[cardId];
        if (!e || e.zoneKey !== zoneKey) {
          indexMismatches++;
        } else {
          if (e.index !== i) posMismatches++;                          // F7: positional index drift
          // F4/F7: owner OR controller corruption in a private zone. A hidden card's owner must
          // be the zone's player; its controller must equal its owner (no foreign control while
          // hidden) — a moved card that kept a stale in-play controller would be flagged here.
          if (ownerScoped && (e.ownerID !== zonePlayer || e.controllerID !== zonePlayer)) badOwner++;
        }
        if (zoneKey.startsWith("inkwell:") && meta[cardId]?.state === "ready") ready++;
      }
      if (zoneKey.startsWith("inkwell:")) inkReady[zoneKey] = ready;
    }
    // F7: reverse-index orphans — a cardIndex entry whose zone does not actually contain it
    let orphanIndex = 0;
    for (const [cardId, e] of Object.entries(idx) as [string, any][]) {
      const z = zc[e.zoneKey];
      if (!z || !z.includes(cardId)) orphanIndex++;
    }
    // F5/F7: public zone summaries (count + revision) — a hidden-only repartition must leave
    // these byte-for-byte unchanged; a caller can diff before/after.
    const summaries: Record<string, { count: number; revision: number }> = {};
    for (const [zoneKey, s] of Object.entries(sum) as [string, any][]) {
      summaries[zoneKey] = { count: s?.count ?? 0, revision: s?.revision ?? 0 };
    }
    // F7: structured reveal windows — id + current zone + index so a test can assert a revealed
    // card kept its exact position (positional reveal knowledge), not just its zone membership.
    const revealedIds: { id: string; zoneKey: string | null; index: number | null }[] = [];
    for (const rv of (st.ctx.zones.reveals?.active ?? [])) {
      for (const cid of (rv.cardIDs ?? [])) {
        const e = idx[cid];
        revealedIds.push({ id: cid, zoneKey: e?.zoneKey ?? null, index: typeof e?.index === "number" ? e.index : null });
      }
    }
    return { indexMismatches, multiZone, total, posMismatches, badOwner, orphanIndex, inkReady, summaries, revealedIds };
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
  /** Execute exactly `key` (no fallback to other candidates). Returns true only
   *  if the requested key was the one that executed. */
  private executeKey(key: string): boolean {
    const server = this.engine.asServer();
    if (key === PASS_KEY) {
      const res = server.takeAutomatedActionForCurrentActor({ strategy: PASS_STRATEGY, searchCaps: SEARCH_CAPS });
      return Boolean(res.finalResult?.success) || res.fallbackTaken === "passTurn";
    }
    const res = server.takeAutomatedActionForCurrentActor({ strategy: exactKeyStrategy(key), searchCaps: SEARCH_CAPS });
    const executed = res.selectedCandidate ? candidateKey(res.selectedCandidate) : (res.fallbackTaken ?? null);
    return executed === key && Boolean(res.finalResult?.success);
  }

  /** Execute a batch of descent paths IN-PROCESS from a root snapshot, returning
   *  each path's leaf observation. One IPC replaces (paths × pathLen) per-step
   *  round-trips — the core of the in-process search (architecture doc §4.2). */
  runPaths(rootId: number, paths: string[][]): any[] {
    const out: any[] = [];
    for (const keys of paths) {
      this.restore(rootId);
      let invalid = false;
      let failedAtDepth = -1;
      for (let d = 0; d < keys.length; d += 1) {
        if (this.observe0Done()) break;   // stop early if a path hit terminal
        if (!this.executeKey(keys[d]!)) {  // exact execution failed -> path is invalid
          invalid = true;
          failedAtDepth = d;
          break;
        }
      }
      const obs: any = this.observe();
      if (invalid) {
        obs.invalidPath = true;            // search must NOT treat this as a real leaf
        obs.failedAtDepth = failedAtDepth;
      }
      out.push(obs);
    }
    return out;
  }

  /** Execute exactly `key` from the CURRENT lane state and observe — WITHOUT
   *  mutating the public history (like runPaths, so search leaves share the root's
   *  history). Returns the next obs; flags `invalidPath` if the exact key did not
   *  execute. Used by the adaptive full-ISMCTS lane driver (run_infoset). */
  stepExact(key: string): any {
    const ok = this.executeKey(key);
    const obs: any = this.observe();
    if (!ok) obs.invalidPath = true;
    return obs;
  }

  private observe0Done(): boolean {
    const b: any = this.engine.getBoard("playerOne");
    return b.status === "finished";
  }

  private currentActor(): string | undefined {
    const enumr = this.engine.asServer().enumerateAutomatedActionsForCurrentActor({ searchCaps: SEARCH_CAPS });
    return enumr.actorId;
  }

  private idxOf(pid?: string): number {
    return pid === CANONICAL_PLAYER_TWO ? 1 : 0;
  }

  /** Public counts both players can legally see (lore, ink count, hand count). */
  private publicCounts() {
    const b: any = this.engine.getBoard("playerOne");
    const get = (pid: string) => {
      const pb = b.players[pid] ?? {};
      return { lore: pb.lore ?? 0, ink: (pb.inkwell ?? []).length, hand: pb.handCount ?? 0 };
    };
    const a = get(CANON[0]!), c = get(CANON[1]!);
    return { lore: [a.lore, c.lore], ink: [a.ink, c.ink], hand: [a.hand, c.hand],
             turn: b.turnNumber ?? 0, cards: b.cards };
  }

  /** Append one PUBLIC event after a real move. The played-card identity is the
   *  key belief signal (a card left the opponent's hand); inked cards stay hidden
   *  (defId null) — we record only that a hidden card moved hand->inkwell. */
  private pushHistoryEvent(actorId: string | undefined, key: string) {
    const parts = key.split(":");
    const family = parts[0] ?? "";
    const cardId = parts[1] ?? null;
    const pc = this.publicCounts();
    let defId: string | null = null;
    if (family === "playCard" && cardId) {
      const c: any = (pc.cards as any)[cardId];
      if (c && !c.hidden) defId = c.definitionId ?? null;
    }
    this.history.push({
      actor: this.idxOf(actorId), family, defId,
      lore: pc.lore, ink: pc.ink, hand: pc.hand, turn: pc.turn,
    });
    if (this.history.length > HIST_MAX) this.history.shift();
  }

  /** Grammar-gap proof: enumerate the current actor's decision three ways —
   *  capped automation (what the bot uses), uncapped automation (Botcana-style
   *  full enumeration), and the engine's RAW legal move ids (the true grammar) —
   *  so we can measure where each yields legal moves the others miss. */
  grammarProbe(): any {
    const server: any = this.engine.asServer();
    const actor = this.currentActor();
    if (!actor) return { actor: null };
    const fam = (r: any) => Array.from(new Set((r.candidates ?? []).map((c: any) => c.family)));
    // unsupported-shape = a legal move the automation CANNOT represent (true grammar
    // gap); overflow-skip = combinations dropped by a cap; validation-reject = illegal.
    const unsupported = (r: any) => (r.unsupportedSkips ?? [])
      .filter((d: any) => d.kind === "unsupported-shape")
      .map((d: any) => ({ family: d.family, reason: String(d.reason).slice(0, 80) }));
    const overflow = (r: any) => (r.unsupportedSkips ?? []).filter((d: any) => d.kind === "overflow-skip").length;
    const capped = server.enumerateAutomatedActionsForCurrentActor({ searchCaps: SEARCH_CAPS });
    const uncapped = server.enumerateAutomatedActionsForCurrentActor({ searchCaps: BIG_CAPS });
    const board: any = this.engine.getBoard(viewFor(actor));
    return {
      actor, turn: board.turnNumber ?? 0, phase: board.phase ?? null, status: board.status,
      cappedCount: (capped.candidates ?? []).length, cappedFamilies: fam(capped),
      cappedOverflow: overflow(capped),
      uncappedCount: (uncapped.candidates ?? []).length, uncappedFamilies: fam(uncapped),
      uncappedOverflow: overflow(uncapped),
      unsupported: unsupported(uncapped),        // the true grammar gap (if any)
      validationRejects: (uncapped.validationSkips ?? []).length,
    };
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
    const oppSlotSeq: Record<string, number> = {};
    for (const c of Object.values(board.cards) as any[]) {
      // Finding #3: an OPPONENT hidden-zone card must NOT leak its identity into the
      // actor-visible obs["cards"]. The fog projection still tags hidden inkwell/limbo rows
      // with the RAW instance id (project-board.ts buildHiddenCard returns `rawCardId` as `id`,
      // and derives real stats for it), so redact every opponent-owned hidden card to a STABLE
      // slot placeholder carrying only public-allowlisted structure. The real ids remain solely
      // in obs["hidden"] (oppHidden), the privileged witness for the search-only determinization
      // sampler — never the trunk / info-set key. The actor's OWN hidden cards are NOT redacted
      // (the actor legally knows its own inkwell/mulligan identities).
      if (Boolean(c.hidden) && c.ownerId !== selfId) {
        // Finding #4: do NOT trust the projection's `zoneIndex` for placeholder uniqueness — the
        // native projection assigns slot 0 to EVERY hidden limbo card, so duplicate placeholders
        // would collide and overwrite token positions in serialization. Use a per-zone running
        // counter, giving each redacted opponent hidden card a distinct, position-stable slot
        // (counts are conserved across a determinization, so the assignment is invariant to a
        // hidden identity swap).
        const slot = (oppSlotSeq[c.zone] = (oppSlotSeq[c.zone] ?? 0) + 1) - 1;
        // Finding #5: redaction is ZONE-SPECIFIC — only fields PROVEN public for that hidden zone
        // survive. Opponent INKWELL exertion is public (spent ink), so it is forwarded; every
        // other hidden zone (hand / deck / limbo / …) exposes no per-card physical state, so those
        // rows are neutral. Identity (id / definition / name / printed stats / keywords) is always
        // nulled; the real ids live solely in obs["hidden"] for the search-only sampler.
        const isInk = c.zone === "inkwell";
        cards.push({
          id: `oppslot:${c.zone}:${slot}`,
          owner: 1, zone: c.zone, cardType: "unknown",
          cost: 0, strength: 0, willpower: 0, lore: 0, damage: 0,
          exerted: isInk ? Boolean(c.exerted) : false,
          drying: false,
          ready: isInk ? !c.exerted : true,
          keywords: [],
          definitionId: null, name: null, hidden: true,
        });
        continue;
      }
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
        name: c.fullName ?? c.name ?? null,   // human-readable (verbose logging)
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
      selfIdx: this.idxOf(selfId),       // index of `self` in CANON order (history frame)
      history: this.history.slice(-HIST_MAX),
      hidden: this.oppHidden(selfId),
    };
  }

  step(stableKey: string) {
    const server = this.engine.asServer();
    const actor = this.currentActor();   // who is about to move (before execution)
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
    this.pushHistoryEvent(actor, executed ?? stableKey);
    const obs = this.observe();
    return { obs, executed, requested: stableKey, matched: executed === stableKey, success };
  }

  stepAuto(strategyName: string) {
    const strategy = STRATEGIES[strategyName] ?? STRATEGIES.best;
    const server = this.engine.asServer();
    const actor = this.currentActor();   // who is about to move (before execution)
    const res = server.takeAutomatedActionForCurrentActor({ strategy, searchCaps: SEARCH_CAPS });
    const executed = res.selectedCandidate ? candidateKey(res.selectedCandidate) : (res.fallbackTaken ?? null);
    const family = res.selectedCandidate?.family ?? (res.fallbackTaken ?? null);
    this.steps += 1;
    this.pushHistoryEvent(actor, executed ?? `${family}:`);
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
    case "grammar_probe": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, probe: session.grammarProbe() };
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
    // DIAGNOSTIC-ONLY (Tier-A): hand-only determinization, INVALID for clean-label
    // training. Phase 3 adds the full-`World` "determinize_world" op that honors the
    // sampled opponent inkwell/deck + self deck order.
    case "determinize": {
      if (!session) return { ok: false, error: "no session" };
      const self = String(req.self);
      const handIds: string[] = Array.isArray(req.handInstanceIds) ? req.handInstanceIds : [];
      session.determinize(self, handIds, req.seed ? String(req.seed) : undefined);
      return { ok: true, obs: session.observe() };
    }
    // Tier-A Phase 3: full-`World` determinization (clean-label capable). Honors the exact
    // opponent hand/inkwell/deck partition + optional self-deck order with NO ambient
    // randomness; throws (-> ok:false) on any invalid spec; returns the realized partition.
    case "determinize_world": {
      if (!session) return { ok: false, error: "no session" };
      const self = String(req.self);
      const realized = session.determinizeWorld(self, req.world ?? {});
      return { ok: true, obs: session.observe(), realized };
    }
    case "check_consistency": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, consistency: session.checkConsistency() };
    }
    case "protected_facts": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, protectedFacts: session.protectedFacts() };
    }
    case "run_paths": {
      if (!session) return { ok: false, error: "no session" };
      const paths: string[][] = Array.isArray(req.paths) ? req.paths : [];
      return { ok: true, obs: session.runPaths(Number(req.root), paths) };
    }
    case "step_exact": {
      if (!session) return { ok: false, error: "no session" };
      return { ok: true, obs: session.stepExact(String(req.stableKey)) };
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
