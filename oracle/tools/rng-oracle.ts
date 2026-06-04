/**
 * Faithful replication of the frozen oracle RNG, for generating golden vectors.
 *
 * Mirrors `lorcana-engine/src/core/runtime/match-runtime.random-apis.ts` exactly:
 *
 *   const random = () => {
 *     draft.ctx.random.draws++;                 // increment BEFORE computing
 *     return seedrandom(`${seed}:${draws}`)();
 *   };
 *   shuffle = (arr) => { result=[...arr];
 *     for (i=len-1; i>0; i--) { j=floor(random()*(i+1)); swap(result[i],result[j]); }
 *     return result; };
 *
 * The `seedrandom` factory is injected so the caller controls the exact (pinned)
 * version. This module imports no oracle gameplay code.
 */

export type SeedrandomFactory = (seed: string) => () => number;

/** A draws-counting random source bound to a seed, matching the oracle draft. */
export function makeOracleRandom(seedrandom: SeedrandomFactory, seed: string, startDraws = 0) {
  let draws = startDraws;
  const random = (): number => {
    draws++;
    return seedrandom(`${seed}:${draws}`)();
  };
  return { random, getDraws: () => draws };
}

/** The float the oracle returns when its draw counter reaches `draws`. */
export function oracleRandomAt(seedrandom: SeedrandomFactory, seed: string, draws: number): number {
  return seedrandom(`${seed}:${draws}`)();
}

export interface RandomSample {
  seed: string;
  draws: number;
  value: number;
}

/** Values for draws = 1..count (counter starts at 0, pre-incremented). */
export function oracleRandomSequence(
  seedrandom: SeedrandomFactory,
  seed: string,
  count: number,
): RandomSample[] {
  const out: RandomSample[] = [];
  for (let draws = 1; draws <= count; draws++) {
    out.push({ seed, draws, value: oracleRandomAt(seedrandom, seed, draws) });
  }
  return out;
}

export interface ShuffleSample {
  seed: string;
  startDraws: number;
  input: number[];
  output: number[];
  drawsAfter: number;
}

export function oracleShuffle(
  seedrandom: SeedrandomFactory,
  seed: string,
  array: number[],
  startDraws = 0,
): ShuffleSample {
  const { random, getDraws } = makeOracleRandom(seedrandom, seed, startDraws);
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return { seed, startDraws, input: array, output: result, drawsAfter: getDraws() };
}
