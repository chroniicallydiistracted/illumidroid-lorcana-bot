"""Tier-A #2 — real-engine `LaneSimulator` for full ISMCTS.

Drives `run_infoset` against the Bun engine using the existing bridge primitives:
each simulation `begin_lane(world)` restores the root snapshot and determinizes the
sampled hidden world (leak-free — the #1 fix randomizes opponent inkwell + both
deck orders), then `step(action_id)` executes exactly that stable key WITHOUT
mutating the engine's public history (`step_exact`), so leaves share the root's
history. The actor-filtered, leak-free `infoSetKey` is computed in Python from the
filtered observation (never from `obs["hidden"]`) plus an observer-projected branch
history; the logical `actionId` is the engine `stableKey`.

This is the correctness-first lane driver (one IPC per transition). The batched
`begin_ismcts_wave`/`step_ismcts_lanes` RPC optimization (spec §9/§16) is a later
performance pass; it must preserve this adaptive per-world selection.

DIAGNOSTIC-ONLY until Tier-A Phases 3/7 land: `begin_lane` currently calls the
HAND-ONLY `engine.determinize` RPC, which discards the sampled world's opponent
inkwell / opponent deck / self deck (Phase 3 adds `determinize_world`; Phase 7 passes
the full `World`). A hand-only determinization is NOT a valid clean-label search, so it
is fail-closed by default: a caller must explicitly opt in with
`hand_only_diagnostic=True` (this is the "explicitly marked invalid for clean-label"
gate the Phase 1 completion criterion requires, at the search/determinization boundary
rather than only at the training sample-writer).
"""

from __future__ import annotations

from search.infoset import info_set_key, history_event


class EngineSimulator:
    def __init__(self, engine, root_snap: int, self_id: str, *,
                 hand_only_diagnostic: bool = False) -> None:
        self.engine = engine
        self.root_snap = root_snap
        self.self_id = self_id
        self.hand_only_diagnostic = bool(hand_only_diagnostic)
        self.branch: list = []
        self.cur: dict = {}

    def _with_key(self, obs: dict, invalid: bool = False) -> dict:
        obs["infoSetKey"] = info_set_key(obs, self.branch)
        if invalid:
            obs["invalid"] = True
        self.cur = obs
        return obs

    def begin_lane(self, world) -> dict:
        if not self.hand_only_diagnostic:
            # fail-closed: the hand-only determinization path discards opponent
            # inkwell/deck + self deck and is invalid for clean-label ISMCTS until
            # Phases 3/7 wire full-world determinization (determinize_world).
            from training.tier_a_guard import TierARemediationIncompleteError
            raise TierARemediationIncompleteError(
                "EngineSimulator.begin_lane uses the HAND-ONLY determinization RPC, which "
                "discards the sampled world's opponent inkwell/deck and self deck. This is "
                "not valid for clean-label ISMCTS; Tier-A Phases 3/7 add full-world "
                "determinize_world. Pass hand_only_diagnostic=True to use the diagnostic path."
            )
        self.engine.restore(self.root_snap)
        obs = self.engine.determinize(self.self_id, list(world.opponent_hand_ids),
                                      seed=(world.seed or None))
        self.branch = []
        return self._with_key(obs)

    def step(self, action_id: str) -> dict:
        # observer-projected branch event from the CURRENT actor (leak-free; face-down
        # inking is redacted by history_event) — appended for perfect-recall keying.
        parts = action_id.split(":")
        seat = int(self.cur.get("selfIdx", 0))
        self.branch = self.branch + [history_event(seat, parts[0], parts[1] if len(parts) > 1 else None)]
        obs = self.engine.step_exact(action_id)
        return self._with_key(obs, invalid=bool(obs.get("invalidPath")))

    def end(self) -> None:
        self.engine.restore(self.root_snap)
