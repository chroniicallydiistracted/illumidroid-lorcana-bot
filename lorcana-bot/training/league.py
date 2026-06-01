"""Phase 3 — league players + registry (PFSP opponent sampling + Elo).

A `Player` advances one decision of a game on the live engine and (if it is a
*learning* seat) returns the data needed to build a training `Sample`. The
`League` holds the population — the trainable main, frozen past mains, permanent
scripted anchors, and exploiters — tracks pairwise results, samples opponents by
**prioritized fictitious self-play (PFSP)**, and maintains **Elo** ratings.
See architecture doc §5.1.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from engine.serialization import encode_obs, encode_belief
from search.ismcts import BISMCTS, SearchConfig
from search.evaluator import NetEvaluator, BeliefEvaluator

SEAT_ONE = "player_one"
SEAT_TWO = "player_two"

# raw record captured at a learning decision (z filled in after the match)
Raw = tuple  # (enc: dict, pi: np.ndarray, belief: dict, actor: str)


class Player:
    """Base player. `act` chooses + executes one decision; returns
    (new_obs, raw|None) where raw is captured only for learning seats."""

    def __init__(self, player_id: str, record: bool = False) -> None:
        self.id = player_id
        self.record = record

    def act(self, engine, obs: dict) -> tuple[dict, Raw | None]:
        raise NotImplementedError


class ScriptedPlayer(Player):
    """A permanent league anchor driven by the shipped automation strategy."""

    def __init__(self, player_id: str, strategy: str = "best") -> None:
        super().__init__(player_id, record=False)
        self.strategy = strategy

    def act(self, engine, obs: dict) -> tuple[dict, Raw | None]:
        r = engine.step_auto(self.strategy)
        return r["obs"], None


class NetPlayer(Player):
    """A neural agent acting via B-ISMCTS (Phase 1 `run` or Phase 2
    `run_belief`). Records (I, π_MCTS, belief, actor) when `record` is set."""

    def __init__(self, player_id: str, net, cfg: SearchConfig | None = None,
                 record: bool = False, use_belief: bool = True, n_worlds: int = 6,
                 greedy: bool = False, rng: np.random.Generator | None = None) -> None:
        super().__init__(player_id, record=record)
        self.net = net
        self.cfg = cfg or SearchConfig(simulations=16, depth_limit=4)
        self.use_belief = use_belief
        self.n_worlds = n_worlds
        self.greedy = greedy
        self.rng = rng or np.random.default_rng()

    def act(self, engine, obs: dict) -> tuple[dict, Raw | None]:
        legal = obs.get("legal", [])
        actor = obs.get("actor")
        mcts = BISMCTS(engine, NetEvaluator(self.net), self.cfg, self.rng)
        if self.use_belief:
            res = mcts.run_belief(obs, BeliefEvaluator(self.net), n_worlds=self.n_worlds,
                                  sims_per_world=self.cfg.simulations)
        else:
            res = mcts.run(obs)
        if res.pi.sum() <= 0:
            a = 0
        elif self.greedy:
            a = int(res.pi.argmax())
        else:
            a = int(self.rng.choice(len(res.pi), p=res.pi))
        new_obs = engine.step(legal[a]["stableKey"])["obs"]
        raw: Raw | None = None
        if self.record and actor is not None and len(res.pi) == len(legal):
            raw = (encode_obs(obs), res.pi.copy(), encode_belief(obs), actor)
        return new_obs, raw


def frozen_net_copy(net):
    """Deep-copy a net and freeze it (for a league snapshot / opponent)."""
    clone = copy.deepcopy(net)
    for p in clone.parameters():
        p.requires_grad_(False)
    clone.eval()
    return clone


@dataclass
class _PairStat:
    games: int = 0
    wins_a: int = 0  # wins for the lexicographically-first id in the key
    wins_b: int = 0  # wins for the second id (draws = games - wins_a - wins_b)


class League:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.players: dict[str, Player] = {}
        self.kind: dict[str, str] = {}            # id -> "main"|"past"|"anchor"|"exploiter"
        self.elo: dict[str, float] = {}
        self._pairs: dict[tuple[str, str], _PairStat] = {}
        self.rng = rng or np.random.default_rng()

    def register(self, player: Player, kind: str) -> None:
        self.players[player.id] = player
        self.kind[player.id] = kind
        self.elo.setdefault(player.id, 1000.0)

    def opponents_for(self, main_id: str) -> list[str]:
        return [pid for pid in self.players if pid != main_id]

    # -- results / Elo --------------------------------------------------------
    def _key(self, a: str, b: str) -> tuple[tuple[str, str], bool]:
        return ((a, b), True) if a <= b else ((b, a), False)

    def record_result(self, a_id: str, b_id: str, winner_id: str | None) -> None:
        key, a_first = self._key(a_id, b_id)
        st = self._pairs.setdefault(key, _PairStat())
        st.games += 1
        if winner_id is not None:                 # draws increment neither (counted as 0.5 later)
            if winner_id == key[0]:
                st.wins_a += 1
            else:
                st.wins_b += 1
        self._update_elo(a_id, b_id, winner_id)

    def _update_elo(self, a: str, b: str, winner: str | None, k: float = 32.0) -> None:
        ra, rb = self.elo[a], self.elo[b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        sa = 0.5 if winner is None else (1.0 if winner == a else 0.0)
        self.elo[a] = ra + k * (sa - ea)
        self.elo[b] = rb + k * ((1.0 - sa) - (1.0 - ea))

    def win_prob(self, main_id: str, opp_id: str, prior: float = 0.5) -> float:
        """Estimated P(main beats opp) from recorded results (prior 0.5)."""
        key, main_first = self._key(main_id, opp_id)
        st = self._pairs.get(key)
        if st is None or st.games == 0:
            return prior
        wins_main = st.wins_a if main_first else st.wins_b
        draws = st.games - st.wins_a - st.wins_b
        return (wins_main + 0.5 * draws + prior) / (st.games + 1.0)

    # -- PFSP -----------------------------------------------------------------
    def pfsp_weights(self, main_id: str, candidates: list[str],
                     mode: str = "hard", floor: float = 0.05) -> np.ndarray:
        w = np.zeros(len(candidates), dtype=np.float64)
        for i, opp in enumerate(candidates):
            p = self.win_prob(main_id, opp)
            if mode == "even":      # weight ~ informativeness (close matchups)
                val = p * (1.0 - p)
            else:                    # "hard": focus where we lose / are even
                val = (1.0 - p)
            w[i] = max(val, floor)
        s = w.sum()
        return w / s if s > 0 else np.full(len(candidates), 1.0 / len(candidates))

    def pfsp_sample(self, main_id: str, mode: str = "hard") -> Player | None:
        cands = self.opponents_for(main_id)
        if not cands:
            return None
        w = self.pfsp_weights(main_id, cands, mode)
        return self.players[cands[int(self.rng.choice(len(cands), p=w))]]
