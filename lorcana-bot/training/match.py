"""Phase 3 — match routing: run one game between two (possibly different) policies.

Each decision is routed to the seat whose turn it is; the observation is already
from that actor's information set, so each Player reasons from its own POV. A
learning seat's decisions are collected and turned into training `Sample`s with
the outcome z assigned from that seat's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from training.league import Player, SEAT_ONE, SEAT_TWO
from training.learner import Sample


@dataclass
class MatchResult:
    winner_seat: str | None                       # "player_one" | "player_two" | None
    seat_to_id: dict[str, str]                    # seat -> player id
    samples: dict[str, list[Sample]] = field(default_factory=dict)  # player id -> samples
    steps: int = 0

    @property
    def winner_id(self) -> str | None:
        return self.seat_to_id.get(self.winner_seat) if self.winner_seat else None


def play_match(engine, seat_players: dict[str, Player], seed: str = "m",
               deck_p1: str | None = None, deck_p2: str | None = None,
               max_steps: int = 400) -> MatchResult:
    """seat_players maps SEAT_ONE/SEAT_TWO -> Player. Returns the winner and each
    learning seat's samples."""
    obs = engine.reset(seed, deck_p1, deck_p2)
    raws: dict[str, list[tuple]] = {p.id: [] for p in seat_players.values()}
    steps = 0
    while not obs.get("done") and steps < max_steps and obs.get("legal"):
        actor = obs.get("actor")
        player = seat_players.get(actor)
        if player is None:
            break
        new_obs, raw = player.act(engine, obs)
        if raw is not None:
            raws[player.id].append(raw)
        obs = new_obs
        steps += 1

    winner = obs.get("winner")
    samples: dict[str, list[Sample]] = {}
    for pid, items in raws.items():
        out: list[Sample] = []
        for enc, pi, belief, actor in items:
            z = 0.0 if winner is None else (1.0 if str(winner) == str(actor) else -1.0)
            out.append(Sample(enc=enc, pi=pi, z=z, belief=belief))
        samples[pid] = out

    return MatchResult(
        winner_seat=str(winner) if winner is not None else None,
        seat_to_id={SEAT_ONE: seat_players[SEAT_ONE].id, SEAT_TWO: seat_players[SEAT_TWO].id},
        samples=samples,
        steps=steps,
    )
