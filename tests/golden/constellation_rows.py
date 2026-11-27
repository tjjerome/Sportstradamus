"""Offer rows and star keys shared by the constellation figure tests: one NBA game, one line."""

from __future__ import annotations

GAME = "NYK/SAS"
BOOST = 1.5


def offer_row(
    player: str,
    team: str,
    ev: float | None,
    *,
    market: str = "PTS",
    game: str = GAME,
    bet: str = "Over",
) -> dict:
    """An offer at ``ev``; ``None`` is a leg the fit did not price.

    ``Kelly`` is 0 on every row and no row carries ``Pick``: the map reads the fitted EV,
    never the model's Kelly or the board's slate picks, so a map drawn off either comes
    out empty.
    """
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": 10.5,
        "Game": game,
        "League": "NBA",
        "Team": team,
        "Kelly": 0.0,
        "Win Prob": 0.6,
        "Boost": BOOST,
        "Fit Prob": None if ev is None else ev / BOOST,
        "EV": ev,
    }


def star_key(player: str, market: str = "PTS", bet: str = "Over") -> str:
    return f"{player}|{market}|{bet}"
