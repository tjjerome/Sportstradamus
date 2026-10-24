"""Offer rows and star keys shared by the constellation figure tests: one NBA game, one line."""

from __future__ import annotations

GAME = "NYK/SAS"


def offer_row(
    player: str,
    team: str,
    kelly: float,
    *,
    market: str = "PTS",
    game: str = GAME,
    bet: str = "Over",
) -> dict:
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": 10.5,
        "Game": game,
        "League": "NBA",
        "Team": team,
        "Kelly": kelly,
        "Win Prob": 0.6,
        "Boost": 1.5,
    }


def star_key(player: str, market: str = "PTS", bet: str = "Over") -> str:
    return f"{player}|{market}|{bet}"
