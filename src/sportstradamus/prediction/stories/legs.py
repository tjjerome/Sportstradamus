"""Parlay-leg parsing for the prophecy-prose package.

Leg string format is ``"{Player} {Bet} {Line} {Market} - {Model P}%, {Boost}x"``
(built in ``prediction/correlation.py``).
"""


def parse_leg(leg: str) -> dict | None:
    """Splitting on ``" Over "`` / ``" Under "`` yields the player name even when
    it contains spaces.  Returns ``None`` for anything that does not parse.
    """
    if not isinstance(leg, str) or not leg.strip():
        return None
    head = leg.split(" - ", 1)[0].strip()
    for bet in ("Over", "Under"):
        token = f" {bet} "
        if token not in head:
            continue
        player, rest = head.split(token, 1)
        rest = rest.strip().split()
        if not rest:
            return None
        try:
            line = float(rest[0])
        except ValueError:
            return None
        market = " ".join(rest[1:]).strip()
        if not player.strip() or not market:
            return None
        return {"Player": player.strip(), "Bet": bet, "Line": line, "Market": market}
    return None
