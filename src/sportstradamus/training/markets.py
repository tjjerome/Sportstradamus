"""Per-league market definitions and selection helpers for the training pipeline."""

import click

ALL_MARKETS: dict[str, list[str]] = {
    "NFL": [
        "targets",
        "carries",
        "attempts",
        "passing yards",
        "rushing yards",
        "receiving yards",
        "yards",
        "qb yards",
        "fantasy points prizepicks",
        "fantasy points underdog",
        "passing tds",
        "tds",
        "rushing tds",
        "receiving tds",
        "qb tds",
        "completions",
        "receptions",
        "interceptions",
        "sacks taken",
        "passing first downs",
    ],
    "NBA": [
        "MIN",
        "PTS",
        "REB",
        "AST",
        "PRA",
        "PR",
        "RA",
        "PA",
        "FG3M",
        "fantasy points prizepicks",
        "FG3A",
        "FTM",
        "FGM",
        "FGA",
        "STL",
        "BLK",
        "BLST",
        "TOV",
        "OREB",
        "DREB",
        "PF",
    ],
    "WNBA": [
        "MIN",
        "AST",
        "FG3M",
        "PA",
        "PR",
        "PTS",
        "RA",
        "REB",
        "OREB",
        "DREB",
        "FGA",
        "BLK",
        "STL",
        "BLST",
        "TOV",
        "FTM",
        "PRA",
        "fantasy points prizepicks",
    ],
    "MLB": [
        "pitches thrown",
        "pitching outs",
        "pitcher strikeouts",
        "hits allowed",
        "runs allowed",
        "walks allowed",
        "hitter fantasy points underdog",
        "pitcher fantasy points underdog",
        "hits+runs+rbi",
        "total bases",
        "walks",
        "stolen bases",
        "hits",
        "runs",
        "rbi",
        "batter strikeouts",
        "singles",
        "doubles",
        "home runs",
    ],
    "NHL": [
        "timeOnIce",
        "shotsAgainst",
        "saves",
        "shots",
        "points",
        "goalsAgainst",
        "goalie fantasy points underdog",
        "skater fantasy points underdog",
        "blocked",
        "powerPlayPoints",
        "sogBS",
        "hits",
        "goals",
        "assists",
        "faceOffWins",
    ],
}


def select_markets(
    active_markets: dict[str, list[str]],
    market_arg: str | None,
) -> dict[str, list[str]]:
    """Narrow each active league's market list to the stems requested via --market.

    Backs the ``meditate --market`` option: trains a chosen subset instead of a
    whole league's market list.

    Args:
        active_markets: Mapping of league -> ordered market stem list (the
            registry slice already narrowed by --league).
        market_arg: Comma-separated market stems, or None for no filtering.
            Whitespace around each stem is stripped and empty tokens dropped.

    Returns:
        A mapping with the same key set as ``active_markets``. Each league's
        list contains only the requested stems that appear in that league,
        preserving the registry's original order. When ``market_arg`` is None
        the input mapping is returned unchanged.

    Raises:
        click.UsageError: If a requested stem is absent from every active
            league (a typo guard). A stem present in some leagues but not
            others is not an error — it simply trains where it exists.
    """
    if market_arg is None:
        return active_markets

    requested = [stem.strip() for stem in market_arg.split(",") if stem.strip()]
    known_stems = {stem for stems in active_markets.values() for stem in stems}
    unknown = [stem for stem in requested if stem not in known_stems]
    if unknown:
        raise click.UsageError(
            f"Unknown market(s) {unknown!r}. "
            f"Valid stems for active league(s): {sorted(known_stems)!r}"
        )

    requested_set = set(requested)
    return {
        league: [stem for stem in stems if stem in requested_set]
        for league, stems in active_markets.items()
    }
