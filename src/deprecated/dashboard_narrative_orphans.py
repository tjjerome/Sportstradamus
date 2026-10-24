# ARCHIVED 2026-10-09 from src/sportstradamus/dashboard/narrative.py
# Reason: The Tonight card's prophecy line dropped its lead-side arrow (owner call),
#         removing this lookup's only call site.
# Last live SHA: f3713b95
# Original imports (now unresolved here):
#   import pandas as pd
#   from sportstradamus.dashboard.narrative import _story_rows


def game_lead_side(stories: pd.DataFrame, *, game: str, date, platform: str | None = None) -> str:
    """The lead story's ``lead_side`` for one ``(Game, Date)``, optionally scoped to ``platform``.

    The max-``model_ev`` lead row's side when both platforms lead the game on different
    sides; ``""`` when no story is marked lead for the key, or there are no stories.
    """
    rows = _story_rows(stories, game, date, platform)
    if rows.empty:
        return ""
    lead_rows = rows.loc[rows["lead"]]
    if lead_rows.empty:
        return ""
    side = lead_rows.loc[lead_rows["model_ev"].idxmax(), "lead_side"]
    return str(side) if pd.notna(side) else ""
