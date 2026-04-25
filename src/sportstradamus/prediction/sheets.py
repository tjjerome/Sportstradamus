"""Google Sheets write helpers for the prediction pipeline.

:func:`save_data` writes per-platform offer tables and the All Parlays
sheet.  It is called once per platform inside :func:`main` after
:func:`process_offers` returns.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from sportstradamus.spiderLogger import logger


def save_data(df, parlay_df, book, gc):
    """Write per-platform offers and parlay candidates to Google Sheets.

    Writes filtered offers to the platform's own worksheet, updates the
    ``All Parlays`` aggregate sheet, and seeds the ``Parlay Search`` sheet
    with the highest-EV row.

    Args:
        df: Scored offers DataFrame (internal columns already dropped by caller).
        parlay_df: Parlay candidates DataFrame.
        book: DFS platform name used as the worksheet name.
        gc: Authorized ``gspread.Client`` instance.
    """
    if len(df) > 0:
        try:
            df.sort_values("Model", ascending=False, inplace=True)
            mask = (df.Books > 0.95) & (df.Model > 1.02) & (df.Boost <= 2.5)
            if book == "Underdog":
                df["Boost"] = df["Boost"] / 1.78
            wks = gc.open("Sportstradamus").worksheet(book)
            wks.clear()
            wks.update(
                [
                    df.columns.values.tolist(),
                    *df.loc[mask].values.tolist(),
                    *df.loc[~mask].values.tolist(),
                ]
            )
            wks.set_basic_filter()
            wks.format("J:K", {"numberFormat": {"type": "NUMBER", "pattern": "0.00"}})
            wks.format("N:P", {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})
            wks.update_cell(
                1,
                19,
                "Last Updated: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            )

            wks = gc.open("Sportstradamus").worksheet("All Parlays")
            sheet_df = pd.DataFrame(wks.get_all_records())
            if not sheet_df.empty:
                sheet_df = sheet_df.loc[sheet_df.Platform != book]
            if not parlay_df.empty:
                bet_ranks = parlay_df.groupby(["Platform", "Game", "Family"]).rank("first", False)[
                    ["Model EV", "Rec Bet", "Fun"]
                ]
                parlay_df = parlay_df.join(bet_ranks, rsuffix=" Rank")
                sheet_df = pd.concat([sheet_df, parlay_df]).sort_values("Model EV", ascending=False)

            wks.clear()
            sheet_df = sheet_df.drop(
                columns=["Leg Probs", "Corr Pairs", "Boost Pairs", "Indep P", "Indep PB"],
                errors="ignore",
            )
            sheet_df = sheet_df.replace([np.inf, -np.inf], np.nan).fillna("")
            wks.update([sheet_df.columns.values.tolist(), *sheet_df.values.tolist()])

            if not sheet_df.empty:
                wks = gc.open("Sportstradamus").worksheet("Parlay Search")
                wks.update_cell(1, 5, sheet_df.iloc[0]["Platform"])
                wks.update_cell(2, 5, sheet_df.iloc[0]["League"])
                wks.update_cell(3, 5, sheet_df.iloc[0]["Game"])
                wks.update_cell(4, 5, "Highest EV")
                wks.update_cell(7, 2, 1)
                wks.update_cell(7, 5, 1)
                wks.update_cell(7, 8, 1)
                wks.update_cell(
                    1, 10, "Last Updated: " + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                )

        except Exception:
            logger.exception(f"Error writing {book} offers")
