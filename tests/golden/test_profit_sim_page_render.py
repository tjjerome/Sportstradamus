"""Regression pin for the ``pages/6_…_Stats_Profit_Sim.py`` summary-table block.

The page shipped two latent ``NameError``s in this block: ``range(N_MONTE_CARLO)``
(only ``N_MONTE_CARLO_DEFAULT`` is imported) and a ``summary[...]`` dict that was
never built (a stale ``summarize_runs`` extraction left the inline computation
dead and the dict reference dangling). Compile / import / the archive-lock test
all passed — only an actual Streamlit run hit the error — so this slices the
block out of the page source and exec's it with a stub ``summarize_runs`` to pin
that it runs and formats every column.
"""

import pathlib

import pandas as pd

_PAGE = next(
    (pathlib.Path(__file__).resolve().parents[2] / "src" / "sportstradamus" / "pages").glob(
        "6_*Stats_Profit_Sim.py"
    )
)

_FAKE_SUMMARY = {
    "mean_final": 1234.0,
    "roi": 0.234,
    "max_drawdown": 0.1,
    "sharpe": 1.5,
    "win_rate": 0.55,
}


def test_summary_table_block_builds_rows_via_summarize_runs() -> None:
    src = _PAGE.read_text(encoding="utf-8")
    start = src.index("summary_rows = []")
    end = src.index("\nst.dataframe(summary_df")
    ns = {
        "pd": pd,
        "all_results": {"Moderate": pd.DataFrame({"run": [0], "bankroll": [1234.0]})},
        "initial_bankroll": 1000,
        "summarize_runs": lambda result, initial_bankroll: _FAKE_SUMMARY,
    }
    exec(compile(src[start:end], str(_PAGE), "exec"), ns)

    df = ns["summary_df"]
    assert list(df["Strategy"]) == ["Moderate"]
    assert list(df["Final Bankroll"]) == ["$1,234"]
    assert list(df["ROI"]) == ["+23.4%"]
    assert list(df["Max Drawdown"]) == ["10.0%"]
    assert list(df["Sharpe Ratio"]) == ["1.500"]
    assert list(df["Win% (daily)"]) == ["55.0%"]
