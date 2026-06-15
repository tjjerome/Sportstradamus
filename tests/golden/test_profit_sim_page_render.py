"""Regression pin for the profit-sim summary-table block.

The block shipped two latent ``NameError``s once (``range(N_MONTE_CARLO)`` and a
``summary[...]`` dict that was never built). Compile / import / the archive-lock test all
passed — only an actual Streamlit run hit the error — so this slices the
``summary_rows`` → ``summary_df`` block out of the source and exec's it with a stub
``summarize_runs`` to pin that it runs and formats every column.

The block now lives in ``components/profit_sim.py:_render_summary_table`` (P7 folded the
standalone Profit Sim page into Receipts); the slice is dedented to module level before exec.
"""

import pathlib
import textwrap

import pandas as pd

_COMPONENT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "sportstradamus"
    / "dashboard"
    / "components"
    / "profit_sim.py"
)

_FAKE_SUMMARY = {
    "mean_final": 1234.0,
    "roi": 0.234,
    "max_drawdown": 0.1,
    "sharpe": 1.5,
    "win_rate": 0.55,
}


def test_summary_table_block_builds_rows_via_summarize_runs() -> None:
    src = _COMPONENT.read_text(encoding="utf-8")
    start = src.rindex("\n", 0, src.index("summary_rows = []")) + 1
    end = src.index("\n", src.index("summary_df = pd.DataFrame(summary_rows)")) + 1
    block = textwrap.dedent(src[start:end])
    ns = {
        "pd": pd,
        "all_results": {"Moderate": pd.DataFrame({"run": [0], "bankroll": [1234.0]})},
        "initial_bankroll": 1000,
        "summarize_runs": lambda result, initial_bankroll: _FAKE_SUMMARY,
    }
    exec(compile(block, str(_COMPONENT), "exec"), ns)

    df = ns["summary_df"]
    assert list(df["Strategy"]) == ["Moderate"]
    assert list(df["Final Bankroll"]) == ["$1,234"]
    assert list(df["ROI"]) == ["+23.4%"]
    assert list(df["Max Drawdown"]) == ["10.0%"]
    assert list(df["Sharpe Ratio"]) == ["1.500"]
    assert list(df["Win% (daily)"]) == ["55.0%"]
