"""The sportsbooks' price at an offer's own line, as the offer's scoring run saw the archive.

``book_inputs`` reads it off one slate's ladder and names the offer's class of sportsbook
coverage, the stratum the EV fit prices it in (``live_fit``; docs/handoffs/profit-engine.md
5.1). ``stamp`` does that for one scoring run's offers, and ``fill_history`` for the history
rows a run did not stamp. Each takes a loaded ``Archive``; none opens one.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus.helpers.archive import archive_market
from sportstradamus.helpers.training_quotes import DFS_PLATFORM_BOOKS
from sportstradamus.realized import COUNT_ONCE_FROM

# A sportsbook quote older than this at scoring counts for nothing: the weights are earned on
# fresh quotes.
BOOK_MAX_AGE = pd.Timedelta(minutes=90)

# What a run stamps on each offer: the books' chance of the side the offer took, how many
# price its line, its class of coverage and the newest of those quotes.
COLS: list[str] = ["Book Prob", "Book Count", "Book Class", "Book Observed At"]

# A run stamps ``Scored At`` as it starts and reads the archive as it stood when it took the
# lock. Quotes stamped inside that last minute were committed after the lock, unless the run's
# own served quotes show it read them (docs/archive/researcher_ev_fit.md, on visibility).
_SAME_BATCH = pd.Timedelta(seconds=60)


def book_inputs(
    rungs: pd.DataFrame, players: Sequence[str], lines: np.ndarray, scored_at: pd.Timestamp
) -> pd.DataFrame:
    """What the sportsbooks said about each offer, as its scoring run saw them.

    ``rungs`` is ``Archive.get_ladder_rungs`` output for the offers' game date, sportsbooks
    only. A quote older than ``BOOK_MAX_AGE`` at ``scored_at`` counts for nothing.

    Returns:
        Positional over ``players`` and ``lines``: ``Book Over`` (the plain mean of the books'
        over-probabilities at the offer's own line, NaN when none prices it), ``Book Count``,
        ``Book Observed At`` (the newest of those quotes) and ``Book Class``: ``A`` when a
        book prices the line, ``B`` when the books price only other lines, ``C`` when none
        quotes the player.
    """
    offers = pd.DataFrame({"entity": players, "offered": lines})
    fresh = rungs[scored_at - rungs["observed_at"] <= BOOK_MAX_AGE]
    quoted = offers.reset_index().merge(fresh, on="entity")
    at_line = (
        # The cast is for the archive's empty read, whose columns carry no type.
        quoted[np.isclose(quoted["line"].astype(float), quoted["offered"])]
        .groupby("index")
        .agg(over=("p_over", "mean"), count=("book", "size"), newest=("observed_at", "max"))
        .reindex(offers.index)
    )
    count = at_line["count"].fillna(0).astype(int)
    return pd.DataFrame(
        {
            "Book Over": at_line["over"],
            "Book Count": count,
            "Book Observed At": pd.to_datetime(at_line["newest"]),
            "Book Class": np.select(
                [count > 0, offers.index.isin(quoted["index"])], ["A", "B"], "C"
            ),
        }
    )


def stamp(
    offers: pd.DataFrame, archive, scored_at: pd.Timestamp, *, at: pd.Timestamp | None = None
) -> pd.DataFrame:
    """``COLS`` for one scoring run's offers, read off each slate's ladder.

    Args:
        offers: ``League``, ``Market`` (the archive's name for it), ``Date``, ``Player``,
            ``Line`` and ``Bet``.
        archive: A loaded ``Archive``.
        scored_at: The run's stamp, which a quote's age is measured from.
        at: The newest quote stamp the run could read, for a run replayed later; None reads
            the archive as it stands.

    Returns:
        ``COLS`` on ``offers``' index, ``Book Prob`` on the side ``Bet`` names.
    """
    slates = []
    for (league, market, day), slate in offers.groupby(["League", "Market", "Date"]):
        rungs = archive.get_ladder_rungs(
            league,
            market,
            day,
            sorted(slate["Player"].unique()),
            exclude_books=DFS_PLATFORM_BOOKS,
            at=at,
        )
        inputs = book_inputs(rungs, slate["Player"].to_numpy(), slate["Line"].to_numpy(), scored_at)
        slates.append(inputs.set_axis(slate.index))
    books = pd.concat(slates).reindex(offers.index)
    over = offers["Bet"].eq("Over")
    books["Book Prob"] = books["Book Over"].where(over, 1 - books["Book Over"])
    return books[COLS]


def fill_history(history: pd.DataFrame, archive) -> pd.DataFrame:
    """History with ``COLS`` filled on the rows no run stamped.

    A row scored before the stamp existed is priced from the archive as its run could read
    it: quotes stamped up to ``Scored At`` when the newest quote the run served is under
    ``_SAME_BATCH`` old, else up to ``_SAME_BATCH`` before it. A row dated before
    ``COUNT_ONCE_FROM`` or with no ``Scored At`` stays unstamped: no fit reads it.
    """
    stamped = history.reindex(columns=COLS)
    pending = history[
        stamped["Book Class"].isna() & (pd.to_datetime(history["Date"]) >= COUNT_ONCE_FROM)
    ]
    newest_quote = history.groupby("Scored At")["Quote Observed At"].max()
    fills = []
    # A row with no ``Scored At`` falls out here: groupby drops the missing key.
    for scored_at, run in tqdm(pending.groupby("Scored At"), desc="Book stamps", unit="run"):
        saw_batch = scored_at - newest_quote[scored_at] < _SAME_BATCH
        offers = run.assign(Market=list(map(archive_market, run["League"], run["Market"])))
        cut = scored_at if saw_batch else scored_at - _SAME_BATCH
        fills.append(stamp(offers, archive, scored_at, at=cut))
    if not fills:
        return history
    filled = pd.concat(fills).reindex(history.index)
    run_stamped = stamped["Book Class"].notna()
    return history.assign(**{col: filled[col].mask(run_stamped, stamped[col]) for col in COLS})
