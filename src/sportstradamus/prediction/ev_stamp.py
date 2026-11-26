"""The EV stamp on a platform's scored records: the books' price, the fit's EV, the picks.

``stamp_records`` lays ``book_stamp`` (what the books said at each offer's own line, as this
run saw the archive) and ``live_fit`` (the win chance fitted on settled posted picks) onto
the records, then flags each slate's picks by that EV. The records carry all of it to the
board snapshot and to history (docs/handoffs/profit-engine.md 5.1, 5.3).
"""

import pandas as pd

from sportstradamus import book_stamp, live_fit
from sportstradamus.helpers.io import LIVE_FIT_PATH, read_parquet_safe
from sportstradamus.prediction.model_prob import normalize_market

# What a scoring run stamps on each record: ``book_stamp.stamp``'s columns, ``live_fit.score``'s,
# then the pick flag. ``history_schema.DECISION_COLS`` spells the same eight out; importing
# them from here would be a cycle, as ``live_fit`` imports ``history_schema``.
EV_STAMP_COLS: list[str] = [*book_stamp.COLS, *live_fit.SCORE_COLS, "Pick"]


def stamp_records(records: list[dict], platform: str, scored_at: pd.Timestamp, archive) -> None:
    """Add ``EV_STAMP_COLS`` to each of one platform's scored records, in place.

    The four book columns are ``book_stamp.stamp``'s: the sportsbooks' price at the record's
    own line, for the side its ``Bet`` names. The next three are ``live_fit.score``'s, read
    off the nightly fit table: before the first table exists every offer prices at the app's
    own fair price, and a record with one side posted gets NaN. ``Pick`` is
    ``live_fit.picks``' flag: the players with the highest ``EV`` on each league's game
    date, per pick kind, as many as the slate has places.

    Args:
        records: The platform's scored records (``offer_records.SCORED_RECORD_COLS``), at
            least one.
        platform: The platform that posted them; its ``stat_map`` names their markets.
        scored_at: The run's stamp, which a quote's age is measured from.
        archive: A loaded ``Archive``.
    """
    offers = pd.DataFrame(records)
    # The archive files a ladder under the league's own market key, not the platform's label.
    archive_names = [
        normalize_market(league, label, platform)
        for league, label in zip(offers["League"], offers["Market"], strict=True)
    ]
    books = book_stamp.stamp(offers.assign(Market=archive_names), archive, scored_at)
    posted = (
        offers.join(books)
        .rename(columns={"Boost_Over": "Payout Over", "Boost_Under": "Payout Under"})
        .assign(Platform=platform)
    )
    priced = posted.join(live_fit.score(posted, read_parquet_safe(LIVE_FIT_PATH)))
    stamps = priced.assign(Pick=live_fit.picks(priced))[EV_STAMP_COLS].to_dict("records")
    for record, stamp in zip(records, stamps, strict=True):
        record.update(stamp)
