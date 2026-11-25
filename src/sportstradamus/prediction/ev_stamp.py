"""The EV stamp on a platform's scored records: the sportsbooks' price and the fit's EV.

``stamp_records`` lays ``book_stamp`` (what the books said at each offer's own line, as this
run saw the archive) and ``live_fit`` (the win chance fitted on settled posted picks) onto
the records, which carry both to the board snapshot and to history
(docs/handoffs/profit-engine.md 5.1).
"""

import pandas as pd

from sportstradamus import book_stamp, live_fit
from sportstradamus.helpers.io import LIVE_FIT_PATH, read_parquet_safe
from sportstradamus.prediction.model_prob import normalize_market

# What a scoring run stamps on each record: ``book_stamp.stamp``'s columns, then
# ``live_fit.score``'s. ``history_schema.DECISION_COLS`` spells the same seven out.
EV_STAMP_COLS: list[str] = [*book_stamp.COLS, "Fit Prob", "Fit Prob Low", "EV"]


def stamp_records(records: list[dict], platform: str, scored_at: pd.Timestamp, archive) -> None:
    """Add ``EV_STAMP_COLS`` to each of one platform's scored records, in place.

    The four book columns are ``book_stamp.stamp``'s: the sportsbooks' price at the record's
    own line, for the side its ``Bet`` names. The other three are ``live_fit.score``'s, read
    off the nightly fit table: before the first table exists every offer prices at the app's
    own fair price, and a record with one side posted gets NaN.

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
    fit = live_fit.score(
        offers.join(books)
        .rename(columns={"Boost_Over": "Payout Over", "Boost_Under": "Payout Under"})
        .assign(Platform=platform),
        read_parquet_safe(LIVE_FIT_PATH),
    )
    stamps = books.join(fit)[EV_STAMP_COLS].to_dict("records")
    for record, stamp in zip(records, stamps, strict=True):
        record.update(stamp)
