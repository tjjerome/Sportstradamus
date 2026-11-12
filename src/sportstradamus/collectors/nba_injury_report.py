"""The newest official NBA injury report, parsed, with its publication time.

The league posts a PDF every quarter hour (ET) at a URL that carries the stamp; a stamp not yet
posted answers 403 (the bucket denies a missing key rather than answering 404). Every
``prophecize`` run with the NBA live probes the last ``PROBE_SLOTS`` quarter-hours (from
``collectors.lineup_capture``), keeps the first that answers and is not stored yet, writes its
rows to ``NBA_INJURY_DIR/reports/date=<ET date>/<stamp>.parquet`` and the PDF itself to
``NBA_INJURY_DIR/pdf/<stamp>.pdf``, so a decision time joins to the latest report stored at or
before it (``fetched_at``). Licence: NBA.com terms.
"""

from __future__ import annotations

import io
import re
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from pypdf import PdfReader

from sportstradamus.helpers.io import NBA_INJURY_DIR, _atomic_write_parquet, read_upcoming_events
from sportstradamus.helpers.scraping import BROWSER_USER_AGENT, CAPTURE_TIMEOUT_S
from sportstradamus.spiderLogger import logger

REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
# Reports are posted on the quarter hour, Eastern; the file name carries that clock.
PUBLISH_STEP = timedelta(minutes=15)
# Quarter-hours probed backward before a run concludes nothing new is posted (two hours).
PROBE_SLOTS = 8
_ET = ZoneInfo("America/New_York")
REPORT_DTYPES = {
    "report_stamp": "string",
    "last_modified": "string",
    "fetched_at": "datetime64[ns]",
    "game_date": "string",
    "game_time": "string",
    "matchup": "string",
    "team": "string",
    "player": "string",
    "status": "string",
    "reason": "string",
}
# The thirty teams as the report spells them: it writes "LA Clippers", so neither nba_api's
# full names nor abbreviations.json are a drop-in. A renamed or expansion team does not fail
# the parse; it surfaces as a ``player`` value beginning with the team's name.
REPORT_TEAMS = (
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "LA Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
)
_STATUSES = "Out|Questionable|Doubtful|Probable|Available"
_TEAM = "|".join(re.escape(team) for team in sorted(REPORT_TEAMS, key=len, reverse=True))
# A row: optional date, time and matchup, then the team (carried forward when absent), the
# player as "Last, First", the status word, the reason. A team that has not filed yet ends in
# NOT YET SUBMITTED instead: no row, but its game heads the rows that follow.
_ROW = re.compile(
    r"^(?:(?P<game_date>\d{2}/\d{2}/\d{4}) )?(?:(?P<game_time>\d{2}:\d{2} \(ET\)) )?"
    rf"(?:(?P<matchup>[A-Z]{{3}}@[A-Z]{{3}}) )?(?:(?P<team>{_TEAM}) )?"
    rf"(?:(?P<player>.+?, \S+(?: \S+)*?) (?P<status>{_STATUSES})(?: (?P<reason>.*))?"
    r"|NOT YET SUBMITTED)$"
)


def candidate_stamps(now: datetime) -> list[str]:
    """The newest ``PROBE_SLOTS`` quarter-hour stamps at or before ``now``, newest first."""
    clock = now.astimezone(_ET).replace(second=0, microsecond=0)
    slot = clock - (timedelta(minutes=clock.minute) % PUBLISH_STEP)
    return [(slot - i * PUBLISH_STEP).strftime("%Y-%m-%d_%I_%M%p") for i in range(PROBE_SLOTS)]


def parse_lines(lines: list[str]) -> list[dict]:
    """Rows of one report from its text lines; game, time, matchup and team carry forward."""
    rows, carried = [], {"game_date": None, "game_time": None, "matchup": None, "team": None}
    for text in lines:
        match = _ROW.match(text.strip())
        if match is None:
            continue
        fields = match.groupdict()
        carried |= {key: fields[key] for key in carried if fields[key]}
        if fields["player"] is None:
            continue
        rows.append(
            {
                **carried,
                "player": fields["player"],
                "status": fields["status"],
                "reason": fields["reason"] or "",
            }
        )
    return rows


def pdf_lines(raw: bytes) -> list[str]:
    """The report PDF's text lines, page by page; pypdf yields one line per table row."""
    return [
        line
        for page in PdfReader(io.BytesIO(raw)).pages
        for line in page.extract_text().splitlines()
    ]


def capture_injury_report(fetched_at: pd.Timestamp) -> None:
    """Store the newest official NBA injury report not yet on disk.

    ``fetched_at`` is the run's ``Scored At``, naive UTC.

    Raises:
        RuntimeError: Nothing answered in the probe window while an NBA game sits on
            ``upcoming_events.json``.
    """
    stamps = candidate_stamps(fetched_at.tz_localize(UTC))
    for stamp in stamps:
        day = stamp[:10]
        out = NBA_INJURY_DIR / "reports" / f"date={day}" / f"{stamp}.parquet"
        if out.exists():
            logger.info(f"NBA injury report {stamp} already stored")
            return
        response = requests.get(
            REPORT_URL.format(stamp=stamp),
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=CAPTURE_TIMEOUT_S,
        )
        if response.status_code == HTTPStatus.FORBIDDEN:
            continue
        response.raise_for_status()
        frame = (
            pd.DataFrame(parse_lines(pdf_lines(response.content)), columns=list(REPORT_DTYPES))
            .assign(
                report_stamp=stamp,
                last_modified=response.headers.get("Last-Modified"),
                fetched_at=fetched_at,
            )
            .astype(REPORT_DTYPES)
        )
        pdf = NBA_INJURY_DIR / "pdf" / f"{stamp}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(response.content)
        _atomic_write_parquet(frame, out)
        logger.info(f"NBA injury report {stamp} stored, {len(frame)} rows")
        return
    window = f"nothing posted from {stamps[-1]} to {stamps[0]}"
    # In season the league posts around the clock and S3 answers 403 for a block as for a
    # missing key, so an empty window on a day with an NBA game on the ledger is the alarm.
    if any(event["league"] == "NBA" for event in read_upcoming_events()):
        raise RuntimeError(f"NBA injury report: {window} with an NBA game on the ledger")
    logger.info(f"NBA injury report: {window}")
