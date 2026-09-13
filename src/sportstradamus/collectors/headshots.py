"""Player-headshot cache — one square WebP per player, filled by ``fetch headshots``.

The constellation ticket card draws a face when this box's cache holds one and its
initials disc when it does not, so the cache is dressing the dashboard never depends on.
It is gitignored and per-box for the reason ``stat_calibration.json`` is: runtime-produced,
binary, and cheap to rebuild.

Two families of source. MLB and NFL are Cloudinary-backed and answer a ``g_face``
transform in the URL, which crops to the face server-side and cuts one NFL transfer from
3.8 MB to about 21 KB. NBA, WNBA and NHL serve transparent torso cutouts cropped here
instead, and their alpha survives into the cache so the card's own disc shows through
rather than a white square on a dark surface.

Neither NBA nor WNBA answers an unknown id with a 404 — both return HTTP 200 and a flat
grey silhouette, and NHL redirects to one — so every decoded image is probed for that
before it reaches the cache. An undetected placeholder would render as a face, which is
worse than the disc it replaced.

Licence clearance per league is the owner's call, made before this job goes on a cron;
nothing here checks it.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import io
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import click
import nflreadpy as nflr
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers.io import _atomic_write_parquet, _gamelog_paths, read_parquet_safe
from sportstradamus.helpers.scraping import Scrape
from sportstradamus.helpers.text import remove_accents

HEADSHOT_DIR = Path(str(pkg_resources.files(data) / "assets" / "headshots"))
INDEX_PATH = HEADSHOT_DIR / "index.parquet"

LEAGUES = ("MLB", "NBA", "WNBA", "NHL", "NFL")

# (id, name, team) per league gamelog. Ids are int64 for MLB/NBA/WNBA and object for
# NHL/NFL, so the index stringifies every one of them or the join back to it breaks.
_GAMELOG_COLUMNS = {
    "MLB": ("playerId", "playerName", "team"),
    "NBA": ("PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"),
    "WNBA": ("PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"),
    "NHL": ("playerId", "playerName", "team"),
    "NFL": ("player id", "player display name", "team"),
}

# NHL's ``latest/`` path answers for traded and retired players alike, unlike the
# season-and-team-stamped one the NHL gamelog carries no season to build. NFL has no
# id-addressable pattern at all and reaches its photos through the roster instead.
_URL_TEMPLATE = {
    "MLB": (
        "https://img.mlbstatic.com/mlb-photos/image/upload/w_256,c_fill,ar_1:1,g_face"
        "/v1/people/{id}/headshot/67/current"
    ),
    "NBA": "https://cdn.nba.com/headshots/nba/latest/1040x760/{id}.png",
    "WNBA": "https://cdn.wnba.com/headshots/wnba/latest/1040x760/{id}.png",
    "NHL": "https://assets.nhle.com/mugs/nhl/latest/{id}.png",
}

# The roster's own URLs carry a bare ``f_auto,q_auto`` over a multi-megabyte original;
# swapping a face crop into that slot is both the crop and the bandwidth fix.
_NFL_TRANSFORM = "w_256,h_256,c_thumb,g_face,z_0.7,f_auto,q_auto"
_NFL_TRANSFORM_SLOT = re.compile(r"(/image/(?:upload|private)/)[^/]+/")

# Square side as a fraction of source width. MLB and NFL arrive already square from their
# CDN's face crop, so 1.0 is the identity there; the other three serve transparent torso
# cutouts whose head fills about this much of the frame, measured off live headshots.
_CROP_WIDTH_FRACTION = {"MLB": 1.0, "NFL": 1.0, "NBA": 0.48, "WNBA": 0.48, "NHL": 0.62}

# The card renders the disc at 34 CSS px, so 128 covers a 3x-DPR phone. Measured at about
# 4.6 KB a face, so a sixty-star figure ships ~360 KiB of base64 against a ~1 MB budget.
_DISC_PX = 128
_DISC_QUALITY = 80

# Flattened to 64x64 a photograph holds well over this many distinct colours while the
# known silhouettes hold 45-187, and ``getcolors`` returns None above its cap — so the
# count is the whole probe.
_PLACEHOLDER_MAX_COLOURS = 400
_PLACEHOLDER_PROBE_PX = 64

_WHITE = (255, 255, 255, 255)


def _nfl_roster(first_season: int) -> pd.DataFrame:
    """Latest roster row per player from ``first_season`` on, as ``id, name, team, url``.

    Carries ``season`` too, so the caller can tell this year's squad from the rest. One
    season covers barely three fifths of the NFL gamelog's ids, hence the whole span.
    """
    seasons = list(range(first_season, nflr.get_current_season() + 1))
    roster = nflr.load_rosters(seasons).to_pandas()
    roster = roster[roster["gsis_id"].notna() & roster["headshot_url"].notna()]
    roster = roster.sort_values("season").groupby("gsis_id", as_index=False).tail(1)
    return pd.DataFrame(
        {
            "id": roster["gsis_id"].astype(str),
            "name": roster["full_name"].map(remove_accents),
            "team": roster["team"],
            "url": roster["headshot_url"].str.replace(
                _NFL_TRANSFORM_SLOT, rf"\g<1>{_NFL_TRANSFORM}/", regex=True
            ),
            "season": roster["season"],
        }
    )


def _enumerate(league: str) -> pd.DataFrame:
    """Every player the league can offer a face for, as ``id, name, team, url``.

    The gamelog parquet is the only enumeration carrying ids — the ``players`` sidecars are
    name-keyed — and its rows are written chronologically, so the last row per id holds the
    current name and team. NFL additionally unions this season's roster, which is both its
    only URL source and the only place a rookie appears before their first gamelog row.
    """
    id_col, name_col, team_col = _GAMELOG_COLUMNS[league]
    columns = [id_col, name_col, team_col] + (["season"] if league == "NFL" else [])
    gamelog = read_parquet_safe(_gamelog_paths(league)["gamelog"], columns=columns)
    players = pd.DataFrame(
        {
            "id": gamelog[id_col].astype(str),
            "name": gamelog[name_col].map(remove_accents),
            "team": gamelog[team_col],
        }
    ).drop_duplicates(subset=["id"], keep="last")

    if league != "NFL":
        players["url"] = [_URL_TEMPLATE[league].format(id=pid) for pid in players["id"]]
        return players

    roster = _nfl_roster(int(gamelog["season"].min()))
    current = roster.loc[roster["season"] == roster["season"].max(), "id"]
    roster = roster[roster["id"].isin(set(players["id"]) | set(current))].drop(columns="season")
    merged = players.merge(roster, on="id", how="outer", suffixes=("", "_roster"))
    merged["name"] = merged["name"].fillna(merged["name_roster"])
    merged["team"] = merged["team"].fillna(merged["team_roster"])
    return merged.loc[merged["url"].notna(), ["id", "name", "team", "url"]]


def _to_disc(raw: bytes, fraction: float) -> bytes | None:
    """Render one downloaded headshot to a cache-ready square WebP, or ``None``.

    ``None`` means the bytes hold no usable face: an image PIL cannot decode (one live NHL
    mug serves a complete but corrupt PNG), or one of the flat silhouettes the basketball
    and hockey CDNs return in place of a 404.

    The crop needs no per-family branch. A transparent cutout's subject starts partway down
    its frame and the alpha bounding box says where, while an opaque photo's bounding box
    starts at zero — which is exactly where an already-square source should be cropped from.
    """
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
    except OSError:
        return None
    side = min(round(image.width * fraction), image.height)
    left = (image.width - side) // 2
    top = min(image.getchannel("A").getbbox()[1], image.height - side)
    disc = image.crop((left, top, left + side, top + side)).resize(
        (_DISC_PX, _DISC_PX), Image.LANCZOS
    )

    flat = Image.alpha_composite(Image.new("RGBA", disc.size, _WHITE), disc).convert("RGB")
    probe = flat.resize((_PLACEHOLDER_PROBE_PX, _PLACEHOLDER_PROBE_PX))
    if probe.getcolors(maxcolors=_PLACEHOLDER_MAX_COLOURS):
        return None

    payload = io.BytesIO()
    disc.save(payload, "WEBP", quality=_DISC_QUALITY, method=6)
    return payload.getvalue()


def _cache_player(scrape: Scrape, league: str, player: dict) -> tuple[str, str]:
    """Download one player's headshot into the cache; return ``(status, file)``.

    ``("missing", "")`` covers every way a CDN declines to hand over a usable face — a 404,
    a redirect to a placeholder, or a 200 whose image is one.
    """
    raw = scrape.get_bytes(player["url"])
    disc = _to_disc(raw, _CROP_WIDTH_FRACTION[league]) if raw else None
    if not disc:
        return "missing", ""
    relative = f"{league.lower()}/{player['id']}.webp"
    path = HEADSHOT_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(disc)
    return "ok", relative


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    type=click.Choice(LEAGUES, case_sensitive=False),
    help="League to fetch; repeatable. Defaults to all five.",
)
@click.option("--force", is_flag=True, help="Re-download players already cached.")
def headshots(leagues: tuple[str, ...], force: bool) -> None:
    """Fill this box's player-headshot cache, one square WebP per player.

    Idempotent: a player already cached is skipped, so a monthly run downloads only
    newcomers and retries the players a CDN had no photo for last time — which is how a
    rookie's face eventually lands.
    """
    leagues = tuple(league.upper() for league in leagues) or LEAGUES
    scrape = Scrape()
    index = {
        (row["league"], row["id"]): row for row in read_parquet_safe(INDEX_PATH).to_dict("records")
    }
    counts: Counter[str] = Counter()
    try:
        for league in leagues:
            for player in tqdm(
                _enumerate(league).to_dict("records"), desc=f"{league} headshots", unit="player"
            ):
                cached = index.get((league, player["id"]))
                if (
                    not force
                    and cached
                    and cached["status"] == "ok"
                    and (HEADSHOT_DIR / cached["file"]).is_file()
                ):
                    counts["skip"] += 1
                    continue
                status, relative = _cache_player(scrape, league, player)
                counts[status] += 1
                index[(league, player["id"])] = {
                    "league": league,
                    "id": player["id"],
                    "name": player["name"],
                    "team": player["team"],
                    "file": relative,
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "status": status,
                }
    finally:
        _atomic_write_parquet(pd.DataFrame(list(index.values())), INDEX_PATH)
        click.echo("headshots: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
