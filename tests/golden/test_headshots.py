"""Golden pins for the player-headshot cache — its fetcher and the card's resolver.

Two contracts meet here. ``collectors.headshots`` must never let a CDN's grey placeholder
into the cache (NBA and WNBA answer an unknown id with HTTP 200 and one, so a status check
alone would not notice), must crop a transparent torso cutout to the head rather than the
chest, and must not re-download a player it already holds. ``dashboard.assets.headshot_uris``
must fall silent — never raise, never guess — for every player the cache cannot answer for,
because the constellation card's fallback is the initials disc it has always drawn.

Every fixture image is built in memory: the crops are geometry, not photographs, and the
repo commits no binaries for a gitignored cache.
"""

from __future__ import annotations

import base64
import io
import random

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from PIL import Image

from sportstradamus.collectors import headshots
from sportstradamus.dashboard import assets

_CUTOUT_SIZE = (1040, 760)  # the NBA/WNBA frame
_NBA_FRACTION = headshots._CROP_WIDTH_FRACTION["NBA"]


def _noise(size: tuple[int, int], channel: str, seed: int = 7) -> Image.Image:
    """A block of noise with one channel dominant — busy enough to clear the ghost probe."""
    rng = random.Random(seed)
    pixels = []
    for _ in range(size[0] * size[1]):
        low, high = rng.randrange(0, 90), rng.randrange(160, 256)
        pixels.append((high, low, low) if channel == "r" else (low, low, high))
    image = Image.new("RGB", size)
    image.putdata(pixels)
    return image


def _cutout(subject_top: int) -> bytes:
    """A transparent torso cutout: a red head band over a blue body, starting at ``subject_top``."""
    canvas = Image.new("RGBA", _CUTOUT_SIZE, (0, 0, 0, 0))
    canvas.paste(_noise((240, 260), "r").convert("RGBA"), (400, subject_top))
    canvas.paste(_noise((400, 400), "b", seed=8).convert("RGBA"), (320, subject_top + 260))
    return _encode(canvas)


def _encode(image: Image.Image) -> bytes:
    payload = io.BytesIO()
    image.save(payload, "PNG")
    return payload.getvalue()


def _decode(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGBA")


def test_a_cutout_is_cropped_from_where_its_subject_starts_not_the_frame_centre():
    # Subject top (200) sits above the naive vertical centre (130 for a 499px square on a
    # 760px frame), so a centred crop would open with a band of empty pixels.
    disc = _decode(headshots._to_disc(_cutout(subject_top=200), _NBA_FRACTION))
    assert disc.size == (headshots._DISC_PX, headshots._DISC_PX)
    assert disc.getchannel("A").getbbox()[1] == 0


def test_a_cutout_crops_to_the_head_rather_than_the_chest():
    disc = _decode(headshots._to_disc(_cutout(subject_top=40), _NBA_FRACTION)).convert("RGB")
    band = np.asarray(disc.crop((0, 0, disc.width, 24)))
    assert band[:, :, 0].mean() > band[:, :, 2].mean()


def test_an_opaque_square_source_is_taken_whole():
    square = _noise((256, 256), "r").convert("RGBA")
    disc = _decode(headshots._to_disc(_encode(square), headshots._CROP_WIDTH_FRACTION["MLB"]))
    assert disc.size == (headshots._DISC_PX, headshots._DISC_PX)
    assert disc.getchannel("A").getbbox() == (0, 0, headshots._DISC_PX, headshots._DISC_PX)


def test_a_flat_silhouette_is_refused():
    ghost = Image.new("RGBA", _CUTOUT_SIZE, (0, 0, 0, 0))
    ghost.paste(Image.new("RGBA", (300, 500), (128, 128, 128, 255)), (370, 60))
    assert headshots._to_disc(_encode(ghost), _NBA_FRACTION) is None


def test_undecodable_bytes_are_refused_rather_than_raising():
    assert headshots._to_disc(b"\x89PNG\r\n\x1a\n not an image at all", _NBA_FRACTION) is None


class _FakeScrape:
    """Stands in for ``Scrape``, recording which URLs a run actually asked for."""

    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.asked: list[str] = []

    def get_bytes(self, url):
        self.asked.append(url)
        return self.payloads.get(url)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point both modules at an empty tmp cache and hand back the fake CDN behind it."""
    monkeypatch.setattr(headshots, "HEADSHOT_DIR", tmp_path)
    monkeypatch.setattr(headshots, "INDEX_PATH", tmp_path / "index.parquet")
    monkeypatch.setattr(assets, "HEADSHOT_DIR", tmp_path)
    monkeypatch.setattr(assets, "HEADSHOT_INDEX_PATH", tmp_path / "index.parquet")
    scrape = _FakeScrape({"u/ok": _cutout(subject_top=40)})
    monkeypatch.setattr(headshots, "Scrape", lambda: scrape)
    monkeypatch.setattr(
        headshots,
        "_enumerate",
        lambda league: pd.DataFrame(
            [
                {"id": "1", "name": "Ada Lovelace", "team": "NYL", "url": "u/ok"},
                {"id": "2", "name": "Grace Hopper", "team": "SEA", "url": "u/gone"},
            ]
        ),
    )
    return scrape


def _run(*args) -> None:
    result = CliRunner().invoke(headshots.headshots, ["--league", "WNBA", *args])
    assert result.exit_code == 0, result.output


def test_a_run_caches_the_hits_and_records_the_misses(cache, tmp_path):
    _run()
    index = pd.read_parquet(tmp_path / "index.parquet")
    assert index["id"].tolist() == ["1", "2"]
    assert index.set_index("id")["status"].to_dict() == {"1": "ok", "2": "missing"}
    assert (tmp_path / "wnba" / "1.webp").is_file()
    assert index.loc[index["id"] == "2", "file"].item() == ""


def test_a_second_run_re_asks_only_for_the_players_it_missed(cache):
    _run()
    cache.asked.clear()
    _run()
    assert cache.asked == ["u/gone"]


def test_force_re_asks_for_everyone(cache):
    _run()
    cache.asked.clear()
    _run("--force")
    assert sorted(cache.asked) == ["u/gone", "u/ok"]


def _pool(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_a_cached_player_resolves_to_a_webp_data_uri(cache):
    _run()
    uris = assets.headshot_uris(_pool({"League": "WNBA", "Player": "Ada Lovelace", "Team": "NYL"}))
    assert uris["Ada Lovelace"].startswith("data:image/webp;base64,")
    assert base64.b64decode(uris["Ada Lovelace"].split(",", 1)[1])[:4] == b"RIFF"


def test_a_player_the_cache_missed_is_absent_rather_than_blank(cache):
    _run()
    pool = _pool({"League": "WNBA", "Player": "Grace Hopper", "Team": "SEA"})
    assert assets.headshot_uris(pool) == {}


def test_a_box_with_no_cache_at_all_answers_nothing(cache):
    pool = _pool({"League": "WNBA", "Player": "Ada Lovelace", "Team": "NYL"})
    assert assets.headshot_uris(pool) == {}


def test_a_wider_lens_star_from_another_league_reads_its_own_row(cache, tmp_path):
    _run()
    # Same name, same file on disk, different league — the resolver must not cross over.
    index = pd.read_parquet(tmp_path / "index.parquet")
    assets._headshot_rows.cache_clear()
    pd.concat([index, index.assign(league="NBA", id="9")]).to_parquet(
        tmp_path / "index.parquet", index=False
    )
    uris = assets.headshot_uris(
        _pool(
            {"League": "WNBA", "Player": "Ada Lovelace", "Team": "NYL"},
            {"League": "NBA", "Player": "Grace Hopper", "Team": "SEA"},
        )
    )
    assert set(uris) == {"Ada Lovelace"}


def test_two_players_sharing_a_name_on_one_figure_both_fall_back_to_initials(cache, tmp_path):
    _run()
    index = pd.read_parquet(tmp_path / "index.parquet")
    twin = index[index["id"] == "1"].assign(id="3", team="LVA", file="wnba/3.webp")
    (tmp_path / "wnba" / "3.webp").write_bytes(
        (tmp_path / "wnba" / "1.webp").read_bytes() + b"\x00"
    )
    assets._headshot_rows.cache_clear()
    pd.concat([index, twin]).to_parquet(tmp_path / "index.parquet", index=False)
    uris = assets.headshot_uris(
        _pool(
            {"League": "WNBA", "Player": "Ada Lovelace", "Team": "NYL"},
            {"League": "WNBA", "Player": "Ada Lovelace", "Team": "LVA"},
        )
    )
    assert uris == {}
