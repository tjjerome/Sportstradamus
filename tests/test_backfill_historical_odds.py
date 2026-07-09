"""Pin the apikey shapes on both backfill fetch paths: ``get_props`` takes the
key string itself, ``get_moneylines`` takes the whole creds dict. The probe
path once passed the dict into ``get_props`` — a guaranteed 401 on every paid
dry run."""

import datetime

from sportstradamus.scripts import backfill_historical_odds as bho

FAKE_KEYS = {bho.HISTORICAL_KEY_NAME: "key-string", "odds_api": "other-key"}
PROPS = {"NFL": {"player_pass_yds": "passing yards"}}
DATES = [datetime.datetime(2023, 10, 29)]


def test_probe_passes_key_string(monkeypatch):
    seen = []
    monkeypatch.setattr(bho, "get_props", lambda archive, apikey, *a, **k: seen.append(apikey))
    bho._probe(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6)
    assert seen == ["key-string"]


def test_backfill_passes_key_string_and_creds_dict(monkeypatch, tmp_path):
    class FakeArchive:
        def write(self):
            pass

    props_keys, moneyline_keys = [], []
    monkeypatch.setattr(bho, "Archive", FakeArchive)
    monkeypatch.setattr(bho, "PROGRESS_PATH", tmp_path / "progress.json")
    monkeypatch.setattr(bho, "get_props", lambda archive, apikey, *a, **k: props_keys.append(apikey))
    monkeypatch.setattr(
        bho, "get_moneylines", lambda archive, apikey, **k: moneyline_keys.append(apikey)
    )
    bho._backfill(FAKE_KEYS, PROPS, "NFL", "americanfootball_nfl", DATES, 6)
    assert props_keys == ["key-string"]
    assert moneyline_keys == [FAKE_KEYS]
