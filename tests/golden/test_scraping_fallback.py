"""``Scrape`` keeps fetching through a ScrapeOps outage on a static browser header."""

import requests

from sportstradamus.helpers import scraping
from sportstradamus.helpers.scraping import BROWSER_USER_AGENT, Scrape


def _response(status, body, content_type):
    res = requests.Response()
    res.status_code = status
    res._content = body.encode()
    res.headers["content-type"] = content_type
    return res


def _serve(monkeypatch, responses):
    """Answer ScrapeOps with the 502 page it served on 2026-10-23, every other GET in order."""
    sent = []

    def get(url, headers=None, **kwargs):
        if "scrapeops" in url:
            return _response(502, "<html><title>502 Bad Gateway</title></html>", "text/html")
        sent.append(headers)
        return responses.pop(0)

    monkeypatch.setattr(scraping.requests, "get", get)
    monkeypatch.setattr(scraping, "sleep", lambda _s: None)
    return sent


def test_scrapeops_outage_falls_back_to_the_static_browser_header(monkeypatch, caplog):
    _serve(monkeypatch, [])
    scrape = Scrape()
    # CI's fixture key is empty, and "" is in every string.
    scrape._scrapeops_key = "secret-scrapeops-key"
    assert scrape.header == {"User-Agent": BROWSER_USER_AGENT}
    assert "ScrapeOps headers unavailable (502)" in caplog.text
    assert "secret-scrapeops-key" not in caplog.text


def test_get_csv_retries_a_soft_block_on_the_fallback_header(monkeypatch):
    sent = _serve(
        monkeypatch,
        [
            _response(200, "<html>data license</html>", "text/html"),
            _response(200, "a,b\n1,2\n", "text/csv"),
        ],
    )
    frame = Scrape().get_csv("https://moneypuck.com/moneypuck/playerData/games/x.csv")
    assert frame.to_dict("records") == [{"a": 1, "b": 2}]
    assert sent == [{"User-Agent": BROWSER_USER_AGENT}] * 2
