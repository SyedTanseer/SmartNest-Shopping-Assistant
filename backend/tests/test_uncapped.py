"""Removing the SerpApi cap on purpose: ``SERPAPI_LIFETIME_LIMIT=unlimited``.

An absent value is NOT the same as unlimited: scripts that spend requests refuse to run when the cap simply is not loaded
(the .env may not have been read), but run (still limited per run by --spend) when the cap was removed deliberately.
"""

from __future__ import annotations

import json

import httpx
import pytest

from smartnest.furnishing import SnapshotCatalog
from smartnest.furnishing.live import LiveSearcher
from smartnest.integrations.products.serpapi import SerpApiClient, SerpApiSettings, load_serpapi_settings

KEY = "test-key-1234567890"


def load(tmp_path, value):
    env = {"SERPAPI_API_KEY": KEY}
    if value is not None:
        env["SERPAPI_LIFETIME_LIMIT"] = value
    return load_serpapi_settings(tmp_path / "none.env", env)


class TestConfig:
    @pytest.mark.parametrize("word", ["unlimited", "UNLIMITED", " none ", "off", "no"])
    def test_the_word_removes_the_cap_on_purpose(self, tmp_path, word):
        s = load(tmp_path, word)
        assert s.lifetime_limit is None and s.uncapped is True

    def test_a_number_is_still_a_cap(self, tmp_path):
        s = load(tmp_path, "20")
        assert s.lifetime_limit == 20 and s.uncapped is False

    def test_absent_is_not_the_same_as_removed(self, tmp_path):
        s = load(tmp_path, None)
        assert s.lifetime_limit is None and s.uncapped is False

    def test_garbage_is_still_rejected(self, tmp_path):
        with pytest.raises(Exception):
            load(tmp_path, "lots")

    def test_the_word_in_the_env_file_works_too(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text(f"SERPAPI_API_KEY={KEY}\nSERPAPI_LIFETIME_LIMIT=unlimited\n", encoding="utf-8")
        assert load_serpapi_settings(f, {}).uncapped is True


class TestBehaviour:
    def transport(self, seen):
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(200, json={"shopping_results": []})
        return httpx.MockTransport(handler)

    def test_the_client_never_refuses_and_keeps_counting(self, tmp_path):
        seen = []
        s = SerpApiSettings(api_key=KEY, usage_file=tmp_path / "u.json", uncapped=True)
        client = SerpApiClient(s, transport=self.transport(seen))
        for i in range(30):
            client.search(f"q{i}")
        assert len(seen) == 30 and client.ledger.used == 30
        assert json.loads((tmp_path / "u.json").read_text())["used"] == 30

    def test_live_search_reports_no_limit_and_never_runs_out(self, tmp_path):
        seen = []
        s = SerpApiSettings(api_key=KEY, usage_file=tmp_path / "u.json", uncapped=True)
        searcher = LiveSearcher(s, SnapshotCatalog(tmp_path / "c"), transport=self.transport(seen))
        assert searcher.status() == {"enabled": True, "remaining": None, "limit": None, "used": 0}
        for cat in ("bookshelf", "study_table", "x_telescope", "rug"):
            searcher.search(cat)
        assert len(seen) == 4 and searcher.used == 4 and searcher.remaining is None

    def test_a_cap_still_works_when_set(self, tmp_path):
        seen = []
        s = SerpApiSettings(api_key=KEY, usage_file=tmp_path / "u.json", lifetime_limit=1)
        searcher = LiveSearcher(s, SnapshotCatalog(tmp_path / "c"), transport=self.transport(seen))
        searcher.search("bookshelf")
        with pytest.raises(Exception):
            searcher.search("rug")
        assert len(seen) == 1


class TestCaptureScriptGuard:
    """capture_catalog.py: refuse when the cap may simply not be loaded; allow when removed on purpose."""

    def run(self, monkeypatch, capsys, settings):
        import capture_catalog
        monkeypatch.setattr(capture_catalog, "load_serpapi_settings", lambda: settings)
        monkeypatch.setattr("sys.argv", ["capture_catalog.py", "--plan"])
        code = capture_catalog.main()
        return code, capsys.readouterr().out

    def test_refuses_when_the_cap_is_absent(self, monkeypatch, capsys, tmp_path):
        code, out = self.run(monkeypatch, capsys, SerpApiSettings(api_key=KEY, usage_file=tmp_path / "u.json"))
        assert code == 2 and "REFUSING" in out and "unlimited" in out

    def test_runs_when_the_cap_was_removed_on_purpose(self, monkeypatch, capsys, tmp_path):
        code, out = self.run(monkeypatch, capsys, SerpApiSettings(api_key=KEY, usage_file=tmp_path / "u.json", uncapped=True))
        assert code == 0 and "none (removed on purpose)" in out and "--spend" in out

    def test_runs_with_a_number(self, monkeypatch, capsys, tmp_path):
        code, out = self.run(monkeypatch, capsys, SerpApiSettings(api_key=KEY, usage_file=tmp_path / "u.json", lifetime_limit=50))
        assert code == 0 and "Lifetime cap 50" in out


class TestTestsNeverTouchTheRealLedger:
    def test_a_client_built_from_default_settings_does_not_write_the_real_usage_file(self):
        default = SerpApiSettings(api_key=KEY).usage_file
        before = default.read_bytes() if default.exists() else None
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            return httpx.Response(200, json={"shopping_results": []})
        client = SerpApiClient(SerpApiSettings(api_key=KEY), transport=httpx.MockTransport(handler))
        for i in range(3):
            client.search(f"q{i}")
        after = default.read_bytes() if default.exists() else None
        assert len(seen) == 3 and client.ledger is None and after == before
