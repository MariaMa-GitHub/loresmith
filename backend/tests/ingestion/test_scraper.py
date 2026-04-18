import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.ingestion.scraper import ScrapedPage, Scraper, _FandomFallbackRobotsPolicy


def test_scraped_page_fields():
    page = ScrapedPage(
        url="https://example.com/wiki/Foo",
        text="Some content about Foo.",
        title="Foo",
        fetched_at=datetime.now(UTC),
    )
    assert page.url == "https://example.com/wiki/Foo"
    assert "Foo" in page.text


@pytest.mark.asyncio
async def test_scraper_fetch_returns_none_if_disallowed(tmp_path):
    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    # Patch robots check to deny
    with patch.object(scraper, "is_allowed", return_value=False):
        result = await scraper.fetch("https://example.com/wiki/Foo")

    assert result is None


@pytest.mark.asyncio
async def test_scraper_fetch_parses_html(tmp_path):
    # Fandom URLs are fetched via the MediaWiki API (Cloudflare blocks direct
    # /wiki/ requests), so the mocked response mirrors the api.php envelope.
    api_html = (
        '<div class="mw-parser-output"><p>Zagreus is the son of Hades.</p></div>'
    )
    api_body = json.dumps({"parse": {"title": "Zagreus", "text": api_html}})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = api_body
    mock_response.raise_for_status = MagicMock()

    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch.object(scraper, "is_allowed", return_value=True):
        with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await scraper.fetch("https://hades.fandom.com/wiki/Zagreus")

    assert result is not None
    assert "Zagreus is the son of Hades" in result.text
    assert result.title == "Zagreus"


@pytest.mark.asyncio
async def test_scraper_uses_cache_on_second_fetch(tmp_path):
    api_html = "<div class='mw-parser-output'><p>Cached content.</p></div>"
    api_body = json.dumps({"parse": {"title": "Nyx", "text": api_html}})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = api_body
    mock_response.raise_for_status = MagicMock()

    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch.object(scraper, "is_allowed", return_value=True):
        with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            url = "https://hades.fandom.com/wiki/Nyx"
            await scraper.fetch(url)
            await scraper.fetch(url)

            # Only one HTTP call was made — second came from cache
            assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_scraper_refetches_when_cache_is_stale(tmp_path):
    api_html = "<div class='mw-parser-output'><p>Fresh content.</p></div>"
    api_body = json.dumps({"parse": {"title": "Nyx", "text": api_html}})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = api_body
    mock_response.raise_for_status = MagicMock()

    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0, cache_ttl=timedelta(hours=1))
    url = "https://hades.fandom.com/wiki/Nyx"
    cache_path = scraper._cache_path(url)
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stale_payload = {
        "url": url,
        "html": "<div class='mw-parser-output'><p>Stale content.</p></div>",
        "fetched_at": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
    }
    cache_path.write_text(json.dumps(stale_payload))

    with patch.object(scraper, "is_allowed", return_value=True):
        with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await scraper.fetch(url)

    assert result is not None
    assert "Fresh content." in result.text
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_scraper_force_refresh_ignores_fresh_cache(tmp_path):
    api_html = "<div class='mw-parser-output'><p>Refetched content.</p></div>"
    api_body = json.dumps({"parse": {"title": "Nyx", "text": api_html}})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = api_body
    mock_response.raise_for_status = MagicMock()

    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0, force_refresh=True)
    url = "https://hades.fandom.com/wiki/Nyx"
    cache_path = scraper._cache_path(url)
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_payload = {
        "url": url,
        "html": "<div class='mw-parser-output'><p>Cached content.</p></div>",
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    cache_path.write_text(json.dumps(fresh_payload))

    with patch.object(scraper, "is_allowed", return_value=True):
        with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await scraper.fetch(url)

    assert result is not None
    assert "Refetched content." in result.text
    assert mock_client.get.call_count == 1


@pytest.mark.asyncio
async def test_scraper_fetch_returns_none_on_http_error(tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)
    )

    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch.object(scraper, "is_allowed", return_value=True):
        with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await scraper.fetch("https://hades.fandom.com/wiki/Nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_scraper_fetch_checks_rewritten_fandom_api_url_against_robots(tmp_path):
    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch.object(scraper, "is_allowed", side_effect=[True, False]) as mock_allowed:
        result = await scraper.fetch("https://hades.fandom.com/wiki/Zagreus")

    assert result is None
    assert mock_allowed.await_count == 2


@pytest.mark.asyncio
async def test_is_allowed_returns_true_when_robots_is_missing(tmp_path):
    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        allowed = await scraper.is_allowed("https://example.com/wiki/Foo")

    assert allowed is True


@pytest.mark.asyncio
async def test_is_allowed_uses_fandom_fallback_on_browser_challenge(tmp_path):
    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {"cf-mitigated": "challenge"}
        mock_response.text = "Just a moment... Enable JavaScript and cookies to continue"
        mock_response.request.url.host = "hades.fandom.com"
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        article_allowed = await scraper.is_allowed("https://hades.fandom.com/wiki/Zagreus")
        api_allowed = await scraper.is_allowed(
            "https://hades.fandom.com/api.php?action=parse&page=Zagreus"
        )
        special_disallowed = await scraper.is_allowed(
            "https://hades.fandom.com/wiki/Special:RecentChanges"
        )
        api_without_action_disallowed = await scraper.is_allowed(
            "https://hades.fandom.com/api.php"
        )

    assert article_allowed is True
    assert api_allowed is True
    assert special_disallowed is False
    assert api_without_action_disallowed is False
    assert mock_client.get.await_count == 1


@pytest.mark.asyncio
async def test_is_allowed_returns_false_when_robots_is_unreachable(tmp_path):
    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("robots.txt not found"))
        mock_client_cls.return_value = mock_client

        allowed = await scraper.is_allowed("https://example.com/wiki/Foo")

    assert allowed is False


# ---------------------------------------------------------------------------
# _FandomFallbackRobotsPolicy unit tests
# ---------------------------------------------------------------------------

class TestFandomFallbackRobotsPolicy:
    UA = "Loresmith/0.1"
    POLICY = _FandomFallbackRobotsPolicy()

    @pytest.mark.parametrize("url, expected", [
        # --- allowed: normal article pages ---
        ("https://hades.fandom.com/wiki/Zagreus", True),
        ("https://hades.fandom.com/wiki/Nyx", True),
        ("https://hades2.fandom.com/wiki/Melin%C3%AB", True),
        # --- allowed: MediaWiki API with action= ---
        ("https://hades.fandom.com/api.php?action=parse&page=Zagreus&format=json", True),
        # --- allowed: whitelisted Special pages ---
        ("https://hades.fandom.com/wiki/Special:CreateNewWiki", True),
        # --- disallowed: /api.php with no query ---
        ("https://hades.fandom.com/api.php", False),
        # --- disallowed: /api.php query without action= ---
        ("https://hades.fandom.com/api.php?format=json", False),
        # --- disallowed: blocked wiki namespaces ---
        ("https://hades.fandom.com/wiki/User:SomeUser", False),
        ("https://hades.fandom.com/wiki/User_talk:SomeUser", False),
        ("https://hades.fandom.com/wiki/Template:Infobox", False),
        ("https://hades.fandom.com/wiki/Template_talk:Infobox", False),
        ("https://hades.fandom.com/wiki/Help:Contents", False),
        ("https://hades.fandom.com/wiki/UserProfile:SomeUser", False),
        # --- disallowed: non-Fandom domain ---
        ("https://example.com/wiki/Zagreus", False),
        # --- disallowed: /wiki/ with no article title ---
        ("https://hades.fandom.com/wiki/", False),
        # --- disallowed: root path ---
        ("https://hades.fandom.com/", False),
    ])
    def test_can_fetch(self, url, expected):
        assert self.POLICY.can_fetch(self.UA, url) == expected

    @pytest.mark.parametrize("agent", [
        "SemrushBot/7.3",
        "GPTBot/1.0",
        "CCBot/2.0",
        "Google-Extended",
    ])
    def test_blocked_user_agents_are_denied(self, agent):
        assert self.POLICY.can_fetch(agent, "https://hades.fandom.com/wiki/Zagreus") is False
