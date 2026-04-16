from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.ingestion.scraper import ScrapedPage, Scraper


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
    html = """<html><head><title>Zagreus</title></head>
    <body><div class="mw-parser-output"><p>Zagreus is the son of Hades.</p></div></body></html>"""

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html
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
    html = "<html><body><div class='mw-parser-output'><p>Cached content.</p></div></body></html>"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html
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
async def test_is_allowed_returns_true_when_no_robots(tmp_path):
    scraper = Scraper(cache_dir=tmp_path, crawl_delay=0)

    with patch("app.ingestion.scraper.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("robots.txt not found"))
        mock_client_cls.return_value = mock_client

        allowed = await scraper.is_allowed("https://example.com/wiki/Foo")

    assert allowed is True
