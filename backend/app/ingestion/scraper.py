import asyncio
import hashlib
import json
import logging
import urllib.robotparser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    url: str
    text: str
    title: str
    fetched_at: datetime


class Scraper:
    """Robots-aware HTTP scraper with disk-based response caching."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        crawl_delay: float = 1.0,
        user_agent: str = "Loresmith/0.1 (+https://github.com/your-org/loresmith)",
    ) -> None:
        self._cache_dir = cache_dir
        self._crawl_delay = crawl_delay
        self._user_agent = user_agent
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        key = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self._cache_dir / f"{key}.json"

    def _load_from_cache(self, url: str) -> str | None:
        path = self._cache_path(url)
        if path and path.exists():
            data = json.loads(path.read_text())
            return data.get("html")
        return None

    def _save_to_cache(self, url: str, html: str) -> None:
        path = self._cache_path(url)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "html": html}))

    async def is_allowed(self, url: str) -> bool:
        """Return True if robots.txt permits crawling this URL."""
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        if base not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{base}/robots.txt",
                        headers={"User-Agent": self._user_agent},
                    )
                    rp.parse(resp.text.splitlines())
            except Exception:
                # If robots.txt is unreachable, assume allowed.
                rp.allow_all = True
            self._robots_cache[base] = rp

        return self._robots_cache[base].can_fetch(self._user_agent, url)

    def _extract_text(self, html: str, url: str) -> tuple[str, str]:
        """Return (title, cleaned_text) from raw HTML."""
        parser = HTMLParser(html)

        # Extract title
        title = ""
        title_node = (
            parser.css_first("h1.page-title")
            or parser.css_first("h1#firstHeading")
            or parser.css_first("title")
        )
        if title_node:
            title = title_node.text(strip=True)

        # Extract main content
        content_node = (
            parser.css_first(".mw-parser-output")
            or parser.css_first("#mw-content-text")
            or parser.css_first("body")
        )
        if not content_node:
            return title, ""

        # Remove navigation noise
        for selector in [".toc", ".navbox", ".references", ".reflist",
                         ".noprint", "script", "style", ".mw-editsection"]:
            for node in content_node.css(selector):
                node.decompose()

        text = content_node.text(separator="\n", strip=True)
        # Collapse excessive blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        return title, "\n".join(lines)

    async def fetch(self, url: str) -> ScrapedPage | None:
        """Fetch a page, respecting robots.txt. Returns None if disallowed."""
        if not await self.is_allowed(url):
            return None

        # Try cache first
        cached_html = self._load_from_cache(url)
        if cached_html:
            title, text = self._extract_text(cached_html, url)
            return ScrapedPage(url=url, text=text, title=title, fetched_at=datetime.now(UTC))

        # Fetch with rate-limit delay
        if self._crawl_delay > 0:
            await asyncio.sleep(self._crawl_delay)

        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

        self._save_to_cache(url, html)
        title, text = self._extract_text(html, url)
        return ScrapedPage(url=url, text=text, title=title, fetched_at=datetime.now(UTC))
