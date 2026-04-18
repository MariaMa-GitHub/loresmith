import asyncio
import hashlib
import json
import logging
import urllib.robotparser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import httpx
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)


class _AllowAllRobotsPolicy:
    """Stand-in for RobotFileParser used when robots.txt is unreachable.

    Using a local sentinel keeps us off urllib.robotparser private attributes
    (the previous code mutated ``rp.allow_all = True``), and the ``can_fetch``
    signature matches the real parser so callers don't need to branch.
    """

    def can_fetch(self, user_agent: str, url: str) -> bool:  # pragma: no cover - trivial
        return True


class _DisallowAllRobotsPolicy:
    """Stand-in for RobotFileParser used when robots.txt cannot be trusted."""

    def can_fetch(self, user_agent: str, url: str) -> bool:  # pragma: no cover - trivial
        return False


def _is_fandom_domain(host: str) -> bool:
    return host.casefold().endswith(".fandom.com")


def _is_fandom_robots_challenge(response: httpx.Response) -> bool:
    """Detect the Cloudflare browser challenge Fandom currently serves."""
    host = response.request.url.host or ""
    if not _is_fandom_domain(host):
        return False
    if response.status_code not in {403, 503}:
        return False
    if response.headers.get("cf-mitigated", "").casefold() == "challenge":
        return True
    body = response.text.casefold()
    return "just a moment" in body and "enable javascript and cookies" in body


class _FandomFallbackRobotsPolicy:
    """Conservative fallback when Fandom blocks robots.txt behind a browser challenge.

    The live robots.txt remains the source of truth whenever it can be fetched.
    This fallback only allows the article and API paths the ingestion pipeline
    depends on and keeps obvious non-content namespaces blocked.
    """

    _blocked_user_agents = frozenset(
        {
            "semrushbot",
            "serpstatbot",
            "gptbot",
            "google-extended",
            "imagesiftbot",
            "ccbot",
        }
    )
    _blocked_wiki_prefixes = (
        "special:",
        "user:",
        "user_talk:",
        "template:",
        "template_talk:",
        "help:",
        "userprofile:",
    )
    _allowed_special_titles = frozenset({"special:createnewwiki", "special:allmaps"})

    def can_fetch(self, user_agent: str, url: str) -> bool:
        parsed = urlparse(url)
        if not _is_fandom_domain(parsed.netloc):
            return False

        agent = user_agent.casefold()
        if any(blocked in agent for blocked in self._blocked_user_agents):
            return False

        segments = [segment for segment in unquote(parsed.path).split("/") if segment]
        if not segments:
            return False

        if segments[-1] == "api.php":
            return bool(parsed.query) and "action=" in parsed.query.casefold()

        if "wiki" not in segments:
            return False

        wiki_index = segments.index("wiki")
        title = "/".join(segments[wiki_index + 1 :]).strip()
        if not title:
            return False

        normalized_title = title.replace(" ", "_").casefold()
        if normalized_title in self._allowed_special_titles:
            return True
        return not normalized_title.startswith(self._blocked_wiki_prefixes)


@dataclass
class ScrapedPage:
    url: str
    text: str
    title: str
    fetched_at: datetime


class Scraper:
    """Robots-aware HTTP scraper with disk-based response caching.

    Cache entries are reused only while fresh. A caller can also bypass cache
    reads entirely (``force_refresh=True``) to guarantee a re-fetch.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        crawl_delay: float = 1.0,
        cache_ttl: timedelta | None = timedelta(hours=24),
        force_refresh: bool = False,
        user_agent: str = "Loresmith/0.1 (+https://github.com/MariaMa-GitHub/loresmith)",
    ) -> None:
        self._cache_dir = cache_dir
        self._crawl_delay = crawl_delay
        self._cache_ttl = cache_ttl
        self._force_refresh = force_refresh
        self._user_agent = user_agent
        self._robots_cache: dict[
            str,
            urllib.robotparser.RobotFileParser
            | _AllowAllRobotsPolicy
            | _DisallowAllRobotsPolicy
            | _FandomFallbackRobotsPolicy,
        ] = {}

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        key = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self._cache_dir / f"{key}.json"

    def _load_from_cache(self, url: str) -> str | None:
        path = self._cache_path(url)
        if self._force_refresh:
            return None
        if path and path.exists():
            data = json.loads(path.read_text())
            if not self._is_cache_fresh(path, data):
                return None
            return data.get("html")
        return None

    def _save_to_cache(self, url: str, html: str) -> None:
        path = self._cache_path(url)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "url": url,
                        "html": html,
                        "fetched_at": datetime.now(UTC).isoformat(),
                    }
                )
            )

    def _is_cache_fresh(self, path: Path, data: dict) -> bool:
        if self._cache_ttl is None:
            return True

        fetched_at_raw = data.get("fetched_at")
        fetched_at: datetime | None = None
        if isinstance(fetched_at_raw, str):
            try:
                fetched_at = datetime.fromisoformat(fetched_at_raw)
            except ValueError:
                fetched_at = None

        if fetched_at is None:
            fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        elif fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)

        return datetime.now(UTC) - fetched_at <= self._cache_ttl

    async def is_allowed(self, url: str) -> bool:
        """Return True if robots.txt permits crawling this URL."""
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        if base not in self._robots_cache:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{base}/robots.txt",
                        headers={"User-Agent": self._user_agent},
                    )
                if resp.status_code == 404:
                    self._robots_cache[base] = _AllowAllRobotsPolicy()
                elif _is_fandom_robots_challenge(resp):
                    logger.warning(
                        "robots.txt for %s returned a browser challenge; using "
                        "conservative Fandom fallback policy for article/API paths",
                        base,
                    )
                    self._robots_cache[base] = _FandomFallbackRobotsPolicy()
                else:
                    resp.raise_for_status()
                    rp = urllib.robotparser.RobotFileParser()
                    rp.parse(resp.text.splitlines())
                    self._robots_cache[base] = rp
            except Exception as exc:
                logger.warning(
                    "Failed to fetch robots.txt for %s; refusing crawl until "
                    "robots policy can be determined: %s",
                    base,
                    exc,
                )
                self._robots_cache[base] = _DisallowAllRobotsPolicy()

        return self._robots_cache[base].can_fetch(self._user_agent, url)

    def _fandom_api_url(self, url: str) -> tuple[str, str] | None:
        """If `url` is a Fandom wiki page, return (api_url, page_title).

        Fandom sits behind Cloudflare bot protection that 403s direct /wiki/
        requests, but the MediaWiki API at /api.php responds normally to a
        plain User-Agent header. We rewrite fetches transparently; the
        caller still sees the original /wiki/ URL as the canonical source.
        """
        parsed = urlparse(url)
        if not parsed.netloc.endswith(".fandom.com"):
            return None
        if not parsed.path.startswith("/wiki/"):
            return None
        page_title = parsed.path[len("/wiki/"):]
        encoded = quote(unquote(page_title), safe="")
        api_url = (
            f"{parsed.scheme}://{parsed.netloc}/api.php"
            f"?action=parse&page={encoded}&format=json&prop=text"
            f"&redirects=1&formatversion=2"
        )
        return api_url, unquote(page_title).replace("_", " ")

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

        fandom_info = self._fandom_api_url(url)
        fetch_url = fandom_info[0] if fandom_info else url
        if fetch_url != url and not await self.is_allowed(fetch_url):
            logger.warning(
                "Robots policy disallows rewritten fetch URL %s for source %s",
                fetch_url,
                url,
            )
            return None

        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            ) as client:
                response = await client.get(fetch_url)
                response.raise_for_status()
                body = response.text
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

        api_title: str | None = None
        if fandom_info:
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                logger.warning("MediaWiki API returned non-JSON for %s: %s", url, exc)
                return None
            if isinstance(data, dict) and "error" in data:
                err = data["error"]
                logger.warning(
                    "MediaWiki API error for %s: code=%s info=%r",
                    url, err.get("code"), err.get("info"),
                )
                return None
            try:
                html = data["parse"]["text"]
                api_title = data["parse"].get("title")
            except (KeyError, TypeError) as exc:
                logger.warning("Unexpected MediaWiki API response shape for %s: %s", url, exc)
                return None
        else:
            html = body

        self._save_to_cache(url, html)
        title, text = self._extract_text(html, url)
        if not title and api_title:
            title = api_title
        return ScrapedPage(url=url, text=text, title=title, fetched_at=datetime.now(UTC))
