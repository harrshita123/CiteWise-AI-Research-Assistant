from urllib.parse import urlparse
from collections import OrderedDict
from dataclasses import dataclass
import re
import time
from threading import Lock

import requests

from .answers import ExternalEvidence


@dataclass(frozen=True)
class SearchOutcome:
    results: list[ExternalEvidence]
    cache_hit: bool
    rejected_irrelevant: int = 0


class TavilyExternalSearch:
    """Return cited search excerpts; never use Tavily's uncited generated answer."""

    endpoint = "https://api.tavily.com/search"
    trusted_hosts = (
        "ncbi.nlm.nih.gov", "arxiv.org", "doi.org", "acm.org", "ieee.org",
        "springer.com", "springerlink.com", "nature.com", "sciencedirect.com",
        "tandfonline.com", "sagepub.com", "wiley.com", "plos.org",
        "frontiersin.org", "mdpi.com", "react.dev", "redux.js.org",
        "developer.mozilla.org", "docs.python.org", "w3.org",
    )
    india_law_hosts = ("meity.gov.in", "indiacode.nic.in", "egazette.gov.in",
                       "uidai.gov.in", "dbtbharat.gov.in")
    search_depth = "advanced"
    max_results = 10

    @classmethod
    def _trusted_host(cls, host: str) -> bool:
        return (host.endswith((".gov", ".gov.in", ".edu", ".ac.uk", ".ac.in"))
                or any(host == domain or host.endswith(f".{domain}") for domain in cls.trusted_hosts))

    def __init__(self, api_key: str, timeout_seconds: int = 15):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._cache: OrderedDict[str, tuple[float, tuple[ExternalEvidence, ...], int]] = OrderedDict()
        self._cache_lock = Lock()

    def available(self) -> bool:
        return bool(self.api_key)

    def _cache_key(self, query: str) -> str:
        normalized = " ".join(query.casefold().split())
        return f"{self.search_depth}|{self.max_results}|{normalized}"

    def _cached_entry(self, query: str) -> tuple[list[ExternalEvidence], int] | None:
        key = self._cache_key(query)
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry[0] <= time.monotonic():
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return list(entry[1]), entry[2]

    def cached(self, query: str) -> list[ExternalEvidence] | None:
        entry = self._cached_entry(query)
        return entry[0] if entry is not None else None

    def search(self, query: str) -> list[ExternalEvidence]:
        return self.search_with_trace(query).results

    @staticmethod
    def _relevant(query: str, title: str, content: str) -> bool:
        ignored = {"about", "additional", "after", "before", "could", "current", "details",
                   "does", "effects", "evidence", "external", "findings", "from", "have",
                   "information", "into", "legal", "missing", "more", "paper", "question",
                   "reported", "requirements", "research", "source", "study", "than",
                   "that", "their", "these", "this", "what",
                   "when", "where", "which", "with", "would"}
        terms = {word for word in re.findall(r"[a-z]{4,}", query.casefold()) if word not in ignored}
        if not terms:
            return True
        source_terms = set(re.findall(r"[a-z]{4,}", f"{title} {content}".casefold()))
        return len(terms & source_terms) >= min(2, len(terms))

    def search_with_trace(self, query: str) -> SearchOutcome:
        if not self.available():
            return SearchOutcome([], False)
        cached = self._cached_entry(query)
        if cached is not None:
            return SearchOutcome(cached[0], True, cached[1])
        official_india_law = bool(re.search(r"\b(india|indian)\b", query, re.I)
                                  and re.search(r"\b(legal|law|regulation|biometric|data protection)\b", query, re.I))
        search_query = f"{query} site:gov.in OR site:nic.in" if official_india_law else query
        response = requests.post(self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"query": search_query[:500], "search_depth": self.search_depth, "max_results": self.max_results,
                  "include_answer": False, "include_published_date": True},
            timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Tavily returned a malformed search response")
        results = []
        rejected_irrelevant = 0
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            url = item.get("url", "")
            if not isinstance(url, str):
                continue
            parsed = urlparse(url)
            content = item.get("content", "")
            if parsed.scheme != "https" or not parsed.netloc or not isinstance(content, str) or not content.strip():
                continue
            host = (parsed.hostname or "").lower()
            if official_india_law and not any(
                    host == domain or host.endswith(f".{domain}") for domain in self.india_law_hosts):
                continue
            if not official_india_law and not self._trusted_host(host):
                continue
            title = item.get("title") if isinstance(item.get("title"), str) else parsed.netloc
            if not self._relevant(query, title, content):
                rejected_irrelevant += 1
                continue
            results.append(ExternalEvidence(
                id=f"ext{len(results) + 1}", title=title or parsed.netloc,
                url=url, content=content[:3000],
                published_date=item.get("published_date") if isinstance(item.get("published_date"), str) else None,
                publisher=item.get("publisher") if isinstance(item.get("publisher"), str) else None,
                author=item.get("author") if isinstance(item.get("author"), str) else None))
            if len(results) >= 5:
                break
        if results:
            key = self._cache_key(query)
            with self._cache_lock:
                self._cache[key] = (time.monotonic() + 1800, tuple(results), rejected_irrelevant)
                self._cache.move_to_end(key)
                if len(self._cache) > 64:
                    self._cache.popitem(last=False)
        return SearchOutcome(results, False, rejected_irrelevant)
