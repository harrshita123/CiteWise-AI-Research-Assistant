from urllib.parse import urlparse
from ipaddress import ip_address
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
    raw_results: int = 0
    invalid_results: int = 0
    invalid_urls: int = 0
    rejected_domains: int = 0
    duplicates: int = 0


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
    search_depth = "advanced"
    max_results = 10

    @classmethod
    def _trusted_host(cls, host: str) -> bool:
        return (bool(re.search(r"\.(?:gov|edu|ac)\.[a-z]{2}$", host))
                or host.endswith((".gov", ".edu"))
                or any(host == domain or host.endswith(f".{domain}") for domain in cls.trusted_hosts))

    def __init__(self, api_key: str, timeout_seconds: int = 15):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._cache: OrderedDict[str, tuple[float, SearchOutcome]] = OrderedDict()
        self._cache_lock = Lock()

    def available(self) -> bool:
        return bool(self.api_key)

    def _cache_key(self, query: str) -> str:
        normalized = " ".join(query.casefold().split())
        return f"{self.search_depth}|{self.max_results}|{normalized}"

    def _cached_entry(self, query: str) -> SearchOutcome | None:
        key = self._cache_key(query)
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry[0] <= time.monotonic():
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry[1]

    def cached(self, query: str) -> list[ExternalEvidence] | None:
        entry = self._cached_entry(query)
        return list(entry.results) if entry is not None else None

    def search(self, query: str) -> list[ExternalEvidence]:
        return self.search_with_trace(query).results

    @staticmethod
    def _terms(value: str) -> set[str]:
        aliases = {"webcam": "camera", "cameras": "camera", "monitoring": "monitor",
                   "surveillance": "monitor", "proctoring": "monitor", "proctored": "monitor",
                   "confidentiality": "privacy", "biometrics": "biometric",
                   "regulations": "law", "legislation": "law", "requirements": "law",
                   "universities": "university", "colleges": "university",
                   "students": "student", "participants": "participant", "effects": "effect"}
        ignored = {"about", "additional", "after", "before", "could", "current", "details",
                   "does", "effects", "evidence", "external", "findings", "from", "have",
                   "information", "into", "legal", "missing", "more", "paper", "question",
                   "reported", "requirements", "research", "source", "study", "than",
                   "that", "their", "these", "this", "what",
                   "when", "where", "which", "with", "would"}
        return {aliases.get(word, word[:-1] if word.endswith("s") and len(word) > 4 else word)
                for word in re.findall(r"[a-z]{4,}", value.casefold()) if word not in ignored}

    @classmethod
    def _relevant(cls, query: str, title: str, content: str) -> bool:
        terms = cls._terms(query)
        if not terms:
            return False
        title_terms = cls._terms(title)
        source_terms = title_terms | cls._terms(content)
        matches = terms & source_terms
        return bool(matches) and (len(matches) >= min(2, len(terms))
                                  or bool(matches & title_terms) and len(terms) == 1)

    @staticmethod
    def _valid_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            if parsed.scheme != "https" or not host or parsed.username or parsed.password:
                return False
            if parsed.port not in (None, 443) or not re.fullmatch(r"[a-z0-9.-]+", host, re.I):
                return False
            if ".." in host or host.startswith(".") or host.endswith("."):
                return False
            try:
                ip_address(host)
                return False
            except ValueError:
                return True
        except ValueError:
            return False

    def search_with_trace(self, query: str) -> SearchOutcome:
        if not self.available():
            return SearchOutcome([], False)
        cached = self._cached_entry(query)
        if cached is not None:
            return SearchOutcome(list(cached.results), True, cached.rejected_irrelevant,
                                 cached.raw_results, cached.invalid_results, cached.invalid_urls,
                                 cached.rejected_domains, cached.duplicates)
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
        candidates: list[tuple[int, ExternalEvidence]] = []
        rejected_irrelevant = 0
        invalid_results = invalid_urls = rejected_domains = duplicates = 0
        seen_urls: set[str] = set()
        for item in payload["results"]:
            if not isinstance(item, dict):
                invalid_results += 1
                continue
            url = item.get("url", "")
            if not isinstance(url, str):
                invalid_urls += 1
                continue
            content = item.get("content", "")
            if not isinstance(content, str) or not content.strip():
                invalid_results += 1
                continue
            if not self._valid_url(url):
                invalid_urls += 1
                continue
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if official_india_law and not host.endswith((".gov.in", ".nic.in")):
                rejected_domains += 1
                continue
            if not official_india_law and not self._trusted_host(host):
                rejected_domains += 1
                continue
            title = item.get("title") if isinstance(item.get("title"), str) else parsed.netloc
            if not self._relevant(query, title, content):
                rejected_irrelevant += 1
                continue
            canonical = f"{host}{parsed.path.rstrip('/') or '/'}"
            if canonical in seen_urls:
                duplicates += 1
                continue
            seen_urls.add(canonical)
            source = ExternalEvidence(
                id="", title=title or parsed.netloc,
                url=url, content=content[:3000],
                published_date=item.get("published_date") if isinstance(item.get("published_date"), str) else None,
                publisher=item.get("publisher") if isinstance(item.get("publisher"), str) else None,
                author=item.get("author") if isinstance(item.get("author"), str) else None)
            query_terms = self._terms(query)
            score = 2 * len(query_terms & self._terms(title)) + len(query_terms & self._terms(content))
            candidates.append((score, source))
        candidates.sort(key=lambda item: item[0], reverse=True)
        diverse = []
        repeated_hosts = []
        selected_hosts = set()
        for _, source in candidates:
            host = urlparse(source.url).hostname
            if host in selected_hosts:
                repeated_hosts.append(source)
            else:
                diverse.append(source)
                selected_hosts.add(host)
        results = [ExternalEvidence(id=f"ext{index}", title=item.title, url=item.url,
                   content=item.content, published_date=item.published_date, publisher=item.publisher,
                   author=item.author) for index, item in enumerate((diverse + repeated_hosts)[:5], 1)]
        outcome = SearchOutcome(results, False, rejected_irrelevant, len(payload["results"]),
                                invalid_results, invalid_urls, rejected_domains, duplicates)
        key = self._cache_key(query)
        with self._cache_lock:
            self._cache[key] = (time.monotonic() + (1800 if results else 300), outcome)
            self._cache.move_to_end(key)
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
        return outcome
