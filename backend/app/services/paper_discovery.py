"""Academic discovery. Provider records are untrusted until registry checks pass."""
from __future__ import annotations

import json
import hashlib
import logging
import re
import random
import time
from .publication_metadata import metadata_conflict, names_agree, normalized_title
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher
from threading import Lock
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree

import requests

from .academic_registries import pubmed, datacite, medical_identifier

DOI = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9]+$", re.I)
ARXIV = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v[1-9]\d*)?$", re.I)
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def normalize_doi(value: str) -> str | None:
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I).lower()
    return value if len(value) <= 255 and DOI.fullmatch(value) else None


def normalize_arxiv(value: str) -> str | None:
    value = re.sub(r"^(?:arxiv:\s*|https?://arxiv\.org/abs/)", "", value.strip(), flags=re.I)
    return value.lower() if ARXIV.fullmatch(value) else None


def search_intent(query: str) -> str:
    if normalize_doi(query): return "doi"
    if normalize_arxiv(query): return "arxiv"
    if query.lower().startswith("author:"): return "author"
    return "title" if len(normalized_title(query).split()) >= 6 else "topic"


def rank_record(record: dict, query: str, intent: str) -> tuple:
    title, target = normalized_title(record["title"]), normalized_title(query)
    exact = title == target
    similarity = SequenceMatcher(None, title, target).ratio() if title and target else 0
    terms = set(target.split())
    relevance = len(terms & set(title.split())) / len(terms) if terms else 0
    if intent == "author":
        author_query = normalized_title(query.removeprefix("author:").strip())
        relevance = max((SequenceMatcher(None, normalized_title(name), author_query).ratio()
                         for name in record.get("authors", [])), default=0)
    ident = medical_identifier(query)
    identifier_match = bool((ident and record.get(ident[0]) == ident[1])
                            or (normalize_doi(query) and record.get("doi") == normalize_doi(query))
                            or (normalize_arxiv(query) and (record.get("arxiv_id") or "").split("v")[0] == normalize_arxiv(query).split("v")[0]))
    verified = record["status"] in {"verified_published", "verified_preprint"}
    return (identifier_match, exact, similarity >= 0.9 if intent == "title" else False,
            relevance, similarity, verified, record.get("publication_type") != "book-chapter", len(record["sources"]), bool(record.get("venue")),
            bool(record.get("year")), record["id"])


def provider_message(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        return str(exc) if str(exc) in {"not configured", "timeout", "connection error", "response too large", "invalid provider response", "rate limited", "authentication error"} or re.fullmatch(r"HTTP \d{3}", str(exc)) else "unavailable"
    return "invalid provider response"


class ProviderError(Exception):
    pass


class AcademicClient:
    """Fixed HTTPS origins, no redirects, bounded response and no user-supplied URLs."""
    hosts = {"crossref": "https://api.crossref.org", "openalex": "https://api.openalex.org",
             "pubmed": "https://eutils.ncbi.nlm.nih.gov", "datacite": "https://api.datacite.org",
             "arxiv": "https://export.arxiv.org"}

    def __init__(self, openalex_key: str = "", crossref_email: str = "", coordinator=None, ncbi_key=""):
        self.ncbi_key = ncbi_key
        self.openalex_key = openalex_key
        self.crossref_email = crossref_email.strip()
        self.coordinator = coordinator
        self._cache = {}
        self._locks = {provider: Lock() for provider in self.hosts}
        self._next_request = {provider: 0.0 for provider in self.hosts}
        self._cooldowns = {provider: 0.0 for provider in self.hosts}
        self._health = {provider: "not checked" for provider in self.hosts}

    def health(self):
        return {provider: {"configured": bool(self.openalex_key) if provider == "openalex" else True,
                           "keyless_available": provider in {"openalex", "pubmed"}, "status": self._health[provider]}
                for provider in self.hosts}

    def retry_delay(self):
        """Remaining observed provider cooldown in seconds, without exposing headers."""
        local = max(0.0, *(deadline - time.monotonic() for deadline in self._cooldowns.values()))
        return max(local, self.coordinator.retry_delay()) if self.coordinator else local

    @staticmethod
    def _request_interval(provider, path):
        if provider == "crossref":
            return 1.0 if path == "/works" else 0.25
        return {"arxiv": 3.0, "pubmed": 0.35}.get(provider, 0.1)

    @staticmethod
    def _retry_after(value):
        try:
            return max(0, float(value))
        except (TypeError, ValueError):
            try:
                return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return 60.0

    def get(self, provider: str, path: str, params: dict | None = None):
        if provider not in self.hosts or not path.startswith("/") or path.startswith("//"):
            raise ProviderError("invalid provider request")
        logging.getLogger(__name__).debug("academic lookup provider=%s", provider)
        if not self.coordinator:
            return self._get_local(provider, path, params)
        if provider == "crossref" and path.startswith("/works/"):
            doi = normalize_doi(unquote(path[len("/works/"):]))
            if doi:
                path = "/works/" + quote(doi, safe="")
        params = dict(params or {})
        if provider == "arxiv" and "id_list" in params:
            identifier = normalize_arxiv(str(params["id_list"]))
            if identifier:
                params["id_list"] = identifier
        for field in ("search", "query.title", "query.author", "query.bibliographic"):
            if isinstance(params.get(field), str):
                params[field] = params[field].strip()
        # The digest avoids storing query contents or credentials in cache keys.
        cache_key = hashlib.sha256(json.dumps([provider, path, params or {}], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        try:
            cached = self.coordinator.cached(provider, cache_key)
            if cached is not None:
                return cached
            deadline = time.monotonic() + 4
            token = None
            while token is None and time.monotonic() < deadline:
                token = self.coordinator.acquire(provider)
                if token is None:
                    cached = self.coordinator.cached(provider, cache_key)
                    if cached is not None:
                        return cached
                    time.sleep(0.1 + random.uniform(0, 0.05))
        except Exception as exc:
            raise ProviderError("connection error") from exc
        if not token:
            raise ProviderError("rate limited")
        interval = self._request_interval(provider, path)
        try:
            with self.coordinator.keep_alive(provider, token) as lost:
                cached = self.coordinator.cached(provider, cache_key)
                if cached is not None:
                    return cached
                data = self._get_local(provider, path, params, lost, token)
                if lost.is_set():
                    raise ProviderError("connection error")
                try:
                    if provider == "arxiv":
                        valid = ElementTree.fromstring(data).tag == ATOM + "feed"
                    elif provider == "pubmed" and path.endswith("efetch.fcgi"):
                        root = ElementTree.fromstring(data)
                        valid = root.tag == "PubmedArticleSet" and root.find(".//ERROR") is None
                    else:
                        document = json.loads(data)
                        if not isinstance(document, dict):
                            valid = False
                        elif provider == "datacite":
                            valid = isinstance(document.get("data"), list if path == "/dois" else dict)
                        elif provider == "pubmed":
                            valid = isinstance(document.get("esearchresult", {}).get("idlist"), list)
                        elif provider == "crossref":
                            message = document.get("message")
                            valid = isinstance(message, dict) and (path != "/works" or isinstance(message.get("items"), list))
                        else:
                            valid = isinstance(document.get("results"), list) if path == "/works" else isinstance(document.get("id"), str)
                except (ValueError, UnicodeDecodeError, ElementTree.ParseError):
                    valid = False
                if not valid:
                    raise ProviderError("invalid provider response")
                if not self.coordinator.publish(provider, token, cache_key, data):
                    raise ProviderError("connection error")
                return data
        finally:
            self.coordinator.release(provider, token, interval)

    def _get_local(self, provider: str, path: str, params: dict | None = None, lost=None, token=None):
        if provider not in self.hosts or not path.startswith("/") or path.startswith("//"):
            raise ProviderError("invalid provider request")
        params = dict(params or {})
        headers = {"User-Agent": "CiteWise/1.0 (academic discovery)"}
        if provider == "openalex" and self.openalex_key:
            headers["Authorization"] = "Bearer " + self.openalex_key
        if provider == "crossref" and self.crossref_email:
            params["mailto"] = self.crossref_email
        key = (provider, path, tuple(sorted(params.items())))
        if provider == "pubmed" and self.ncbi_key:
            params["api_key"] = self.ncbi_key
        with self._locks[provider]:
            cached = self._cache.get(key) if not self.coordinator else None
            if cached and cached[0] > time.monotonic():
                return cached[1]
            if self._cooldowns[provider] > time.monotonic():
                raise ProviderError("rate limited")
            delay = self._next_request[provider] - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            interval = self._request_interval(provider, path)
            self._next_request[provider] = time.monotonic() + interval
            try:
                for attempt in range(2):
                    if lost is not None and lost.is_set():
                        raise ProviderError("connection error")
                    with requests.get(self.hosts[provider] + path, params=params, timeout=(5, 18),
                                      allow_redirects=False, stream=True, headers=headers) as response:
                        if response.status_code == 429:
                            cooldown = self._retry_after(response.headers.get("Retry-After"))
                            self._cooldowns[provider] = time.monotonic() + cooldown
                            if self.coordinator:
                                self.coordinator.defer(provider, cooldown, token)
                            self._health[provider] = "rate limited"
                            raise ProviderError("rate limited")
                        if response.status_code in (401, 403):
                            self._health[provider] = "authentication error" if provider == "openalex" and self.openalex_key else "temporarily unavailable"
                            raise ProviderError("authentication error" if provider == "openalex" and self.openalex_key else f"HTTP {response.status_code}")
                        if 500 <= response.status_code <= 599 and attempt == 0:
                            time.sleep(max(interval, 0.5) + random.uniform(0, 0.25))
                            self._next_request[provider] = time.monotonic() + interval
                            continue
                        if response.status_code == 404:
                            self._health[provider] = "healthy"
                            raise ProviderError("HTTP 404")
                        if response.status_code != 200:
                            self._health[provider] = "temporarily unavailable"
                            raise ProviderError(f"HTTP {response.status_code}")
                        chunks, size = [], 0
                        for chunk in response.iter_content(65536):
                            if lost is not None and lost.is_set():
                                raise ProviderError("connection error")
                            size += len(chunk)
                            if size > 2_000_000:
                                raise ProviderError("response too large")
                            chunks.append(chunk)
                        data = b"".join(chunks)
                        break
            except requests.Timeout as exc:
                self._health[provider] = "timed out"
                raise ProviderError("timeout") from exc
            except requests.RequestException as exc:
                self._health[provider] = "temporarily unavailable"
                raise ProviderError("connection error") from exc
            self._health[provider] = "healthy"
            if len(self._cache) >= 512:
                self._cache.clear()
            if not self.coordinator:
                self._cache[key] = (time.monotonic() + 300, data)
            return data

    def json(self, provider, path, params=None):
        try:
            return json.loads(self.get(provider, path, params))
        except (ValueError, UnicodeDecodeError) as exc:
            self._health[provider] = "invalid response"
            raise ProviderError("invalid provider response") from exc


class PaperDiscoveryService:
    def __init__(self, client: AcademicClient):
        self.client = client

    def _crossref(self, doi: str, candidate_title: str | None = None, candidate_authors: list[str] | None = None, diagnostics=None):
        message = self.client.json("crossref", "/works/" + quote(doi, safe=""))
        item = message.get("message", {}) if isinstance(message, dict) else {}
        actual = normalize_doi(str(item.get("DOI", "")))
        titles = item.get("title") or []
        title = titles[0] if isinstance(titles, list) and titles and isinstance(titles[0], str) else ""
        if actual != doi:
            return None
        kind = item.get("type", "")
        if kind not in {"journal-article", "proceedings-article", "posted-content", "book-chapter"}:
            if diagnostics is not None and kind == "component":
                diagnostics.append({"id": "doi:" + doi, "reason": "component_identifier", "source": "Crossref"})
            return None
        subtitles = item.get("subtitle") or []
        if title and isinstance(subtitles, list) and subtitles and isinstance(subtitles[0], str):
            if not normalized_title(title).endswith(normalized_title(subtitles[0])):
                title = f"{title}: {subtitles[0]}"
        if not title or (candidate_title and normalized_title(title) != normalized_title(candidate_title)):
            return None
        # Crossref registers corrections/retractions as separate records. Check every
        # linked update, failing closed when the notice cannot be retrieved.
        notices = item.get("updated-by") or []
        retracted = False
        for notice in notices:
            if not isinstance(notice, dict):
                return None
            if str(notice.get("type", "")).lower() in {"retraction", "withdrawal", "withdrawn"}:
                retracted = True
                break
            notice_doi = normalize_doi(str(notice.get("DOI", "")))
            if not notice_doi:
                return None
            detail = self.client.json("crossref", "/works/" + quote(notice_doi, safe=""))
            update = detail.get("message", {})
            if any(str(ref.get("type", "")).lower() in {"retraction", "withdrawal"}
                   for ref in update.get("update-to", []) if isinstance(ref, dict)):
                retracted = True
                break
        authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) or a.get("name")
                   for a in item.get("author", [])
                   if isinstance(a, dict) and (a.get("family") or a.get("given") or a.get("name"))]
        if candidate_authors and authors:
            registry_names = {normalized_title(name).split()[-1] for name in authors if normalized_title(name)}
            candidate_names = {normalized_title(name).split()[-1] for name in candidate_authors if normalized_title(name)}
            if registry_names and candidate_names and not any(names_agree(x,y) for x in authors for y in candidate_authors):
                return None
        years = []
        for field in ("published", "published-online", "published-print", "issued"):
            dates = (item.get(field) or {}).get("date-parts") or []
            if dates and dates[0] and isinstance(dates[0][0], int):
                years.append(dates[0][0])
        year = years[0] if years else None
        venues = item.get("container-title") or []
        return {"id": "doi:" + doi, "doi": doi, "arxiv_id": None, "title": title,
                "authors": authors, "year": year, "venue": venues[0] if venues else None,
                "publisher": item.get("publisher"), "publication_type": kind, "publication_years": sorted(set(years)),
                "status": "retracted" if retracted else "registry_record",
                "landing_url": "https://doi.org/" + quote(doi, safe="/"), "pdf_url": None,
                "sources": ["Crossref"], "status_note": "Crossref reports a withdrawal or retraction." if retracted else "No retraction identified in checked Crossref metadata; coverage may be incomplete.",
                "verified_at": datetime.now(timezone.utc).isoformat()}

    def _arxiv(self, entry):
        raw_id = entry.findtext(ATOM + "id", "")
        match = re.fullmatch(r"https?://arxiv\.org/abs/(.+)", raw_id)
        identifier = normalize_arxiv(match.group(1)) if match else None
        title = " ".join(entry.findtext(ATOM + "title", "").split())
        comments = entry.findtext(ARXIV_NS + "comment", "")
        if not identifier or not title:
            return None
        published = entry.findtext(ATOM + "published", "")
        return {"id": "arxiv:" + identifier, "doi": None, "arxiv_id": identifier, "title": title,
                "authors": [a.findtext(ATOM + "name", "") for a in entry.findall(ATOM + "author") if a.findtext(ATOM + "name")],
                "year": int(published[:4]) if re.match(r"^\d{4}-", published) else None,
                "venue": None, "publisher": "arXiv", "publication_type": "preprint",
                "version_group": "arxiv:" + re.sub(r"v\d+$", "", identifier),
                "published_doi": normalize_doi(entry.findtext(ARXIV_NS + "doi", "")),
                "status": ("retracted" if re.search(r"^\s*(?:withdrawn|retracted)\b", title, re.I)
                           or re.search(r"\b(?:paper|article|submission) (?:has been |was |is )?withdrawn\b", comments, re.I)
                           else "verified_preprint" if entry.findall(ATOM + "author") else "registry_record"),
                "landing_url": "https://arxiv.org/abs/" + identifier,
                "pdf_url": "https://arxiv.org/pdf/" + identifier, "sources": ["arXiv"],
                "status_note": "Official preprint; peer review and completeness of withdrawal reporting are not established.",
                "verified_at": datetime.now(timezone.utc).isoformat()}

    @staticmethod
    def _published_location(item, names):
        """Surface indexed conference editions whose work-level DOI is a preprint DOI."""
        work = item.get("id") or ""
        if (item.get("type") != "conference-paper" or item.get("is_retracted") is not False
                or not names or not item.get("publication_year")
                or not re.fullmatch(r"https://openalex\.org/W\d+", work)):
            return None
        for location in item.get("locations") or []:
            source = location.get("source") or {}
            if (location.get("version") != "publishedVersion" or source.get("type") != "conference"
                    or not source.get("display_name")):
                continue
            url = location.get("landing_page_url") or ""
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                continue
            if parsed.hostname in {"doi.org", "arxiv.org", "dx.doi.org"}:
                continue
            return {"id": "openalex:" + work.rsplit("/", 1)[1], "title": item["title"],
                    "doi": None, "arxiv_id": None, "authors": names, "year": item["publication_year"],
                    "venue": source["display_name"], "publisher": None,
                    "publication_type": "proceedings-article", "status": "registry_record",
                    "landing_url": url, "pdf_url": None, "sources": ["OpenAlex"],
                    "status_note": "OpenAlex indexes a published proceedings location. Follow the official record to check the edition."}
        return None

    def search(self, query: str, offset: int = 0, limit: int = 10):
        query = query.strip()
        doi = normalize_doi(query)
        arxiv = normalize_arxiv(query)
        intent = search_intent(query)
        if (query.lower().startswith(("doi:", "https://doi.org/", "http://doi.org/", "10.")) and not doi
                or query.lower().startswith(("arxiv:", "https://arxiv.org/abs/")) and not arxiv):
            raise ValueError("Invalid DOI or arXiv identifier.")
        errors, candidates, results, excluded, diagnostics = {}, [], {}, set(), []
        has_more = False
        medical = medical_identifier(query)
        alex_records = {}
        alex_retracted = set()
        if not arxiv and not medical:
            try:
                if doi:
                    candidates.append((doi, None, None))
                else:
                    field = "query.title" if intent == "title" else "query.author" if intent == "author" else "query.bibliographic"
                    search_text = re.sub(r"^author:\s*", "", query, flags=re.I)
                    data = self.client.json("crossref", "/works", {field: search_text, "rows": limit, "offset": offset})
                    items = data.get("message", {}).get("items", [])
                    has_more = len(items) >= limit
                    for item in items:
                        found = normalize_doi(str(item.get("DOI", ""))) if isinstance(item, dict) else None
                        titles = item.get("title") or [] if isinstance(item, dict) else []
                        if found and titles and isinstance(titles[0], str):
                            candidates.append((found, titles[0], None))
            except (ProviderError, AttributeError, TypeError) as exc:
                errors["crossref"] = provider_message(exc)
            if not doi:
                try:
                    data = self.client.json("openalex", "/works", {"search": re.sub(r"^author:\s*", "", query, flags=re.I), "per-page": limit, "page": offset // limit + 1})
                    has_more = has_more or len(data.get("results", [])) >= limit
                    for item in data.get("results", []):
                        if not isinstance(item, dict):
                            continue
                        found = normalize_doi(str(item.get("doi", "")))
                        if found and item.get("is_retracted") is True:
                            alex_retracted.add(found)
                            diagnostics.append({"id": "doi:" + found, "reason": "reported_retracted", "source": "OpenAlex"})
                            continue
                        if isinstance(item.get("title"), str):
                            names = [author.get("author", {}).get("display_name", "") for author in item.get("authorships", [])
                                     if isinstance(author, dict) and isinstance(author.get("author"), dict)]
                            if normalized_title(item["title"]) == normalized_title(query):
                                edition = self._published_location(item, list(filter(None, names)))
                                if edition:
                                    results[edition["id"]] = edition
                            if not found:
                                continue
                            alex_records[found] = {"doi": found, "title": item["title"], "authors": names,
                                                   "year": item.get("publication_year")}
                            candidates.append((found, item["title"], names))
                except (ProviderError, AttributeError, TypeError) as exc:
                    errors["openalex"] = provider_message(exc)
            for identifier, title, names in candidates[:limit * 2]:
                if identifier in excluded:
                    continue
                try:
                    if "doi:" + identifier in results:
                        continue
                    record = self._crossref(identifier, diagnostics=diagnostics)
                    if record:
                        alex = alex_records.get(identifier)
                        if identifier in alex_retracted:
                            record.update(status="retracted", status_note="OpenAlex flags this DOI as retracted.")
                            record["sources"].append("OpenAlex")
                        elif alex and record["status"] != "retracted":
                            if metadata_conflict(record, alex):
                                record.update(status="conflicting", status_note="Academic sources disagree on the metadata for this DOI.")
                            elif alex["authors"] and record["authors"] and record["year"] and record.get("venue"):
                                record["status"] = "verified_published" if record["publication_type"] != "posted-content" else "registry_record"
                                record["sources"].append("OpenAlex")
                        results[record["id"]] = record
                    else:
                        excluded.add(identifier)
                        results.pop("doi:" + identifier, None)
                        diagnostics.append({"id": "doi:" + identifier, "reason": "identity_or_status_check_failed", "source": "Crossref"})
                except (ProviderError, AttributeError, TypeError, KeyError, IndexError) as exc:
                    if not isinstance(exc, ProviderError) or str(exc) != "HTTP 404":
                        errors["crossref_verification"] = provider_message(exc)
        if not doi and not medical:
            try:
                safe_query = re.sub(r'["\\]', ' ', re.sub(r"^author:\s*", "", query, flags=re.I)).strip()
                field = "ti" if intent == "title" else "au" if intent == "author" else "all"
                params = {"id_list": arxiv} if arxiv else {"search_query": f'{field}:"{safe_query}"', "start": offset, "max_results": limit}
                root = ElementTree.fromstring(self.client.get("arxiv", "/api/query", params))
                entries = root.findall(ATOM + "entry")
                if not entries and intent == "title":
                    params["search_query"] = f'all:"{safe_query}"'
                    root = ElementTree.fromstring(self.client.get("arxiv", "/api/query", params))
                    entries = root.findall(ATOM + "entry")
                has_more = has_more or len(entries) >= limit
                for entry in entries:
                    record = self._arxiv(entry)
                    if record and (not arxiv or record["arxiv_id"] == arxiv or re.sub(r"v\d+$", "", record["arxiv_id"]) == arxiv):
                        results[record["id"]] = record
            except (ProviderError, ElementTree.ParseError, ValueError, TypeError) as exc:
                if isinstance(exc, ElementTree.ParseError) and hasattr(self.client, "_health"):
                    self.client._health["arxiv"] = "invalid response"
                errors["arxiv"] = provider_message(exc)
        if not arxiv:
            for provider, fetch in (("pubmed", pubmed), ("datacite", datacite)):
                if medical and provider != "pubmed":
                    continue
                try:
                    records, rejected, more = fetch(self.client, query, intent, offset, limit, normalize_doi)
                    has_more |= more
                    excluded.update(rejected)
                    for record in records:
                        key = record["id"]
                        existing = results.get(key)
                        if existing:
                            if "retracted" in {existing["status"], record["status"]}:
                                existing.update(status="retracted", status_note="A checked source reports a retraction or withdrawal.")
                                existing["sources"].extend(record["sources"])
                                continue
                            type_conflict = (existing["publication_type"] == "posted-content" and record["status"] == "verified_published"
                                             or record["publication_type"] == "preprint" and existing["publication_type"] in {"journal-article", "proceedings-article"})
                            if metadata_conflict(existing, record) or type_conflict or existing["status"] == "conflicting":
                                existing.update(status="conflicting", status_note="Academic sources disagree on the metadata for this DOI.")
                                existing["sources"].extend(record["sources"])
                                continue
                            existing["sources"].extend(record["sources"])
                            if record["status"] == "verified_published" and existing["publication_type"] != "posted-content":
                                existing["status"] = "verified_published"
                            for field in ("pmid", "pmcid"):
                                if record.get(field):
                                    existing[field] = record[field]
                            existing["status_note"] += " " + record["status_note"]
                        else:
                            results[key] = record
                    diagnostics.extend({"id": value, "reason": "excluded_by_provider_status", "source": provider} for value in rejected)
                except (ProviderError, ValueError, AttributeError, KeyError, TypeError, ElementTree.ParseError) as exc:
                    if str(exc) != "HTTP 404":
                        errors[provider] = provider_message(exc)
        for identifier in excluded:
            results.pop(identifier if identifier.startswith("pmid:") else "doi:" + identifier, None)
        for record in results.values():
            if record.get("doi") in alex_retracted:
                record.update(status="retracted", status_note="OpenAlex flags this DOI as retracted.")
            published_doi = record.get("published_doi")
            other = results.get("doi:" + published_doi) if published_doi else None
            if other and other["status"] not in {"conflicting", "retracted"} and any(
                    names_agree(a, b) for a in record["authors"] for b in other["authors"]):
                other["version_group"] = record["version_group"]
        for record in results.values():
            reference = normalize_arxiv(record.get("repository_identifier", ""))
            if not reference:
                continue
            for other in results.values():
                arxiv_id = other.get("arxiv_id")
                if not arxiv_id or reference not in {arxiv_id, re.sub(r"v\d+$", "", arxiv_id)}:
                    continue
                if (other["status"] == "verified_preprint" and record["status"] == "registry_record"
                        and normalized_title(record["title"]) == normalized_title(other["title"])
                        and any(names_agree(a, b) for a in record["authors"] for b in other["authors"])):
                    record["version_group"] = other["version_group"]
                    break
        published = [record for record in results.values()
                     if record.get("doi") and record["publication_type"] == "proceedings-article"]
        for key, record in list(results.items()):
            if key.startswith("openalex:") and any(
                    other["year"] == record["year"]
                    and normalized_title(other["title"]) == normalized_title(record["title"])
                    and normalized_title(other["venue"] or "") == normalized_title(record["venue"])
                    and any(names_agree(a, b) for a in record["authors"] for b in other["authors"])
                    for other in published):
                del results[key]
        ranked = sorted(results.values(), key=lambda item: rank_record(item, query, intent), reverse=True)
        if errors:
            logging.getLogger(__name__).info("academic_provider_status %s", errors)
        return {"results": ranked[:limit], "provider_errors": errors, "offset": offset, "limit": limit, "has_more": has_more,
                "_diagnostics": diagnostics[:40]}
