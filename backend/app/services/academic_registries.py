"""PubMed and DataCite adapters for the shared academic transport."""
import re
from datetime import datetime, timezone
from urllib.parse import quote
from xml.etree import ElementTree as ET


def medical_identifier(value):
    match = re.fullmatch(r"(?:PMID\s*:\s*|https://pubmed\.ncbi\.nlm\.nih\.gov/)([1-9]\d{0,8})/?", value, re.I)
    if match:
        return "pmid", match[1]
    match = re.fullmatch(r"(?:PMCID\s*:\s*)?(PMC[1-9]\d{0,9})", value, re.I)
    if match:
        return "pmcid", match[1].upper()
    return None


def _text(node, path):
    element = node.find(path)
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _record(title, authors, year, venue, publisher, kind, source, url, note, **ids):
    doi, pmid = ids.get("doi"), ids.get("pmid")
    return dict(id="doi:" + doi if doi else "pmid:" + pmid, title=title, authors=authors,
                year=year, venue=venue, publisher=publisher, publication_type=kind,
                status="verified_preprint" if kind == "preprint" else "verified_published",
                landing_url=url, sources=[source], status_note=note, pdf_url=None, arxiv_id=None,
                verified_at=datetime.now(timezone.utc).isoformat(), **ids)


def pubmed(client, query, intent, offset, limit, normalize_doi):
    ident = medical_identifier(query)
    doi = normalize_doi(query)
    term = re.sub(r'^author:\s*', '', query, flags=re.I).replace('"', ' ')
    if ident and ident[0] == "pmid":
        ids, more = [ident[1]], False
    else:
        field = "pmc" if ident else "AID" if doi else {"author": "Author", "title": "Title"}.get(intent)
        term = ident[1] if ident else doi or term
        expression = f'"{term}"[{field}]' if field else term
        if intent == "author" and not ident and not doi:
            words = term.split()
            if len(words) >= 2 and "," not in term:
                # PubMed indexes full names in surname-first order.
                surname_first = " ".join([words[-1], *words[:-1]])
                expression += f' OR "{surname_first}"[Full Author Name]'
        params = dict(db="pubmed", term=expression,
                      retstart=offset, retmax=limit, retmode="json")
        search = client.json("pubmed", "/entrez/eutils/esearch.fcgi", params)["esearchresult"]
        ids = [str(i) for i in search["idlist"] if re.fullmatch(r"[1-9]\d{0,8}", str(i))]
        more = offset + len(ids) < int(search["count"])
    records, rejected = [], set()
    if not ids:
        return records, rejected, more
    root = ET.fromstring(client.get("pubmed", "/entrez/eutils/efetch.fcgi",
                         dict(db="pubmed", id=",".join(ids), retmode="xml")))
    if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
        raise ValueError("invalid provider response")
    for item in root.findall("PubmedArticle"):
        pmid = _text(item, "MedlineCitation/PMID")
        if pmid not in ids:
            continue
        article = item.find("MedlineCitation/Article")
        if article is None:
            continue
        identifiers = {node.get("IdType"): node.text or "" for node in item.findall("PubmedData/ArticleIdList/ArticleId")}
        found = normalize_doi(identifiers.get("doi", ""))
        pmc = identifiers.get("pmc")
        if doi and doi != found or ident and ident[0] == "pmcid" and pmc != ident[1]:
            continue
        types = {node.text for node in article.findall("PublicationTypeList/PublicationType")}
        title = _text(article, "ArticleTitle")
        retracted = "Retracted Publication" in types or item.find(".//CommentsCorrections[@RefType='RetractionIn']") is not None
        if "Retraction of Publication" in types:
            rejected.update(filter(None, [found, "pmid:" + pmid]))
            continue
        if not title or not retracted and not types & {"Journal Article", "Review", "Preprint", "Systematic Review", "Meta-Analysis"}:
            continue
        authors = [" ".join(filter(None, [_text(a, "ForeName") or _text(a, "Initials"), _text(a, "LastName")]))
                   or _text(a, "CollectiveName") for a in article.findall("AuthorList/Author")]
        date = article.find("Journal/JournalIssue/PubDate")
        year = re.search(r"\b\d{4}\b", " ".join(date.itertext())) if date is not None else None
        records.append(_record(title, list(filter(None, authors)), int(year[0]) if year else None,
            _text(article, "Journal/Title"), None, "preprint" if "Preprint" in types else "journal-article",
            "PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "PubMed publication types and linked retraction notices checked.", doi=found, pmid=pmid, pmcid=pmc))
        if retracted:
            records[-1].update(status="retracted", status_note="PubMed reports a retracted publication.")
        elif not authors or not year or not _text(article, "Journal/Title"):
            records[-1]["status"] = "registry_record"
    return records, rejected, more


def datacite(client, query, intent, offset, limit, normalize_doi):
    doi = normalize_doi(query)
    if doi:
        rows = [client.json("datacite", "/dois/" + quote(doi, safe="/"))["data"]]
        more = False
    else:
        field = "creators.name" if intent == "author" else "titles.title"
        terms = re.findall(r"\w+", re.sub(r"^author:\s*", "", query, flags=re.I))
        if not terms:
            return [], set(), False
        expression = f'{field}:("' + " ".join(terms) + '")'
        data = client.json("datacite", "/dois", {"query": expression, "page[size]": limit,
                           "page[number]": offset // limit + 1})
        rows = data["data"]
        more = offset + len(rows) < data.get("meta", {}).get("total", 0)
    records, rejected = [], set()
    for row in rows:
        attr = row["attributes"]
        found = normalize_doi(attr.get("doi", ""))
        if not found or found != normalize_doi(row["id"]) or doi and found != doi or attr.get("state") != "findable":
            continue
        types = attr.get("types", {})
        general = (types.get("resourceTypeGeneral") or "").casefold()
        kind = (types.get("resourceType") or "").strip().casefold() if general == "text" else general
        title = next((t["title"] for t in attr.get("titles", []) if t.get("title") and not t.get("titleType")), "")
        if re.search(r"^\s*(?:\[?(?:withdrawn|retracted)\b|retraction[: ])", title, re.I) or any(
                r.get("relationType") == "IsObsoletedBy" for r in attr.get("relatedIdentifiers", [])):
            rejected.add(found)
            continue
        if not title or kind not in {"journalarticle", "journal article", "conferencepaper", "conference paper", "preprint"}:
            continue
        authors = [" ".join(filter(None, [c.get("givenName"), c.get("familyName")])) or c.get("name")
                   for c in attr.get("creators", [])]
        records.append(_record(title, list(filter(None, authors)), attr.get("publicationYear"), None,
            attr.get("publisher"), kind, "DataCite", "https://commons.datacite.org/doi.org/" + quote(found, safe="/"),
            "DataCite registration and resource type checked. Retraction coverage is limited.", doi=found))
        records[-1]["status"] = "registry_record"
        # A registry landing URL identifies a repository edition. Require an
        # independently retrieved arXiv record before assigning a shared group.
        if kind == "preprint":
            match = re.fullmatch(r"https?://arxiv\.org/abs/([^?#]+?)/?", attr.get("url") or "", re.I)
            if match:
                records[-1]["repository_identifier"] = match[1]
    return records, rejected, more
