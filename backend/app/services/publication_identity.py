"""Conservative, document-scoped publication identity checks."""
from collections import Counter
import logging
import re
from datetime import datetime, timezone

import pymupdf

from .publication_metadata import fold_accents, name_parts, names_agree
from .academic_registries import medical_identifier
from .paper_discovery import normalize_arxiv, normalize_doi, normalized_title

logger = logging.getLogger(__name__)

def layout_title(page):
    candidates = []
    previous = None
    blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]
    for block in blocks:
        lines = block.get("lines", [])
        x0, top, x1, bottom = block["bbox"]
        if not lines or top > page.rect.height * .6 or any(line["dir"][0] < .99 for line in lines):
            previous = None
            continue
        spans = [span for line in lines for span in line["spans"] if span["text"].strip()]
        if not spans:
            previous = None
            continue
        size = max(span["size"] for span in spans)
        value = " ".join("".join(span["text"] for span in line["spans"]
                                 if not (span.get("flags", 0) & pymupdf.TEXT_FONT_SUPERSCRIPT
                                         and re.fullmatch(r"\s*[*†‡]+\s*", span["text"]))).strip()
                         for line in lines if any(span["size"] >= size * .94 and span["text"].strip()
                                                  for span in line["spans"]))
        if re.match(r"(?i)^(?:arxiv:|doi:)", value):
            previous = None
            continue
        center = (x0 + x1) / 2
        # Some PDFs place each line of a centered title in its own text block.
        if previous:
            prior_size, prior_top, prior_bottom, prior_center, prior_value = previous
            if abs(prior_size - size) < .3 and abs(center - prior_center) < size and 0 <= top - prior_bottom <= size:
                top, value = prior_top, prior_value + " " + value
        if 3 <= len(value.split()) <= 50:
            candidates.append((size, -top, len(value), value))
        previous = (size, top, bottom, center, value)
    return max(candidates)[3][:500] if candidates else ""


def matched_authors(names, byline):
    # Strip typeset affiliation numbers only in the candidate byline.
    byline = re.sub(r"(?<=[^\W\d_])[\d¹²³⁴⁵⁶⁷⁸⁹⁰]+", "", byline)
    words = " ".join(re.findall(r"[^\W\d_]+", fold_accents(byline).casefold()))
    found = []
    for name in names:
        parts = name_parts(name)
        if not parts:
            continue
        if len(parts) == 1:
            continue
        given, family = parts[0], parts[-1]
        # Match a full given name or its initial, allowing middle initials.
        first = rf"(?:{re.escape(given)}|{re.escape(given[0])})" if len(given) > 1 else re.escape(given) + r"[^\W\d_]*"
        last = re.escape(family)
        pattern = rf"\b(?:{first}(?: [^\W\d_]{{1,2}}){{0,3}} {last}|{last} {first})\b"
        if re.search(pattern, words) or " " + " ".join(parts) + " " in " " + words + " ":
            found.append(name)
    return found


def plausible_author_name(name):
    return (re.fullmatch(r"[^\W\d_][^@\d:()]{0,75}", name) is not None
            and 2 <= len(name.split()) <= 6
            and all(word[0].isupper() or word in {"de", "van", "der", "von", "da", "del", "di"}
                    for word in name.split())
            and not re.search(
                r"(?i)\b(?:university|department|institute|laboratory|press|abstract|research|journal|association|college|academy|faculty|school|proceedings)\b",
                name))


def layout_authors(page, first, title):
    """Read a name list immediately below the visible title, before affiliations."""
    hint = normalized_title(title)
    if not hint or hint not in normalized_title(first):
        return []
    blocks = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]
    title_blocks = [block for block in blocks if block.get("lines") and block["bbox"][1] < page.rect.height * .45
                    and (words := normalized_title(" ".join(span["text"] for line in block["lines"] for span in line["spans"])))
                    and (hint in words or len(words.split()) >= 2 and words in hint)]
    if not title_blocks:
        return []
    bottom = max(block["bbox"][3] for block in title_blocks)
    authors = []
    for block in sorted(blocks, key=lambda item: item["bbox"][1]):
        top = block["bbox"][1]
        if not block.get("lines") or not -1 <= top - bottom <= page.rect.height * .075:
            continue
        lines = block["lines"]
        row = [line for line in lines if abs(line["bbox"][1] - top) < 2]
        # A typeset byline can share one block with affiliations and email below.
        # Keep horizontally separated names on the top row in reading order.
        if len(row) > 1:
            ordered = sorted(row, key=lambda line: line["bbox"][0])
            if all(right["bbox"][0] >= left["bbox"][2] for left, right in zip(ordered, ordered[1:])):
                row_names = [" ".join("".join(span["text"] for span in line["spans"]
                                             if not span.get("flags", 0) & pymupdf.TEXT_FONT_SUPERSCRIPT).split())
                             for line in ordered]
                if all(plausible_author_name(name) for name in row_names):
                    authors.extend(row_names)
                    break
        value = " ".join(span["text"] for line in block["lines"] for span in line["spans"]
                         if not span.get("flags", 0) & pymupdf.TEXT_FONT_SUPERSCRIPT)
        value = re.sub(r"(?i),\s*(?:Ph\.?D\.?|M\.?D\.?|MSc|MPH|MBBS|MB,BChir|DO|RN|DDS|DPhil)(?=[;,]|\s*$)", "", value.strip())
        value = re.sub(r"(?<=[^\W\d_])\d+(?:\s*,\s*\d+)*", "", value)
        value = re.sub(r"(?<=[a-z])\s*ID(?=\s*[,;*&]|\s*$)", "", value)
        value = re.sub(r"\s+et al\.?(?=\s*[*†‡]|\s*$)", "", value)
        value = re.sub(r"[*†‡✉∗]", "", value)
        if re.fullmatch(r"\([^()\n]{3,200}\)", value.strip()) and re.search(r"(?i)\b(collaboration|consortium|study group)\b", value):
            authors.extend(re.split(r"\s+and\s+", value.strip()[1:-1]))
            bottom = block["bbox"][3]
            continue
        segments = re.split(r"\s*(?:[;,;&]|\band\b)\s*", value)
        names = [" ".join(name.split()).strip() for name in segments if name.strip()]
        if names and all(plausible_author_name(name) for name in names):
            authors.extend(names)
            bottom = block["bbox"][3]
        elif authors:
            break
    return list(dict.fromkeys(authors))


def first_page_dois(first):
    """Prefer labeled identifiers and standalone resolver URLs before body citations."""
    front = re.split(r"(?im)^\s*(?:references|bibliography|works cited)\s*$", first, maxsplit=1)[0]
    # Typeset identifiers can wrap immediately after the DOI prefix.
    front = re.sub(r"(10\.\d{4,9}/)[ \t]*\n[ \t]*(?=\w)", r"\1", front)
    pattern = r"10\.\d{4,9}/[-._;()/:a-z0-9]+"
    standalone = re.findall(rf"(?im)^[ \t]*(?:doi[ \t]*:[ \t]*|https?://(?:dx\.)?doi\.org/)({pattern})[ \t]*$", front)
    footer = re.findall(rf"(?im)^[^\n]{{0,100}}\|[^\S\n]*https?://(?:dx\.)?doi\.org/({pattern})(?=[^\S\n]|$)", front)
    locators = standalone or footer or re.findall(rf"(?i)\bdoi[ \t]*:[ \t]*({pattern})", front)
    return {doi for raw in locators if (doi := normalize_doi(raw.rstrip(".,;:")))}


def layout_summary(page, title, author):
    """Read a visually distinct opening summary after a corroborated byline."""
    if not title or not author:
        return None
    blocks = []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]:
        lines = [[span for span in line["spans"] if span["text"].strip()]
                 for line in block.get("lines", []) if line["dir"][0] > .99]
        if any(lines):
            text = "\n".join(" ".join(span["text"].strip() for span in line) for line in lines if line)
            size = max(span["size"] for line in lines for span in line)
            blocks.append((block["bbox"], text, size))
    title_block = next((box for box, text, size in blocks
                        if normalized_title(title) in normalized_title(text)), None)
    if not title_block:
        # Layout extraction can split a wrapped headline into separate blocks.
        title_block = next((right for (left, start, size), (right, end, next_size) in zip(blocks, blocks[1:])
                            if abs(size - next_size) < .3 and 0 <= right[1] - left[3] <= size
                            and normalized_title(title) == normalized_title(start + " " + end)), None)
    if not title_block:
        return None
    byline = next((box for box, text, size in blocks if box[1] >= title_block[3]
                   and box[1] < page.rect.height * .4 and matched_authors([author], text)), None)
    if not byline:
        return None
    for box, text, size in blocks:
        text = re.split(r"(?im)^\s*(?:\d+\.?\s+)?(?:introduction|keywords|index terms)\s*$", text, maxsplit=1)[0]
        if not (0 <= box[1] - byline[3] < page.rect.height * .1
                and box[1] < page.rect.height * .55 and 65 <= len(text.split()) <= 450):
            continue
        # Body text below a standfirst usually changes type size or column width.
        if any(next_box[1] >= box[3] and next_box[1] - box[3] < page.rect.height * .15
               and len(next_text.split()) >= 20
               and (next_size < size - .35 or next_box[2] - next_box[0] < (box[2] - box[0]) * .8)
               for next_box, next_text, next_size in blocks):
            return re.sub(r"(?<=\w)- (?=\w)", "", " ".join(text.split()))[:3000]
    return None


REPOSITORY = re.compile(r"\b(?:institutional repository|university repository|research repository|institutional archive)\b", re.I)
DEPOSIT_NOTICE = re.compile(r"\b(?:downloaded from|deposited in|repository copy|version of record|accepted manuscript|author accepted version|rights statement)\b", re.I)
MANUSCRIPT_VERSION = re.compile(r"\b(?:author[- ]accepted manuscript|accepted manuscript|author's accepted version|author accepted version|version\s*:\s*accepted version)\b", re.I)
ABSTRACT_HEADING = re.compile(r"(?im)^[ \t]*(?:abstract|summary|résumé|resumen|resumo|zusammenfassung)[ \t]*(?:[:.\u2014\u2013-]|$)")


def manuscript_front_page(document):
    """Skip a bounded number of explicitly marked deposit cover sheets."""
    first = document[0].get_text("text")[:30000]
    opening = first[:3500]
    described_copy = re.search(r"\b(?:this is|this document is) (?:an? )?repository copy\b", opening, re.I)
    deposit_fields = re.search(r"(?im)^\s*(?:deposited (?:in|via)|repository record)\b", opening) and MANUSCRIPT_VERSION.search(opening)
    series_cover = re.search(r"(?im)^.{0,100}\bworking paper series\s*$", opening) and re.search(
        r"(?im)^[ \t]*working paper[ \t]+(?:no\.?[ \t]*)?\d+", opening)
    citation_cover = re.search(r"(?im)^\s*(?:document|publication) version\s*:", opening) and re.search(
        r"(?im)^\s*(?:citation for (?:the )?published version|publisher rights statement)\s*:", opening)
    repository_cover = citation_cover or deposit_fields or ((REPOSITORY.search(opening) or described_copy) and DEPOSIT_NOTICE.search(opening))
    if not (repository_cover or series_cover):
        return 0, first, False
    viable = None
    for index in range(1, min(4, document.page_count)):
        page = document[index]
        text = page.get_text("text")[:30000].replace("\u00a0", " ")
        title = layout_title(page)
        if not title or REPOSITORY.search(text[:1000]):
            continue
        if ABSTRACT_HEADING.search(text):
            return index, text, bool(MANUSCRIPT_VERSION.search(opening))
        if viable is None and layout_authors(page, text, title):
            viable = (index, text, bool(MANUSCRIPT_VERSION.search(opening)))
    return viable or (0, first, False)


def repeated_cover_title(cover, first):
    """A series cover can confirm a manuscript title printed in a combined block."""
    cover_lines = {normalized_title(line) for line in cover.splitlines()[:16]}
    lines = [line.strip() for line in first.splitlines()[:12] if line.strip()]
    for index, line in enumerate(lines[:-1]):
        author = lines[index + 1]
        if (3 <= len(line.split()) <= 30 and normalized_title(line) in cover_lines
                and re.fullmatch(r"[^\W\d_][^\d@:()]{2,65}", author)
                and 2 <= len(author.split()) <= 6 and matched_authors([author], cover)):
            return line, [author]
    return "", []


def repository_cover_matches(cover, title, authors):
    return bool(title and authors and normalized_title(title) in normalized_title(cover)
                and matched_authors(authors, cover))


def extract_identity_input(content: bytes, include_tail: bool = False) -> dict:
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if not document.page_count:
            return {"first": "", "text": "", "metadata_title": ""}
        front_index, first, accepted_copy = manuscript_front_page(document)
        front_page = document[front_index]
        candidate = layout_title(front_page) if len(first.strip().splitlines()) > 1 else ""
        metadata_title = " ".join((document.metadata.get("title") or "").split())
        metadata_author = (document.metadata.get("author") or "")[:1500]
        # Only inspect early pages and stop at a references heading. Identifiers in
        # bibliographies are not evidence of the uploaded document's identity.
        pages = [document[index].get_text("text")[:15000] for index in range(front_index, min(front_index + 3, document.page_count))]
        tail = "\n".join(document[index].get_text("text")[:16000]
                         for index in range(max(0, document.page_count - 6), document.page_count)) if include_tail else ""
        author = re.split(r"[;,]|\band\b", metadata_author, maxsplit=1)[0].strip()
        title = metadata_title if metadata_title and normalized_title(metadata_title) in normalized_title(first) else candidate
        visible_authors = layout_authors(front_page, first, title)
        group = re.search(r"(?im)^[ \t]*(?:group author|collaboration|consortium)[ \t]*:[ \t]*([^\n]+)", first)
        if group and len(group[1].split()) <= 14:
            visible_authors = [" ".join(group[1].split())]
        cover = document[0].get_text("text")[:3500] if front_index else ""
        if cover and not visible_authors:
            repeated_title, repeated_authors = repeated_cover_title(cover, first)
            if repeated_title:
                candidate = title = repeated_title
                visible_authors = repeated_authors
        summary = layout_summary(front_page, title, visible_authors[0] if visible_authors else author)
        cover_doi = None
        if cover and repository_cover_matches(cover, title, visible_authors):
            identifiers = first_page_dois(cover)
            if len(identifiers) == 1:
                cover_doi = next(iter(identifiers))
        text = re.split(r"(?im)^\s*(?:references|bibliography|works cited)\s*$", "\n".join(pages), maxsplit=1)[0]
    return {"first": first, "text": text, "metadata_title": metadata_title[:500], "layout_title": candidate, "metadata_author": metadata_author, "tail": tail, "layout_summary": summary, "layout_authors": visible_authors, "cover_doi": cover_doi,
            "next_page_text": pages[1][:6000] if len(pages) > 1 else "",
            "cover_page": 1 if front_index else None,
            "front_page": front_index + 1, "front_source": "manuscript" if front_index else "first_page",
            "manuscript_version": "accepted_manuscript" if accepted_copy or MANUSCRIPT_VERSION.search(first[:600]) else None}


def abstract_page_text(passage):
    """Remove publication sidebars that the PDF reader places after the main column."""
    sidebar = re.search(
        r"(?im)^\s*(?:open access\s*$|citation\s*:|academic editor\s*:|received\s*:|"
        r"copyright\s*:|data availability\s*:|competing interests\s*:|published\s*:)", passage)
    if sidebar:
        passage = passage[:sidebar.start()]
    passage = re.sub(r"(?im)^.*(?:https?://(?:dx\.)?doi\.org/|\|\s*Vol\b).*$", "", passage)
    return re.sub(r"(?m)^\s*\d+\s*(?:/\s*\d+)?\s*$", "", passage)


def extracted_paper_metadata(identity_input: dict) -> dict:
    """Project displayable PDF metadata without contacting academic providers."""
    first = identity_input.get("first", "")
    if not first:
        return {}
    first = first.replace("\u00a0", " ")
    front = re.split(r"(?im)^\s*(?:abstract|résumé|resumen|resumo|zusammenfassung|introduction|references|bibliography|works cited)\b", first, maxsplit=1)[0]
    visible = normalized_title(front)
    spaced_visible = normalized_title(front.replace("\n", " "))
    title = next((value for field in ("metadata_title", "layout_title")
                  if (value := identity_input.get(field, "").strip())
                  and 3 <= len(value.split()) <= 35
                  and not value.endswith((".", ";", ","))
                  and not re.match(r"(?i)^(?:arxiv:|doi:|proceedings of|abstract\b|has (?:successfully )?completed\b|this is to certify\b|awarded to\b)", value)
                  and any(f" {normalized_title(value)} " in f" {rendering} " for rendering in (visible, spaced_visible))), None)
    authors = []
    if title:
        byline = visible.partition(normalized_title(title))[2] or spaced_visible.partition(normalized_title(title))[2]
        names = re.split(r"\s*(?:;|\n|\band\b)\s*", identity_input.get("metadata_author", ""))
        # A comma can separate names or indicate surname-first order.
        for name in names:
            whole_name = matched_authors([name], byline) if name else []
            authors.extend(whole_name or matched_authors(name.split(","), byline))

    if title and identity_input.get("layout_authors"):
        authors = identity_input["layout_authors"]

    abstract = re.search(
        r"(?im)^[ \t]*(?:abstract|résumé|resumen|resumo|zusammenfassung)(?:[ \t]*[:.\u2014\u2013-][ \t]*|[ \t]*\n+)(.*)", first, re.S)
    abstract_text = None
    if abstract:
        boundary = re.compile(
            r"(?im)^[ \t]*(?:\d+(?:\.\d+)*\.?[ \t\n]+)?(?:introduction|main text|author summary|abstract word count|key\s*words|index terms|references)\b")
        body_start = boundary.search(abstract[1])
        passage = abstract_page_text(abstract[1][:body_start.start()] if body_start else abstract[1])
        following = identity_input.get("next_page_text", "")
        continuation = boundary.search(following) if not body_start else None
        if continuation and 0 < continuation.start() < 4500:
            passage += "\n" + abstract_page_text(following[:continuation.start()])
        passage = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", passage)
        for author in authors:
            contact = re.search(rf"(?im)^\s*{re.escape(author)}\s*\n(?=[^\n]*(?:department|university|institute|@))", passage)
            if contact:
                passage = passage[:contact.start()]
        abstract_text = " ".join(passage.strip().split())[:5000] or None
    if not abstract_text:
        structured = re.search(
            r"(?ms)^[ \t]*(?:(?:IMPORTANCE|BACKGROUND)[ \t]+.{20,800}?\n)?"
            r"[ \t]*(?:OBJECTIVES?|AIMS?|PURPOSE)[ \t]+.{25,3000}?\n"
            r"[ \t]*(?:CONCLUSIONS?(?: AND RELEVANCE)?)[ \t]+.*?"
            r"(?=\n[ \t]*(?:(?:[A-Z][^\n]{1,100}\.\s+(?:19|20)\d{2};)|Published online|Author Affiliations|Introduction|Trial Registration|Keywords|References)|\Z)",
            first)
        if structured:
            abstract_text = " ".join(structured[0].split())[:4500]
    if not abstract_text and not identity_input.get("layout_summary"):
        # Some structured abstracts are made of standalone headings, without
        # an Abstract heading before the content in the extracted reading order.
        sections = list(re.finditer(r"(?m)^[ \t]*(BACKGROUND|OBJECTIVES?|METHODS|RESULTS|CONCLUSIONS)[ \t]*$", first))
        labels = [section[1] for section in sections]
        if (labels and labels[0] in {"BACKGROUND", "OBJECTIVE", "OBJECTIVES"}
                and labels[-3:] == ["METHODS", "RESULTS", "CONCLUSIONS"]):
            start = sections[0].start()
            end = re.search(r"(?im)^[ \t]*(?:abstr\s*act|introduction|downloaded from|copyright)\b",
                            first[sections[-1].end():])
            passage = first[start:sections[-1].end() + end.start()] if end else first[start:]
            passage = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", passage)
            abstract_text = " ".join(passage.split())[:5000]
    dois = first_page_dois(first) or ({identity_input["cover_doi"]} if identity_input.get("cover_doi") else set())
    arxiv = {value for raw in re.findall(r"(?im)^\s*arxiv\s*:\s*([^\s]+)", front)
             if (value := normalize_arxiv(raw))}
    year = re.search(r"(?im)^[ \t]*(?:publication year|published(?: online)?)[ \t]*:?[ \t]*[^\n]{0,30}?\b((?:19|20)\d{2})\b", first)
    venue = re.search(r"(?im)^\s*(?:journal|conference)\s*:[ \t]*([^\n]+)", front)
    publisher = re.search(r"(?im)^\s*publisher\s*:[ \t]*([^\n]+)", front)
    pmid = re.search(r"(?im)^\s*PMID\s*:\s*([1-9]\d{0,8})\b", front)
    pmcid = re.search(r"(?im)^\s*PMCID\s*:\s*(PMC[1-9]\d{0,9})\b", front)
    issue = re.search(r"(?im)^([\w][^\n|:]{2,90}?)\.\s+((?:19|20)\d{2});\s*\d[^\n]*\bdoi\s*:", first) if len(dois) == 1 else None
    footer = None
    if len(dois) == 1:
        doi = re.escape(next(iter(dois)))
        doi_text = re.sub(r"(10\.\d{4,9}/)[ \t]*\n[ \t]*(?=\w)", r"\1", first)
        footer = re.search(rf"(?im)^([^\n|]{{3,90}})\|[^\S\n]*https?://(?:dx\.)?doi\.org/{doi}(?=[^\S\n]|$)([^\n]*)", doi_text)
    proceedings = re.search(r"(?im)^((?:Proceedings|Findings) of .{5,160}?)(?=, (?:pages|pp\.)|$)", front)
    proceedings_year = re.findall(r"\b(?:19|20)\d{2}\b", proceedings[1]) if proceedings else []
    first_date = re.search(r"(?im)^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+((?:19|20)\d{2})\s*$", front) if title and authors else None
    footer_year = re.search(r"\b((?:19|20)\d{2})\s*$", footer[2]) if footer else None
    publication_year = (year[1] if year else issue[2] if issue else first_date[1] if first_date else footer_year[1] if footer_year
                        else proceedings_year[0] if len(set(proceedings_year)) == 1 else None)
    publication_venue = venue[1].strip()[:250] if venue else issue[1] if issue else footer[1].strip() if footer else proceedings[1] if proceedings else None
    watermark = re.search(r"(?im)^\s*arxiv\s*:\s*([^\s]+)[ \t]+\[[\w.-]+\]", first)
    if watermark and (stamped := normalize_arxiv(watermark[1])):
        arxiv.add(stamped)
    result = {"year": int(publication_year) if publication_year else None, "venue": publication_venue,
              "pmid": pmid[1] if pmid else None, "pmcid": pmcid[1].upper() if pmcid else None,
              "publisher": publisher[1].strip()[:250] if publisher else None,
              "version": identity_input.get("manuscript_version"),
              "title": title, "authors": list(dict.fromkeys(authors)) or None, "abstract": abstract_text or (identity_input.get("layout_summary") if authors else None),
              "doi": next(iter(dois)) if len(dois) == 1 else None,
              "arxiv_id": next(iter(arxiv)) if len(arxiv) == 1 else None}
    return {field: value for field, value in result.items() if value is not None}


def verify_pdf_identity(content: bytes, discovery) -> dict:
    return verify_identity_input(extract_identity_input(content), discovery)


def verify_identity_input(identity_input: dict, discovery) -> dict:
    result = {"status": "not_verified", "reason": "No reliable matching academic record was found.",
              "checked_at": datetime.now(timezone.utc).isoformat()}
    first, text, metadata_title = (identity_input.get(key, "") for key in ("first", "text", "metadata_title"))
    if len(text.strip()) < 40:
        result["reason_code"] = "insufficient_metadata"
        result["reason"] = "There is not enough readable text to identify this document; OCR is not available."
        return result
    # Exclude labelled captions and supplement lines before reading identifiers.
    text = re.sub(r"(?im)^\s*(?:fig(?:ure)?\.?|table|supplement(?:ary)?(?: material)?)\s+[A-Z]?\d+[^\n]*", "", text)
    text = re.sub(r"(10\.\d{4,9}/)\s*\n\s*(?=\w)", r"\1", text)
    dois = {value for match in re.findall(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text, re.I)
            if (value := normalize_doi(match.rstrip(".,;:").removesuffix(")") if match.count(")") > match.count("(") else match.rstrip(".,;:")))}
    # A standalone DOI field on the title page takes precedence over identifiers
    # scattered through the article, including figures and cited papers.
    labelled_dois = first_page_dois(first) or ({identity_input["cover_doi"]} if identity_input.get("cover_doi") else set())
    if len(labelled_dois) == 1:
        dois = labelled_dois
    arxivs = {value for match in re.findall(r"(?:arxiv\s*:\s*|arxiv\.org/abs/)((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?)", text, re.I)
              if (value := normalize_arxiv(match))}
    medical_ids = {f"PMID:{value}" for value in re.findall(r"\bPMID\s*:\s*([1-9]\d{0,8})\b", text, re.I)}
    medical_ids.update(value.upper() for value in re.findall(r"\bPMC[1-9]\d{0,9}\b", text, re.I))
    visible = extracted_paper_metadata(identity_input)
    # A repository watermark identifies that edition even when the article
    # mentions other identifiers in its introduction.
    stamped_arxiv = visible.get("arxiv_id")
    trusted_ids = set(labelled_dois)
    if visible.get("pmid"):
        trusted_ids.add("PMID:" + visible["pmid"])
    if visible.get("pmcid"):
        trusted_ids.add(visible["pmcid"])
    if stamped_arxiv:
        trusted_ids.add(stamped_arxiv)
    identifiers = sorted(trusted_ids or (dois | arxivs | medical_ids))
    if len(identifiers) > 3:
        # Citation-heavy body text is weak evidence. Search by visible title instead.
        identifiers = sorted(labelled_dois)[:3] if len(labelled_dois) == 1 else []
    # PDF title metadata is only a search hint, never proof. Require it to be
    # visible on the title page before sending it to a provider.
    hint = normalized_title(metadata_title)
    front = re.split(r"(?im)^\s*(?:abstract|introduction|references|bibliography|works cited)\s*(?:[:.\n]|$)", first, maxsplit=1)[0]
    page = normalized_title(front)
    spaced_page = normalized_title(front.replace("\n", " "))
    candidate = identity_input.get("layout_title", "")
    if (len(hint.split()) < 3 or hint not in page) and candidate and normalized_title(candidate) in page:
        metadata_title, hint = candidate, normalized_title(candidate)
    if not identifiers and (not hint or hint not in page):
        lines = [line.strip() for line in front.splitlines() if line.strip()]
        if len(lines) >= 2 and 3 <= len(lines[0].split()) <= 30:
            metadata_title = lines[0]
            hint = normalized_title(metadata_title)
    if not identifiers and (len(hint.split()) < 3 or hint not in page):
        result["reason_code"] = "insufficient_metadata"
        result["reason"] = "A reliable title or publication identifier was not found."
        return result
    if not identifiers and visible.get("title"):
        metadata_title, hint = visible["title"], normalized_title(visible["title"])
    queries = list(identifiers) or [metadata_title]
    title_fallback = bool(visible.get("title") and (trusted_ids or len(identifiers) == 1))
    if title_fallback:
        metadata_title = visible["title"]
        hint = normalized_title(metadata_title)
    matches = {}
    child_identifiers = set()
    vetoed = set()
    unavailable = False
    trusted_record_found = False
    rejected = Counter()
    for query in queries:
        identifier_query = query in identifiers
        response = discovery.search(query, 0, 1 if identifier_query else 5)
        logger.info("publication lookup strategy=%s candidate_count=%d provider_failures=%s",
                    "identifier" if identifier_query else "title", len(response.get("results", [])),
                    sorted(response.get("provider_errors", {})))
        if (identifier_query and query == identifiers[-1] and not matches and len(hint.split()) >= 3 and hint in page
                and all(i in child_identifiers for i in identifiers[:-1])
                and any(d.get("reason") == "component_identifier" for d in response.get("_diagnostics", []))):
            queries.append(metadata_title)
        for check in response.get("_diagnostics", []):
            if check.get("reason") == "component_identifier":
                child_identifiers.add(query)
            if check.get("reason") in {"excluded_by_provider_status", "identity_or_status_check_failed"}:
                vetoed.add(check["id"].removeprefix("doi:"))
        unavailable = unavailable or bool(response.get("provider_errors"))
        for record in response.get("results", []):
            if not identifier_query and trusted_ids:
                record_ids = {record.get("doi"), record.get("arxiv_id"), record.get("pmcid")}
                if record.get("pmid"):
                    record_ids.add("PMID:" + record["pmid"])
                if record.get("arxiv_id"):
                    record_ids.add(re.sub(r"v\d+$", "", record["arxiv_id"]))
                if not trusted_ids.intersection(record_ids):
                    rejected["unconfirmed_identifier"] += 1
                    continue
                if labelled_dois and record.get("doi") not in labelled_dois:
                    rejected["different_printed_doi"] += 1
                    continue
            if identifier_query:
                medical = medical_identifier(query)
                if medical and record.get(medical[0]) != medical[1]:
                    rejected["different_identifier"] += 1
                    continue
                expected_doi = normalize_doi(query)
                returned_doi = normalize_doi(record.get("doi") or "")
                expected_arxiv = normalize_arxiv(query)
                returned_arxiv = normalize_arxiv(record.get("arxiv_id") or "")
                if expected_doi and returned_doi != expected_doi:
                    rejected["different_identifier"] += 1
                    continue
                if expected_arxiv and (not returned_arxiv or returned_arxiv != expected_arxiv and re.sub(r"v\d+$", "", returned_arxiv) != expected_arxiv):
                    rejected["different_identifier"] += 1
                    continue
                trusted_record_found |= query in trusted_ids
            title = normalized_title(record.get("title") or "")
            byline = page.partition(title)[2][:2500] if title else ""
            extracted_authors = identity_input.get("layout_authors") or []
            authors = ([name for name in record.get("authors", [])
                        if any(names_agree(name, visible_name) for visible_name in extracted_authors)]
                       if extracted_authors else matched_authors(record.get("authors", []), byline))
            if not normalized_title(candidate) == title and candidate and title in normalized_title(candidate) and len(normalized_title(candidate).split()) > len(title.split()):
                # Uniform-font text blocks may contain the byline as well as the title.
                if not extracted_authors and not matched_authors(authors, normalized_title(candidate).partition(title)[2]):
                    rejected["partial_title"] += 1
                    continue
            if not identifier_query and hint != title:
                rejected["different_title"] += 1
                continue
            required_authors = min(2, len(extracted_authors), len(record.get("authors", []))) if extracted_authors else 1
            if (title and len(title.split()) >= 3 and any(f" {title} " in f" {rendering} " for rendering in (page, spaced_page))
                    and authors and len(authors) >= required_authors
                    and record.get("status") in {"verified_published", "verified_preprint", "registry_record", "conflicting", "retracted"}):
                matches[record.get("id") or record.get("landing_url") or title] = {**record, "matched_authors": authors}
            else:
                rejected["title_or_author_not_corroborated"] += 1
        if (title_fallback and identifier_query and query == identifiers[-1] and not matches
                and not trusted_record_found and metadata_title not in queries
                and not trusted_ids.intersection(vetoed)):
            queries.append(metadata_title)
    matches = {key: record for key, record in matches.items()
               if key not in vetoed and record.get("doi") not in vetoed
               and not (record.get("pmid") and "pmid:" + record["pmid"] in vetoed)}
    for rec in matches.values():
        target = matches.get("doi:" + (rec.get("published_doi") or ""))
        if target and rec.get("version_group") and set(rec["matched_authors"]) & set(target["matched_authors"]):
            target["version_group"] = rec["version_group"]
    if len(matches) > 1 and not identifiers and visible.get("venue") and visible.get("year"):
        editions = {key: record for key, record in matches.items()
                    if record.get("year") == visible["year"]
                    and normalized_title(record.get("venue") or "") == normalized_title(visible["venue"])}
        if len(editions) == 1:
            matches = editions
    if len(matches) > 1 and not identifiers:
        # Proceedings headers identify the printed edition even when the PDF
        # carries no DOI and an earlier preprint has the same title and byline.
        header = first.partition("\n")[0]
        if re.match(r"\s*Proceedings\b", header, re.I):
            years = set(re.findall(r"\b(?:19|20)\d{2}\b", header))
            editions = {key: record for key, record in matches.items()
                        if record.get("publication_type") == "proceedings-article"
                        and str(record.get("year")) in years}
            if len(editions) == 1:
                matches = editions
    if len(matches) > 1 and stamped_arxiv and all(r.get("version_group") for r in matches.values()) and len({r["version_group"] for r in matches.values()}) == 1:
        specific = {k:r for k,r in matches.items() if r.get("arxiv_id")}
        if len(specific) == 1:
            matches = specific
    if len(matches) != 1:
        logger.info("publication matching unresolved candidate_rejections=%s matching_records=%d registry_vetoes=%d",
                    dict(rejected), len(matches), len(vetoed))
        result["status"] = "conflicting" if matches else "verification_unavailable" if unavailable else "not_verified"
        if matches:
            result["reason_code"] = "ambiguous_editions"
            result["reason"] = "Several academic records match the title and authors, but the PDF does not distinguish their editions."
        elif unavailable:
            result["reason_code"] = "provider_unavailable"
            result["reason"] = "The academic source could not be checked."
        elif trusted_record_found:
            result["reason_code"] = "identifier_not_corroborated"
            result["reason"] = "The record for the printed identifier did not match enough evidence from this PDF."
        else:
            result["reason_code"] = "no_corroborated_record"
            result["reason"] = "No academic record matched enough evidence from this PDF."
        return result
    record = next(iter(matches.values()))
    status = record["status"]
    if status == "registry_record" and record.get("year") and record.get("venue") and record.get("publication_type") in {"journal-article", "proceedings-article", "book-chapter"}:
        status = "verified_published"
    reason = {"conflicting": "Academic sources disagree about the record.",
              "registry_record": "Matching registry metadata found. Publication status is unconfirmed.",
              "retracted": "The academic source reports a retraction or withdrawal."}
    corroborated = {} if status == "conflicting" else {
        field: record[field] for field in ("authors", "year", "venue", "publisher")
        if record.get(field)
    }
    if status == "verified_published" and visible.get("version") == "accepted_manuscript":
        reason[status] = "The accepted manuscript matches a published record."
    return {**corroborated, "status": status, "version": visible.get("version"),
            "verification_scope": "publication_record" if status in {"verified_published", "verified_preprint"} else None, "title": record["title"], "doi": record.get("doi"),
            "arxiv_id": record.get("arxiv_id"), "pmid": record.get("pmid"), "pmcid": record.get("pmcid"),
            "evidence": {"authors_matched": record["matched_authors"],
                         "year_on_title_page": str(record["year"]) in page.split() if record.get("year") else None,
                         "venue_on_title_page": normalized_title(record["venue"]) in page if record.get("venue") else None,
                         "status_check": record.get("status_note")},
            "source": ", ".join(record.get("sources", [])), "source_url": record.get("landing_url"),
            "reason_code": "conflicting_sources" if status == "conflicting" else None,
            "reason": reason.get(status, "Publication identity was matched; scientific correctness was not assessed."),
            "checked_at": result["checked_at"]}
