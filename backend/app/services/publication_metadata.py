"""Shared metadata comparisons. Titles and names are evidence, never record keys."""
import re
import unicodedata
from html import unescape


def normalized_title(value):
    value = unicodedata.normalize("NFKC", unescape(value)).casefold().replace("\u00ad", "")
    value = re.sub(r"(?<=\w)[’'˝ˇ˘˙¸˛](?=\w)", "", value)
    value = fold_accents(value)
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    # Preserve mathematical operators and signs which can distinguish two titles.
    return " ".join(re.findall(r"\w+|[^\w\s]", value.translate(str.maketrans(
        {c: " " for c in '.,:;!?“”"‘’()[]{}–—-'})), re.UNICODE))


def fold_accents(value):
    return "".join(char for char in unicodedata.normalize("NFKD", value)
                   if not unicodedata.combining(char))


def name_parts(value):
    value = re.sub(r"(?i)(?:,?\s+(?:Ph\.?D\.?|M\.?D\.?|MSc|MPH|MBBS|MB,?BChir|DO|RN|DDS|DPhil))+$", "", value.strip())
    value = fold_accents(value).replace("\u200b", "")
    if "," in value:
        family, given = value.split(",", 1)
        value = given + " " + family
    return normalized_title(value).split()


def names_agree(left, right):
    a, b = name_parts(left), name_parts(right)
    if not a or not b or a[-1] != b[-1]:
        return False
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if len(short) >= 2 and len(short[0]) > 1 and long[-len(short):] == short:
        return True
    return len(a) > 1 and len(b) > 1 and a[0][0] == b[0][0] and (
        len(a[0]) == 1 or len(b[0]) == 1 or a[0] == b[0])


def metadata_conflict(a, b):
    author_agreement = any(names_agree(x, y) for x in a["authors"] for y in b["authors"])
    if a["authors"] and b["authors"] and not author_agreement:
        return True
    title_a, title_b = normalized_title(a["title"]), normalized_title(b["title"])
    if title_a != title_b:
        # An explicitly delimited subtitle is sometimes stored separately or
        # omitted. Compare that variant only for the same DOI and byline.
        same_doi = a.get("doi") and a["doi"] == b.get("doi")
        omit_subtitle = any(
            (parts := re.split(r"[:–—]", record["title"], maxsplit=1)) and len(parts) == 2
            and len(normalized_title(parts[0]).split()) >= 4
            and normalized_title(parts[0]) == other
            for record, other in ((a, title_b), (b, title_a))
        )
        if not (same_doi and author_agreement and omit_subtitle):
            return True
    for field in ("doi", "pmid", "pmcid"):
        if a.get(field) and b.get(field) and a[field] != b[field]:
            return True
    if a.get("venue") and b.get("venue") and normalized_title(a["venue"]) != normalized_title(b["venue"]):
        return True
    years_a = set(a.get("publication_years") or [a.get("year")]) - {None}
    years_b = set(b.get("publication_years") or [b.get("year")]) - {None}
    return bool(years_a and years_b and not years_a & years_b
                and min(abs(x-y) for x in years_a for y in years_b) > 1)
