from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

import pymupdf
from tokenizers import Tokenizer


class PdfValidationError(ValueError):
    pass


class PdfTextUnavailable(PdfValidationError):
    pass


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    headings: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int
    text: str
    content_hash: str
    section: str | None = None


def _margin_key(text: str, top: float, bottom: float, height: float) -> tuple[str, str] | None:
    if bottom <= height * 0.12:
        margin = "header"
    elif top >= height * 0.82:
        margin = "footer"
    else:
        return None
    # Running page numbers change, while the surrounding header/footer stays the same.
    normalized = re.sub(r"\d+", "#", text.lower())
    return margin, " ".join(normalized.split())


def extract_pages(source: Path | bytes, max_pages: int, max_characters: int) -> tuple[int, list[PageText]]:
    try:
        document = pymupdf.open(source) if isinstance(source, Path) else pymupdf.open(stream=source, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise PdfValidationError("This file could not be read as a PDF.") from error

    with document:
        if not document.is_pdf or document.needs_pass:
            raise PdfValidationError("Upload a PDF that is not password protected.")
        if document.is_repaired or not document.page_count:
            raise PdfValidationError("This PDF is damaged or contains no pages.")
        if document.page_count > max_pages:
            raise PdfValidationError(f"Use a PDF with at most {max_pages} pages for this milestone.")

        extracted = []
        margin_counts: Counter[tuple[str, str]] = Counter()
        character_count = 0
        for page in document:
            lines = []
            margin_keys = set()
            for block in page.get_text("dict", sort=True,
                    flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if not text:
                        continue
                    key = _margin_key(text, line["bbox"][1], line["bbox"][3], page.rect.height)
                    spans = [span for span in line["spans"] if span.get("text", "").strip()]
                    size = max((span.get("size", 0) for span in spans), default=0)
                    lines.append((text, key, size))
                    if key:
                        margin_keys.add(key)
                    character_count += len(text)
            if character_count > max_characters:
                raise PdfValidationError("This PDF contains too much text. Please upload a smaller paper.")
            extracted.append((page.number + 1, lines))
            margin_counts.update(margin_keys)

        repeat_threshold = max(2, (document.page_count * 3 + 4) // 5)
        repeated_margins = {key for key, count in margin_counts.items() if count >= repeat_threshold}
        sizes = Counter()
        for _, lines in extracted:
            for text, key, size in lines:
                if key not in repeated_margins and size:
                    sizes[round(size, 1)] += len(text)
        body_size = sizes.most_common(1)[0][0] if sizes else 0
        pages = []
        for page_number, lines in extracted:
            parts = []
            headings = []
            position = 0
            for line, key, size in lines:
                if key in repeated_margins:
                    continue
                normalized = " ".join(line.split())
                if parts:
                    position += 1
                # Font-size is a layout hint, not proof of a semantic section.
                if (body_size and size >= body_size * 1.2 and 3 <= len(normalized) <= 120
                        and len(normalized.split()) <= 12 and re.search(r"[A-Za-z]", normalized)
                        and not normalized.endswith((".", "?", "!"))):
                    headings.append((position, normalized))
                parts.append(normalized)
                position += len(normalized)
            text = " ".join(parts)
            if text:
                pages.append(PageText(page_number, text, tuple(headings)))

        if not pages:
            raise PdfTextUnavailable("No readable text was found. Scanned PDFs need OCR, which is not available yet.")
        return document.page_count, pages


def create_chunks(
    pages: list[PageText], tokenizer: Tokenizer, chunk_tokens: int = 220, overlap_tokens: int = 30,
) -> list[TextChunk]:
    if not 0 <= overlap_tokens < chunk_tokens <= 254:
        raise ValueError("Chunk size must fit MiniLM, with overlap smaller than the chunk.")

    chunks = []
    section = None
    for page in pages:
        boundaries = [(0, section)]
        for position, heading in page.headings:
            if position == 0:
                boundaries[0] = (0, heading)
            elif 0 < position < len(page.text):
                boundaries.append((position, heading))
        for segment_index, (segment_start, section) in enumerate(boundaries):
            segment_end = boundaries[segment_index + 1][0] if segment_index + 1 < len(boundaries) else len(page.text)
            segment = page.text[segment_start:segment_end]
            offsets = tokenizer.encode(segment, add_special_tokens=False).offsets
            start = 0
            while start < len(offsets):
                end = min(start + chunk_tokens, len(offsets))
                # Slice source text so punctuation and citation excerpts survive tokenization.
                text = segment[offsets[start][0]:offsets[end - 1][1]].strip()
                if text:
                    chunks.append(TextChunk(
                        chunk_index=len(chunks), page_number=page.page_number, text=text,
                        content_hash=sha256(text.encode("utf-8")).hexdigest(), section=section,
                    ))
                if end == len(offsets):
                    break
                start = end - overlap_tokens
    if not chunks:
        raise PdfValidationError("No usable text chunks were found in this PDF.")
    return chunks
