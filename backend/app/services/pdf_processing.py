from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pymupdf
from tokenizers import Tokenizer


class PdfValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    page_number: int
    text: str
    content_hash: str


def extract_pages(path: Path, max_pages: int, max_characters: int) -> tuple[int, list[PageText]]:
    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise PdfValidationError("This file could not be read as a PDF.") from error

    with document:
        if not document.is_pdf or document.needs_pass:
            raise PdfValidationError("Upload a PDF that is not password protected.")
        if document.page_count > max_pages:
            raise PdfValidationError(f"Use a PDF with at most {max_pages} pages for this milestone.")

        pages = []
        character_count = 0
        for page in document:
            text = " ".join(page.get_text("text", sort=True).split())
            character_count += len(text)
            if character_count > max_characters:
                raise PdfValidationError("This PDF contains too much text. Please upload a smaller paper.")
            if text:
                pages.append(PageText(page.number + 1, text))

        if not pages:
            raise PdfValidationError("No readable text was found. Scanned PDFs need OCR, which is not available yet.")
        return document.page_count, pages


def create_chunks(
    pages: list[PageText], tokenizer: Tokenizer, chunk_tokens: int = 220, overlap_tokens: int = 30,
) -> list[TextChunk]:
    if not 0 <= overlap_tokens < chunk_tokens <= 254:
        raise ValueError("Chunk size must fit MiniLM, with overlap smaller than the chunk.")

    chunks = []
    for page in pages:
        offsets = tokenizer.encode(page.text, add_special_tokens=False).offsets
        start = 0
        while start < len(offsets):
            end = min(start + chunk_tokens, len(offsets))
            # Slice the original text so punctuation and citation excerpts survive tokenization.
            text = page.text[offsets[start][0]:offsets[end - 1][1]].strip()
            if text:
                chunks.append(TextChunk(
                    chunk_index=len(chunks), page_number=page.page_number, text=text,
                    content_hash=sha256(text.encode("utf-8")).hexdigest(),
                ))
            if end == len(offsets):
                break
            start = end - overlap_tokens
    if not chunks:
        raise PdfValidationError("No usable text chunks were found in this PDF.")
    return chunks
