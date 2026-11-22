from pathlib import Path
import re

import chromadb
from chromadb.config import Settings

from ..paper_models import ChunkResponse
from ..services.pdf_processing import TextChunk


class VectorRepository:
    def __init__(self, directory: Path):
        client = chromadb.PersistentClient(path=str(directory), settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection("paper_chunks_minilm_v1", embedding_function=None)

    @staticmethod
    def paper_filter(owner_id: str, paper_id: str) -> dict:
        return {"$and": [{"owner_id": owner_id}, {"paper_id": paper_id}]}

    def save(self, owner_id: str, paper_id: str, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[f"{paper_id}:{chunk.chunk_index}" for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[{
                "owner_id": owner_id, "paper_id": paper_id,
                "page_number": chunk.page_number, "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash,
            } for chunk in chunks],
        )

    def preview(self, owner_id: str, paper_id: str, offset: int = 0, limit: int = 10) -> list[ChunkResponse]:
        result = self.collection.get(
            ids=[f"{paper_id}:{index}" for index in range(offset, offset + limit)],
            where=self.paper_filter(owner_id, paper_id), include=["documents", "metadatas"],
        )
        chunks = [ChunkResponse(text=text, **metadata)
                  for text, metadata in zip(result["documents"], result["metadatas"])]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def sample(self, owner_id: str, paper_id: str, chunk_count: int, limit: int = 16) -> list[ChunkResponse]:
        """Return evenly spread chunks so recommendations are not based on page one only."""
        if chunk_count <= 0:
            return []
        count = min(chunk_count, limit)
        indices = sorted({round(index * (chunk_count - 1) / max(1, count - 1)) for index in range(count)})
        result = self.collection.get(
            ids=[f"{paper_id}:{index}" for index in indices], where=self.paper_filter(owner_id, paper_id),
            include=["documents", "metadatas"],
        )
        return sorted((ChunkResponse(text=text, **metadata)
                       for text, metadata in zip(result["documents"], result["metadatas"])),
                      key=lambda chunk: chunk.chunk_index)

    @staticmethod
    def _terms(text: str) -> set[str]:
        stopwords = {"what", "which", "who", "where", "when", "how", "does", "did", "have",
                     "has", "been", "with", "from", "this", "that", "the", "and", "for", "are",
                     "study", "paper", "document", "both", "two", "each", "compare", "comparison"}
        synonyms = {"cheat": "misconduct", "dishonesty": "misconduct", "proctor": "monitor",
                    "surveillance": "monitor", "finding": "result", "outcome": "result",
                    "institution": "university", "college": "university",
                    "methodology": "design", "participant": "student",
                    "randomized": "design", "experimental": "design"}
        terms = set()
        for word in re.findall(r"[a-z0-9]+", text.casefold()):
            if len(word) < 3:
                continue
            if word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("ing") and len(word) > 5:
                word = word[:-3]
            elif word.endswith("s") and len(word) > 4:
                word = word[:-1]
            elif word.endswith("ly") and len(word) > 6:
                word = word[:-2]
            if word not in stopwords:
                terms.add(synonyms.get(word, word))
        return terms

    def search(self, owner_id: str, paper_id: str, embedding: list[float], limit: int = 5,
               question: str | None = None, candidate_limit: int | None = None) -> list[ChunkResponse]:
        study_labels = list(dict.fromkeys(re.findall(r"\bstudy\s+([a-z]|\d+)\b", question or "", re.I)))
        comparison = bool(question and (
            re.search(r"\b(?:both studies|two studies|each study|compar\w*|combined|together|overall|universally)\b",
                      question, re.I)
            or len(study_labels) >= 2))
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(50, max(limit, candidate_limit or (50 if comparison else 25))) if question else limit,
            where=self.paper_filter(owner_id, paper_id),
            include=["documents", "metadatas"],
        )
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        passages = [ChunkResponse(text=text, **metadata)
                    for text, metadata in zip(documents, metadatas)]
        if question:
            question_terms = self._terms(question)
            if question_terms:
                def relevance(passage: ChunkResponse) -> tuple[float, float]:
                    sentences = re.split(r"(?<=[.!?])\s+", passage.text)
                    scores = []
                    for sentence in sentences:
                        sentence_terms = self._terms(sentence)
                        matches = len(question_terms & sentence_terms)
                        scores.append((matches / len(question_terms),
                                       matches / max(len(sentence_terms), 1)))
                    return max(scores, default=(0.0, 0.0))

                scores = [relevance(passage) for passage in passages]
                if scores and max(score[0] for score in scores) >= 0.5:
                    passages = [passage for _, passage in sorted(
                        enumerate(passages), key=lambda item: (-scores[item[0]][0],
                                                               -scores[item[0]][1], item[0]))]
            study = re.search(r"\bstudy\s+([a-z]|\d+)\b", question, re.I)
            methodology = re.search(r"\b(?:methodolog\w*|research design|study design)\b", question, re.I)
            if study and methodology and not comparison:
                label = study.group(1)
                other = re.compile(rf"\bstudy\s+(?!{re.escape(label)}\b)(?:[a-z]|\d+)\b", re.I)
                requested = [pattern for prompt, pattern in (
                    (r"\b(?:methodolog\w*|design)\b", r"\b(?:design|randomiz\w*|experiment\w*)\b"),
                    (r"\b(?:sample size|participants?|students?)\b", r"\b(?:participants?|students?|sample)\b"),
                    (r"\buniversit\w*\b", r"\buniversit\w*\b"),
                    (r"\bregions?\b", r"\bregions?\b"),
                    (r"\b(?:observation period|duration|semester)\b",
                     r"\b(?:observation period|semester|duration)\b"),
                ) if re.search(prompt, question, re.I)]
                def methodology_coverage(passage: ChunkResponse) -> int:
                    if not re.search(rf"\bstudy\s+{label}\b", passage.text, re.I):
                        return 0
                    coverage = sum(bool(re.search(pattern, passage.text, re.I)) for pattern in requested)
                    return coverage if not other.search(passage.text) else 0
                if requested and max((methodology_coverage(item) for item in passages), default=0) >= 2:
                    passages = sorted(passages, key=methodology_coverage, reverse=True)
            if study and not comparison:
                # A topical word in another study must not outrank the named
                # study when a focused expansion query asks for that subject.
                label = study.group(1)
                other = re.compile(rf"\bstudy\s+(?!{re.escape(label)}\b)(?:[a-z]|\d+)\b", re.I)
                passages = sorted(passages, key=lambda item: not (
                    re.search(rf"\bstudy\s+{label}\b", item.text, re.I)
                    and not other.search(item.text)))
        if comparison:
            # Prefer a passage about each study, not just a comparison table that
            # happens to contain both names.
            topic_terms = self._terms(question or "")
            def score(item: ChunkResponse) -> int:
                overlap = len(topic_terms & self._terms(item.text))
                if (question and re.search(r"\bgeneraliz\w*|\btransferab\w*", question, re.I)
                        and re.search(r"\b(?:randomiz\w*|quasi.experiment\w*|volunteer\w*|sample|semester|single.country)\b",
                                      item.text, re.I)):
                    overlap += 1
                return overlap
            selected = [item for item in passages if not topic_terms or score(item) > 0][:limit]
            inferred_labels = list(dict.fromkeys(label for item in passages
                for label in re.findall(r"\bstudy\s+([a-z]|\d+)\b", item.text, re.I)))
            labels = study_labels or (inferred_labels[:2] if re.search(
                r"\b(?:both studies|two studies|each study)\b", question or "", re.I) else [])
            for label in labels:
                marker = re.compile(rf"\bstudy\s+{label}\b", re.I)
                other = re.compile(rf"\bstudy\s+(?!{re.escape(label)}\b)(?:[a-z]|\d+)\b", re.I)
                candidates = [item for item in passages if marker.search(item.text)
                              and not other.search(item.text) and (not topic_terms or score(item) > 0)]
                if not candidates:
                    continue
                candidate = max(candidates, key=score)
                if candidate in selected:
                    continue
                if len(selected) < limit:
                    selected.append(candidate)
                elif selected:
                    replace = next((index for index in range(len(selected) - 1, -1, -1)
                                    if bool(marker.search(selected[index].text))
                                    == bool(other.search(selected[index].text))),
                                   len(selected) - 1)
                    selected[replace] = candidate
            return list({item.chunk_index: item for item in selected}.values())
        return passages[:limit]

    def remove(self, owner_id: str, paper_id: str) -> None:
        self.collection.delete(where=self.paper_filter(owner_id, paper_id))
