"""PDF-grounded question recommendations without changing answer generation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
import re
import time
from typing import Callable, Iterable

from ..paper_models import ChunkResponse
from ..providers.answers import Evidence
from ..repositories.vectors import VectorRepository


@dataclass(frozen=True)
class RecommendedQuestion:
    id: str
    text: str
    recommendation_type: str
    source: ChunkResponse


class QuestionRecommendationService:
    """Build small, verifiable suggestions from chunks already stored in ChromaDB.

    Groq is optional: a provider can supply more specific candidates through
    ``recommend_questions``. Every candidate is still checked against PDF
    retrieval before it reaches the client.
    """

    _cache: OrderedDict[tuple[str, str, str, int], tuple[float, list[RecommendedQuestion]]] = OrderedDict()
    _cache_lock = Lock()
    _generation_lock = Lock()
    _cache_limit = 128
    _cache_seconds = 60 * 60

    def __init__(self, vectors: VectorRepository, embed: Callable[[list[str]], list[list[float]]], provider: object | None):
        self.vectors = vectors
        self.embed = embed
        self.provider = provider
        self.provider_used = False

    @staticmethod
    def _normalise(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))

    @classmethod
    def _different(cls, candidate: str, existing: Iterable[str]) -> bool:
        candidate_terms = VectorRepository._terms(candidate)
        if len(cls._normalise(candidate).split()) < 3:
            return False
        for item in existing:
            other_terms = VectorRepository._terms(item)
            if cls._normalise(candidate) == cls._normalise(item):
                return False
            union = candidate_terms | other_terms
            if union and len(candidate_terms & other_terms) / len(union) >= 0.8:
                return False
        return True

    @staticmethod
    def _question_type(question: str) -> str:
        lowered = question.casefold()
        if re.search(r"\b(?:method\w*|design\w*|participant\w*|sample\w*|data|procedure\w*)\b", lowered):
            return "methodology"
        if re.search(r"\b(?:limit\w*|caveat\w*|bias\w*|generaliz\w*)\b", lowered):
            return "limitations"
        if re.search(r"\b(?:result\w*|finding\w*|outcome\w*|effect\w*)\b", lowered):
            return "findings"
        return "paper"

    def _provider_candidates(self, mode: str, context: str, samples: list[ChunkResponse], limit: int) -> list[str]:
        recommend = getattr(self.provider, "recommend_questions", None)
        available = getattr(self.provider, "available", None)
        if not callable(recommend) or (callable(available) and not available()):
            raise RuntimeError("Question suggestions are temporarily unavailable.")
        evidence = [Evidence(f"c{item.chunk_index}", item.text[:1200]) for item in samples]
        self.provider_used = True
        result = recommend(mode, context[:3000], evidence, limit)
        return [item.strip() for item in result if isinstance(item, str)]

    def _supported(self, owner_id: str, paper_id: str, question: str) -> ChunkResponse | None:
        if not question.endswith("?") or len(question) < 12 or len(question) > 240:
            return None
        if re.search(r"\b(?:tell me more|anything else|what about it)\b", question, re.I):
            return None
        try:
            vector = self.embed([question])[0]
            result = self.vectors.search(owner_id, paper_id, vector, limit=1, question=question, candidate_limit=12)
        except Exception as exc:
            raise RuntimeError("Question suggestions could not be checked.") from exc
        if not result:
            return None
        terms = VectorRepository._terms(question)
        evidence_terms = VectorRepository._terms(result[0].text)
        # A question category (for example "methodology") can be supported by
        # the matching evidence vocabulary even when no rare term is shared.
        return result[0] if terms & evidence_terms else None

    @staticmethod
    def _identifier(paper_id: str, kind: str, question: str) -> str:
        return "rq-" + sha256(f"{paper_id}\0{kind}\0{question.casefold()}".encode()).hexdigest()[:16]

    def _recommend(self, owner_id: str, paper_id: str, mode: str, samples: list[ChunkResponse],
                   context: str, previous_questions: list[str], limit: int) -> list[RecommendedQuestion]:
        if not samples:
            return []
        history = "\n".join(previous_questions[-10:])
        raw = self._provider_candidates(mode, f"{context}\nPreviously asked questions:\n{history}", samples, limit * 3)
        result: list[RecommendedQuestion] = []
        seen = list(previous_questions)
        for candidate in raw:
            candidate = " ".join(candidate.split())
            if not candidate.endswith("?"):
                candidate += "?"
            kind = self._question_type(candidate)
            if not self._different(candidate, seen):
                continue
            source = self._supported(owner_id, paper_id, candidate)
            if source is None:
                continue
            result.append(RecommendedQuestion(self._identifier(paper_id, kind, candidate), candidate, kind, source))
            seen.append(candidate)
            if len(result) == limit:
                break
        return result

    def starter(self, owner_id: str, paper_id: str, paper: dict) -> list[RecommendedQuestion]:
        version = f'{paper.get("file_hash", "")}:{paper.get("updated_at", "")}:{paper.get("embedding_model", "")}'
        key = (owner_id, paper_id, version, int(paper.get("chunk_count", 0)))
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self._cache_seconds:
                self._cache.move_to_end(key)
                return cached[1]
        # This prevents duplicate LLM work when several browser renders request
        # suggestions for the same selected paper at once.
        with self._generation_lock:
            with self._cache_lock:
                cached = self._cache.get(key)
                if cached and time.monotonic() - cached[0] < self._cache_seconds:
                    self._cache.move_to_end(key)
                    return cached[1]
            samples = self.vectors.sample(owner_id, paper_id, int(paper.get("chunk_count", 0)), 16)
            recommendations = self._recommend(owner_id, paper_id, "starter", samples, "", [], 4)
            with self._cache_lock:
                if recommendations:
                    self._cache[key] = (time.monotonic(), recommendations)
                else:
                    return []
                self._cache.move_to_end(key)
                if len(self._cache) > self._cache_limit:
                    self._cache.popitem(last=False)
            return recommendations

    def follow_up(self, owner_id: str, paper_id: str, question: str, answer: str,
                  sources: list[dict], previous_questions: list[str]) -> list[RecommendedQuestion]:
        samples = [ChunkResponse(**{key: item[key] for key in ("chunk_index", "page_number", "text", "content_hash")})
                   for item in sources if all(key in item for key in ("chunk_index", "page_number", "text", "content_hash"))]
        if not samples:
            return []
        context = f"User question: {question}\nAnswer: {answer}"
        return self._recommend(owner_id, paper_id, "follow_up", samples, context, previous_questions, 3)

    def alternatives(self, owner_id: str, paper_id: str, question: str, paper: dict,
                     previous_questions: list[str]) -> list[RecommendedQuestion]:
        samples = self.vectors.sample(owner_id, paper_id, int(paper.get("chunk_count", 0)), 16)
        return self._recommend(owner_id, paper_id, "alternative", samples,
                               f"The user asked an unsupported question: {question}",
                               previous_questions + [question], 3)
