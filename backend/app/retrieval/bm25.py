import re
from dataclasses import dataclass

from rank_bm25 import BM25Plus

_TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b")


@dataclass
class BM25Hit:
    passage_id: int
    score: float
    content: str
    spoiler_tier: int = 0


class BM25Index:
    """In-memory BM25 index over passage texts.

    Rebuilt from the Passages table at startup or after ingestion.
    Not persisted to disk — cheap to reconstruct from Postgres.

    Uses BM25Plus rather than BM25Okapi: BM25Okapi's IDF formula yields
    negative scores when a term appears in every document in the corpus,
    which causes the zero-score filter to incorrectly suppress relevant results
    in small or single-document indexes. BM25Plus floors IDF at 0.
    """

    def __init__(self) -> None:
        self._index: BM25Plus | None = None
        self._passage_ids: list[int] = []
        self._texts: list[str] = []
        self._spoiler_tiers: list[int] = []

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def build(
        self,
        passage_ids: list[int],
        texts: list[str],
        spoiler_tiers: list[int] | None = None,
    ) -> None:
        """(Re)build the index from a list of passage IDs and their text content."""
        self._passage_ids = list(passage_ids)
        self._texts = list(texts)
        if spoiler_tiers is None:
            self._spoiler_tiers = [0] * len(passage_ids)
        else:
            self._spoiler_tiers = list(spoiler_tiers)
        tokenized = [self._tokenize(text) for text in texts]
        self._index = BM25Plus(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 10,
        max_spoiler_tier: int | None = None,
    ) -> list[BM25Hit]:
        """Return up to top_k results with positive BM25 scores."""
        if self._index is None or not self._passage_ids:
            return []

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        indexed = sorted(
            [
                (i, score)
                for i, score in enumerate(scores)
                if score > 0
                and (
                    max_spoiler_tier is None
                    or self._spoiler_tiers[i] <= max_spoiler_tier
                )
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            BM25Hit(
                passage_id=self._passage_ids[i],
                score=float(score),
                content=self._texts[i],
                spoiler_tier=self._spoiler_tiers[i],
            )
            for i, score in indexed
        ]
