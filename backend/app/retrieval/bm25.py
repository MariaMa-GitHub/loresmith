from dataclasses import dataclass

from rank_bm25 import BM25Plus


@dataclass
class BM25Hit:
    passage_id: int
    score: float
    content: str


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

    def build(self, passage_ids: list[int], texts: list[str]) -> None:
        """(Re)build the index from a list of passage IDs and their text content."""
        self._passage_ids = list(passage_ids)
        self._texts = list(texts)
        tokenized = [text.lower().split() for text in texts]
        self._index = BM25Plus(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[BM25Hit]:
        """Return up to top_k results with positive BM25 scores."""
        if self._index is None or not self._passage_ids:
            return []

        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)

        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            BM25Hit(
                passage_id=self._passage_ids[i],
                score=float(score),
                content=self._texts[i],
            )
            for i, score in indexed
            if score > 0
        ]
