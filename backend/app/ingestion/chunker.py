import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    source_url: str
    title: str
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


class Chunker:
    """Splits plain text into overlapping word-window passages."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str, source_url: str, title: str = "") -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = self._chunk_size - self._overlap
        start = 0

        while start < len(words):
            end = min(start + self._chunk_size, len(words))
            content = " ".join(words[start:end])
            chunks.append(Chunk(content=content, source_url=source_url, title=title))
            if end == len(words):
                break
            start += step

        return chunks
