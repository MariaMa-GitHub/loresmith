import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.section_extractor import SectionRecord


@dataclass
class Chunk:
    content: str
    source_url: str
    title: str
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


class Chunker:
    """Splits plain text into overlapping word-window passages.

    If a ``title`` is provided, it is prepended as a single-line header to
    every chunk's content so that the embedder and BM25 both see the entity
    name the article is about — critical for lore queries where the title
    *is* the entity (e.g. "Zagreus", "Melinoë").
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    def chunk(self, text: str, source_url: str, title: str = "") -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        title_prefix = f"{title.strip()}\n\n" if title and title.strip() else ""
        chunks: list[Chunk] = []
        step = self._chunk_size - self._overlap
        start = 0

        while start < len(words):
            end = min(start + self._chunk_size, len(words))
            window = " ".join(words[start:end])
            content = f"{title_prefix}{window}"
            chunks.append(Chunk(content=content, source_url=source_url, title=title))
            if end == len(words):
                break
            start += step

        return chunks

    def chunk_sections(
        self,
        records: list["SectionRecord"],
        title: str,
        source_url: str,
    ) -> list[Chunk]:
        """Chunk a list of SectionRecords into Chunk objects.

        Short sections (≤ chunk_size words) become a single chunk. Oversized
        sections fall back to the same word-window approach as ``chunk()``.
        """
        chunks: list[Chunk] = []
        for record in records:
            section_label = " > ".join(record.section_path) if record.section_path else ""
            prefix = (
                f"{title.strip()}\n{section_label}\n\n" if section_label else f"{title.strip()}\n\n"
            )
            words = record.text.split()
            if not words:
                continue
            if len(words) <= self._chunk_size:
                chunks.append(
                    Chunk(
                        content=f"{prefix}{record.text}",
                        source_url=source_url,
                        title=title,
                    )
                )
            else:
                step = self._chunk_size - self._overlap
                start = 0
                while start < len(words):
                    end = min(start + self._chunk_size, len(words))
                    window = " ".join(words[start:end])
                    chunks.append(
                        Chunk(content=f"{prefix}{window}", source_url=source_url, title=title)
                    )
                    if end == len(words):
                        break
                    start += step
        return chunks
