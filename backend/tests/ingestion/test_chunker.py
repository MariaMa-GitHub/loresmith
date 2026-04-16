import hashlib

from app.ingestion.chunker import Chunk, Chunker


def test_chunk_has_content_hash():
    c = Chunk(content="hello world", source_url="https://example.com", title="Test")
    assert c.content_hash == hashlib.sha256("hello world".encode()).hexdigest()


def test_chunker_short_text_yields_one_chunk():
    chunker = Chunker(chunk_size=400, overlap=50)
    text = "Zagreus is the son of Hades. " * 5  # 30 words — well below chunk_size
    chunks = chunker.chunk(text, "https://example.com/wiki/Foo", title="Foo")
    assert len(chunks) == 1
    assert "Zagreus" in chunks[0].content


def test_chunker_long_text_yields_multiple_chunks():
    chunker = Chunker(chunk_size=20, overlap=5)
    # 80 words → should produce at least 3 chunks with size=20, overlap=5
    text = " ".join(["word"] * 80)
    chunks = chunker.chunk(text, "https://example.com/wiki/Bar", title="Bar")
    assert len(chunks) >= 3


def test_chunker_chunks_overlap():
    chunker = Chunker(chunk_size=10, overlap=3)
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunker.chunk(text, "https://example.com", title="Overlap Test")
    # The last 3 words of chunk N should appear at the start of chunk N+1
    assert len(chunks) >= 2
    end_of_first = chunks[0].content.split()[-3:]
    start_of_second = chunks[1].content.split()[:3]
    assert end_of_first == start_of_second


def test_chunker_all_chunks_have_source_url():
    chunker = Chunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk(" ".join(["x"] * 50), "https://example.com/wiki/Nyx", title="Nyx")
    for chunk in chunks:
        assert chunk.source_url == "https://example.com/wiki/Nyx"


def test_chunker_empty_text_yields_no_chunks():
    chunker = Chunker(chunk_size=400, overlap=50)
    chunks = chunker.chunk("", "https://example.com", title="Empty")
    assert chunks == []


def test_chunker_hashes_are_unique_across_chunks():
    chunker = Chunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk(" ".join([f"unique{i}" for i in range(40)]), "https://x.com", title="X")
    hashes = [c.content_hash for c in chunks]
    assert len(hashes) == len(set(hashes))


def test_chunker_raises_if_overlap_not_less_than_chunk_size():
    import pytest
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, overlap=10)
