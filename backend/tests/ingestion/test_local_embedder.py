"""LocalEmbedder unit tests.

We stub `sentence_transformers` at sys.modules level so the real library (and
its heavy torch + transformers transitive imports) never loads during tests.
This keeps the suite fast and CI offline.
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _install_stub_sentence_transformers(dim: int = 768) -> MagicMock:
    """Replace `sentence_transformers` in sys.modules with a lightweight stub.

    Returns the MagicMock standing in for SentenceTransformer so tests can
    inspect call counts / arguments.
    """
    captured_kwargs: dict = {}

    def model_factory(model_name, device=None):
        # Fake numpy array that supports .tolist() without importing numpy at
        # module top-level (numpy is already a transitive dep but we keep the
        # import local for isolation).
        import numpy as np

        model = MagicMock()
        # LocalEmbedder tries `get_embedding_dimension` first (new-style in
        # sentence-transformers 5.x) and falls back to the old name. Configure
        # both so the test works regardless of which branch runs.
        model.get_embedding_dimension = MagicMock(return_value=dim)
        model.get_sentence_embedding_dimension = MagicMock(return_value=dim)

        def encode(texts, **kw):
            captured_kwargs.update(kw)
            return np.zeros((len(texts), dim), dtype=np.float32)

        model.encode = MagicMock(side_effect=encode)
        return model

    stub_cls = MagicMock(side_effect=model_factory)
    stub_cls.captured_encode_kwargs = captured_kwargs  # type: ignore[attr-defined]

    stub_module = ModuleType("sentence_transformers")
    stub_module.SentenceTransformer = stub_cls  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = stub_module
    return stub_cls


@pytest.fixture
def stub_st():
    """Install the stub for a single test and clean up after."""
    prior = sys.modules.get("sentence_transformers")
    stub_cls = _install_stub_sentence_transformers(dim=768)
    yield stub_cls
    if prior is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = prior


@pytest.fixture
def stub_st_wrong_dim():
    """Same as stub_st but the stub reports dim=384, used to test validation."""
    prior = sys.modules.get("sentence_transformers")
    stub_cls = _install_stub_sentence_transformers(dim=384)
    yield stub_cls
    if prior is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = prior


@pytest.mark.asyncio
async def test_local_embedder_returns_768d_vector_per_text(stub_st):
    from app.ingestion.local_embedder import LocalEmbedder

    embedder = LocalEmbedder()
    results = await embedder.embed(["text one", "text two", "text three"])

    assert len(results) == 3
    assert all(len(v) == 768 for v in results)


@pytest.mark.asyncio
async def test_local_embedder_empty_list_returns_empty(stub_st):
    from app.ingestion.local_embedder import LocalEmbedder

    embedder = LocalEmbedder()
    results = await embedder.embed([])

    assert results == []
    stub_st.assert_not_called()  # no model load on empty input


@pytest.mark.asyncio
async def test_local_embedder_loads_model_once(stub_st):
    from app.ingestion.local_embedder import LocalEmbedder

    embedder = LocalEmbedder()
    await embedder.embed(["a"])
    await embedder.embed(["b", "c"])

    assert stub_st.call_count == 1


@pytest.mark.asyncio
async def test_local_embedder_rejects_wrong_dimension(stub_st_wrong_dim):
    from app.ingestion.local_embedder import LocalEmbedder

    embedder = LocalEmbedder(model_name="BAAI/bge-small-en-v1.5")
    with pytest.raises(RuntimeError, match="384-dim"):
        await embedder.embed(["text"])


@pytest.mark.asyncio
async def test_local_embedder_passes_normalize_flag(stub_st):
    from app.ingestion.local_embedder import LocalEmbedder

    embedder = LocalEmbedder()
    await embedder.embed(["x"])

    assert stub_st.captured_encode_kwargs.get("normalize_embeddings") is True
