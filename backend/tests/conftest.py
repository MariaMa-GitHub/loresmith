import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.db.session import get_engine, get_session_factory
from app.main import app


@pytest.fixture(autouse=True)
def clear_cached_settings_and_db_handles():
    """Prevent cached mocked settings/engines from leaking between tests."""
    get_session_factory.cache_clear()
    get_engine.cache_clear()
    get_settings.cache_clear()
    yield
    get_session_factory.cache_clear()
    get_engine.cache_clear()
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
