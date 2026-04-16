from unittest.mock import patch

from app.db.session import get_engine, get_session_factory


def test_get_engine_uses_database_url():
    with patch("app.db.session.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "postgresql+asyncpg://u:p@host/db"
        engine = get_engine()
        assert "postgresql" in str(engine.url)


def test_get_session_factory_returns_callable():
    with patch("app.db.session.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "postgresql+asyncpg://u:p@host/db"
        factory = get_session_factory()
        assert callable(factory)
