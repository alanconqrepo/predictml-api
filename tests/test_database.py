"""
Tests pour src/db/database.py.

Couvre :
- init_db()  : exécute SELECT 1 via le moteur → vérifié par mock sur l'engine entier
- close_db() : appelle engine.dispose() → vérifié par mock sur l'engine entier
- get_db()   : génère une AsyncSession et la ferme correctement
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# init_db — vérifie l'exécution de SELECT 1
# ---------------------------------------------------------------------------


def test_init_db_calls_execute():
    """init_db() doit appeler conn.execute() exactement une fois."""
    from src.db.database import init_db

    mock_execute = AsyncMock()
    mock_conn = AsyncMock()
    mock_conn.execute = mock_execute
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    mock_engine = MagicMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)

    with patch("src.db.database.engine", mock_engine):
        asyncio.run(init_db())

    mock_execute.assert_called_once()


def test_init_db_execute_called_with_text_clause():
    """L'argument passé à execute() est un objet TextClause SQLAlchemy (SELECT 1)."""
    from sqlalchemy.sql.elements import TextClause
    from src.db.database import init_db

    captured = []
    mock_execute = AsyncMock(side_effect=lambda arg: captured.append(arg))
    mock_conn = AsyncMock()
    mock_conn.execute = mock_execute
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    mock_engine = MagicMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)

    with patch("src.db.database.engine", mock_engine):
        asyncio.run(init_db())

    assert len(captured) == 1
    assert isinstance(captured[0], TextClause)
    assert "1" in str(captured[0])


def test_init_db_propagates_connection_error():
    """init_db() propage l'exception si la connexion échoue."""
    from src.db.database import init_db

    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(side_effect=Exception("connexion refusée"))
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    mock_engine = MagicMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)

    with patch("src.db.database.engine", mock_engine):
        with pytest.raises(Exception, match="connexion refusée"):
            asyncio.run(init_db())


# ---------------------------------------------------------------------------
# close_db — vérifie l'appel à engine.dispose()
# ---------------------------------------------------------------------------


def test_close_db_calls_dispose():
    """close_db() appelle engine.dispose() exactement une fois."""
    from src.db.database import close_db

    mock_dispose = AsyncMock()
    mock_engine = MagicMock()
    mock_engine.dispose = mock_dispose

    with patch("src.db.database.engine", mock_engine):
        asyncio.run(close_db())

    mock_dispose.assert_called_once()


def test_close_db_does_not_raise_normally():
    """close_db() ne lève pas d'exception en conditions normales."""
    from src.db.database import close_db

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()

    with patch("src.db.database.engine", mock_engine):
        asyncio.run(close_db())  # ne doit pas lever


# ---------------------------------------------------------------------------
# get_db — vérification du cycle de vie de la session
# ---------------------------------------------------------------------------


def test_get_db_yields_session():
    """get_db() doit yield l'objet de session retourné par AsyncSessionLocal."""
    from src.db.database import get_db

    mock_session = AsyncMock()
    mock_session.close = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("src.db.database.AsyncSessionLocal", return_value=mock_ctx):
        async def _collect():
            gen = get_db()
            session = await gen.__anext__()
            await gen.aclose()
            return session

        result = asyncio.run(_collect())

    assert result is mock_session


def test_get_db_context_manager_exited():
    """get_db() ferme bien le contexte manager de la session (finally)."""
    from src.db.database import get_db

    mock_session = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("src.db.database.AsyncSessionLocal", return_value=mock_ctx):
        async def _exhaust():
            gen = get_db()
            await gen.__anext__()
            await gen.aclose()

        asyncio.run(_exhaust())

    mock_ctx.__aexit__.assert_called()
