"""
Tests unitaires pour src/services/webhook_service.py

Couvre :
- Envoi réussi (200, 201)
- Erreurs HTTP (4xx, 5xx) — loguées sans exception
- Erreurs réseau (ConnectError, Timeout) — loguées sans exception
- Vérification du payload et de l'URL
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(status_code: int) -> MagicMock:
    """Crée un mock de réponse httpx avec le status_code donné."""
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _make_async_client_mock(response: MagicMock) -> MagicMock:
    """
    Crée un mock d'AsyncClient compatible avec le context manager async.
    Le mock.post() retourne le response donné.
    """
    client_mock = MagicMock()
    client_mock.post = AsyncMock(return_value=response)
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=client_mock)
    async_cm.__aexit__ = AsyncMock(return_value=False)
    return async_cm, client_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_webhook_success_200():
    """POST réussi (200) — ne lève pas d'exception."""
    from src.services.webhook_service import send_webhook

    async_cm, _ = _make_async_client_mock(_make_mock_response(200))
    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        await send_webhook("http://example.com/hook", {"event": "prediction"})


@pytest.mark.asyncio
async def test_send_webhook_success_201():
    """POST réussi (201) — ne lève pas d'exception."""
    from src.services.webhook_service import send_webhook

    async_cm, _ = _make_async_client_mock(_make_mock_response(201))
    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        await send_webhook("http://example.com/hook", {"event": "created"})


@pytest.mark.asyncio
async def test_send_webhook_http_4xx_logged_no_raise():
    """Réponse 400 — loguée sans lever d'exception."""
    from src.services.webhook_service import send_webhook

    async_cm, _ = _make_async_client_mock(_make_mock_response(400))
    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        # Ne doit pas lever d'exception
        await send_webhook("http://example.com/hook", {"event": "test"})


@pytest.mark.asyncio
async def test_send_webhook_http_5xx_logged_no_raise():
    """Réponse 500 — loguée sans lever d'exception."""
    from src.services.webhook_service import send_webhook

    async_cm, _ = _make_async_client_mock(_make_mock_response(500))
    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        await send_webhook("http://example.com/hook", {"event": "test"})


@pytest.mark.asyncio
async def test_send_webhook_connection_error_logged():
    """ConnectError réseau — loguée sans lever d'exception."""
    import httpx
    from src.services.webhook_service import send_webhook

    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        await send_webhook("http://unreachable.local/hook", {"event": "test"})


@pytest.mark.asyncio
async def test_send_webhook_timeout_logged():
    """TimeoutException — loguée sans lever d'exception."""
    import httpx
    from src.services.webhook_service import send_webhook

    async_cm = MagicMock()
    client_mock = MagicMock()
    client_mock.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    async_cm.__aenter__ = AsyncMock(return_value=client_mock)
    async_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        await send_webhook("http://slow.example.com/hook", {"event": "test"})


@pytest.mark.asyncio
async def test_send_webhook_sends_json_payload():
    """Le payload dict est bien transmis à client.post()."""
    from src.services.webhook_service import send_webhook

    payload = {"model": "my_model", "result": 42, "probability": 0.99}
    async_cm, client_mock = _make_async_client_mock(_make_mock_response(200))

    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        await send_webhook("http://example.com/hook", payload)

    client_mock.post.assert_called_once()
    call_kwargs = client_mock.post.call_args
    # Le payload doit être transmis via json=
    assert call_kwargs.kwargs.get("json") == payload or (
        len(call_kwargs.args) > 1 and call_kwargs.args[1] == payload
    )


@pytest.mark.asyncio
async def test_send_webhook_uses_correct_url():
    """L'URL fournie est bien passée à client.post()."""
    from src.services.webhook_service import send_webhook

    target_url = "http://myapp.io/callbacks/prediction"
    async_cm, client_mock = _make_async_client_mock(_make_mock_response(200))

    with patch("src.services.webhook_service.httpx.AsyncClient", return_value=async_cm):
        await send_webhook(target_url, {"event": "done"})

    client_mock.post.assert_called_once()
    call_args = client_mock.post.call_args
    # L'URL doit être le premier argument positionnel ou keyword 'url'
    assert call_args.args[0] == target_url or call_args.kwargs.get("url") == target_url
