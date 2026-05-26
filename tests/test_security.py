"""
Unit tests — authentication functions (src/core/security.py).

Strategy:
  Call async functions directly via asyncio.run().
  DBService is mocked with unittest.mock.patch.
  User/HTTPAuthorizationCredentials objects are built with MagicMock.

Covers:
- verify_token(): valid token, invalid, inactive user, last_login update
- check_prediction_rate_limit(): quota not reached, exactly reached, exceeded
- require_admin(): admin, user, readonly
"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _make_user(
    user_id=1,
    token="valid-token",
    is_active=True,
    role="user",
    rate_limit=100,
    token_expires_at=None,
):
    """Build a minimal mock User object."""
    from src.db.models import UserRole

    user = MagicMock()
    user.id = user_id
    user.api_token = token
    user.is_active = is_active
    user.role = UserRole.ADMIN if role == "admin" else (
        UserRole.READONLY if role == "readonly" else UserRole.USER
    )
    user.rate_limit_per_day = rate_limit
    user.token_expires_at = token_expires_at
    return user


def _make_credentials(token="valid-token"):
    """Build a mock HTTPAuthorizationCredentials."""
    creds = MagicMock()
    creds.credentials = token
    return creds


# ---------------------------------------------------------------------------
# Tests verify_token
# ---------------------------------------------------------------------------


class TestVerifyToken:
    """Tests for verify_token()."""

    def test_valid_token_returns_user(self):
        """Valid token with active user → returns the user."""
        from src.core.security import verify_token

        mock_user = _make_user(is_active=True)
        mock_db = MagicMock()

        async def _run():
            with (
                patch(
                    "src.core.security.DBService.get_user_by_token",
                    new=AsyncMock(return_value=mock_user),
                ),
                patch(
                    "src.core.security.DBService.update_user_last_login",
                    new=AsyncMock(return_value=None),
                ),
            ):
                result = await verify_token(_make_credentials("valid-token"), mock_db)
                return result

        result = asyncio.run(_run())
        assert result is mock_user

    def test_invalid_token_raises_401(self):
        """Unknown token (None from get_user_by_token) → HTTPException 401."""
        from src.core.security import verify_token

        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_by_token",
                new=AsyncMock(return_value=None),
            ):
                await verify_token(_make_credentials("bad-token"), mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 401

    def test_inactive_user_raises_403(self):
        """User found but is_active=False → HTTPException 403."""
        from src.core.security import verify_token

        mock_user = _make_user(is_active=False)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_by_token",
                new=AsyncMock(return_value=mock_user),
            ):
                await verify_token(_make_credentials("inactive-token"), mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 403

    def test_valid_token_calls_update_last_login(self):
        """Successful authentication → update_user_last_login is called."""
        from src.core.security import verify_token

        mock_user = _make_user(is_active=True, user_id=42)
        mock_db = MagicMock()
        mock_update = AsyncMock(return_value=None)

        async def _run():
            with (
                patch(
                    "src.core.security.DBService.get_user_by_token",
                    new=AsyncMock(return_value=mock_user),
                ),
                patch(
                    "src.core.security.DBService.update_user_last_login",
                    new=mock_update,
                ),
            ):
                await verify_token(_make_credentials("valid-token"), mock_db)

        asyncio.run(_run())
        mock_update.assert_called_once_with(mock_db, 42)

    def test_invalid_token_detail_message(self):
        """HTTPException 401 contains an error message."""
        from src.core.security import verify_token

        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_by_token",
                new=AsyncMock(return_value=None),
            ):
                await verify_token(_make_credentials("bad-token"), mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.detail != ""

    def test_expired_token_raises_401(self):
        """Token with token_expires_at in the past → HTTPException 401."""
        from src.core.security import verify_token

        past = datetime.utcnow() - timedelta(days=1)
        mock_user = _make_user(is_active=True, token_expires_at=past)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_by_token",
                new=AsyncMock(return_value=mock_user),
            ):
                await verify_token(_make_credentials("expired-token"), mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 401

    def test_expired_token_detail_says_expire(self):
        """The 401 message for expired token contains 'expired'."""
        from src.core.security import verify_token

        past = datetime.utcnow() - timedelta(seconds=1)
        mock_user = _make_user(is_active=True, token_expires_at=past)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_by_token",
                new=AsyncMock(return_value=mock_user),
            ):
                await verify_token(_make_credentials("expired-token"), mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert "expired" in exc_info.value.detail.lower()

    def test_non_expired_token_passes(self):
        """Token with token_expires_at in the future → successful authentication."""
        from src.core.security import verify_token

        future = datetime.utcnow() + timedelta(days=89)
        mock_user = _make_user(is_active=True, token_expires_at=future)
        mock_db = MagicMock()

        async def _run():
            with (
                patch(
                    "src.core.security.DBService.get_user_by_token",
                    new=AsyncMock(return_value=mock_user),
                ),
                patch(
                    "src.core.security.DBService.update_user_last_login",
                    new=AsyncMock(return_value=None),
                ),
            ):
                return await verify_token(_make_credentials("valid-token"), mock_db)

        result = asyncio.run(_run())
        assert result is mock_user

    def test_no_expiry_token_passes(self):
        """Token without token_expires_at (None) → no expiry check."""
        from src.core.security import verify_token

        mock_user = _make_user(is_active=True, token_expires_at=None)
        mock_db = MagicMock()

        async def _run():
            with (
                patch(
                    "src.core.security.DBService.get_user_by_token",
                    new=AsyncMock(return_value=mock_user),
                ),
                patch(
                    "src.core.security.DBService.update_user_last_login",
                    new=AsyncMock(return_value=None),
                ),
            ):
                return await verify_token(_make_credentials("valid-token"), mock_db)

        result = asyncio.run(_run())
        assert result is mock_user


# ---------------------------------------------------------------------------
# Tests check_prediction_rate_limit
# ---------------------------------------------------------------------------


class TestCheckPredictionRateLimit:
    """Tests for check_prediction_rate_limit()."""

    def test_below_limit_returns_user(self):
        """Count < rate_limit → returns the user."""
        from src.core.security import check_prediction_rate_limit

        mock_user = _make_user(rate_limit=100)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_prediction_count_today",
                new=AsyncMock(return_value=50),
            ):
                return await check_prediction_rate_limit(mock_user, mock_db)

        result = asyncio.run(_run())
        assert result is mock_user

    def test_at_limit_raises_429(self):
        """Count == rate_limit_per_day → HTTPException 429."""
        from src.core.security import check_prediction_rate_limit

        mock_user = _make_user(rate_limit=10)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_prediction_count_today",
                new=AsyncMock(return_value=10),
            ):
                await check_prediction_rate_limit(mock_user, mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 429

    def test_above_limit_raises_429(self):
        """Count > rate_limit_per_day → HTTPException 429."""
        from src.core.security import check_prediction_rate_limit

        mock_user = _make_user(rate_limit=5)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_prediction_count_today",
                new=AsyncMock(return_value=7),
            ):
                await check_prediction_rate_limit(mock_user, mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 429

    def test_rate_limit_detail_includes_limit_and_count(self):
        """The 429 message mentions the quota and the counter."""
        from src.core.security import check_prediction_rate_limit

        mock_user = _make_user(rate_limit=5)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_prediction_count_today",
                new=AsyncMock(return_value=7),
            ):
                await check_prediction_rate_limit(mock_user, mock_db)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        detail = exc_info.value.detail
        assert "5" in detail  # the quota
        assert "7" in detail  # the count

    def test_zero_predictions_passes_non_zero_limit(self):
        """Count=0 with rate_limit=10 → returns user (no quota consumed)."""
        from src.core.security import check_prediction_rate_limit

        mock_user = _make_user(rate_limit=10)
        mock_db = MagicMock()

        async def _run():
            with patch(
                "src.core.security.DBService.get_user_prediction_count_today",
                new=AsyncMock(return_value=0),
            ):
                return await check_prediction_rate_limit(mock_user, mock_db)

        result = asyncio.run(_run())
        assert result is mock_user


# ---------------------------------------------------------------------------
# Tests require_admin
# ---------------------------------------------------------------------------


class TestRequireAdmin:
    """Tests for require_admin()."""

    def test_admin_role_returns_user(self):
        """User with role=admin → returns the user."""
        from src.core.security import require_admin

        mock_user = _make_user(role="admin")

        async def _run():
            return await require_admin(mock_user)

        result = asyncio.run(_run())
        assert result is mock_user

    def test_user_role_raises_403(self):
        """User with role=user → HTTPException 403."""
        from src.core.security import require_admin

        mock_user = _make_user(role="user")

        async def _run():
            await require_admin(mock_user)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 403

    def test_readonly_role_raises_403(self):
        """User with role=readonly → HTTPException 403."""
        from src.core.security import require_admin

        mock_user = _make_user(role="readonly")

        async def _run():
            await require_admin(mock_user)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 403

    def test_admin_error_message_contains_administrateurs(self):
        """The 403 message mentions administrators."""
        from src.core.security import require_admin

        mock_user = _make_user(role="user")

        async def _run():
            await require_admin(mock_user)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert "admin" in exc_info.value.detail.lower()
