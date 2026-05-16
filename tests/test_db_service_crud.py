"""
Tests unitaires — DBService CRUD (src/services/db_service.py).

Stratégie :
  Utilise _TestSessionLocal de conftest (SQLite in-memory) pour tester les
  méthodes de DBService en isolation sans passer par l'API HTTP.
  N'inclut PAS upsert_observed_results (PostgreSQL-only, déjà couvert via test_observed_results.py).

Tokens : préfixe "test-token-dbcrud-" pour éviter les collisions.

Couvre :
- Utilisateurs : create, get_by_token, get_by_id, get_all, delete, update, last_login
- Prédictions : create, count_today
- Modèles : create_metadata, get_metadata, get_all_active, deactivate
"""

import asyncio
import secrets
from datetime import datetime, timedelta

import pytest

from src.services.db_service import DBService
from tests.conftest import _TestSessionLocal

# Tokens uniques pour ce module
ADMIN_TOKEN = "test-token-dbcrud-admin-zz00"
USER_TOKEN = "test-token-dbcrud-user-zz01"

# Modèle de référence pour les tests
MODEL_NAME = "dbcrud_test_model"
MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Setup : créer les utilisateurs de base une seule fois
# ---------------------------------------------------------------------------


async def _setup():
    """Crée les utilisateurs admin et user si absents."""
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="dbcrud_admin",
                email="dbcrud_admin@test.com",
                api_token=ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_user_by_token(db, USER_TOKEN):
            await DBService.create_user(
                db,
                username="dbcrud_user",
                email="dbcrud_user@test.com",
                api_token=USER_TOKEN,
                role="user",
                rate_limit=100,
            )


asyncio.run(_setup())


# ---------------------------------------------------------------------------
# Tests utilisateurs
# ---------------------------------------------------------------------------


class TestUserCRUD:
    """Tests CRUD sur les utilisateurs."""

    def test_create_user_returns_user_with_id(self):
        """create_user() retourne un User avec id, username, email, api_token."""

        async def _run():
            async with _TestSessionLocal() as db:
                token = secrets.token_urlsafe(16)
                user = await DBService.create_user(
                    db,
                    username=f"dbcrud_new_{token[:6]}",
                    email=f"new_{token[:6]}@test.com",
                    api_token=token,
                    role="user",
                    rate_limit=50,
                )
                assert user.id is not None
                assert user.username == f"dbcrud_new_{token[:6]}"
                assert user.api_token == token
                return user.id

        asyncio.run(_run())

    def test_get_user_by_token_returns_user(self):
        """get_user_by_token() retourne l'utilisateur pour un token connu."""

        async def _run():
            async with _TestSessionLocal() as db:
                user = await DBService.get_user_by_token(db, ADMIN_TOKEN)
                assert user is not None
                assert user.api_token == ADMIN_TOKEN

        asyncio.run(_run())

    def test_get_user_by_token_returns_none_for_unknown(self):
        """get_user_by_token() retourne None pour un token inconnu."""

        async def _run():
            async with _TestSessionLocal() as db:
                user = await DBService.get_user_by_token(db, "totally-unknown-token-xyz-99")
                assert user is None

        asyncio.run(_run())

    def test_get_user_by_id_returns_user(self):
        """get_user_by_id() retourne l'utilisateur pour un id connu."""

        async def _run():
            async with _TestSessionLocal() as db:
                admin = await DBService.get_user_by_token(db, ADMIN_TOKEN)
                user = await DBService.get_user_by_id(db, admin.id)
                assert user is not None
                assert user.id == admin.id

        asyncio.run(_run())

    def test_get_user_by_id_returns_none_for_unknown(self):
        """get_user_by_id() retourne None pour un id inexistant."""

        async def _run():
            async with _TestSessionLocal() as db:
                user = await DBService.get_user_by_id(db, 999999)
                assert user is None

        asyncio.run(_run())

    def test_get_all_users_includes_admin(self):
        """get_all_users() retourne une liste incluant l'admin créé."""

        async def _run():
            async with _TestSessionLocal() as db:
                users = await DBService.get_all_users(db)
                tokens = [u.api_token for u in users]
                assert ADMIN_TOKEN in tokens

        asyncio.run(_run())

    def test_delete_user_removes_from_db(self):
        """delete_user() supprime l'utilisateur et get_user_by_id retourne None ensuite."""

        async def _run():
            async with _TestSessionLocal() as db:
                token = secrets.token_urlsafe(16)
                user = await DBService.create_user(
                    db,
                    username=f"todel_{token[:8]}",
                    email=f"todel_{token[:8]}@test.com",
                    api_token=token,
                )
                user_id = user.id
                result = await DBService.delete_user(db, user_id)
                assert result is True

            async with _TestSessionLocal() as db2:
                deleted = await DBService.get_user_by_id(db2, user_id)
                assert deleted is None

        asyncio.run(_run())

    def test_delete_user_returns_false_for_unknown_id(self):
        """delete_user() retourne False si l'utilisateur n'existe pas."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.delete_user(db, 9999999)
                assert result is False

        asyncio.run(_run())

    def test_update_user_changes_email(self):
        """update_user() modifie l'email et le persiste en base."""

        async def _run():
            async with _TestSessionLocal() as db:
                token = secrets.token_urlsafe(16)
                user = await DBService.create_user(
                    db,
                    username=f"toupdate_{token[:6]}",
                    email=f"old_{token[:6]}@test.com",
                    api_token=token,
                )
                new_email = f"new_{token[:6]}@test.com"
                updated = await DBService.update_user(db, user.id, email=new_email)
                assert updated is not None
                assert updated.email == new_email

        asyncio.run(_run())

    def test_update_user_regenerates_token(self):
        """update_user(regenerate_token=True) génère un token différent de l'original."""

        async def _run():
            async with _TestSessionLocal() as db:
                token = secrets.token_urlsafe(16)
                user = await DBService.create_user(
                    db,
                    username=f"toregen_{token[:6]}",
                    email=f"regen_{token[:6]}@test.com",
                    api_token=token,
                )
                original_token = user.api_token
                updated = await DBService.update_user(db, user.id, regenerate_token=True)
                assert updated is not None
                assert updated.api_token != original_token

        asyncio.run(_run())

    def test_update_user_returns_none_for_unknown_id(self):
        """update_user() retourne None pour un id inexistant."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.update_user(db, 9999999, email="x@test.com")
                assert result is None

        asyncio.run(_run())

    def test_update_user_last_login_sets_timestamp(self):
        """update_user_last_login() met à jour le champ last_login."""

        async def _run():
            async with _TestSessionLocal() as db:
                admin = await DBService.get_user_by_token(db, ADMIN_TOKEN)
                assert admin is not None
                await DBService.update_user_last_login(db, admin.id)

            async with _TestSessionLocal() as db2:
                admin2 = await DBService.get_user_by_token(db2, ADMIN_TOKEN)
                assert admin2 is not None
                assert admin2.last_login is not None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests prédictions
# ---------------------------------------------------------------------------


class TestPredictionCRUD:
    """Tests sur la création et le comptage des prédictions."""

    def test_create_prediction_returns_prediction_with_id(self):
        """create_prediction() retourne une Prediction avec id non null."""

        async def _run():
            async with _TestSessionLocal() as db:
                admin = await DBService.get_user_by_token(db, ADMIN_TOKEN)
                pred = await DBService.create_prediction(
                    db,
                    user_id=admin.id,
                    model_name="dbcrud_pred_model",
                    model_version="1.0.0",
                    input_features={"x": 1.0, "y": 2.0},
                    prediction_result=0,
                    probabilities=[0.8, 0.2],
                    response_time_ms=12.5,
                )
                assert pred.id is not None
                assert pred.model_name == "dbcrud_pred_model"

        asyncio.run(_run())

    def test_count_predictions_today_counts_only_today(self):
        """get_user_prediction_count_today() compte uniquement les prédictions d'aujourd'hui."""

        async def _run():
            async with _TestSessionLocal() as db:
                # Créer un utilisateur dédié avec token unique
                token = "test-token-dbcrud-countday-aa"
                if not await DBService.get_user_by_token(db, token):
                    await DBService.create_user(
                        db,
                        username="dbcrud_countday",
                        email="countday@test.com",
                        api_token=token,
                    )
                user = await DBService.get_user_by_token(db, token)

                count_before = await DBService.get_user_prediction_count_today(db, user.id)

                # Créer 2 prédictions aujourd'hui
                for _ in range(2):
                    await DBService.create_prediction(
                        db,
                        user_id=user.id,
                        model_name="count_model",
                        model_version="1.0.0",
                        input_features={"a": 1},
                        prediction_result=1,
                        probabilities=None,
                        response_time_ms=5.0,
                    )

                count_after = await DBService.get_user_prediction_count_today(db, user.id)
                assert count_after == count_before + 2

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests métadonnées modèles
# ---------------------------------------------------------------------------


class TestModelMetadataCRUD:
    """Tests sur le CRUD des métadonnées de modèles."""

    def test_create_model_metadata_persists(self):
        """create_model_metadata() retourne un objet avec name et version corrects."""

        async def _run():
            async with _TestSessionLocal() as db:
                suffix = secrets.token_hex(4)
                meta = await DBService.create_model_metadata(
                    db,
                    name=f"dbcrud_model_{suffix}",
                    version="1.0.0",
                    minio_bucket="models",
                    minio_object_key=f"dbcrud_model_{suffix}/v1.0.0.joblib",
                    accuracy=0.92,
                )
                assert meta.id is not None
                assert meta.name == f"dbcrud_model_{suffix}"
                assert meta.version == "1.0.0"
                assert meta.accuracy == 0.92

        asyncio.run(_run())

    def test_get_model_metadata_by_version(self):
        """get_model_metadata() avec version spécifique retourne le bon modèle."""

        async def _run():
            async with _TestSessionLocal() as db:
                suffix = secrets.token_hex(4)
                model_name = f"dbcrud_getmeta_{suffix}"
                await DBService.create_model_metadata(
                    db,
                    name=model_name,
                    version="2.0.0",
                    minio_bucket="models",
                    minio_object_key=f"{model_name}/v2.0.0.joblib",
                )
                meta = await DBService.get_model_metadata(db, model_name, "2.0.0")
                assert meta is not None
                assert meta.version == "2.0.0"

        asyncio.run(_run())

    def test_get_model_metadata_unknown_version_returns_none(self):
        """get_model_metadata() retourne None pour une version inconnue."""

        async def _run():
            async with _TestSessionLocal() as db:
                meta = await DBService.get_model_metadata(
                    db, "nonexistent_model_xyz", "9.9.9"
                )
                assert meta is None

        asyncio.run(_run())

    def test_get_all_active_models_includes_active(self):
        """get_all_active_models() liste uniquement les modèles is_active=True."""

        async def _run():
            async with _TestSessionLocal() as db:
                suffix = secrets.token_hex(4)
                model_name = f"dbcrud_active_{suffix}"
                await DBService.create_model_metadata(
                    db,
                    name=model_name,
                    version="1.0.0",
                    minio_bucket="models",
                    minio_object_key=f"{model_name}/v1.0.0.joblib",
                )
                models = await DBService.get_all_active_models(db)
                names = [m.name for m in models]
                assert model_name in names

        asyncio.run(_run())

    def test_deactivate_model_sets_is_active_false(self):
        """deactivate_model() désactive le modèle et get_model_metadata ne le retourne plus."""

        async def _run():
            async with _TestSessionLocal() as db:
                suffix = secrets.token_hex(4)
                model_name = f"dbcrud_deact_{suffix}"
                await DBService.create_model_metadata(
                    db,
                    name=model_name,
                    version="1.0.0",
                    minio_bucket="models",
                    minio_object_key=f"{model_name}/v1.0.0.joblib",
                )
                result = await DBService.deactivate_model(db, model_name, "1.0.0")
                assert result is True
                meta = await DBService.get_model_metadata(db, model_name, "1.0.0")
                assert meta is None

        asyncio.run(_run())

    def test_deactivate_model_returns_false_for_unknown(self):
        """deactivate_model() retourne False si le modèle n'existe pas."""

        async def _run():
            async with _TestSessionLocal() as db:
                result = await DBService.deactivate_model(db, "unknown_xyz", "9.9.9")
                assert result is False

        asyncio.run(_run())
