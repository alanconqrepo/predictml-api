"""
Tests unitaires pour GoldenTestService.

Couvre les branches non exercées par les tests d'intégration existants :
- parse_csv()    : BOM, coercions de types, valeurs vides, CSV vide, description
- run_for_policy(): retour None si 0 tests, propagation HTTPException, wrapping 500
- run_tests()    : modèle sans feature_names_in_ (fallback), exception de prédiction
"""

import asyncio
import io
import joblib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.schemas.golden_test import GoldenTestRunResponse
from src.services.db_service import DBService
from src.services.golden_test_service import GoldenTestService
from src.services.model_service import model_service
from tests.conftest import _TestSessionLocal

# ---------------------------------------------------------------------------
# Constantes et helpers
# ---------------------------------------------------------------------------

_MODEL = "gts_unit_iris"
_VERSION = "3.0.0"
_ADMIN_TOKEN = "test-token-gts-unit-001"


def _make_df_model() -> LogisticRegression:
    """LogisticRegression entraîné sur DataFrame → feature_names_in_ disponible."""
    df = pd.DataFrame(
        {
            "f1": [5.1, 4.9, 6.3, 5.8],
            "f2": [3.5, 3.0, 2.9, 2.7],
        }
    )
    y = [0, 0, 1, 1]
    return LogisticRegression(max_iter=500).fit(df, y)


def _make_np_model() -> LogisticRegression:
    """LogisticRegression entraîné sur array NumPy → pas de feature_names_in_."""
    X = np.array([[5.1, 3.5], [4.9, 3.0], [6.3, 2.9], [5.8, 2.7]])
    y = [0, 0, 1, 1]
    m = LogisticRegression(max_iter=500).fit(X, y)
    assert not hasattr(m, "feature_names_in_")
    return m


def _inject_cache(name: str, version: str, model) -> str:
    key = f"{name}:{version}"
    data = {
        "model": model,
        "metadata": SimpleNamespace(
            name=name,
            version=version,
            confidence_threshold=None,
            webhook_url=None,
            feature_baseline=None,
        ),
    }
    _jbuf = io.BytesIO()
    joblib.dump(data, _jbuf)
    asyncio.run(model_service._redis.set(f"model:{key}", _jbuf.getvalue()))
    return key


async def _clear_cache(key: str):
    await model_service._redis.delete(f"model:{key}")


async def _setup_db():
    async with _TestSessionLocal() as db:
        if not await DBService.get_user_by_token(db, _ADMIN_TOKEN):
            await DBService.create_user(
                db,
                username="gts_unit_admin",
                email="gts_unit_admin@test.com",
                api_token=_ADMIN_TOKEN,
                role="admin",
                rate_limit=10000,
            )
        if not await DBService.get_model_metadata(db, _MODEL, _VERSION):
            await DBService.create_model_metadata(
                db,
                name=_MODEL,
                version=_VERSION,
                minio_bucket="models",
                minio_object_key=f"{_MODEL}/v{_VERSION}.joblib",
                is_active=True,
                is_production=True,
            )


asyncio.run(_setup_db())


# ---------------------------------------------------------------------------
# parse_csv — tests unitaires purs
# ---------------------------------------------------------------------------


class TestParseCsv:
    """Tests unitaires pour GoldenTestService.parse_csv()."""

    def test_basic_csv(self):
        csv = b"f1,f2,expected_output\n1.0,2.0,setosa\n"
        rows = GoldenTestService.parse_csv(csv)
        assert len(rows) == 1
        assert rows[0]["expected_output"] == "setosa"
        assert rows[0]["input_features"] == {"f1": 1.0, "f2": 2.0}

    def test_bom_prefix_ignored(self):
        """UTF-8 BOM (\\ufeff) ne doit pas polluer le nom de la première colonne."""
        # encode("utf-8-sig") ajoute les bytes BOM \xef\xbb\xbf en tête
        csv = "sepal_length,expected_output\n5.1,setosa\n".encode("utf-8-sig")
        rows = GoldenTestService.parse_csv(csv)
        assert len(rows) == 1
        assert "sepal_length" in rows[0]["input_features"]

    def test_int_coercion(self):
        """Valeur sans point décimal → int."""
        csv = b"count,expected_output\n3,ok\n"
        rows = GoldenTestService.parse_csv(csv)
        assert rows[0]["input_features"]["count"] == 3
        assert isinstance(rows[0]["input_features"]["count"], int)

    def test_float_coercion(self):
        """Valeur avec point décimal → float."""
        csv = b"score,expected_output\n3.14,yes\n"
        rows = GoldenTestService.parse_csv(csv)
        assert rows[0]["input_features"]["score"] == pytest.approx(3.14)
        assert isinstance(rows[0]["input_features"]["score"], float)

    def test_string_value_kept_as_str(self):
        """Valeur non numérique → conservée telle quelle (str)."""
        csv = b"category,expected_output\nabc,yes\n"
        rows = GoldenTestService.parse_csv(csv)
        assert rows[0]["input_features"]["category"] == "abc"
        assert isinstance(rows[0]["input_features"]["category"], str)

    def test_description_absent_defaults_to_none(self):
        """Colonne 'description' absente → description=None."""
        csv = b"f1,expected_output\n1.0,cat\n"
        rows = GoldenTestService.parse_csv(csv)
        assert rows[0]["description"] is None

    def test_description_value_stripped(self):
        """Description avec espaces superflus → trimmée."""
        csv = b"f1,expected_output,description\n1.0,cat, cas normal \n"
        rows = GoldenTestService.parse_csv(csv)
        assert rows[0]["description"] == "cas normal"

    def test_description_empty_string_returns_none(self):
        """Description vide → None (pas chaîne vide)."""
        csv = b"f1,expected_output,description\n1.0,cat,\n"
        rows = GoldenTestService.parse_csv(csv)
        assert rows[0]["description"] is None

    def test_blank_line_skipped(self):
        """Ligne entièrement vide ignorée."""
        csv = b"f1,expected_output\n1.0,a\n\n2.0,b\n"
        rows = GoldenTestService.parse_csv(csv)
        assert len(rows) == 2
        assert [r["expected_output"] for r in rows] == ["a", "b"]

    def test_multiple_rows(self):
        """Plusieurs lignes valides → liste complète."""
        csv = b"f1,expected_output\n1.0,a\n2.0,b\n3.0,c\n"
        rows = GoldenTestService.parse_csv(csv)
        assert len(rows) == 3
        assert [r["expected_output"] for r in rows] == ["a", "b", "c"]

    def test_missing_expected_output_column_raises(self):
        """CSV sans colonne 'expected_output' → ValueError mentionnant expected_output."""
        csv = b"f1,f2\n1.0,2.0\n"
        with pytest.raises(ValueError, match="expected_output"):
            GoldenTestService.parse_csv(csv)

    def test_empty_expected_output_value_raises(self):
        """Valeur expected_output vide dans une ligne → ValueError."""
        csv = b"f1,expected_output\n1.0,\n"
        with pytest.raises(ValueError, match="vide"):
            GoldenTestService.parse_csv(csv)

    def test_empty_csv_no_data_rows_raises(self):
        """CSV avec uniquement le header → ValueError."""
        csv = b"f1,expected_output\n"
        with pytest.raises(ValueError, match="aucune ligne"):
            GoldenTestService.parse_csv(csv)


# ---------------------------------------------------------------------------
# run_for_policy — tests unitaires
# ---------------------------------------------------------------------------


class TestRunForPolicy:
    """Tests pour GoldenTestService.run_for_policy()."""

    def test_returns_none_when_total_tests_is_zero(self):
        """run_for_policy retourne None si aucun golden test n'est enregistré."""
        empty_result = GoldenTestRunResponse(
            model_name=_MODEL,
            version=_VERSION,
            total_tests=0,
            passed=0,
            failed=0,
            pass_rate=1.0,
            details=[],
        )
        with patch(
            "src.services.golden_test_service.GoldenTestService.run_tests",
            new=AsyncMock(return_value=empty_result),
        ):
            result = asyncio.run(
                GoldenTestService.run_for_policy(None, _MODEL, _VERSION)
            )
        assert result is None

    def test_returns_result_when_tests_exist(self):
        """run_for_policy retourne le résultat quand il y a des tests."""
        run_result = GoldenTestRunResponse(
            model_name=_MODEL,
            version=_VERSION,
            total_tests=5,
            passed=5,
            failed=0,
            pass_rate=1.0,
            details=[],
        )
        with patch(
            "src.services.golden_test_service.GoldenTestService.run_tests",
            new=AsyncMock(return_value=run_result),
        ):
            result = asyncio.run(
                GoldenTestService.run_for_policy(None, _MODEL, _VERSION)
            )
        assert result is run_result

    def test_propagates_http_exception_unchanged(self):
        """HTTPException re-levée sans modification (statut conservé)."""
        from fastapi import HTTPException

        exc = HTTPException(status_code=404, detail="not found")
        with patch(
            "src.services.golden_test_service.GoldenTestService.run_tests",
            new=AsyncMock(side_effect=exc),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    GoldenTestService.run_for_policy(None, _MODEL, _VERSION)
                )
        assert exc_info.value.status_code == 404

    def test_wraps_unexpected_exception_as_500(self):
        """Exception inattendue → HTTPException 500."""
        from fastapi import HTTPException

        with patch(
            "src.services.golden_test_service.GoldenTestService.run_tests",
            new=AsyncMock(side_effect=RuntimeError("crash inattendu")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    GoldenTestService.run_for_policy(None, _MODEL, _VERSION)
                )
        assert exc_info.value.status_code == 500
        assert "crash inattendu" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# run_tests — cas limites
# ---------------------------------------------------------------------------


class TestRunTestsEdgeCases:
    """Branches inhabituelles de run_tests()."""

    def test_model_without_feature_names_uses_values_order(self):
        """Modèle sans feature_names_in_ → fallback list(features.values()), pas de crash."""
        model = _make_np_model()
        mock_data = {
            "model": model,
            "metadata": SimpleNamespace(
                name=_MODEL,
                version=_VERSION,
                confidence_threshold=None,
                webhook_url=None,
                feature_baseline=None,
            ),
        }

        async def _run():
            async with _TestSessionLocal() as db:
                gt = await GoldenTestService.create_test(
                    db,
                    model_name=_MODEL,
                    input_features={"a": 5.1, "b": 3.5},
                    expected_output="some_class",
                    description=None,
                    user_id=None,
                )
                # Le lazy import dans run_tests() fait `from src.services.model_service import model_service`
                # → patcher l'attribut sur l'objet source suffit
                with patch(
                    "src.services.model_service.model_service.load_model",
                    new=AsyncMock(return_value=mock_data),
                ):
                    result = await GoldenTestService.run_tests(db, _MODEL, _VERSION)
                await GoldenTestService.delete_test(db, gt.id, _MODEL)
                return result

        result = asyncio.run(_run())
        assert result.total_tests >= 1
        # Vérifier qu'aucun détail n'indique une exception Python (pas de crash)
        for d in result.details:
            assert not d.actual.startswith("ERROR:"), (
                f"La prédiction a levé une exception inattendue: {d.actual}"
            )

    def test_prediction_exception_marks_test_as_failed_with_error_prefix(self):
        """Exception lors de la prédiction → actual='ERROR: ...', passed=False."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["f1", "f2"]
        mock_model.predict.side_effect = RuntimeError("modèle cassé")

        mock_data = {
            "model": mock_model,
            "metadata": SimpleNamespace(
                name=_MODEL,
                version=_VERSION,
                confidence_threshold=None,
                webhook_url=None,
                feature_baseline=None,
            ),
        }

        async def _run():
            async with _TestSessionLocal() as db:
                gt = await GoldenTestService.create_test(
                    db,
                    model_name=_MODEL,
                    input_features={"f1": 1.0, "f2": 2.0},
                    expected_output="target_class",
                    description=None,
                    user_id=None,
                )
                with patch(
                    "src.services.model_service.model_service.load_model",
                    new=AsyncMock(return_value=mock_data),
                ):
                    result = await GoldenTestService.run_tests(db, _MODEL, _VERSION)
                await GoldenTestService.delete_test(db, gt.id, _MODEL)
                return result

        result = asyncio.run(_run())
        assert result.total_tests >= 1
        failing = [d for d in result.details if not d.passed]
        assert len(failing) >= 1, "Au moins un test doit être marqué échoué"
        assert failing[0].actual.startswith("ERROR:"), (
            f"actual devrait commencer par 'ERROR:', obtenu: {failing[0].actual}"
        )
