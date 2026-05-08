"""
Tests pour les branches non couvertes de src/services/shap_service.py.

Couvre :
- _resolve_class_index : prediction_result absent de classes_ → 0
- _extract_vals_and_base : format 3D (n_samples, n_features, n_classes)
- _extract_vals_and_base : format 2D avec base multi-classe
- _extract_vals_and_base : format list[array]
- _explain_linear : sans feature_baseline → background de zéros
- compute_shap_explanation : modèle non supporté → HTTPException 422
"""

import numpy as np
import pytest
from unittest.mock import MagicMock


class TestResolveClassIndex:
    """Tests pour _resolve_class_index."""

    def test_prediction_in_classes_returns_correct_index(self):
        """prediction_result présent dans classes_ → index retourné."""
        from src.services.shap_service import _resolve_class_index

        model = MagicMock()
        model.classes_ = np.array(["cat", "dog", "bird"])
        assert _resolve_class_index(model, "dog") == 1

    def test_prediction_not_in_classes_returns_zero(self):
        """prediction_result absent de classes_ → 0."""
        from src.services.shap_service import _resolve_class_index

        model = MagicMock()
        model.classes_ = np.array(["cat", "dog"])
        assert _resolve_class_index(model, "unknown_class") == 0

    def test_model_without_classes_attribute_returns_zero(self):
        """Modèle sans attribut classes_ → 0."""
        from src.services.shap_service import _resolve_class_index

        model = object()  # no classes_ attribute
        assert _resolve_class_index(model, "anything") == 0


class TestExtractValsAndBase:
    """Tests pour _extract_vals_and_base — formats SHAP variés."""

    def test_3d_array_format_extracts_correct_class(self):
        """Format 3D (n_samples, n_features, n_classes) → extraction correcte."""
        from src.services.shap_service import _extract_vals_and_base

        # (1, 3 features, 2 classes)
        shap_vals = np.zeros((1, 3, 2))
        shap_vals[0, :, 1] = [0.1, 0.2, 0.3]  # class_idx=1
        base_vals = np.array([0.4, 0.6])  # per-class base

        vals, base = _extract_vals_and_base(shap_vals, base_vals, class_idx=1)

        np.testing.assert_array_almost_equal(vals, [0.1, 0.2, 0.3])
        assert base == pytest.approx(0.6)

    def test_3d_array_class_zero(self):
        """Format 3D, class_idx=0 → première colonne extraite."""
        from src.services.shap_service import _extract_vals_and_base

        shap_vals = np.zeros((1, 3, 2))
        shap_vals[0, :, 0] = [0.5, 0.6, 0.7]
        base_vals = np.array([0.3, 0.8])

        vals, base = _extract_vals_and_base(shap_vals, base_vals, class_idx=0)

        np.testing.assert_array_almost_equal(vals, [0.5, 0.6, 0.7])
        assert base == pytest.approx(0.3)

    def test_2d_array_multiclass_base_uses_correct_base(self):
        """Format 2D avec base multi-classe → base[class_idx] retourné."""
        from src.services.shap_service import _extract_vals_and_base

        shap_vals = np.array([[0.1, 0.2, 0.3]])  # (1, 3)
        base_vals = np.array([0.5, 0.3])  # two classes

        vals, base = _extract_vals_and_base(shap_vals, base_vals, class_idx=0)

        np.testing.assert_array_almost_equal(vals, [0.1, 0.2, 0.3])
        assert base == pytest.approx(0.5)

    def test_list_format_extracts_correct_class(self):
        """Format list[array] → valeurs de la classe demandée extraites."""
        from src.services.shap_service import _extract_vals_and_base

        shap_vals = [
            np.array([[0.1, 0.2]]),  # class 0
            np.array([[0.3, 0.4]]),  # class 1
        ]
        base_vals = np.array([0.5, 0.6])

        vals, base = _extract_vals_and_base(shap_vals, base_vals, class_idx=1)

        np.testing.assert_array_almost_equal(vals, [0.3, 0.4])
        assert base == pytest.approx(0.6)

    def test_list_format_scalar_base(self):
        """Format list avec base scalaire → base correctement extraite."""
        from src.services.shap_service import _extract_vals_and_base

        shap_vals = [np.array([[0.1, 0.2]])]  # single class
        base_vals = np.array(0.5)  # scalar (0D)

        vals, base = _extract_vals_and_base(shap_vals, base_vals, class_idx=0)

        np.testing.assert_array_almost_equal(vals, [0.1, 0.2])
        assert isinstance(base, float)


class TestExplainLinearNoBAseline:
    """Tests pour _explain_linear sans feature_baseline."""

    def test_explain_linear_without_baseline_uses_zeros_background(self):
        """Sans feature_baseline → background=zéros, pas d'exception."""
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from src.services.shap_service import _explain_linear

        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.1, 0.2, 0.3, 0.4]})
        y = [0, 0, 1, 1]
        model = LogisticRegression(max_iter=500).fit(df, y)

        x = np.array([[1.5, 0.15]])
        result = _explain_linear(model, ["f1", "f2"], x, 0, feature_baseline=None)

        assert "shap_values" in result
        assert result["model_type"] == "linear"
        assert isinstance(result["base_value"], float)
        assert set(result["shap_values"].keys()) == {"f1", "f2"}

    def test_explain_linear_with_baseline_uses_means(self):
        """Avec feature_baseline → background construit à partir des means."""
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from src.services.shap_service import _explain_linear

        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.1, 0.2, 0.3, 0.4]})
        y = [0, 0, 1, 1]
        model = LogisticRegression(max_iter=500).fit(df, y)

        baseline = {"f1": {"mean": 2.5, "std": 1.0}, "f2": {"mean": 0.25, "std": 0.1}}
        x = np.array([[1.5, 0.15]])
        result = _explain_linear(model, ["f1", "f2"], x, 0, feature_baseline=baseline)

        assert "shap_values" in result
        assert result["model_type"] == "linear"


class TestComputeShapExplanationUnsupportedModel:
    """Tests pour compute_shap_explanation avec modèle non supporté."""

    def test_unsupported_model_raises_http_422(self):
        """Modèle hors liste TREE_TYPES/LINEAR_TYPES → HTTPException 422."""
        from fastapi import HTTPException
        from src.services.shap_service import compute_shap_explanation

        class UnsupportedModel:
            pass

        model = UnsupportedModel()
        x = np.array([[1.0, 2.0]])

        with pytest.raises(HTTPException) as exc_info:
            compute_shap_explanation(model, ["f1", "f2"], x, 0, None)

        assert exc_info.value.status_code == 422
        assert "non supporté" in exc_info.value.detail
