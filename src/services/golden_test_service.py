"""
Service Golden Test Set — exécution de cas de test de régression pré-déploiement.

Chaque GoldenTest stocke un (input_features, expected_output) pour un modèle.
Lors d'un run, le service charge le modèle, exécute les prédictions et compare
les sorties aux valeurs attendues.
"""

import csv
import io
from typing import List, Optional

import numpy as np
import structlog
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.golden_test import GoldenTest
from src.schemas.golden_test import GoldenTestRunDetail, GoldenTestRunResponse

logger = structlog.get_logger(__name__)


class GoldenTestService:
    """Service CRUD + exécution pour le Golden Test Set."""

    @staticmethod
    async def create_test(
        db: AsyncSession,
        model_name: str,
        input_features: dict,
        expected_output: str,
        description: Optional[str],
        user_id: Optional[int],
    ) -> GoldenTest:
        gt = GoldenTest(
            model_name=model_name,
            input_features=input_features,
            expected_output=expected_output,
            description=description,
            created_by_user_id=user_id,
        )
        db.add(gt)
        await db.flush()
        await db.refresh(gt)
        return gt

    @staticmethod
    async def get_tests(db: AsyncSession, model_name: str) -> List[GoldenTest]:
        result = await db.execute(
            select(GoldenTest)
            .where(GoldenTest.model_name == model_name)
            .order_by(GoldenTest.created_at)
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete_test(db: AsyncSession, test_id: int, model_name: str) -> bool:
        result = await db.execute(
            select(GoldenTest).where(
                GoldenTest.id == test_id,
                GoldenTest.model_name == model_name,
            )
        )
        gt = result.scalar_one_or_none()
        if gt is None:
            return False
        await db.delete(gt)
        await db.flush()
        return True

    @staticmethod
    async def run_tests(
        db: AsyncSession,
        model_name: str,
        version: str,
    ) -> GoldenTestRunResponse:
        """Charge le modèle et exécute tous les golden tests enregistrés pour ce modèle."""
        from src.services.db_service import DBService
        from src.services.model_service import model_service

        # Vérifier que le modèle existe avant de procéder
        metadata = await DBService.get_model_metadata(db, model_name, version)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modèle '{model_name}' version '{version}' non trouvé.",
            )

        tests = await GoldenTestService.get_tests(db, model_name)

        if not tests:
            return GoldenTestRunResponse(
                model_name=model_name,
                version=version,
                total_tests=0,
                passed=0,
                failed=0,
                pass_rate=1.0,
                details=[],
            )

        model_data = await model_service.load_model(db, model_name, version)
        model = model_data["model"]

        details: List[GoldenTestRunDetail] = []

        for gt in tests:
            features: dict = gt.input_features

            try:
                if hasattr(model, "feature_names_in_"):
                    x = np.array(
                        [[features[name] for name in model.feature_names_in_]],
                        dtype=object,
                    )
                else:
                    x = np.array([list(features.values())], dtype=object)

                raw = model.predict(x)[0]
                actual = str(raw.item() if hasattr(raw, "item") else raw)
                passed = actual == str(gt.expected_output)
            except Exception as exc:
                logger.warning(
                    "Erreur lors du golden test",
                    test_id=gt.id,
                    model=model_name,
                    version=version,
                    error=str(exc),
                )
                actual = f"ERROR: {exc}"
                passed = False

            details.append(
                GoldenTestRunDetail(
                    test_id=gt.id,
                    description=gt.description,
                    input=features,
                    expected=str(gt.expected_output),
                    actual=actual,
                    passed=passed,
                )
            )

        n_passed = sum(1 for d in details if d.passed)
        n_failed = len(details) - n_passed
        pass_rate = n_passed / len(details) if details else 1.0

        logger.info(
            "Golden tests exécutés",
            model=model_name,
            version=version,
            total=len(details),
            passed=n_passed,
            pass_rate=f"{pass_rate:.2%}",
        )

        return GoldenTestRunResponse(
            model_name=model_name,
            version=version,
            total_tests=len(details),
            passed=n_passed,
            failed=n_failed,
            pass_rate=pass_rate,
            details=details,
        )

    @staticmethod
    def parse_csv(content: bytes) -> List[dict]:
        """
        Parse un CSV dont les colonnes sont : features... + expected_output (requis) + description (optionnel).

        Retourne une liste de dicts avec les clés :
        - input_features: dict des features
        - expected_output: str
        - description: str | None
        """
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))

        if reader.fieldnames is None or "expected_output" not in reader.fieldnames:
            raise ValueError(
                "Le CSV doit contenir une colonne 'expected_output'. "
                "Format attendu : feature1,feature2,...,expected_output[,description]"
            )

        reserved = {"expected_output", "description"}
        feature_cols = [f for f in reader.fieldnames if f not in reserved]

        rows = []
        for i, row in enumerate(reader, start=2):
            if not any(row.values()):
                continue
            features = {}
            for col in feature_cols:
                val = row[col]
                try:
                    features[col] = float(val) if "." in val else int(val)
                except (ValueError, TypeError):
                    features[col] = val
            rows.append(
                {
                    "input_features": features,
                    "expected_output": row["expected_output"].strip(),
                    "description": row.get("description", "").strip() or None,
                }
            )
            if not rows[-1]["expected_output"]:
                raise ValueError(f"Ligne {i} : 'expected_output' vide.")

        if not rows:
            raise ValueError("Le CSV ne contient aucune ligne de données.")

        return rows

    @staticmethod
    async def run_for_policy(
        db: AsyncSession,
        model_name: str,
        version: str,
    ) -> Optional[GoldenTestRunResponse]:
        """
        Exécute les golden tests pour l'évaluation de la policy d'auto-promotion.
        Retourne None si aucun test n'est enregistré (ne bloque pas).
        """
        try:
            result = await GoldenTestService.run_tests(db, model_name, version)
            return result if result.total_tests > 0 else None
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Erreur inattendue lors des golden tests pour la policy",
                model=model_name,
                version=version,
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur lors de l'exécution des golden tests : {exc}",
            )
