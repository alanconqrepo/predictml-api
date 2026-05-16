"""
Client HTTP pour l'API predictml
"""

from typing import Optional, Tuple, Union

import requests
import streamlit as st


@st.cache_data(ttl=30, show_spinner=False)
def get_models(api_url: str, token: str) -> list:
    """Liste tous les modèles — résultat mis en cache 30 secondes."""
    return APIClient(api_url, token).list_models()


@st.cache_data(ttl=10, show_spinner=False)
def get_model_detail(api_url: str, token: str, name: str, version: str) -> dict:
    """Détail d'une version de modèle — résultat mis en cache 10 secondes."""
    return APIClient(api_url, token).get_model(name, version)


@st.cache_data(ttl=10, show_spinner=False)
def get_golden_tests(api_url: str, token: str, model_name: str) -> list:
    """Liste les golden tests d'un modèle — résultat mis en cache 10 secondes."""
    return APIClient(api_url, token).list_golden_tests(model_name)


class APIClient:
    def __init__(self, base_url: str, token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _get(self, path: str, params: Optional[dict] = None) -> requests.Response:
        return requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            params={k: v for k, v in (params or {}).items() if v is not None},
            timeout=10,
        )

    def _post(
        self, path: str, json: Optional[dict] = None, params: Optional[dict] = None
    ) -> requests.Response:
        return requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=json,
            params={k: v for k, v in (params or {}).items() if v is not None},
            timeout=10,
        )

    def _patch(self, path: str, json: Optional[dict] = None) -> requests.Response:
        return requests.patch(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=json,
            timeout=10,
        )

    def _delete(self, path: str) -> requests.Response:
        return requests.delete(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=10,
        )

    # --- Auth ---

    def check_auth(self) -> Tuple[bool, bool]:
        """
        Vérifie le token et le rôle.
        Retourne (is_valid, is_admin).
        """
        try:
            r = self._get("/users")
            if r.status_code == 200:
                return True, True
            if r.status_code == 403:
                return True, False
            return False, False
        except requests.RequestException:
            return False, False

    # --- Health ---

    def get_health(self) -> dict:
        r = self._get("/health")
        r.raise_for_status()
        return r.json()

    # --- Users ---

    def list_users(self) -> list:
        r = self._get("/users")
        r.raise_for_status()
        return r.json()

    def create_user(self, data: dict) -> dict:
        r = self._post("/users", json=data)
        r.raise_for_status()
        return r.json()

    def update_user(self, user_id: int, data: dict) -> dict:
        r = self._patch(f"/users/{user_id}", json=data)
        r.raise_for_status()
        return r.json()

    def delete_user(self, user_id: int) -> bool:
        r = self._delete(f"/users/{user_id}")
        return r.status_code == 204

    def get_me(self) -> dict:
        r = self._get("/users/me")
        r.raise_for_status()
        return r.json()

    def get_my_quota(self) -> dict:
        r = self._get("/users/me/quota")
        r.raise_for_status()
        return r.json()

    def get_user_usage(self, user_id: int, days: int = 30) -> dict:
        r = self._get(f"/users/{user_id}/usage", params={"days": days})
        r.raise_for_status()
        return r.json()

    # --- Models ---

    def upload_model(
        self,
        name: str,
        version: str,
        file_bytes: bytes,
        filename: str,
        description: Optional[str] = None,
        algorithm: Optional[str] = None,
        accuracy: Optional[float] = None,
        f1_score: Optional[float] = None,
        tags: Optional[list] = None,
        train_file_bytes: Optional[bytes] = None,
        train_filename: Optional[str] = None,
    ) -> dict:
        """Upload un modèle .joblib avec ses métadonnées (multipart/form-data)."""
        import json

        data: dict = {"name": name, "version": version}
        if description:
            data["description"] = description
        if algorithm:
            data["algorithm"] = algorithm
        if accuracy is not None:
            data["accuracy"] = str(accuracy)
        if f1_score is not None:
            data["f1_score"] = str(f1_score)
        if tags:
            data["tags"] = json.dumps(tags)

        files: dict = {"file": (filename, file_bytes, "application/octet-stream")}
        if train_file_bytes is not None and train_filename:
            files["train_file"] = (train_filename, train_file_bytes, "text/x-python")

        r = requests.post(
            f"{self.base_url}/models",
            headers=self._headers(),
            files=files,
            data=data,
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    def list_models(self) -> list:
        r = self._get("/models")
        r.raise_for_status()
        return r.json()

    def get_model(self, name: str, version: str) -> dict:
        r = self._get(f"/models/{name}/{version}")
        r.raise_for_status()
        return r.json()

    def download_model(self, name: str, version: str) -> bytes:
        r = requests.get(
            f"{self.base_url}/models/{name}/{version}/download",
            headers=self._headers(),
            timeout=60,
        )
        r.raise_for_status()
        return r.content

    def download_train_script(self, name: str, version: str) -> bytes:
        r = requests.get(
            f"{self.base_url}/models/{name}/{version}/download-script",
            headers=self._headers(),
            timeout=30,
        )
        r.raise_for_status()
        return r.content

    def download_training_dataset(self, name: str, version: str) -> bytes:
        r = requests.get(
            f"{self.base_url}/models/{name}/{version}/download-dataset",
            headers=self._headers(),
            timeout=60,
        )
        r.raise_for_status()
        return r.content

    def update_model(self, name: str, version: str, data: dict) -> dict:
        r = self._patch(f"/models/{name}/{version}", json=data)
        r.raise_for_status()
        return r.json()

    def delete_model_version(self, name: str, version: str) -> bool:
        r = self._delete(f"/models/{name}/{version}")
        return r.status_code == 204

    def get_model_history(
        self,
        name: str,
        version: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Retourne l'historique d'un modèle (toutes versions ou version spécifique)."""
        if version:
            path = f"/models/{name}/{version}/history"
        else:
            path = f"/models/{name}/history"
        r = self._get(path, params={"limit": limit, "offset": offset})
        r.raise_for_status()
        return r.json()

    def get_retrain_history(self, name: str, limit: int = 50, offset: int = 0) -> dict:
        """Retourne l'historique des ré-entraînements d'un modèle."""
        r = self._get(
            f"/models/{name}/retrain-history",
            params={"limit": limit, "offset": offset},
        )
        r.raise_for_status()
        return r.json()

    def rollback_model(self, name: str, version: str, history_id: int) -> dict:
        """Restaure les métadonnées d'un modèle à un état antérieur (admin requis)."""
        r = self._post(f"/models/{name}/{version}/rollback/{history_id}")
        r.raise_for_status()
        return r.json()

    def validate_input(self, name: str, version: str, features: dict) -> dict:
        """Valide le schéma d'entrée sans lancer de prédiction."""
        r = self._post(f"/models/{name}/{version}/validate-input", json=features)
        r.raise_for_status()
        return r.json()

    def warmup_model(self, name: str, version: str) -> dict:
        """Préchauffe le cache Redis pour un modèle (admin requis)."""
        r = self._post(f"/models/{name}/{version}/warmup")
        r.raise_for_status()
        return r.json()

    def compute_baseline(
        self, name: str, version: str, days: int = 30, dry_run: bool = True
    ) -> dict:
        """Calcule le baseline de features depuis les prédictions de production (admin requis)."""
        r = self._post(
            f"/models/{name}/{version}/compute-baseline",
            params={"days": days, "dry_run": dry_run},
        )
        r.raise_for_status()
        return r.json()

    def retrain_model(
        self,
        name: str,
        version: str,
        start_date: str,
        end_date: str,
        new_version: Optional[str] = None,
        set_production: bool = False,
    ) -> dict:
        """
        Lance le ré-entraînement d'un modèle via son script train.py stocké dans MinIO.

        Timeout de 660s pour accommoder les entraînements longs (jusqu'à 10 min).
        Requiert un token admin.
        """
        payload: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "set_production": set_production,
        }
        if new_version:
            payload["new_version"] = new_version
        r = requests.post(
            f"{self.base_url}/models/{name}/{version}/retrain",
            headers=self._headers(),
            json=payload,
            timeout=660,
        )
        r.raise_for_status()
        return r.json()

    def set_schedule(
        self,
        name: str,
        version: str,
        cron: Optional[str],
        lookback_days: int = 30,
        auto_promote: bool = False,
        enabled: bool = True,
    ) -> dict:
        """Configure le planning cron de ré-entraînement d'un modèle (admin requis)."""
        r = self._patch(
            f"/models/{name}/{version}/schedule",
            json={
                "cron": cron,
                "lookback_days": lookback_days,
                "auto_promote": auto_promote,
                "enabled": enabled,
            },
        )
        r.raise_for_status()
        return r.json()

    def set_policy(
        self,
        name: str,
        min_accuracy: Optional[float] = None,
        max_mae: Optional[float] = None,
        max_latency_p95_ms: Optional[float] = None,
        min_sample_validation: int = 10,
        auto_promote: bool = False,
    ) -> dict:
        """Définit la politique d'auto-promotion post-retrain d'un modèle (admin requis)."""
        r = self._patch(
            f"/models/{name}/policy",
            json={
                "min_accuracy": min_accuracy,
                "max_mae": max_mae,
                "max_latency_p95_ms": max_latency_p95_ms,
                "min_sample_validation": min_sample_validation,
                "auto_promote": auto_promote,
            },
        )
        r.raise_for_status()
        return r.json()

    # --- Predictions ---

    def predict(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        features: Optional[dict] = None,
        explain: bool = False,
        store: bool = False,
    ) -> dict:
        payload: dict = {"model_name": model_name, "features": features or {}}
        if model_version:
            payload["model_version"] = model_version
        params: dict = {"store": "true" if store else "false"}
        if explain:
            params["explain"] = "true"
        r = self._post("/predict", json=payload, params=params)
        r.raise_for_status()
        return r.json()

    def get_predictions(
        self,
        model_name: str,
        start: str,
        end: str,
        version: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
    ) -> dict:
        params: dict = {
            "name": model_name,
            "start": start,
            "end": end,
            "version": version,
            "limit": limit,
            "offset": offset,
        }
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if max_confidence is not None:
            params["max_confidence"] = max_confidence
        r = self._get("/predictions", params=params)
        r.raise_for_status()
        return r.json()

    def explain_prediction(self, prediction_id: int) -> dict:
        """Retourne l'explication SHAP post-hoc pour une prédiction (GET /predictions/{id}/explain)."""
        r = self._get(f"/predictions/{prediction_id}/explain")
        r.raise_for_status()
        return r.json()

    def get_prediction_stats(
        self,
        days: int = 30,
        model_name: Optional[str] = None,
    ) -> list:
        r = self._get(
            "/predictions/stats",
            params={"days": days, "model_name": model_name},
        )
        r.raise_for_status()
        return r.json().get("stats", [])

    def get_leaderboard(self, metric: str = "accuracy", days: int = 30) -> list:
        r = self._get("/models/leaderboard", params={"metric": metric, "days": days})
        r.raise_for_status()
        return r.json()

    # --- Performance ---

    def get_model_performance(
        self,
        model_name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        version: Optional[str] = None,
        granularity: Optional[str] = "day",
    ) -> dict:
        r = self._get(
            f"/models/{model_name}/performance",
            params={"start": start, "end": end, "version": version, "granularity": granularity},
        )
        r.raise_for_status()
        return r.json()

    # --- A/B Testing & Shadow Deployment ---

    def update_model_deployment(
        self,
        name: str,
        version: str,
        deployment_mode: str,
        traffic_weight: Optional[float] = None,
    ) -> dict:
        """Configure le deployment_mode et le traffic_weight d'une version de modèle."""
        payload: dict = {"deployment_mode": deployment_mode}
        if traffic_weight is not None:
            payload["traffic_weight"] = traffic_weight
        return self.update_model(name, version, payload)

    def get_feature_importance(
        self,
        name: str,
        version: Optional[str] = None,
        last_n: int = 100,
        days: int = 7,
    ) -> dict:
        """Récupère l'importance globale des features via SHAP agrégé."""
        r = self._get(
            f"/models/{name}/feature-importance",
            params={"version": version, "last_n": last_n, "days": days},
        )
        r.raise_for_status()
        return r.json()

    def get_ab_comparison(self, model_name: str, days: int = 30) -> dict:
        """Retourne les statistiques de comparaison A/B / shadow pour un modèle."""
        r = self._get(f"/models/{model_name}/ab-compare", params={"days": days})
        r.raise_for_status()
        return r.json()

    def get_shadow_comparison(self, model_name: str, period_days: int = 30) -> dict:
        """Retourne les métriques enrichies shadow vs production pour un modèle."""
        r = self._get(
            f"/models/{model_name}/shadow-compare", params={"period_days": period_days}
        )
        r.raise_for_status()
        return r.json()

    def compare_model_versions(
        self,
        model_name: str,
        versions: Optional[str] = None,
        days: int = 7,
    ) -> dict:
        """Comparaison multi-versions d'un modèle en un seul appel."""
        r = self._get(
            f"/models/{model_name}/compare",
            params={"versions": versions, "days": days},
        )
        r.raise_for_status()
        return r.json()

    def get_confidence_trend(
        self,
        model_name: str,
        version: Optional[str] = None,
        days: int = 30,
        granularity: str = "day",
    ) -> dict:
        """Tendance de confiance du modèle (max proba) sur une fenêtre glissante."""
        params = {
            k: v
            for k, v in {"version": version, "days": days, "granularity": granularity}.items()
            if v is not None
        }
        r = self._get(f"/models/{model_name}/confidence-trend", params=params)
        r.raise_for_status()
        return r.json()

    def get_model_calibration(
        self,
        model_name: str,
        version: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        n_bins: int = 10,
    ) -> dict:
        """Calibration des probabilités d'un modèle (Brier score, reliability diagram)."""
        params = {
            k: v
            for k, v in {"version": version, "start": start, "end": end, "n_bins": n_bins}.items()
            if v is not None
        }
        r = self._get(f"/models/{model_name}/calibration", params=params)
        r.raise_for_status()
        return r.json()

    # --- Observed Results ---

    def get_observed_results_stats(self, model_name: Optional[str] = None) -> dict:
        """Taux de couverture du ground truth (labeled / total prédictions)."""
        r = self._get(
            "/observed-results/stats",
            params={"model_name": model_name},
        )
        r.raise_for_status()
        return r.json()

    # --- Monitoring / Supervision Dashboard ---

    def get_monitoring_overview(self, start: str, end: str) -> dict:
        """Vue d'ensemble de la santé de tous les modèles sur une plage calendaire."""
        r = self._get("/monitoring/overview", params={"start": start, "end": end})
        r.raise_for_status()
        return r.json()

    def get_monitoring_model(self, name: str, start: str, end: str) -> dict:
        """Détail complet de supervision pour un modèle."""
        r = self._get(f"/monitoring/model/{name}", params={"start": start, "end": end})
        r.raise_for_status()
        return r.json()

    def get_predictions_anomalies(
        self,
        model_name: str,
        days: int = 7,
        z_threshold: float = 3.0,
        limit: int = 200,
    ) -> dict:
        """Prédictions avec features aberrantes (z-score ≥ seuil)."""
        r = self._get(
            "/predictions/anomalies",
            params={
                "model_name": model_name,
                "days": days,
                "z_threshold": z_threshold,
                "limit": limit,
            },
        )
        r.raise_for_status()
        return r.json()

    # --- Observed Results ---

    def get_observed_results(
        self,
        model_name: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        id_obs: Optional[str] = None,
    ) -> dict:
        r = self._get(
            "/observed-results",
            params={
                "model_name": model_name,
                "start": start,
                "end": end,
                "limit": limit,
                "offset": offset,
                "id_obs": id_obs,
            },
        )
        r.raise_for_status()
        return r.json()

    def submit_observed_result(
        self,
        id_obs: str,
        model_name: str,
        observed_result,
    ) -> dict:
        """Soumet un résultat observé unique via POST /observed-results."""
        from datetime import datetime

        payload = {
            "data": [
                {
                    "id_obs": id_obs,
                    "model_name": model_name,
                    "date_time": datetime.utcnow().isoformat(),
                    "observed_result": observed_result,
                }
            ]
        }
        r = self._post("/observed-results", json=payload)
        r.raise_for_status()
        return r.json()

    def export_observed_results(
        self,
        start: str,
        end: str,
        model_name: Optional[str] = None,
        export_format: str = "csv",
    ) -> bytes:
        r = requests.get(
            f"{self.base_url}/observed-results/export",
            headers=self._headers(),
            params={
                k: v
                for k, v in {
                    "model_name": model_name,
                    "start": start,
                    "end": end,
                    "format": export_format,
                }.items()
                if v is not None
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.content

    def export_predictions(
        self,
        start: str,
        end: str,
        model_name: Optional[str] = None,
        export_format: str = "csv",
        status: Optional[str] = None,
    ) -> bytes:
        params: dict = {"start": start, "end": end, "format": export_format}
        if model_name:
            params["model_name"] = model_name
        if status:
            params["status"] = status
        r = requests.get(
            f"{self.base_url}/predictions/export",
            headers=self._headers(),
            params=params,
            timeout=120,
        )
        r.raise_for_status()
        return r.content

    def delete_prediction(self, prediction_id: int) -> None:
        r = self._delete(f"/predictions/{prediction_id}")
        r.raise_for_status()

    def purge_predictions(
        self,
        older_than_days: int,
        model_name: Optional[str] = None,
        dry_run: bool = True,
    ) -> dict:
        params: dict = {"older_than_days": older_than_days, "dry_run": dry_run}
        if model_name:
            params["model_name"] = model_name
        r = requests.delete(
            f"{self.base_url}/predictions/purge",
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def upload_observed_results_csv(
        self,
        file_bytes: bytes,
        filename: str,
        model_name: Optional[str] = None,
    ) -> dict:
        data = {}
        if model_name:
            data["model_name"] = model_name
        r = requests.post(
            f"{self.base_url}/observed-results/upload-csv",
            headers=self._headers(),
            files={"file": (filename, file_bytes, "text/csv")},
            data=data,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def predict_batch_from_df(
        self,
        model_name: str,
        rows: list,
        model_version: Optional[str] = None,
    ) -> dict:
        """
        Envoie une liste de dicts de features à POST /predict-batch et retourne les prédictions.

        rows: liste de dicts {"feature1": val, "feature2": val, ...}
        Timeout de 120s pour accommoder les gros batches.
        """
        payload: dict = {
            "model_name": model_name,
            "inputs": [{"features": row} for row in rows],
        }
        if model_version:
            payload["model_version"] = model_version
        r = requests.post(
            f"{self.base_url}/predict-batch",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def get_model_readiness(self, name: str, version: str) -> dict:
        r = self._get(f"/models/{name}/readiness", params={"version": version})
        r.raise_for_status()
        return r.json()

    def get_model_card(self, name: str, version: str, format: str = "json") -> Union[dict, str]:
        accept = "text/markdown" if format == "markdown" else "application/json"
        r = requests.get(
            f"{self.base_url}/models/{name}/{version}/card",
            headers={**self._headers(), "Accept": accept},
            timeout=30,
        )
        r.raise_for_status()
        return r.text if format == "markdown" else r.json()

    # --- Golden Test Set ---

    def list_golden_tests(self, name: str) -> list:
        r = self._get(f"/models/{name}/golden-tests")
        r.raise_for_status()
        return r.json()

    def create_golden_test(self, name: str, payload: dict) -> dict:
        r = self._post(f"/models/{name}/golden-tests", json=payload)
        r.raise_for_status()
        return r.json()

    def upload_golden_tests_csv(self, name: str, file_bytes: bytes, filename: str) -> dict:
        r = requests.post(
            f"{self.base_url}/models/{name}/golden-tests/upload-csv",
            headers=self._headers(),
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def delete_golden_test(self, name: str, test_id: int) -> bool:
        r = self._delete(f"/models/{name}/golden-tests/{test_id}")
        return r.status_code == 204

    def run_golden_tests(self, name: str, version: str) -> dict:
        r = self._post(f"/models/{name}/{version}/golden-tests/run")
        r.raise_for_status()
        return r.json()

    # --- Gestion du token (self-service) ---

    def regenerate_my_token(self) -> dict:
        r = self._post("/users/me/regenerate-token")
        r.raise_for_status()
        return r.json()

    # --- Demandes de création de compte ---

    def submit_account_request(self, data: dict) -> dict:
        r = requests.post(
            f"{self.base_url}/account-requests",
            json=data,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def get_account_requests(self, status: Optional[str] = None) -> list:
        params = {"status": status} if status else None
        r = self._get("/account-requests", params=params)
        r.raise_for_status()
        return r.json()

    def get_pending_account_requests_count(self) -> int:
        r = self._get("/account-requests/pending-count")
        r.raise_for_status()
        return r.json().get("pending_count", 0)

    def approve_account_request(self, request_id: int) -> dict:
        r = self._patch(f"/account-requests/{request_id}/approve")
        r.raise_for_status()
        return r.json()

    def reject_account_request(self, request_id: int, reason: Optional[str] = None) -> dict:
        r = self._patch(f"/account-requests/{request_id}/reject", json={"reason": reason})
        r.raise_for_status()
        return r.json()
