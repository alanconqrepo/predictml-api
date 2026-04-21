"""
Client HTTP pour l'API predictml
"""

from typing import Optional, Tuple

import requests


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

    # --- Models ---

    def list_models(self) -> list:
        r = self._get("/models")
        r.raise_for_status()
        return r.json()

    def get_model(self, name: str, version: str) -> dict:
        r = self._get(f"/models/{name}/{version}")
        r.raise_for_status()
        return r.json()

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

    def rollback_model(self, name: str, version: str, history_id: int) -> dict:
        """Restaure les métadonnées d'un modèle à un état antérieur (admin requis)."""
        r = self._post(f"/models/{name}/{version}/rollback/{history_id}")
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

    def get_predictions(
        self,
        model_name: str,
        start: str,
        end: str,
        version: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        r = self._get(
            "/predictions",
            params={
                "name": model_name,
                "start": start,
                "end": end,
                "version": version,
                "limit": limit,
                "offset": offset,
            },
        )
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

    # --- Performance ---

    def get_model_performance(
        self,
        model_name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        granularity: Optional[str] = "day",
    ) -> dict:
        r = self._get(
            f"/models/{model_name}/performance",
            params={"start": start, "end": end, "granularity": granularity},
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

    def get_ab_comparison(self, model_name: str, days: int = 30) -> dict:
        """Retourne les statistiques de comparaison A/B / shadow pour un modèle."""
        r = self._get(f"/models/{model_name}/ab-compare", params={"days": days})
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

    # --- Observed Results ---

    def get_observed_results(
        self,
        model_name: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        r = self._get(
            "/observed-results",
            params={
                "model_name": model_name,
                "start": start,
                "end": end,
                "limit": limit,
                "offset": offset,
            },
        )
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
