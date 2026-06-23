"""
HTTP client for the predictml API
"""

from typing import Optional, Tuple, Union

import requests
import streamlit as st


@st.cache_data(ttl=30, show_spinner=False)
def get_models(api_url: str, token: str) -> list:
    """List all models — result cached for 30 seconds."""
    return APIClient(api_url, token).list_models()


@st.cache_data(ttl=10, show_spinner=False)
def get_model_detail(api_url: str, token: str, name: str, version: str) -> dict:
    """Detail of a model version — result cached for 10 seconds."""
    return APIClient(api_url, token).get_model(name, version)


@st.cache_data(ttl=10, show_spinner=False)
def get_golden_tests(api_url: str, token: str, model_name: str) -> list:
    """List a model's golden tests — result cached for 10 seconds."""
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
        Verify the token and role.
        Returns (is_valid, is_admin).
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

    def get_user_usage(
        self,
        user_id: int,
        days: int = 30,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        params = {"days": days}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        r = self._get(f"/users/{user_id}/usage", params=params)
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
        auc: Optional[float] = None,
        f1_score: Optional[float] = None,
        tags: Optional[list] = None,
        train_file_bytes: Optional[bytes] = None,
        train_filename: Optional[str] = None,
    ) -> dict:
        """Upload a .joblib model with its metadata (multipart/form-data)."""
        import json

        data: dict = {"name": name, "version": version}
        if description:
            data["description"] = description
        if algorithm:
            data["algorithm"] = algorithm
        if accuracy is not None:
            data["accuracy"] = str(accuracy)
        if auc is not None:
            data["auc"] = str(auc)
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
        """Return the history of a model (all versions or a specific version)."""
        if version:
            path = f"/models/{name}/{version}/history"
        else:
            path = f"/models/{name}/history"
        r = self._get(path, params={"limit": limit, "offset": offset})
        r.raise_for_status()
        return r.json()

    def get_retrain_history(self, name: str, limit: int = 50, offset: int = 0) -> dict:
        """Return the retraining history of a model."""
        r = self._get(
            f"/models/{name}/retrain-history",
            params={"limit": limit, "offset": offset},
        )
        r.raise_for_status()
        return r.json()

    def rollback_model(self, name: str, version: str, history_id: int) -> dict:
        """Restore a model's metadata to a previous state (admin required)."""
        r = self._post(f"/models/{name}/{version}/rollback/{history_id}")
        r.raise_for_status()
        return r.json()

    def validate_input(self, name: str, version: str, features: dict) -> dict:
        """Validate the input schema without running a prediction."""
        r = self._post(f"/models/{name}/{version}/validate-input", json=features)
        r.raise_for_status()
        return r.json()

    def warmup_model(self, name: str, version: str) -> dict:
        """Warm up the Redis cache for a model (admin required)."""
        r = self._post(f"/models/{name}/{version}/warmup")
        r.raise_for_status()
        return r.json()

    def compute_baseline(
        self, name: str, version: str, days: int = 30, dry_run: bool = True
    ) -> dict:
        """Compute the feature baseline from production predictions (admin required)."""
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
        """Enqueue a model retraining — returns immediately with job_id (202)."""
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
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def get_job_status(self, job_id: str) -> dict:
        """Return the current status of an ARQ job (GET /jobs/{job_id})."""
        r = self._get(f"/jobs/{job_id}")
        r.raise_for_status()
        return r.json()

    def stream_job_logs(self, job_id: str):
        """Stream SSE logs from GET /jobs/{job_id}/logs.

        Yields (is_done: bool, data: str) tuples.
        - is_done=False: a plain log line to display.
        - is_done=True:  terminal event; data is the JSON payload
                         (e.g. '{"status": "success", "new_version": "..."}').
        On network error, yields one terminal tuple with an error payload.
        """
        url = f"{self.base_url}/jobs/{job_id}/logs"
        try:
            with requests.get(
                url,
                headers=self._headers(),
                stream=True,
                timeout=800,
            ) as resp:
                resp.raise_for_status()
                is_terminal_event = False
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        is_terminal_event = False
                        continue
                    decoded = (
                        raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    )
                    if decoded.startswith("event:"):
                        is_terminal_event = True
                        continue
                    if decoded.startswith("data: "):
                        payload = decoded[6:]
                        yield is_terminal_event, payload
                        if is_terminal_event:
                            return
        except Exception as exc:
            yield True, f'{{"error": "{exc}", "status": "failed"}}'

    def set_schedule(
        self,
        name: str,
        version: str,
        cron: Optional[str],
        lookback_days: int = 30,
        auto_promote: bool = False,
        enabled: bool = True,
    ) -> dict:
        """Configure the retraining cron schedule for a model (admin required)."""
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
        # Auto-promotion

        auto_promote: bool = False,
        min_accuracy: Optional[float] = None,
        min_auc: Optional[float] = None,
        max_mae: Optional[float] = None,
        max_latency_p95_ms: Optional[float] = None,
        min_sample_validation: int = 10,
        min_golden_test_pass_rate: Optional[float] = None,
        # Circuit breaker (auto-demotion)
        auto_demote: bool = False,

        demote_on_drift: str = "critical",
        demote_on_accuracy_below: Optional[float] = None,
        demote_cooldown_hours: int = 24,
    ) -> dict:
        """Set the auto-promotion / circuit breaker policy for a model (admin required)."""
        r = self._patch(
            f"/models/{name}/policy",
            json={
                "auto_promote": auto_promote,
                "min_accuracy": min_accuracy,
                "min_auc": min_auc,
                "max_mae": max_mae,
                "max_latency_p95_ms": max_latency_p95_ms,
                "min_sample_validation": min_sample_validation,
                "min_golden_test_pass_rate": min_golden_test_pass_rate,
                "auto_demote": auto_demote,
                "demote_on_drift": demote_on_drift,
                "demote_on_accuracy_below": demote_on_accuracy_below,
                "demote_cooldown_hours": demote_cooldown_hours,
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
        user: Optional[str] = None,
        id_obs: Optional[str] = None,
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
        if user:
            params["user"] = user
        if id_obs:
            params["id_obs"] = id_obs
        r = self._get("/predictions", params=params)
        r.raise_for_status()
        return r.json()

    def explain_prediction(self, prediction_id: int) -> dict:
        """Return the post-hoc SHAP explanation for a prediction (GET /predictions/{id}/explain)."""
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
        """Configure the deployment_mode and traffic_weight for a model version."""
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
        """Retrieve global feature importance via aggregated SHAP."""
        r = self._get(
            f"/models/{name}/feature-importance",
            params={"version": version, "last_n": last_n, "days": days},
        )
        r.raise_for_status()
        return r.json()

    def get_ab_comparison(self, model_name: str, days: int = 30, metric: str | None = None) -> dict:
        """Return A/B / shadow comparison statistics for a model."""
        params: dict = {"days": days}
        if metric:
            params["metric"] = metric
        r = self._get(f"/models/{model_name}/ab-compare", params=params)
        r.raise_for_status()
        return r.json()

    def get_shadow_comparison(self, model_name: str, period_days: int = 30) -> dict:
        """Return enriched shadow vs production metrics for a model."""
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
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """Multi-version comparison of a model in a single call."""
        params: dict = {"days": days}
        if versions:
            params["versions"] = versions
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        r = self._get(f"/models/{model_name}/compare", params=params)
        r.raise_for_status()
        return r.json()

    def get_confidence_trend(
        self,
        model_name: str,
        version: Optional[str] = None,
        days: int = 30,
        granularity: str = "day",
    ) -> dict:
        """Model confidence trend (max probability) over a rolling window."""
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
        """Probability calibration of a model (Brier score, reliability diagram)."""
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
        """Ground truth coverage rate (labeled / total predictions)."""
        r = self._get(
            "/observed-results/stats",
            params={"model_name": model_name},
        )
        r.raise_for_status()
        return r.json()

    def get_unlabeled_predictions(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        strategy: str = "uncertainty",
        limit: int = 50,
    ) -> dict:
        """Predictions without an observed result, ordered by labeling value."""
        r = self._get(
            "/predictions/unlabeled",
            params={
                "model_name": model_name,
                "model_version": model_version,
                "strategy": strategy,
                "limit": limit,
            },
        )
        r.raise_for_status()
        return r.json()

    # --- Monitoring / Supervision Dashboard ---

    def get_monitoring_overview(self, start: str, end: str) -> dict:
        """Health overview of all models over a date range."""
        r = self._get("/monitoring/overview", params={"start": start, "end": end})
        r.raise_for_status()
        return r.json()

    def get_monitoring_model(self, name: str, start: str, end: str) -> dict:
        """Full supervision detail for a model."""
        r = self._get(f"/monitoring/model/{name}", params={"start": start, "end": end})
        r.raise_for_status()
        return r.json()

    def get_alert_checks(
        self,
        model_name: Optional[str] = None,
        check_type: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        """History of alerting checks from alert_check_logs."""
        r = self._get(
            "/monitoring/alert-checks",
            params={
                "model_name": model_name,
                "check_type": check_type,
                "start": start,
                "end": end,
                "limit": limit,
                "offset": offset,
            },
        )
        r.raise_for_status()
        return r.json()

    def get_predictions_anomalies(
        self,
        model_name: str,
        days: int = 7,
        z_threshold: float = 3.0,
        limit: int = 200,
    ) -> dict:
        """Predictions with anomalous features (z-score ≥ threshold)."""
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
        """Submit a single observed result via POST /observed-results."""
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
        Send a list of feature dicts to POST /predict-batch and return predictions.

        rows: list of dicts {"feature1": val, "feature2": val, ...}
        Timeout of 120 s to accommodate large batches.
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

    # --- Token management (self-service) ---

    def regenerate_my_token(self) -> dict:
        r = self._post("/users/me/regenerate-token")
        r.raise_for_status()
        return r.json()

    # --- Account creation requests ---

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
