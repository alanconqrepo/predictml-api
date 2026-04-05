"""
Client HTTP pour l'API predictml
"""
import requests
from typing import Optional, Tuple


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

    def _post(self, path: str, json: Optional[dict] = None) -> requests.Response:
        return requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=json,
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
