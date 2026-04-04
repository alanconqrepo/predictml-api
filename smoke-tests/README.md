# Smoke Tests

Tests manuels contre un environnement live. Nécessitent que les services Docker soient démarrés.

## Prérequis

```bash
docker-compose up -d
```

## Lancer les smoke tests

```bash
# Avec le token admin par défaut
python smoke-tests/test_multimodel_api.py

# Avec un token ou une URL personnalisés
API_TOKEN=mon-token API_BASE_URL=http://localhost:8000 python smoke-tests/test_multimodel_api.py
```

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | URL de base de l'API |
| `API_TOKEN` | token admin | Bearer token d'authentification |

## Différence avec `tests/`

| | `smoke-tests/` | `tests/` |
|---|---|---|
| Serveur requis | Oui (Docker) | Non (TestClient mocké) |
| Lancé par | Manuel | `pytest` |
| But | Vérifier l'env live | Valider la logique métier |
