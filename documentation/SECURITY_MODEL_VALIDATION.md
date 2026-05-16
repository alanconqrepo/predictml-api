# Validation des noms de modèles et versions (sécurité)

## Contexte

Les noms de modèles et les versions sont interpolés directement dans les chemins d'objets MinIO :

```python
object_name = f"{name}/v{version}.joblib"
```

Sans validation, un attaquant peut envoyer une valeur comme `../admin_model` ou
`../../../../etc/cron.d/malicious`, provoquant une **traversée de répertoire** (_path traversal_)
dans le bucket MinIO — écrasement d'objets arbitraires ou lecture hors de l'espace prévu.

## Règles de validation

### Nom de modèle

| Propriété | Valeur |
|-----------|--------|
| Regex | `^[a-zA-Z0-9_-]{1,64}$` |
| Caractères autorisés | lettres (`a-z`, `A-Z`), chiffres (`0-9`), tiret (`-`), underscore (`_`) |
| Longueur | 1 à 64 caractères |
| Exemples valides | `iris`, `wine_model`, `my-model-v2` |
| Exemples refusés | `../admin`, `model/evil`, `<script>`, `` (vide) |

### Version

| Propriété | Valeur |
|-----------|--------|
| Regex | `^\d+\.\d+(\.\d+)?$` |
| Format | `X.Y` ou `X.Y.Z` (semver simplifié) |
| Exemples valides | `1.0`, `2.3.1`, `10.0.0` |
| Exemples refusés | `../etc`, `1.0.0.0`, `v1.0`, `latest` |

## Implémentation

### Constantes et fonctions utilitaires

Définies dans `src/api/models.py` (juste après l'instanciation du routeur) :

```python
import re

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")

def validate_model_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise HTTPException(422, "Nom de modèle invalide (caractères autorisés : a-z A-Z 0-9 _ -)")
    return name

def validate_version(version: str) -> str:
    if not VERSION_RE.match(version):
        raise HTTPException(422, "Version invalide (format attendu : X.Y ou X.Y.Z)")
    return version
```

### Paramètres de chemin (path params)

Tous les endpoints FastAPI qui acceptent `{name}` ou `{version}` dans leur URL utilisent des
types `Annotated` avec `Path(pattern=...)` :

```python
ModelNamePath = Annotated[str, Path(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
ModelVersionPath = Annotated[str, Path(pattern=r"^\d+\.\d+(\.\d+)?$")]
```

FastAPI valide le pattern **avant** d'appeler le handler et renvoie une erreur 422 automatique
si la valeur ne correspond pas.

### Paramètres de requête (query params)

Les paramètres `version` optionnels transmis via query string utilisent
`Query(pattern=...)` :

```python
version: Optional[str] = Query(None, pattern=r"^\d+\.\d+(\.\d+)?$")
```

### Corps de requête (body params)

Les schémas Pydantic `PredictionInput`, `BatchPredictionInput` et `ExplainInput`
(dans `src/schemas/prediction.py`) intègrent des `field_validator` :

```python
@field_validator("model_name")
@classmethod
def validate_model_name(cls, v: str) -> str:
    if not _NAME_RE.match(v):
        raise ValueError("Nom de modèle invalide (caractères autorisés : a-z A-Z 0-9 _ -)")
    return v

@field_validator("model_version")
@classmethod
def validate_model_version(cls, v: Optional[str]) -> Optional[str]:
    if v is not None and not _VERSION_RE.match(v):
        raise ValueError("Version invalide (format attendu : X.Y ou X.Y.Z)")
    return v
```

## Fichiers concernés

| Fichier | Modification |
|---------|-------------|
| `src/api/models.py` | Constantes `NAME_RE`/`VERSION_RE`, fonctions `validate_model_name`/`validate_version`, types `ModelNamePath`/`ModelVersionPath`, appliqués sur 35 endpoints |
| `src/schemas/prediction.py` | `field_validator` sur `model_name` et `model_version` dans `PredictionInput`, `BatchPredictionInput`, `ExplainInput` |

## Réponse d'erreur

Lorsque la validation échoue, FastAPI retourne :

```json
HTTP 422 Unprocessable Entity
{
  "detail": [
    {
      "type": "string_pattern_mismatch",
      "loc": ["path", "name"],
      "msg": "String should match pattern '^[a-zA-Z0-9_-]{1,64}$'",
      "input": "../evil"
    }
  ]
}
```

## Tests

Les tests de validation sont dans `tests/test_models_get.py`, `tests/test_models_create.py`,
`tests/test_predict_post.py` et `tests/test_input_validation.py`.
