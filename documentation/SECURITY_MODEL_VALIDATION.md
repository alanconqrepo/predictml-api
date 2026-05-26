# Model Name and Version Validation (Security)

## Context

Model names and versions are interpolated directly into MinIO object paths:

```python
object_name = f"{name}/v{version}.joblib"
```

Without validation, an attacker can send a value like `../admin_model` or
`../../../../etc/cron.d/malicious`, causing a **path traversal** attack
in the MinIO bucket — overwriting arbitrary objects or reading outside the intended scope.

## Validation Rules

### Model Name

| Property | Value |
|-----------|--------|
| Regex | `^[a-zA-Z0-9_-]{1,64}$` |
| Allowed characters | letters (`a-z`, `A-Z`), digits (`0-9`), dash (`-`), underscore (`_`) |
| Length | 1 to 64 characters |
| Valid examples | `iris`, `wine_model`, `my-model-v2` |
| Rejected examples | `../admin`, `model/evil`, `<script>`, `` (empty) |

### Version

| Property | Value |
|-----------|--------|
| Regex | `^\d+\.\d+(\.\d+)?$` |
| Format | `X.Y` or `X.Y.Z` (simplified semver) |
| Valid examples | `1.0`, `2.3.1`, `10.0.0` |
| Rejected examples | `../etc`, `1.0.0.0`, `v1.0`, `latest` |

## Implementation

### Constants and Utility Functions

Defined in `src/api/models.py` (just after router instantiation):

```python
import re

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")

def validate_model_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise HTTPException(422, "Invalid model name (allowed characters: a-z A-Z 0-9 _ -)")
    return name

def validate_version(version: str) -> str:
    if not VERSION_RE.match(version):
        raise HTTPException(422, "Invalid version (expected format: X.Y or X.Y.Z)")
    return version
```

### Path Parameters (path params)

All FastAPI endpoints that accept `{name}` or `{version}` in their URL use
`Annotated` types with `Path(pattern=...)`:

```python
ModelNamePath = Annotated[str, Path(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
ModelVersionPath = Annotated[str, Path(pattern=r"^\d+\.\d+(\.\d+)?$")]
```

FastAPI validates the pattern **before** calling the handler and returns an automatic 422 error
if the value does not match.

### Query Parameters (query params)

Optional `version` parameters passed via query string use
`Query(pattern=...)`:

```python
version: Optional[str] = Query(None, pattern=r"^\d+\.\d+(\.\d+)?$")
```

### Request Body (body params)

The Pydantic schemas `PredictionInput`, `BatchPredictionInput` and `ExplainInput`
(in `src/schemas/prediction.py`) include `field_validator`s:

```python
@field_validator("model_name")
@classmethod
def validate_model_name(cls, v: str) -> str:
    if not _NAME_RE.match(v):
        raise ValueError("Invalid model name (allowed characters: a-z A-Z 0-9 _ -)")
    return v

@field_validator("model_version")
@classmethod
def validate_model_version(cls, v: Optional[str]) -> Optional[str]:
    if v is not None and not _VERSION_RE.match(v):
        raise ValueError("Invalid version (expected format: X.Y or X.Y.Z)")
    return v
```

## Files Involved

| File | Modification |
|---------|-------------|
| `src/api/models.py` | Constants `NAME_RE`/`VERSION_RE`, functions `validate_model_name`/`validate_version`, types `ModelNamePath`/`ModelVersionPath`, applied to 35 endpoints |
| `src/schemas/prediction.py` | `field_validator` on `model_name` and `model_version` in `PredictionInput`, `BatchPredictionInput`, `ExplainInput` |

## Error Response

When validation fails, FastAPI returns:

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

Validation tests are in `tests/test_models_get.py`, `tests/test_models_create.py`,
`tests/test_predict_post.py` and `tests/test_input_validation.py`.
