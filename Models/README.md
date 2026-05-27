# Models Folder

This folder contains all the scikit-learn models available to the API.

## Structure

Each model must be saved in `.joblib` format in this folder.

The file name (without the extension) will be used as the model identifier in the API.

## Examples

- `iris_model.joblib` → accessible via the name `iris_model`
- `sentiment_model.joblib` → accessible via the name `sentiment_model`
- `regression_model.joblib` → accessible via the name `regression_model`

## Adding a new model

1. Train your scikit-learn model:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# Train the model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, "Models/mon_modele.joblib")
```

2. Place the `.joblib` file in this `Models/` folder

3. The model will be automatically available in the API

4. Use it in your requests with the file name (without extension):
```json
{
  "model_name": "mon_modele",
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

## Checking available models

Call the `/models` endpoint to see all available models:
```bash
curl http://localhost:8000/models
```

## Notes

- Models are loaded dynamically on first use
- Once loaded, they remain cached in memory
- Only `.joblib` files are recognised
- The folder is mounted read-only in the Docker container
