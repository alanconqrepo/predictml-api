# Dossier Models

Ce dossier contient tous les modèles scikit-learn disponibles pour l'API.

## Structure

Chaque modèle doit être sauvegardé au format `.pkl` (pickle) dans ce dossier.

Le nom du fichier (sans l'extension) sera utilisé comme identifiant du modèle dans l'API.

## Exemples

- `iris_model.pkl` → accessible via le nom `iris_model`
- `sentiment_model.pkl` → accessible via le nom `sentiment_model`
- `regression_model.pkl` → accessible via le nom `regression_model`

## Ajouter un nouveau modèle

1. Entraînez votre modèle scikit-learn:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

# Entraîner le modèle
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarder le modèle
with open("Models/mon_modele.pkl", "wb") as f:
    pickle.dump(model, f)
```

2. Placez le fichier `.pkl` dans ce dossier `Models/`

3. Le modèle sera automatiquement disponible dans l'API

4. Utilisez-le dans vos requêtes avec le nom du fichier (sans extension):
```json
{
  "model_name": "mon_modele",
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

## Vérifier les modèles disponibles

Appelez l'endpoint `/models` pour voir tous les modèles disponibles:
```bash
curl http://localhost:8000/models
```

## Notes

- Les modèles sont chargés dynamiquement lors de la première utilisation
- Une fois chargés, ils restent en cache en mémoire
- Seuls les fichiers `.pkl` sont reconnus
- Le dossier est monté en lecture seule dans le conteneur Docker
