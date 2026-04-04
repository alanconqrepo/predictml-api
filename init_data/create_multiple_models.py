"""
Script pour créer plusieurs modèles d'exemple dans le dossier Models/
"""
import pickle
from pathlib import Path
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def create_models():
    """Crée plusieurs modèles d'exemple"""

    # Créer le dossier Models s'il n'existe pas
    models_dir = Path("Models")
    models_dir.mkdir(exist_ok=True)

    print("Creation des modeles d'exemple...\n")

    # 1. Modèle Iris avec Random Forest
    print("1. Creation du modele iris_model (Random Forest)...")
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(models_dir / "iris_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"   Sauvegarde: Models/iris_model.pkl")
    print(f"   Precision: {model.score(X, y):.2%}")
    print(f"   Features: 4 (sepal length, sepal width, petal length, petal width)")
    print(f"   Classes: 3 (setosa, versicolor, virginica)\n")

    # 2. Modèle Wine avec Logistic Regression
    print("2. Creation du modele wine_model (Logistic Regression)...")
    X, y = load_wine(return_X_y=True)
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X, y)

    with open(models_dir / "wine_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"   Sauvegarde: Models/wine_model.pkl")
    print(f"   Precision: {model.score(X, y):.2%}")
    print(f"   Features: 13 (alcohol, malic acid, ash, etc.)")
    print(f"   Classes: 3 (class_0, class_1, class_2)\n")

    # 3. Modèle Cancer avec Decision Tree
    print("3. Creation du modele cancer_model (Decision Tree)...")
    X, y = load_breast_cancer(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X, y)

    with open(models_dir / "cancer_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"   Sauvegarde: Models/cancer_model.pkl")
    print(f"   Precision: {model.score(X, y):.2%}")
    print(f"   Features: 30 (mean radius, mean texture, mean perimeter, etc.)")
    print(f"   Classes: 2 (malignant=0, benign=1)\n")

    print("=" * 60)
    print("Tous les modeles ont ete crees avec succes!")
    print(f"Dossier: {models_dir.absolute()}")
    print("\nLancez ensuite init_db.py pour les uploader vers MinIO.")


if __name__ == "__main__":
    create_models()
