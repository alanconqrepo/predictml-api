"""
Script de smoke test pour l'API multi-modèles.
Nécessite que les services Docker soient démarrés : docker-compose up -d
"""
import os
import requests
import json

# Configuration
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("API_TOKEN", "ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA")

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}


def test_root():
    """Test endpoint racine"""
    print("Test: Endpoint racine")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")


def test_models_list():
    """Test l'endpoint /models"""
    print("Test: Liste des modeles disponibles")
    response = requests.get(f"{BASE_URL}/models")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")


def test_iris_prediction():
    """Test prediction avec iris_model"""
    print("Test: Prediction Iris")
    data = {
        "model_name": "iris_model",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data, headers=headers)
    print(f"   Status: {response.status_code}")
    print(f"   Request: {json.dumps(data, indent=2)}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")


def test_wine_prediction():
    """Test prediction avec wine_model"""
    print("Test: Prediction Wine")
    data = {
        "model_name": "wine_model",
        "features": [13.0, 2.0, 2.4, 20.0, 100.0, 2.0, 2.0, 0.3, 2.0, 5.0, 1.0, 3.0, 800.0]
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=data, headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"   Modele wine_model non disponible: {e}\n")


def test_invalid_model():
    """Test avec un modele inexistant"""
    print("Test: Modele inexistant")
    data = {
        "model_name": "nonexistent_model",
        "features": [1, 2, 3, 4]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data, headers=headers)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke Tests - API Multi-Modeles")
    print(f"URL: {BASE_URL}")
    print("=" * 60 + "\n")

    try:
        test_root()
        test_models_list()
        test_iris_prediction()
        test_wine_prediction()
        test_invalid_model()

        print("=" * 60)
        print("Smoke tests termines!")
        print("=" * 60)
    except Exception as e:
        print(f"\nErreur: {e}")
        print("Assurez-vous que l'API est lancee: docker-compose up -d")
