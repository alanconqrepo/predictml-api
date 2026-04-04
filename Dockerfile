# Utiliser une image Python officielle
FROM python:3.13-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application
COPY src/ ./src/

# Créer le dossier Models
RUN mkdir -p /app/Models

# Exposer le port 8000
EXPOSE 8000

# Définir les variables d'environnement par défaut
ENV API_TOKEN=your-secret-token-here
ENV MODELS_DIR=/app/Models
ENV PYTHONPATH=/app

# Commande pour démarrer l'application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
