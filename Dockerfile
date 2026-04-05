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
COPY init_data/ ./init_data/
COPY entrypoint.sh ./entrypoint.sh

# Créer le dossier Models et rendre l'entrypoint exécutable
RUN mkdir -p /app/Models && chmod +x /app/entrypoint.sh

# Exposer le port 8000
EXPOSE 8000

# Définir les variables d'environnement par défaut
ENV MODELS_DIR=/app/Models
ENV PYTHONPATH=/app

# Lance init_db.py puis démarre l'API
CMD ["/app/entrypoint.sh"]
