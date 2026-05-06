# Utiliser une image Python officielle
FROM python:3.13-slim

# Installer curl et dépendances nécessaires
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application
COPY src/ ./src/
COPY init_data/ ./init_data/
COPY alembic/ ./alembic/
COPY alembic.ini ./alembic.ini
COPY entrypoint.sh ./entrypoint.sh

# Créer l'utilisateur non-root et le dossier Models, rendre l'entrypoint exécutable
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/Models && \
    chmod +x /app/entrypoint.sh && \
    chown -R appuser:appuser /app

# Exposer le port 8000
EXPOSE 8000

# Définir les variables d'environnement par défaut
ENV MODELS_DIR=/app/Models
ENV PYTHONPATH=/app

# Exécuter en tant qu'utilisateur non-root
USER appuser

# Lance init_db.py puis démarre l'API
CMD ["/app/entrypoint.sh"]
