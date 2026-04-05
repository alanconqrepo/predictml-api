-- Script d'initialisation PostgreSQL
-- Crée la base de données mlflow si elle n'existe pas déjà.
-- Ce fichier est monté dans /docker-entrypoint-initdb.d/ et s'exécute
-- automatiquement au premier démarrage du container postgres.

SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
