# init_data

Scripts d'initialisation à exécuter **une seule fois** lors du premier déploiement.

## Ordre d'exécution

### Étape 1 — Créer les modèles sklearn localement

```bash
python init_data/create_multiple_models.py
```

**Ce que ça fait :** entraîne 3 modèles scikit-learn (iris, wine, cancer) et les sauvegarde en `.pkl` dans `Models/`.

**Quand :** avant `init_db.py`, si le dossier `Models/` est vide ou inexistant.

---

### Étape 2 — Initialiser la base de données et uploader les modèles

```bash
# Dans le conteneur Docker (recommandé)
docker exec predictml-api python init_data/init_db.py

# Ou en local si les services sont accessibles
python init_data/init_db.py
```

**Ce que ça fait :**
1. Crée les tables PostgreSQL (`users`, `predictions`, `model_metadata`)
2. Crée l'utilisateur admin avec un token généré aléatoirement
3. Upload tous les `.pkl` de `Models/` vers MinIO et les enregistre en base

**Quand :** une seule fois après le premier `docker-compose up -d --build`.

> **Important :** le token admin est affiché une seule fois dans le terminal. Sauvegardez-le immédiatement.

---

## Prérequis

Les services Docker doivent être démarrés avant l'étape 2 :

```bash
docker-compose up -d
```

## Ré-initialisation

Si vous avez besoin de repartir de zéro :

```bash
# Supprimer les volumes et recréer
docker-compose down -v
docker-compose up -d --build
docker exec predictml-api python init_data/init_db.py
```

## Résumé

| Script | Prérequis | Fréquence |
|---|---|---|
| `create_multiple_models.py` | Python + scikit-learn | Une fois (ou si `Models/` est vide) |
| `init_db.py` | Docker démarré | Une fois par déploiement |
