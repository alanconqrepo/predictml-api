# Contribuer à predictml-api

## Workflow

Chaque session de travail = une branche. On ne pousse jamais directement sur `main`.

### 1. Partir de main à jour

```bash
git checkout main
git pull origin main
```

### 2. Créer une branche

```bash
git checkout -b feature/nom-feature
# ou
git checkout -b fix/nom-bug
```

### 3. Travailler, commiter

```bash
git add fichier.py
git commit -m "feat: description courte"
```

Préfixes de commit : `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`

### 4. Pousser et ouvrir une PR

```bash
git push -u origin feature/nom-feature
```

Puis ouvrir une Pull Request vers `main` sur GitHub.

### 5. CI et merge

Le CI lance `pytest tests/ -v` automatiquement.
Le merge est bloqué tant que les tests ne passent pas.
Une fois le CI vert, merger la PR et supprimer la branche.

### 6. Nettoyer en local

```bash
git checkout main
git pull origin main
git branch -d feature/nom-feature
```

## Lancer les tests en local

```bash
pytest tests/ -v
```

Les tests utilisent `TestClient` FastAPI — aucun Docker requis.
