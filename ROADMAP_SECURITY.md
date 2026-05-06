# Roadmap Sécurité — predictml-api

Audit réalisé le 2026-05-06. Périmètre : déploiement on-premises, réseau interne entreprise.

Les préconisations sont classées par sévérité et regroupées en 4 phases d'implémentation.

---

## Phase 1 — Correctifs critiques (code applicatif)

### C1 — Signature HMAC des fichiers modèles avant `pickle.loads()`

**Pourquoi**
`pickle.loads()` peut exécuter du code Python arbitraire lors de la désérialisation. Si un attaquant parvient à écrire un fichier `.pkl` malveillant dans MinIO (accès direct au stockage, credential MinIO compromis) ou à empoisonner le cache Redis, il obtient une exécution de code distante (RCE) avec les droits du processus API. La présence du commentaire `# noqa: S301` dans `model_service.py:145` indique que le linter de sécurité Bandit signale déjà ce problème — il est délibérément ignoré.

**Fichiers concernés**
- `src/services/model_service.py` (ligne ~145)
- `src/services/minio_service.py` (ligne ~133)

**Comment**
1. À l'upload d'un modèle (`POST /models`), calculer un HMAC-SHA256 du contenu `.pkl` avec `SECRET_KEY` :
   ```python
   import hmac, hashlib
   signature = hmac.new(settings.SECRET_KEY.encode(), model_bytes, hashlib.sha256).hexdigest()
   ```
2. Stocker la signature dans les métadonnées du modèle (colonne `ModelMetadata` ou champ JSON `tags`).
3. Au chargement du modèle (depuis MinIO et depuis le cache Redis), vérifier la signature avant tout `pickle.loads()` :
   ```python
   expected = hmac.new(settings.SECRET_KEY.encode(), raw_bytes, hashlib.sha256).hexdigest()
   if not hmac.compare_digest(expected, stored_signature):
       raise SecurityError("Model signature mismatch — refusing to load")
   ```
4. Migration : les modèles existants sans signature doivent être re-signés via un script one-shot ou refusés avec un message explicite.

---

### C2 — Restreindre `POST /models` aux administrateurs

**Pourquoi**
L'endpoint `POST /models` ne requiert qu'un token utilisateur valide (`Depends(verify_token)`), pas un rôle admin. N'importe quel utilisateur authentifié peut donc uploader un fichier `.pkl` arbitraire — y compris un modèle malveillant conçu pour exécuter du code lors de sa désérialisation (voir C1). Il peut aussi uploader un script `train.py` qui sera exécuté côté serveur lors d'un retrain.

**Fichier concerné**
- `src/api/models.py` (ligne ~3060)

**Comment**
Remplacer la dépendance d'authentification sur l'endpoint de création :
```python
# Avant
user: User = Depends(verify_token),

# Après
user: User = Depends(require_admin),
```
Adapter les tests `tests/test_models_create.py` qui utilisent un token non-admin pour la création : ils doivent désormais s'attendre à un 403.

---

### C3 — Restreindre `DELETE /models/{name}/{version}` aux administrateurs

**Pourquoi**
Même problème que C2 : la suppression d'un modèle n'est protégée que par `verify_token`. Un utilisateur normal peut supprimer un modèle en production, causant une interruption de service et une perte de données définitive.

**Fichier concerné**
- `src/api/models.py` (ligne ~3364)

**Comment**
```python
# Avant
user: User = Depends(verify_token),

# Après
user: User = Depends(require_admin),
```
Adapter les tests `tests/test_models_delete.py` en conséquence.

---

### C4 — Isoler l'exécution des scripts de retrain dans un sandbox

**Pourquoi**
Le script `train.py` uploadé par un admin est validé syntaxiquement via `ast.parse()`, mais cette validation ne garantit pas l'innocuité du code : un script valide syntaxiquement peut exfiltrer des données, établir une connexion réseau sortante, modifier des fichiers système, ou exécuter des commandes shell (`os.system(...)` passe la validation). Le script est ensuite exécuté dans un sous-processus avec les mêmes droits que l'API.

**Fichiers concernés**
- `src/api/models.py` (lignes ~1189–1220)
- `src/tasks/retrain_scheduler.py` (lignes ~193–210)

**Comment — deux niveaux de protection complémentaires**

1. **Validation statique renforcée des imports** (rapide, à implémenter en priorité) :
   Parcourir l'AST pour lister tous les `import` et `from ... import` et rejeter si un module hors allowlist est présent :
   ```python
   ALLOWED_MODULES = {"os", "json", "pickle", "joblib", "pandas", "numpy",
                      "sklearn", "datetime", "pathlib", "sys"}
   # Parcourir ast.walk(tree) pour les nœuds Import et ImportFrom
   ```

2. **Contraintes de ressources sur le sous-processus** (Linux, `resource` stdlib) :
   ```python
   import resource
   def set_limits():
       resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))  # 2 Go RAM
       resource.setrlimit(resource.RLIMIT_NOFILE, (50, 50))               # 50 fd max
   subprocess.Popen([...], preexec_fn=set_limits)
   ```

3. **Option avancée — container Docker éphémère** (isolation réseau complète) :
   ```bash
   docker run --rm --network=none --memory=2g --cpus=1 \
     -v /tmp/script:/script:ro -v /tmp/output:/output \
     python:3.13-slim python /script/train.py
   ```
   Nécessite que le service API ait accès au socket Docker — à évaluer selon la politique de sécurité.

---

### H1 — Rendre `SECRET_KEY` obligatoire au démarrage

**Pourquoi**
Si la variable d'environnement `SECRET_KEY` n'est pas définie, `config.py` génère une clé aléatoire via `secrets.token_urlsafe(32)`. Conséquence : à chaque redémarrage de l'API, tous les tokens HMAC des modèles (voir C1) et tout token interne lié à cette clé sont invalidés. De plus, l'absence de `SECRET_KEY` en production passe inaperçue — il n'y a aucune erreur au démarrage.

**Fichier concerné**
- `src/core/config.py` (ligne ~40)

**Comment**
```python
# Avant
SECRET_KEY: str = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)

# Après — lever une exception si absent
SECRET_KEY: str = _require_env("SECRET_KEY", insecure_values=set())
```
Documenter la variable dans `.env.example` avec une commande de génération :
```bash
# Générer une clé : python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=<générer_avec_la_commande_ci-dessus>
```

---

### H3 — Valider le format des noms de modèles et versions

**Pourquoi**
Les noms de modèles et versions sont interpolés directement dans les chemins MinIO :
```python
object_name = f"{name}/v{version}.pkl"
```
Un nom comme `../admin_model` ou `../../../../etc/cron.d/malicious` peut provoquer une traversée de répertoire (path traversal) dans le bucket MinIO, permettant d'écraser des fichiers arbitraires ou de lire des objets en dehors de l'espace prévu.

**Fichier concerné**
- `src/api/models.py` (lignes ~3109, 3142 et toutes les routes `{name}/{version}`)

**Comment**
Ajouter une validation par regex dans les paramètres de route et de body :
```python
import re

NAME_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
VERSION_RE = re.compile(r'^\d+\.\d+(\.\d+)?$')

def validate_model_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise HTTPException(422, "Nom de modèle invalide (caractères autorisés : a-z A-Z 0-9 _ -)")
    return name

def validate_version(version: str) -> str:
    if not VERSION_RE.match(version):
        raise HTTPException(422, "Version invalide (format attendu : X.Y ou X.Y.Z)")
    return version
```
Appeler ces fonctions au début de chaque endpoint recevant `name` et `version`.

---

## Phase 2 — Durcissement de l'infrastructure Docker

### H4 — Activer l'authentification Redis

**Pourquoi**
Redis est exposé sur `0.0.0.0:6379` sans mot de passe. N'importe quelle machine du réseau interne peut lire, écrire ou supprimer des données en cache — y compris les modèles sérialisés en pickle (voir C1). Un attaquant interne ou une machine compromise peut empoisonner le cache pour déclencher un RCE.

**Fichier concerné**
- `docker-compose.yml` (ligne ~60-70)

**Comment**
```yaml
redis:
  image: redis:7-alpine
  command: redis-server --requirepass ${REDIS_PASSWORD}
  environment:
    - REDIS_PASSWORD=${REDIS_PASSWORD}
```
Mettre à jour l'URL de connexion dans `src/core/config.py` :
```python
REDIS_URL: str = os.getenv("REDIS_URL", "redis://:${REDIS_PASSWORD}@redis:6379/0")
```
Ajouter `REDIS_PASSWORD` dans `.env.example` avec une valeur placeholder.

---

### H5 — Remplacer les credentials MinIO par défaut

**Pourquoi**
`minioadmin:minioadmin` sont les identifiants par défaut de MinIO, documentés publiquement. Sur un réseau interne, ces identifiants donnent un accès complet au stockage des modèles (lecture, écriture, suppression) à toute machine pouvant atteindre le port 9000 ou la console 9001.

**Fichier concerné**
- `docker-compose.yml` (lignes ~28-29)
- `.env.example`

**Comment**
```yaml
minio:
  environment:
    MINIO_ROOT_USER: ${MINIO_ROOT_USER}
    MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
```
Mettre à jour `src/core/config.py` pour utiliser ces variables. Documenter dans `.env.example` :
```bash
MINIO_ROOT_USER=<choisir_un_utilisateur>
MINIO_ROOT_PASSWORD=<mot_de_passe_fort_32_caracteres_minimum>
```
La validation de `config.py` (`_require_env` avec l'ensemble `{"minioadmin"}`) existe déjà — elle émet un avertissement en mode DEBUG ; la rendre bloquante en production.

---

### M5/M6/M7 — Restreindre les ports exposés sur l'hôte

**Pourquoi**
PostgreSQL (5433), MLflow (5000), Grafana (3000) et MinIO console (9001) sont exposés sur `0.0.0.0`, c'est-à-dire accessibles depuis toutes les interfaces réseau de la machine hôte. Sur un réseau interne, cela signifie que n'importe quel poste du réseau peut accéder directement à la base de données ou à l'interface d'administration MLflow sans passer par l'API.

**Fichier concerné**
- `docker-compose.yml`

**Comment**
Pour les services qui ne doivent être accessibles qu'en local ou depuis les autres containers Docker, supprimer l'exposition host ou la restreindre à `127.0.0.1` :
```yaml
# Avant — exposé sur tout le réseau
ports:
  - "5433:5432"

# Après — accessible uniquement depuis localhost
ports:
  - "127.0.0.1:5433:5432"
```
Appliquer ce principe à PostgreSQL, MLflow, Grafana et MinIO console (9001). L'API (8000) et Streamlit (8501) restent accessibles selon les besoins.

Compléter avec un réseau Docker isolé :
```yaml
networks:
  internal:        # DB, Redis, MLflow — pas d'accès direct depuis l'hôte
  frontend:        # API, Streamlit — exposés
```

---

### F1 — Exécuter Streamlit en utilisateur non-root

**Pourquoi**
Le container Streamlit tourne actuellement en `root`. Si une vulnérabilité dans Streamlit ou une dépendance est exploitée, l'attaquant obtient un shell root dans le container, facilitant les escalades de privilèges et les sorties de container.

**Fichier concerné**
- `streamlit_app/Dockerfile`

**Comment**
```dockerfile
# Ajouter avant la dernière ligne CMD
RUN addgroup --system streamlit && adduser --system --ingroup streamlit streamlit
RUN chown -R streamlit:streamlit /app
USER streamlit
```
Le Dockerfile de l'API (`Dockerfile`) crée déjà un `appuser` — appliquer le même patron.

---

## Phase 3 — Réduction de l'exposition d'informations

### M1 — Authentifier `/health/dependencies`

**Pourquoi**
L'endpoint `GET /health/dependencies` retourne sans authentification le statut détaillé de tous les services internes : PostgreSQL (latence, erreur de connexion), Redis, MinIO, MLflow. Ces informations aident un attaquant à cartographier l'infrastructure, à identifier des services dégradés et à choisir ses vecteurs d'attaque.

**Fichier concerné**
- `src/main.py` (ligne ~294)

**Comment**
```python
# Ajouter la dépendance admin sur l'endpoint
@app.get("/health/dependencies")
async def health_dependencies(user: User = Depends(require_admin)):
    ...
```
L'endpoint `/health` de base (statut API simple, sans détail des services) peut rester public pour les sondes de disponibilité.

---

### M2 — Rendre `METRICS_TOKEN` obligatoire en production

**Pourquoi**
Si `METRICS_TOKEN` n'est pas défini dans l'environnement, l'endpoint `/metrics` est accessible publiquement (la comparaison `hmac.compare_digest(token, "")` passe pour un header Authorization vide). Les métriques Prometheus exposent les noms des modèles, le nombre de requêtes par endpoint, les latences et les codes d'erreur — informations utiles pour cibler une attaque.

**Fichiers concernés**
- `src/core/config.py` (ligne ~74)
- `src/main.py` (ligne ~181)

**Comment**
Au démarrage, logguer un avertissement critique si `METRICS_TOKEN` est vide :
```python
if not settings.METRICS_TOKEN:
    logger.warning("METRICS_TOKEN non défini — endpoint /metrics accessible publiquement")
```
En production (`DEBUG=false`), le rendre bloquant ou désactiver l'endpoint :
```python
if not settings.DEBUG and not settings.METRICS_TOKEN:
    raise RuntimeError("METRICS_TOKEN doit être défini en production")
```

---

### M3 — Retirer la liste des modèles de l'endpoint `/`

**Pourquoi**
L'endpoint racine `GET /` retourne les noms des modèles disponibles sans authentification. Cette information permet à un observateur externe (ou interne non autorisé) de découvrir la surface d'attaque (quels modèles sont déployés, quelles versions sont en production).

**Fichier concerné**
- `src/main.py` (ligne ~197)

**Comment**
```python
# Avant — liste les modèles
return {"status": "ok", "models": [...]}

# Après — statut minimal uniquement
return {"status": "ok", "version": settings.APP_VERSION, "docs": "/docs"}
```

---

### M4 — Paginer `GET /users`

**Pourquoi**
`GET /users` retourne la liste complète des utilisateurs sans limite. Sur une instance avec des milliers d'utilisateurs, cela peut saturer la mémoire du processus API et constitue un vecteur de DoS pour un admin malveillant ou un token admin compromis.

**Fichier concerné**
- `src/api/users.py` (ligne ~107)

**Comment**
```python
@router.get("/users")
async def list_users(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    users = await db_service.get_users(db, skip=skip, limit=limit)
    return users
```
Implémenter `skip`/`limit` dans `DBService.get_users()`.

---

### M9 — Valider le parsing JSON des champs de formulaire

**Pourquoi**
Les champs `classes`, `feature_baseline`, `training_params` et `tags` de l'endpoint `POST /models` sont parsés avec `json.loads()` sans bloc `try/except`. Un JSON malformé lève une exception non gérée qui se traduit par une erreur 500 au lieu d'une erreur 400 explicite, ce qui peut masquer des erreurs et donner des indices sur l'implémentation interne.

**Fichier concerné**
- `src/api/models.py` (lignes ~3152–3155)

**Comment**
```python
def _parse_json_field(value: str | None, field_name: str) -> dict | list | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"JSON invalide dans '{field_name}': {exc.msg}")

classes_parsed = _parse_json_field(classes, "classes")
feature_baseline_parsed = _parse_json_field(feature_baseline, "feature_baseline")
```

---

### M10 — Filtrer les informations internes dans les messages d'erreur

**Pourquoi**
Les erreurs de retrain retournent directement le `stderr` du sous-processus au client. Ce stderr peut contenir des chemins de fichiers internes, des traces de stack, des noms d'hôtes de services internes ou des credentials partiels — autant d'informations utiles pour un attaquant.

**Fichier concerné**
- `src/api/models.py` (lignes ~1171, ~1295)

**Comment**
```python
# Logguer le détail complet côté serveur
logger.error("Retrain failed", model=name, version=version, stderr=stderr_text)

# Retourner un message générique au client
raise HTTPException(status_code=500, detail="Le retrain a échoué. Consulter les logs serveur.")
```
Conserver le `stderr` dans la réponse uniquement pour les admins, via un flag conditionnel :
```python
if settings.DEBUG:
    response["stderr"] = stderr_text  # Debug seulement
```

---

## Phase 4 — Logging d'audit et rate limiting

### F3 — Logging d'audit des opérations sensibles

**Pourquoi**
Aucune trace n'est conservée pour les opérations admin : qui a créé quel modèle, qui a supprimé quel utilisateur, qui a déclenché un retrain. En cas d'incident (modèle malveillant uploadé, utilisateur supprimé), il est impossible de retrouver l'acteur responsable et l'horodatage de l'action.

**Fichiers concernés**
- `src/api/models.py`, `src/api/users.py`
- `src/core/logging.py` (ou nouveau `src/core/audit.py`)

**Comment**
Créer une fonction `audit_log()` qui émet une ligne de log structurée dédiée :
```python
def audit_log(action: str, actor_id: int, resource: str, details: dict = {}):
    logger.info("AUDIT", action=action, actor_id=actor_id, resource=resource, **details)
```
L'appeler aux points clés :
```python
audit_log("model.upload", actor_id=user.id, resource=f"{name}:{version}")
audit_log("model.delete", actor_id=user.id, resource=f"{name}:{version}")
audit_log("user.create", actor_id=user.id, resource=f"user:{new_user.id}")
audit_log("user.token_regen", actor_id=user.id, resource=f"user:{target_id}")
audit_log("retrain.trigger", actor_id=user.id, resource=f"{name}:{version}")
```

---

### H2 — Ajouter une durée de vie aux tokens

**Pourquoi**
Les tokens Bearer sont valides indéfiniment tant que l'utilisateur est actif. Si un token est compromis (interception, fuite de logs, machine volée), il reste exploitable jusqu'à ce qu'un admin le révoque manuellement. Sur un réseau interne, cette fenêtre peut durer des mois.

**Fichiers concernés**
- `src/db/models/user.py` (schéma User)
- `src/core/security.py`
- Nouvelle migration Alembic

**Comment**
Ajouter une colonne `token_expires_at` (nullable) dans la table `users` :
```python
token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
```
Dans `verify_token()`, vérifier l'expiration :
```python
if user.token_expires_at and user.token_expires_at < datetime.utcnow():
    raise HTTPException(status_code=401, detail="Token expiré")
```
À la création/régénération de token, fixer l'expiration (ex. 90 jours) :
```python
user.token_expires_at = datetime.utcnow() + timedelta(days=settings.TOKEN_LIFETIME_DAYS)
```
Documenter `TOKEN_LIFETIME_DAYS` dans `.env.example` (défaut suggéré : 90).

---

### F5 — Rate limiting par minute sur les endpoints d'authentification

**Pourquoi**
Le rate limiting actuel est uniquement journalier et par utilisateur authentifié. Il ne protège pas contre le brute-force de tokens sur les endpoints publics, ni contre un flood de requêtes depuis une IP compromise sur le réseau interne.

**Fichiers concernés**
- `src/main.py` (configuration middleware)
- `requirements.txt` / `pyproject.toml`

**Comment**
Utiliser la bibliothèque `slowapi` (compatible FastAPI) :
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/predict")
@limiter.limit("60/minute")
async def predict(request: Request, ...):
    ...
```
Appliquer une limite stricte sur les endpoints sensibles :
- `/predict` : 60 requêtes/minute par IP
- `POST /models` : 10 requêtes/minute par IP (admin)
- `POST /users` : 10 requêtes/minute par IP (admin)

---

## Récapitulatif par priorité

| Priorité | ID | Action | Effort |
|----------|----|--------|--------|
| **P0 — Immédiat** | C2 | `require_admin` sur `POST /models` | 5 min |
| **P0 — Immédiat** | C3 | `require_admin` sur `DELETE /models` | 5 min |
| **P0 — Immédiat** | H1 | `SECRET_KEY` obligatoire | 15 min |
| **P0 — Immédiat** | H3 | Validation format nom/version modèle | 30 min |
| **P1 — Court terme** | C1 | Signature HMAC des pickles | 2h |
| **P1 — Court terme** | H4 | Redis avec mot de passe | 30 min |
| **P1 — Court terme** | H5 | Credentials MinIO via variables d'env | 30 min |
| **P1 — Court terme** | M5/M6/M7 | Restriction ports Docker | 1h |
| **P1 — Court terme** | M9 | Parsing JSON avec gestion d'erreur | 30 min |
| **P2 — Moyen terme** | C4 | Sandbox scripts retrain | 3h |
| **P2 — Moyen terme** | M1 | Auth sur `/health/dependencies` | 15 min |
| **P2 — Moyen terme** | M2 | `METRICS_TOKEN` obligatoire | 15 min |
| **P2 — Moyen terme** | M3 | Endpoint `/` sans liste modèles | 10 min |
| **P2 — Moyen terme** | M4 | Pagination `GET /users` | 30 min |
| **P2 — Moyen terme** | M10 | Filtrer erreurs exposées au client | 30 min |
| **P2 — Moyen terme** | F1 | Streamlit non-root | 15 min |
| **P3 — Long terme** | F3 | Logging d'audit | 2h |
| **P3 — Long terme** | H2 | Expiration des tokens | 2h |
| **P3 — Long terme** | F5 | Rate limiting par minute | 1h |
