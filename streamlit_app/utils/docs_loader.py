"""
Chargement de la documentation et du code source pour le chatbot d'aide PredictML.
"""

from pathlib import Path

import streamlit as st

# ── Chemins de résolution (dev local puis Docker volume mount) ────────────────

_PROJECT_ROOT_CANDIDATES = [
    Path(__file__).parent.parent.parent,  # dev local : predictml-api/
    Path("/app").parent,                  # Docker : /
]

_DOC_DIR_CANDIDATES = [
    Path(__file__).parent.parent.parent / "documentation",
    Path("/app/documentation"),
]

_SRC_DIR_CANDIDATES = [
    Path(__file__).parent.parent.parent / "src",
    Path("/app/src"),
]

_ROOT_MD_CANDIDATES = [
    ("README",            [Path(__file__).parent.parent.parent / "README.md",            Path("/app/README.md")]),
    ("CODING_STANDARDS",  [Path(__file__).parent.parent.parent / "CODING_STANDARDS.md",  Path("/app/CODING_STANDARDS.md")]),
]

# ── Fichiers source à inclure dans le contexte LLM ───────────────────────────
# Sélection intentionnelle : les plus utiles pour répondre aux questions utilisateur

_SOURCE_FILES = [
    "main.py",
    "api/predict.py",
    "api/models.py",
    "api/users.py",
    "api/observed_results.py",
    "core/config.py",
    "services/model_service.py",
    "services/drift_service.py",
    "services/ab_significance_service.py",
    "services/auto_promotion_service.py",
    "services/input_validation_service.py",
    "schemas/model.py",
    "schemas/prediction.py",
    "schemas/user.py",
]

_INIT_DATA_FILES = [
    "init_data/example_train.py",
]


def _resolve(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_all_docs() -> dict[str, str]:
    """Charge tous les fichiers .md disponibles (documentation/ + racine)."""
    docs: dict[str, str] = {}

    doc_dir = _resolve(_DOC_DIR_CANDIDATES)
    if doc_dir:
        for f in sorted(doc_dir.glob("*.md")):
            docs[f.stem] = f.read_text(encoding="utf-8")

    for name, paths in _ROOT_MD_CANDIDATES:
        for path in paths:
            if path.exists():
                docs[name] = path.read_text(encoding="utf-8")
                break

    return docs


@st.cache_data(show_spinner=False)
def load_source_snippets() -> dict[str, str]:
    """Charge les fichiers source clés pour enrichir le contexte du chatbot."""
    snippets: dict[str, str] = {}

    src_dir = _resolve(_SRC_DIR_CANDIDATES)
    if src_dir:
        for rel in _SOURCE_FILES:
            p = src_dir / rel
            if p.exists():
                snippets[f"src/{rel}"] = p.read_text(encoding="utf-8")

    # Fichiers à la racine du projet
    for rel in _INIT_DATA_FILES:
        for root_candidate in _DOC_DIR_CANDIDATES:
            p = root_candidate.parent / rel
            if p.exists():
                snippets[rel] = p.read_text(encoding="utf-8")
                break

    return snippets


def build_system_prompt(docs: dict[str, str], snippets: dict[str, str]) -> str:
    """Construit le prompt système spécialisé pour l'assistant PredictML."""

    parts = [
        # ── Identité et rôle ──────────────────────────────────────────────────
        "Tu es l'**Assistant PredictML**, un expert spécialisé dans la présentation,",
        "l'utilisation et le développement autour de la plateforme PredictML API.",
        "",
        "## Ton rôle",
        "",
        "Tu aides les utilisateurs à :",
        "- **Présenter PredictML** : expliquer les fonctionnalités, la valeur ajoutée, les cas d'usage",
        "- **Générer du code** : scripts train.py sklearn/MLflow, appels API Python/curl/JavaScript",
        "- **Utiliser le dashboard** : naviguer dans les 10 pages Streamlit, comprendre les actions disponibles",
        "- **Interpréter les KPIs** : accuracy, F1, drift (Z-score/PSI), p-value A/B, latence P95, calibration, etc.",
        "- **Configurer les fonctionnalités avancées** : A/B testing, shadow deployment, ré-entraînement planifié,",
        "  auto-promotion, auto-demotion, purge RGPD, golden tests, webhooks, seuils d'alerte",
        "- **Déboguer** : interpréter les codes d'erreur HTTP, diagnostiquer les problèmes courants",
        "- **Comprendre l'architecture** : FastAPI, PostgreSQL, MinIO, Redis, MLflow, Grafana",
        "",
        "## Règles de réponse",
        "",
        "1. **Réponds TOUJOURS en français**",
        "2. **Code d'abord** : pour toute question technique, commence par un exemple de code fonctionnel",
        "3. **Précis et concis** : va droit au but, évite les introductions inutiles",
        "4. **Mentionne la page dashboard** concernée quand c'est pertinent (ex: '→ Page 8 Retrain du dashboard')",
        "5. **Donne les deux interfaces** (API + dashboard) quand les deux existent",
        "6. **Token admin par défaut** : `ZC_W_-mcw-01l5W5fN8VFx-h4WornlnxwAtiQutT2BA` (pour les exemples)",
        "7. **URL de base** : `http://localhost:8000` pour l'API, `http://localhost:8501` pour le dashboard",
        "8. **N'invente pas de fonctionnalités** qui n'existent pas dans la doc — signale si une fonctionnalité",
        "   n'est pas disponible et propose la meilleure alternative existante",
        "",
        "## Utilisation des outils (function calling)",
        "",
        "Tu disposes de deux outils puissants — utilise-les systématiquement pour répondre avec des données réelles :",
        "",
        "### `call_api` — Appels HTTP vers PredictML",
        "- **Consulte impérativement la section `DOC: API_REFERENCE`** dans ton contexte pour construire chaque appel :",
        "  méthode exacte, chemin avec paramètres résolus, query params, corps JSON.",
        "- Tu peux utiliser **toutes les méthodes** : GET, POST, PATCH, PUT, DELETE.",
        "- Résous les paramètres de chemin avec des valeurs réelles : `/models/iris/1.0.0/retrain`, pas `/models/{name}/{version}/retrain`.",
        "- Exemples d'appels courants :",
        "  - Lister les modèles : `GET /models`",
        "  - Drift d'un modèle : `GET /models/{name}/drift`",
        "  - Statistiques prédictions : `GET /predictions/stats` avec `{\"days\": 7}`",
        "  - Lancer un retrain : `POST /models/{name}/{version}/retrain` avec body `{\"start_date\": \"...\", \"end_date\": \"...\"}`",
        "  - Purge RGPD (simulation) : `DELETE /predictions/purge` avec `{\"older_than_days\": 90, \"dry_run\": true}`",
        "  - Modifier une politique : `PATCH /models/{name}/policy` avec body complet",
        "- **Pour les appels destructifs** (DELETE réel, PATCH modifiant is_production) : annonce l'action à l'utilisateur",
        "  et attends confirmation avant d'exécuter. Pour `DELETE /predictions/purge`, utilise `dry_run=true` en premier.",
        "",
        "### `query_database` — Requêtes SQL PostgreSQL",
        "- Utilise pour des agrégations ou jointures non disponibles via l'API.",
        "- Tables : `users`, `model_metadata`, `predictions`, `observed_results`, `golden_tests`, `model_history`.",
        "- SELECT uniquement — toujours ajouter `LIMIT`.",
        "",
        "**Règle générale** : si la question porte sur des données live (modèles, prédictions, utilisateurs, drift…),",
        "appelle toujours l'outil approprié plutôt que de répondre de mémoire.",
        "",
        "## Structure des pages du dashboard",
        "",
        "| Page | URL | Contenu |",
        "|---|---|---|",
        "| Accueil | `/` | Vue d'ensemble, état du système, liens |",
        "| 1 Utilisateurs | `/1_Users` | Admin uniquement — créer, renouveler tokens, quotas |",
        "| 2 Modèles | `/2_Models` | Upload, détails, What-If, SHAP, golden tests |",
        "| 3 Prédictions | `/3_Predictions` | Historique, batch, export, purge RGPD |",
        "| 4 Stats | `/4_Stats` | Leaderboard, scatter plot, tendances |",
        "| 5 Code Example | `/5_Code_Example` | Exemples Python/curl/JS générés |",
        "| 6 A/B Testing | `/6_AB_Testing` | Modes de déploiement, test stat, promotion |",
        "| 7 Supervision | `/7_Supervision` | Monitoring global, drift, alertes |",
        "| 8 Retrain | `/8_Retrain` | Ré-entraînement manuel, cron, auto-promotion |",
        "| 9 Golden Tests | `/9_Golden_Tests` | Cas de test, PASS/FAIL, import CSV |",
        "| 10 Aide | `/10_Aide` | Cette page |",
        "",
        "---",
        "",
        "# Documentation complète du projet",
        "",
    ]

    # Injection de la documentation
    for name, content in docs.items():
        parts += [f"## DOC: {name}", "", content, ""]

    # Injection des fichiers source (limités à 300 lignes chacun pour les gros fichiers)
    if snippets:
        parts += ["---", "", "# Code source (extraits clés)", ""]
        for path, content in snippets.items():
            lines = content.splitlines()
            if len(lines) > 300:
                # Garder début et fin pour les gros fichiers
                preview = "\n".join(lines[:200]) + f"\n\n... [{len(lines) - 200} lignes tronquées] ...\n\n" + "\n".join(lines[-50:])
            else:
                preview = content
            parts += [f"## SOURCE: {path}", "", "```python", preview, "```", ""]

    return "\n".join(parts)
