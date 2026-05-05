"""
Chargement de la documentation projet pour le chatbot d'aide.
"""

from pathlib import Path

import streamlit as st

# Dossier documentation/ : chemin local (dev) puis chemin Docker (volume mount)
_DOC_DIRS = [
    Path(__file__).parent.parent.parent / "documentation",
    Path("/app/documentation"),
]

# Fichiers .md à la racine du projet
_ROOT_MD_FILES = [
    ("README", [Path(__file__).parent.parent.parent / "README.md", Path("/app/README.md")]),
    (
        "CODING_STANDARDS",
        [
            Path(__file__).parent.parent.parent / "CODING_STANDARDS.md",
            Path("/app/CODING_STANDARDS.md"),
        ],
    ),
]


@st.cache_data(show_spinner=False)
def load_all_docs() -> dict[str, str]:
    """Charge tous les fichiers .md disponibles (documentation/ + racine)."""
    docs: dict[str, str] = {}

    for doc_dir in _DOC_DIRS:
        if doc_dir.exists():
            for f in sorted(doc_dir.glob("*.md")):
                docs[f.stem] = f.read_text(encoding="utf-8")
            break

    for name, paths in _ROOT_MD_FILES:
        for path in paths:
            if path.exists():
                docs[name] = path.read_text(encoding="utf-8")
                break

    return docs


def build_system_prompt(docs: dict[str, str]) -> str:
    """Construit le prompt système à partir de la documentation chargée."""
    parts = [
        "Tu es l'assistant IA de PredictML, une plateforme MLOps de production.",
        "Stack : FastAPI (port 8000) · Streamlit Admin (port 8501) · PostgreSQL · MinIO · MLflow · Redis.",
        "",
        "Rôle : aider les utilisateurs à :",
        "- Générer des scripts train.py compatibles (sklearn, MLflow, contrat de variables d'env)",
        "- Formuler des appels API REST (Python requests, curl, JavaScript fetch)",
        "- Comprendre les KPIs et métriques du dashboard (accuracy, F1, drift, p-value A/B…)",
        "- Naviguer dans les pages du dashboard Streamlit",
        "- Configurer le ré-entraînement, l'auto-promotion, les tests golden, la purge RGPD",
        "",
        "Règles de réponse :",
        "- Réponds TOUJOURS en français",
        "- Inclus des blocs de code quand c'est pertinent",
        "- Mentionne le nom de la page Streamlit concernée (ex : '📈 Stats', '🤖 Modèles')",
        "- Si une réponse implique un endpoint API, donne l'exemple curl ET Python",
        "- Sois précis, concis et pratique",
        "",
        "## Documentation complète du projet",
        "",
    ]

    for name, content in docs.items():
        parts += [f"### {name}", "", content, ""]

    return "\n".join(parts)
