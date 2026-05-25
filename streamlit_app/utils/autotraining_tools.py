"""
Specialized tools for the AutoTrain PredictML chatbot.

Three additional tools (in addition to query_database and call_api reused from tools.py):
  - execute_python      : executes a train.py script in an isolated subprocess
  - fetch_training_data : fetches predictions + observed results from the API
  - upload_model        : uploads the last produced model to the PredictML API
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from utils.api_client import APIClient
from utils.tools import (
    TOOL_DEFINITIONS,
    build_tool_summary,
    execute_api_call,
    execute_sql,
    render_tool_input,
    render_tool_result,
    tool_expander_label,
)

# ── Resolution paths ──────────────────────────────────────────────────────────

_PROJECT_ROOT_CANDIDATES = [
    Path(__file__).parent.parent.parent,  # local dev: predictml-api/
    Path("/app").parent,  # Docker: /
]

_EXAMPLE_TRAIN_CANDIDATES = [
    Path(__file__).parent.parent.parent / "init_data" / "example_train.py",
    Path("/app/init_data/example_train.py"),
]


def _load_example_train_script() -> str:
    for path in _EXAMPLE_TRAIN_CANDIDATES:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return "# (example not available)"


# ── Date constants ────────────────────────────────────────────────────────────


def _default_train_start() -> str:
    return (date.today() - timedelta(days=90)).isoformat()


def _default_train_end() -> str:
    return date.today().isoformat()


# ── Tool definitions ────────────────────────────────────────────────────────

_EXECUTE_PYTHON_DEF: dict = {
    "name": "execute_python",
    "description": (
        "Executes a Python script (train.py) in an isolated subprocess with a 120-second timeout. "
        "Automatically injects the TRAIN_START_DATE, TRAIN_END_DATE, "
        "OUTPUT_MODEL_PATH and TRAIN_DATA_PATH environment variables (if a dataset is loaded in session). "
        "Returns the exit code, truncated stdout/stderr, JSON metrics extracted "
        "from the last JSON line of stdout, and whether a .joblib file was produced. "
        "Use this tool BEFORE any upload to validate that the script runs correctly. "
        "If the script fails (exit_code != 0), analyse stderr and fix the script."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python code of the train.py script to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 120).",
                "default": 120,
            },
        },
        "required": ["code"],
    },
}

_FETCH_TRAINING_DATA_DEF: dict = {
    "name": "fetch_training_data",
    "description": (
        "Fetches predictions and observed results from the PredictML API, "
        "merges them by id_obs (predictions without an observed_result will have an empty value), "
        "and saves a CSV conforming to the TRAIN_DATA_PATH format expected by train.py. "
        "Automatically stores the CSV path in session so that execute_python can use it. "
        "Use this tool when the user wants to train on real production data. "
        "Returns n_rows, n_labeled, feature_names and a preview of the first 3 rows."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the source model for predictions.",
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format.",
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of predictions to fetch (default: 1000, max: 5000).",
                "default": 1000,
            },
        },
        "required": ["model_name", "start_date", "end_date"],
    },
}

_UPLOAD_MODEL_DEF: dict = {
    "name": "upload_model",
    "description": (
        "Uploads the last .joblib model produced by execute_python to the PredictML API, "
        "with the source train.py script as train_file (to enable future retraining). "
        "Automatically reads the model file from session (autotrain_last_model_path) "
        "and the script from session (autotrain_last_script). "
        "Only use this tool AFTER validating the script with execute_python "
        "(exit_code=0 AND model_produced=true). "
        "The accuracy and f1_score metrics are extracted from the last executed script."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Model name in PredictML (e.g. 'iris_classifier').",
            },
            "version": {
                "type": "string",
                "description": "Semantic version (e.g. '1.0.0').",
            },
            "description": {
                "type": "string",
                "description": "Optional model description.",
            },
            "algorithm": {
                "type": "string",
                "description": "Algorithm used (e.g. 'RandomForestClassifier').",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags (e.g. ['classification', 'iris']).",
            },
        },
        "required": ["name", "version"],
    },
}

# Full list of 5 tools: 3 new + 2 reused from tools.py
AUTOTRAINING_TOOL_DEFINITIONS: list[dict] = [
    _EXECUTE_PYTHON_DEF,
    _FETCH_TRAINING_DATA_DEF,
    _UPLOAD_MODEL_DEF,
    *TOOL_DEFINITIONS,  # query_database + call_api
]


# ── New tool implementations ───────────────────────────────────────────────────


def _run_execute_python(tool_input: dict) -> dict[str, Any]:
    """Executes a Python script in a subprocess with environment variable injection."""
    code = tool_input["code"]
    timeout = int(tool_input.get("timeout", 120))

    # 1. Write the script to a temporary file
    script_path = None
    out_dir = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        # 2. Output directory for the model
        out_dir = tempfile.mkdtemp(prefix="autotrain_model_")
        model_path = os.path.join(out_dir, "model.joblib")

        # 3. Environment variables
        env = os.environ.copy()
        env["TRAIN_START_DATE"] = st.session_state.get(
            "autotrain_train_start", _default_train_start()
        )
        env["TRAIN_END_DATE"] = st.session_state.get("autotrain_train_end", _default_train_end())
        env["OUTPUT_MODEL_PATH"] = model_path

        dataset_path = st.session_state.get("autotrain_dataset_path")
        if dataset_path and os.path.exists(dataset_path):
            env["TRAIN_DATA_PATH"] = dataset_path

        # 4. Execution
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                timeout=timeout,
                env=env,
                text=True,
            )
            elapsed_ms = int((time.time() - t0) * 1000)
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "error": f"Timeout: script exceeded {timeout} seconds.",
                "execution_time_ms": timeout * 1000,
                "stdout": "",
                "stderr": "",
                "metrics": {},
                "model_produced": False,
                "model_path": None,
            }

        # 5. Extract the last JSON line from stdout (metrics)
        metrics: dict = {}
        stdout_lines = proc.stdout.strip().splitlines()
        for line in reversed(stdout_lines):
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    metrics = json.loads(stripped)
                    break
                except json.JSONDecodeError:
                    pass

        # 6. Check whether the model was produced
        model_produced = os.path.exists(model_path) and os.path.getsize(model_path) > 0

        # 7. Store in session state on success
        if model_produced and proc.returncode == 0:
            st.session_state["autotrain_last_model_path"] = model_path
            st.session_state["autotrain_last_script"] = code

            scripts: list = st.session_state.setdefault("autotrain_scripts", [])
            scripts.append(
                {
                    "id": len(scripts) + 1,
                    "code": code,
                    "metrics": metrics,
                    "model_path": model_path,
                    "execution_time_ms": elapsed_ms,
                }
            )

        # 8. Truncate stdout/stderr to avoid excessively large messages
        stdout_preview = proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout
        stderr_preview = proc.stderr[-4000:] if len(proc.stderr) > 4000 else proc.stderr

        return {
            "exit_code": proc.returncode,
            "stdout": stdout_preview,
            "stderr": stderr_preview,
            "execution_time_ms": elapsed_ms,
            "metrics": metrics,
            "model_produced": model_produced,
            "model_path": model_path if model_produced else None,
        }

    finally:
        # Always delete the temporary script file
        if script_path:
            try:
                os.unlink(script_path)
            except OSError:
                pass


def _run_fetch_training_data(tool_input: dict, api_url: str, token: str) -> dict[str, Any]:
    """Fetches predictions + observed_results and builds a training CSV."""
    model_name = tool_input["model_name"]
    start_date = tool_input["start_date"]
    end_date = tool_input["end_date"]
    limit = min(int(tool_input.get("limit", 1000)), 5000)

    headers = {"Authorization": f"Bearer {token}"}
    base = api_url.rstrip("/")

    # Fetch predictions
    try:
        r_pred = requests.get(
            f"{base}/predictions",
            params={
                "name": model_name,
                "start": start_date,
                "end": end_date,
                "limit": limit,
            },
            headers=headers,
            timeout=30,
        )
        r_pred.raise_for_status()
        pred_data = r_pred.json()
        if isinstance(pred_data, dict):
            predictions = pred_data.get(
                "predictions", pred_data.get("data", pred_data.get("items", []))
            )
        else:
            predictions = pred_data
    except Exception as e:
        return {"error": f"Error fetching predictions: {e}"}

    # Fetch observed results
    try:
        r_obs = requests.get(
            f"{base}/observed-results",
            params={
                "model_name": model_name,
                "start": start_date,
                "end": end_date,
                "limit": limit,
            },
            headers=headers,
            timeout=30,
        )
        r_obs.raise_for_status()
        obs_data = r_obs.json()
        if isinstance(obs_data, dict):
            observed = obs_data.get("results", obs_data.get("data", obs_data.get("items", [])))
        else:
            observed = obs_data
    except Exception as e:
        return {"error": f"Error fetching observed results: {e}"}

    if not predictions:
        return {"error": "No predictions found for this date range."}

    # Build DataFrames and merge
    df_pred = pd.DataFrame(predictions)
    df_obs = (
        pd.DataFrame(observed) if observed else pd.DataFrame(columns=["id_obs", "observed_result"])
    )

    # S'assurer que les colonnes nécessaires existent
    if "id_obs" not in df_pred.columns:
        df_pred["id_obs"] = None
    if "observed_result" not in df_obs.columns:
        df_obs["observed_result"] = None

    # Fusionner par id_obs (left join depuis les prédictions)
    if not df_obs.empty and "id_obs" in df_obs.columns:
        obs_map = df_obs.set_index("id_obs")["observed_result"].to_dict()
        df_pred["observed_result"] = df_pred["id_obs"].map(obs_map)
    else:
        df_pred["observed_result"] = None

    n_rows = len(df_pred)
    labeled_mask = df_pred["observed_result"].notna() & (
        df_pred["observed_result"].astype(str).str.strip() != ""
    )
    n_labeled = int(labeled_mask.sum())

    # Extraire les noms de features à partir de la première ligne
    feature_names: list[str] = []
    try:
        sample_feat = df_pred["input_features"].dropna().iloc[0]
        if isinstance(sample_feat, str):
            sample_feat = json.loads(sample_feat)
        if isinstance(sample_feat, dict):
            feature_names = sorted(sample_feat.keys())
    except Exception:
        pass

    # Colonnes requises par le contrat train.py
    columns_needed = [
        "id_obs",
        "input_features",
        "prediction_result",
        "observed_result",
        "timestamp",
        "model_version",
        "response_time_ms",
    ]
    for col in columns_needed:
        if col not in df_pred.columns:
            df_pred[col] = None

    # Sérialiser les colonnes JSON si nécessaire
    for col in ["input_features", "prediction_result", "observed_result"]:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
            )

    # Écrire le CSV dans un fichier temporaire
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    df_pred[columns_needed].to_csv(tmp, index=False)
    csv_path = tmp.name
    tmp.close()

    # Stocker en session state
    st.session_state["autotrain_dataset_path"] = csv_path
    st.session_state["autotrain_dataset_info"] = {
        "n_rows": n_rows,
        "n_labeled": n_labeled,
        "feature_names": feature_names,
        "preview": df_pred[columns_needed].head(3).to_dict("records"),
    }

    return {
        "n_rows": n_rows,
        "n_labeled": n_labeled,
        "feature_names": feature_names,
        "sample_preview": df_pred.head(3).to_dict("records"),
        "csv_path": csv_path,
        "saved": True,
    }


def _run_upload_model(tool_input: dict, api_url: str, token: str) -> dict[str, Any]:
    """Uploads the last produced model to the PredictML API."""
    model_path = st.session_state.get("autotrain_last_model_path")
    script_code = st.session_state.get("autotrain_last_script")

    if not model_path or not os.path.exists(model_path):
        return {
            "error": (
                "No model file available. "
                "First run a script with execute_python and make sure "
                "exit_code=0 and model_produced=true."
            )
        }

    name = tool_input["name"]
    version = tool_input["version"]
    description = tool_input.get("description")
    algorithm = tool_input.get("algorithm")
    tags = tool_input.get("tags")

    # Récupérer les métriques du dernier script exécuté
    scripts: list = st.session_state.get("autotrain_scripts", [])
    last_metrics: dict = scripts[-1]["metrics"] if scripts else {}
    accuracy = last_metrics.get("accuracy")
    f1_score = last_metrics.get("f1_score")

    with open(model_path, "rb") as f:
        file_bytes = f.read()

    train_file_bytes = None
    train_filename = None
    if script_code:
        train_file_bytes = script_code.encode("utf-8")
        train_filename = "train.py"

    try:
        client = APIClient(api_url, token)
        result = client.upload_model(
            name=name,
            version=version,
            file_bytes=file_bytes,
            filename=f"{name}_{version}.joblib",
            description=description,
            algorithm=algorithm,
            accuracy=accuracy,
            f1_score=f1_score,
            tags=tags,
            train_file_bytes=train_file_bytes,
            train_filename=train_filename,
        )
        return {
            "status_code": 200,
            "ok": True,
            "model_name": name,
            "version": version,
            "data": result,
        }
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = str(e)
        return {"error": f"HTTP {e.response.status_code} : {detail}"}
    except Exception as e:
        return {"error": str(e)}


# ── Dispatcher principal ──────────────────────────────────────────────────────


def execute_autotraining_tool(
    tool_name: str,
    tool_input: dict,
    api_url: str,
    token: str,
) -> dict[str, Any]:
    """Dispatche l'exécution vers le bon outil."""
    if tool_name == "execute_python":
        return _run_execute_python(tool_input)

    if tool_name == "fetch_training_data":
        return _run_fetch_training_data(tool_input, api_url, token)

    if tool_name == "upload_model":
        return _run_upload_model(tool_input, api_url, token)

    # Outils partagés depuis tools.py
    if tool_name == "call_api":
        return execute_api_call(
            method=tool_input["method"],
            endpoint=tool_input["endpoint"],
            api_url=api_url,
            token=token,
            params=tool_input.get("params"),
            body=tool_input.get("body"),
        )

    if tool_name == "query_database":
        return execute_sql(
            query=tool_input["query"],
            limit=tool_input.get("limit", 20),
        )

    return {"error": f"Outil inconnu : {tool_name}"}


# ── Helpers de rendu ─────────────────────────────────────────────────────────


def autotraining_tool_expander_label(tool_name: str, tool_input: dict) -> str:
    """Génère un label court pour le titre de l'expander."""
    if tool_name == "execute_python":
        code = tool_input.get("code", "")
        short = code.replace("\n", " ").strip()[:60]
        return f"🐍 execute_python — `{short}{'…' if len(code) > 60 else ''}`"

    if tool_name == "fetch_training_data":
        model = tool_input.get("model_name", "?")
        start = tool_input.get("start_date", "?")
        end = tool_input.get("end_date", "?")
        return f"📥 fetch_training_data — `{model}` ({start} → {end})"

    if tool_name == "upload_model":
        name = tool_input.get("name", "?")
        version = tool_input.get("version", "?")
        return f"🚀 upload_model — `{name}` v{version}"

    # Déléguer aux rendeurs de tools.py pour les outils partagés
    return tool_expander_label(tool_name, tool_input)


def render_autotraining_tool_input(tool_name: str, tool_input: dict) -> None:
    """Affiche les paramètres d'un appel d'outil."""
    if tool_name == "execute_python":
        code = tool_input.get("code", "")
        st.code(code, language="python")
        timeout = tool_input.get("timeout", 120)
        st.caption(f"Timeout : {timeout}s | {len(code.splitlines())} lignes")

    elif tool_name == "fetch_training_data":
        col1, col2, col3 = st.columns(3)
        col1.metric("Modèle", tool_input.get("model_name", "?"))
        col2.metric("Début", tool_input.get("start_date", "?"))
        col3.metric("Fin", tool_input.get("end_date", "?"))
        st.caption(f"Limite : {tool_input.get('limit', 1000)} lignes")

    elif tool_name == "upload_model":
        col1, col2 = st.columns(2)
        col1.metric("Nom", tool_input.get("name", "?"))
        col2.metric("Version", tool_input.get("version", "?"))
        if tool_input.get("algorithm"):
            st.caption(f"Algorithme : {tool_input['algorithm']}")
        if tool_input.get("description"):
            st.caption(f"Description : {tool_input['description']}")

    else:
        # Déléguer à tools.py pour call_api et query_database
        render_tool_input(tool_name, tool_input)


def render_autotraining_tool_result(tool_name: str, result: dict) -> None:
    """Affiche le résultat d'un appel d'outil de façon lisible."""
    if "error" in result:
        st.error(f"Erreur : {result['error']}")
        return

    if tool_name == "execute_python":
        exit_code = result.get("exit_code", -1)

        # Badge exit code
        if exit_code == 0:
            st.success(f"✅ Exit code : {exit_code} — succès")
        else:
            st.error(f"❌ Exit code : {exit_code} — échec")

        # Métriques
        metrics = result.get("metrics", {})
        if metrics:
            m_cols = st.columns(3)
            m_cols[0].metric(
                "Accuracy",
                f"{metrics['accuracy']:.4f}" if "accuracy" in metrics else "—",
            )
            m_cols[1].metric(
                "F1 Score",
                f"{metrics['f1_score']:.4f}" if "f1_score" in metrics else "—",
            )
            m_cols[2].metric("Lignes d'entraînement", metrics.get("n_rows", "—"))

        # Temps d'exécution
        exec_ms = result.get("execution_time_ms", 0)
        st.caption(f"⏱️ Temps d'exécution : {exec_ms} ms")

        # Modèle produit
        if result.get("model_produced"):
            st.success("✅ Modèle .joblib produit et prêt pour l'upload")
        else:
            st.warning("⚠️ Aucun fichier .joblib produit")

        # Stdout (tronqué)
        if result.get("stdout"):
            with st.expander("📤 Stdout", expanded=False):
                st.code(result["stdout"], language="text")

        # Stderr (logs de progression)
        if result.get("stderr"):
            with st.expander("📋 Logs (stderr)", expanded=exit_code != 0):
                st.code(result["stderr"], language="text")

    elif tool_name == "fetch_training_data":
        col1, col2 = st.columns(2)
        col1.metric("Lignes totales", result.get("n_rows", 0))
        col2.metric("Labellisées", result.get("n_labeled", 0))

        feature_names = result.get("feature_names", [])
        if feature_names:
            st.caption(
                f"Features détectées : `{', '.join(feature_names[:10])}`"
                + (" ..." if len(feature_names) > 10 else "")
            )

        if result.get("saved"):
            st.success(
                "✅ CSV sauvegardé — TRAIN_DATA_PATH sera injecté lors du prochain execute_python"
            )

        preview = result.get("sample_preview", [])
        if preview:
            with st.expander("👁️ Aperçu (3 premières lignes)", expanded=False):
                try:
                    df_preview = pd.DataFrame(preview)
                    # Tronquer les colonnes JSON longues pour l'affichage
                    for col in ["input_features", "prediction_result", "observed_result"]:
                        if col in df_preview.columns:
                            df_preview[col] = df_preview[col].astype(str).str[:50]
                    st.dataframe(df_preview, use_container_width=True, hide_index=True)
                except Exception:
                    st.json(preview)

    elif tool_name == "upload_model":
        ok = result.get("ok", False)
        if ok:
            name = result.get("model_name", "?")
            version = result.get("version", "?")
            st.success(f"✅ Modèle **{name}** v{version} uploadé avec succès !")
            st.info(
                "→ Consultez la page **2 Modèles** pour le visualiser et le mettre en production."
            )
            data = result.get("data", {})
            if data:
                raw = json.dumps(data, ensure_ascii=False, default=str)
                if len(raw) > 2000:
                    st.json(json.loads(raw[:2000]))
                    st.caption("… réponse tronquée")
                else:
                    st.json(data)
        else:
            st.error(f"❌ Upload échoué : {result.get('error', 'Erreur inconnue')}")

    else:
        # Déléguer à tools.py pour call_api et query_database
        render_tool_result(tool_name, result)


def build_autotraining_tool_summary(tool_name: str, tool_input: dict, result: dict) -> dict:
    """Construit un résumé compact pour l'historique de conversation."""
    if tool_name == "execute_python":
        exit_code = result.get("exit_code", "?") if "error" not in result else "erreur"
        metrics = result.get("metrics", {})
        acc_str = f"acc={metrics['accuracy']:.3f}" if "accuracy" in metrics else ""
        f1_str = f"f1={metrics['f1_score']:.3f}" if "f1_score" in metrics else ""
        metrics_str = " | ".join(filter(None, [acc_str, f1_str]))
        label = (
            f"🐍 execute_python (exit={exit_code}"
            + (f" | {metrics_str}" if metrics_str else "")
            + ")"
        )
        code_preview = tool_input.get("code", "")[:100]
        return {
            "type": "execute_python",
            "label": label,
            "query": code_preview,
            "result_preview": json.dumps(
                {k: v for k, v in result.items() if k not in ("stdout", "stderr", "model_path")},
                ensure_ascii=False,
                default=str,
            )[:300],
        }

    if tool_name == "fetch_training_data":
        model = tool_input.get("model_name", "?")
        n_rows = result.get("n_rows", "?") if "error" not in result else "erreur"
        n_labeled = result.get("n_labeled", "?") if "error" not in result else ""
        label = f"📥 fetch_training_data — {model} ({n_rows} lignes, {n_labeled} labellisées)"
        return {
            "type": "fetch_training_data",
            "label": label,
            "query": f"{model} {tool_input.get('start_date')} → {tool_input.get('end_date')}",
            "result_preview": json.dumps(
                {k: v for k, v in result.items() if k not in ("sample_preview",)},
                ensure_ascii=False,
                default=str,
            )[:300],
        }

    if tool_name == "upload_model":
        name = tool_input.get("name", "?")
        version = tool_input.get("version", "?")
        ok = result.get("ok", False) if "error" not in result else False
        status = "✅ succès" if ok else "❌ erreur"
        label = f"🚀 upload_model — {name} v{version} → {status}"
        return {
            "type": "upload_model",
            "label": label,
            "query": f"POST /models ({name} v{version})",
            "result_preview": json.dumps(
                {k: v for k, v in result.items() if k != "data"},
                ensure_ascii=False,
                default=str,
            )[:300],
        }

    # Déléguer à tools.py pour les outils partagés
    return build_tool_summary(tool_name, tool_input, result)


# ── Prompt système ────────────────────────────────────────────────────────────


def build_autotraining_system_prompt() -> str:
    """Construit le prompt système spécialisé AutoTrain ML Coach."""
    example_train_code = _load_example_train_script()

    _allowed_modules = (
        "os, sys, json, pickle, joblib, pandas, numpy, sklearn, mlflow, datetime, "
        "pathlib, math, statistics, collections, typing, warnings, logging, time, "
        "copy, functools, itertools, re, io, abc, enum, dataclasses, csv, dotenv, "
        "boto3, botocore, importlib"
    )

    return f"""Tu es **AutoTrain ML Coach**, un assistant spécialisé dans la génération et l'itération \
de scripts `train.py` compatibles avec la plateforme PredictML.

## Ton rôle

Tu aides l'utilisateur à :
1. Comprendre ses données d'entraînement (CSV local ou données de production via l'API)
2. Écrire un script `train.py` conforme au **contrat PredictML**
3. Exécuter et tester le script localement via l'outil `execute_python`
4. Analyser les métriques et itérer pour les améliorer
5. Uploader la meilleure version du modèle via `upload_model`

## Contrat train.py — Règles OBLIGATOIRES

### Variables d'environnement (lire via `os.environ`)
```python
TRAIN_START_DATE  = os.environ["TRAIN_START_DATE"]   # YYYY-MM-DD
TRAIN_END_DATE    = os.environ["TRAIN_END_DATE"]      # YYYY-MM-DD
OUTPUT_MODEL_PATH = os.environ["OUTPUT_MODEL_PATH"]   # chemin absolu .joblib
TRAIN_DATA_PATH   = os.environ.get("TRAIN_DATA_PATH") # optionnel — CSV de production
```

### Sauvegarde obligatoire
```python
joblib.dump(model, OUTPUT_MODEL_PATH)
```

### Sortie stdout (dernière ligne JSON obligatoire)
```json
{{"accuracy": 0.95, "f1_score": 0.94, "n_rows": 1200, "feature_stats": {{}}, "label_distribution": {{}}}}
```
- `accuracy` et `f1_score` sont **obligatoires** pour la mise à jour en base de données
- `n_rows`, `feature_stats`, `label_distribution` sont optionnels mais recommandés (activent le drift monitoring)

### Format du CSV TRAIN_DATA_PATH
Colonnes : `id_obs, input_features, prediction_result, observed_result, timestamp, model_version, response_time_ms`
- `input_features` : dict JSON des features
- `observed_result` : valeur réelle (vide si non labellisée)
- Pour l'entraînement supervisé : filtrer `if row["observed_result"]`

### Imports autorisés
```
{_allowed_modules}
```

## Script de référence complet

```python
{example_train_code}
```

## Utilisation des outils

### `execute_python`
- Utilise-le **TOUJOURS** pour tester un script avant l'upload
- Injecte automatiquement TRAIN_START/END_DATE, OUTPUT_MODEL_PATH, TRAIN_DATA_PATH
- Si `exit_code != 0` : analyse `stderr`, identifie l'erreur, corrige et réexécute
- Si `model_produced = false` : vérifier l'appel à `joblib.dump()`
- Si métriques insuffisantes : propose des améliorations (algo, hyperparamètres, features)

### `fetch_training_data`
- Utilise-le quand l'utilisateur veut des données de production réelles
- Les données seront automatiquement disponibles via TRAIN_DATA_PATH
- Toujours proposer un fallback dataset synthétique si TRAIN_DATA_PATH est absent

### `upload_model`
- N'utilise ce tool QU'APRÈS `exit_code=0` ET `model_produced=true`
- Demande à l'utilisateur le nom et la version avant d'uploader

### `call_api` et `query_database`
- Pour explorer les modèles existants, les métriques de production, les utilisateurs
- Tables disponibles : users, model_metadata, predictions, observed_results, golden_tests

## Workflow recommandé

1. **Comprendre les données** → appelle `fetch_training_data` ou analyse le CSV chargé
2. **Écrire un premier script** → explique les choix (algo, features, split train/test)
3. **Tester** → `execute_python` → affiche les métriques
4. **Itérer** → si erreur : corriger ; si métriques faibles : améliorer
5. **Proposer des variantes** → au moins 2 scripts (ex: RandomForest vs GradientBoosting)
6. **Uploader le meilleur** → `upload_model` avec les bonnes métadonnées

## Règles générales

- Réponds **toujours en français**
- Inclus **toujours** `feature_stats` et `label_distribution` dans la sortie JSON (drift monitoring)
- Écris des scripts avec un **fallback dataset synthétique** si TRAIN_DATA_PATH est absent
- Avant tout upload, vérifie explicitement que `exit_code=0` ET `model_produced=true`
- Si TRAIN_DATA_PATH est disponible, l'utiliser **en priorité** avec fallback synthétique
- Pour les modèles de régression, utiliser `accuracy=r2_score` et `f1_score=0.0`
"""
