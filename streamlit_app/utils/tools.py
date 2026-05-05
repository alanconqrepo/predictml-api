"""
Outils natifs (function calling) pour le chatbot d'aide PredictML.

Deux tools disponibles pour Claude :
  - query_database : exécute une requête SQL SELECT en lecture seule sur PostgreSQL
  - call_api       : effectue un appel HTTP vers l'API PredictML avec le token utilisateur
"""

import json
import os
from typing import Any

import psycopg2
import psycopg2.extras
import requests

# ── Définitions des outils (schéma Anthropic tool use) ───────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "query_database",
        "description": (
            "Exécute une requête SQL SELECT en lecture seule sur la base de données PostgreSQL "
            "de PredictML. Utilise ce tool pour répondre à des questions basées sur des données "
            "réelles : modèles déployés, historique des prédictions, statistiques d'utilisation, "
            "performances mesurées, utilisateurs actifs, résultats observés, golden tests, etc. "
            "Tables disponibles : users, model_metadata, predictions, observed_results, "
            "golden_tests, model_history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Requête SQL SELECT à exécuter. Doit commencer par SELECT. "
                        "Une clause LIMIT sera ajoutée automatiquement si absente. "
                        "Exemples : "
                        "'SELECT name, version, accuracy, is_production FROM model_metadata WHERE is_active=true ORDER BY created_at DESC', "
                        "'SELECT model_name, COUNT(*) as nb FROM predictions WHERE timestamp > NOW()-INTERVAL\\'7 days\\' GROUP BY model_name', "
                        "'SELECT username, role, rate_limit_per_day FROM users WHERE is_active=true'."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Nombre maximum de lignes à retourner (défaut : 20, max : 100).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "call_api",
        "description": (
            "Effectue n'importe quelle requête HTTP vers l'API PredictML au nom de l'utilisateur "
            "connecté (token Bearer de sa session). "
            "Tu as accès à TOUS les endpoints documentés — consulte la section DOC: API_REFERENCE "
            "dans ton contexte pour connaître l'endpoint exact, la méthode, les paramètres de "
            "chemin, les query params et le corps JSON attendus. "
            "Endpoints disponibles (non exhaustif) : "
            "GET /models, GET /models/{name}/drift, GET /models/{name}/performance, "
            "GET /models/leaderboard, GET /predictions, GET /predictions/stats, "
            "GET /models/{name}/ab-compare, GET /models/{name}/feature-importance, "
            "GET /users, GET /users/me, GET /health, "
            "POST /predict, POST /models/{name}/{version}/retrain, "
            "POST /models/{name}/{version}/validate-input, "
            "PATCH /models/{name}/{version}, PATCH /models/{name}/policy, "
            "PATCH /models/{name}/{version}/schedule, PATCH /users/{id}, "
            "DELETE /predictions/purge, DELETE /models/{name}/{version}. "
            "Pour les appels destructifs (DELETE, PATCH modifiant la production), "
            "annonce à l'utilisateur l'action que tu vas effectuer avant de l'exécuter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "description": (
                        "Méthode HTTP. Consulte la documentation API_REFERENCE pour la méthode "
                        "correcte de chaque endpoint. GET = lecture, POST = créer/exécuter, "
                        "PATCH = modifier partiellement, PUT = remplacer, DELETE = supprimer."
                    ),
                },
                "endpoint": {
                    "type": "string",
                    "description": (
                        "Chemin complet de l'endpoint avec les paramètres de chemin résolus "
                        "(sans l'URL de base). Exemples : '/models', '/models/iris/drift', "
                        "'/models/iris/1.0.0/retrain', '/predictions/purge', '/users/42'. "
                        "Remplace {name}, {version}, {id} par les valeurs réelles."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Paramètres de requête (query string) sous forme de dict. "
                        "Ex: {\"days\": 7, \"model_name\": \"iris\", \"dry_run\": true}."
                    ),
                    "additionalProperties": True,
                },
                "body": {
                    "type": "object",
                    "description": (
                        "Corps JSON pour POST/PUT/PATCH. "
                        "Ex: {\"features\": {\"sepal_length\": 5.1}, \"model_name\": \"iris\"}. "
                        "Consulte la doc API_REFERENCE pour le schéma exact attendu."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["method", "endpoint"],
        },
    },
]


# ── Connexion PostgreSQL ───────────────────────────────────────────────────────


def _get_db_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "sklearn_api"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        connect_timeout=5,
    )


# ── Implémentation des outils ─────────────────────────────────────────────────


def execute_sql(query: str, limit: int = 20) -> dict[str, Any]:
    """Exécute une requête SELECT et retourne colonnes + lignes."""
    limit = min(max(1, int(limit)), 100)

    clean = query.strip()
    if not clean.upper().startswith("SELECT"):
        return {"error": "Seules les requêtes SELECT sont autorisées pour des raisons de sécurité."}

    # Ajouter LIMIT si absent
    if "LIMIT" not in clean.upper():
        clean = clean.rstrip(";") + f" LIMIT {limit}"

    try:
        conn = _get_db_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(clean)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
        conn.close()

        return {
            "columns": columns,
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
            "query_executed": clean,
        }
    except psycopg2.Error as e:
        return {"error": f"Erreur SQL : {e.pgerror or str(e)}"}
    except Exception as e:
        return {"error": f"Erreur de connexion à PostgreSQL : {str(e)}"}


def execute_api_call(
    method: str,
    endpoint: str,
    api_url: str,
    token: str,
    params: dict | None = None,
    body: dict | None = None,
) -> dict[str, Any]:
    """Effectue un appel HTTP vers l'API PredictML."""
    url = api_url.rstrip("/") + "/" + endpoint.lstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params={k: v for k, v in (params or {}).items() if v is not None},
            json=body if body else None,
            timeout=15,
        )
        try:
            data = resp.json()
        except Exception:
            data = resp.text

        return {"status_code": resp.status_code, "ok": resp.ok, "data": data}
    except requests.Timeout:
        return {"error": "Timeout : l'API ne répond pas dans les 15 secondes."}
    except requests.ConnectionError:
        return {"error": f"Impossible de contacter l'API à {api_url}. Vérifiez que le service est démarré."}
    except Exception as e:
        return {"error": str(e)}


def execute_tool(
    tool_name: str,
    tool_input: dict,
    api_url: str,
    token: str,
) -> dict[str, Any]:
    """Dispatche l'exécution vers le bon outil."""
    if tool_name == "query_database":
        return execute_sql(
            query=tool_input["query"],
            limit=tool_input.get("limit", 20),
        )
    if tool_name == "call_api":
        return execute_api_call(
            method=tool_input["method"],
            endpoint=tool_input["endpoint"],
            api_url=api_url,
            token=token,
            params=tool_input.get("params"),
            body=tool_input.get("body"),
        )
    return {"error": f"Outil inconnu : {tool_name}"}


# ── Helpers de rendu ─────────────────────────────────────────────────────────


def tool_expander_label(tool_name: str, tool_input: dict) -> str:
    """Génère un label court pour le titre de l'expander."""
    if tool_name == "query_database":
        # Extrait les premières tables mentionnées
        q = tool_input.get("query", "")
        short = q.replace("\n", " ").strip()[:80]
        return f"🗄️ SQL — `{short}…`" if len(q) > 80 else f"🗄️ SQL — `{short}`"
    if tool_name == "call_api":
        m = tool_input.get("method", "GET")
        ep = tool_input.get("endpoint", "")
        return f"🌐 API — `{m} {ep}`"
    return f"🔧 {tool_name}"


def render_tool_input(tool_name: str, tool_input: dict) -> None:
    """Affiche les paramètres d'un appel d'outil."""
    import streamlit as st

    if tool_name == "query_database":
        st.code(tool_input.get("query", ""), language="sql")
        if tool_input.get("limit"):
            st.caption(f"Limite : {tool_input['limit']} lignes")
    elif tool_name == "call_api":
        method = tool_input.get("method", "GET")
        endpoint = tool_input.get("endpoint", "")
        st.code(f"{method} {endpoint}", language="http")
        if tool_input.get("params"):
            st.caption("Paramètres :")
            st.json(tool_input["params"])
        if tool_input.get("body"):
            st.caption("Corps :")
            st.json(tool_input["body"])
    else:
        st.json(tool_input)


def render_tool_result(tool_name: str, result: dict) -> None:
    """Affiche le résultat d'un appel d'outil de façon lisible."""
    import pandas as pd
    import streamlit as st

    if "error" in result:
        st.error(f"Erreur : {result['error']}")
        return

    if tool_name == "query_database":
        row_count = result.get("row_count", 0)
        st.caption(f"{row_count} ligne(s) retournée(s)")
        if row_count > 0:
            try:
                df = pd.DataFrame(result["rows"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception:
                st.json(result["rows"])
        else:
            st.info("Aucune ligne retournée.")

    elif tool_name == "call_api":
        status = result.get("status_code", "?")
        ok = result.get("ok", False)
        color = "green" if ok else "red"
        st.markdown(f"**Status :** :{color}[{status}]")
        data = result.get("data")
        if isinstance(data, (dict, list)):
            # Tronquer les grosses réponses
            raw = json.dumps(data, ensure_ascii=False, default=str)
            if len(raw) > 3000:
                st.json(json.loads(raw[:3000]))
                st.caption(f"… réponse tronquée ({len(raw)} caractères au total)")
            else:
                st.json(data)
        else:
            st.text(str(data)[:1000])

    else:
        st.json(result)


def build_tool_summary(tool_name: str, tool_input: dict, result: dict) -> dict:
    """Construit un résumé compact pour l'historique de conversation."""
    if tool_name == "query_database":
        q = tool_input.get("query", "")
        rows = result.get("row_count", "?") if "error" not in result else "erreur"
        return {
            "type": "sql",
            "label": f"🗄️ SQL ({rows} lignes)",
            "query": q,
            "result_preview": (
                json.dumps(result.get("rows", [])[:3], ensure_ascii=False, default=str)
                if "error" not in result
                else result["error"]
            ),
        }
    if tool_name == "call_api":
        ep = tool_input.get("endpoint", "")
        status = result.get("status_code", "?") if "error" not in result else "erreur"
        return {
            "type": "api",
            "label": f"🌐 API {tool_input.get('method','GET')} {ep} → {status}",
            "method": tool_input.get("method", "GET"),
            "endpoint": ep,
            "result_preview": (
                json.dumps(result.get("data", {}), ensure_ascii=False, default=str)[:300]
                if "error" not in result
                else result["error"]
            ),
        }
    return {"type": "other", "label": f"🔧 {tool_name}", "result_preview": str(result)[:200]}
