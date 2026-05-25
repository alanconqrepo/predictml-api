"""
Native tools (function calling) for the PredictML help chatbot.

Two tools available for Claude:
  - query_database : executes a read-only SQL SELECT query on PostgreSQL
  - call_api       : makes an HTTP call to the PredictML API with the user's token
"""

import json
import os
from typing import Any

import psycopg2
import psycopg2.extras
import requests

# ── Tool definitions (Anthropic tool use schema) ──────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "query_database",
        "description": (
            "Executes a read-only SQL SELECT query on the PredictML PostgreSQL database. "
            "Use this tool to answer questions based on real data: deployed models, "
            "prediction history, usage statistics, measured performance, active users, "
            "observed results, golden tests, etc. "
            "Available tables: users, model_metadata, predictions, observed_results, "
            "golden_tests, model_history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "SQL SELECT query to execute. Must start with SELECT. "
                        "A LIMIT clause will be added automatically if absent. "
                        "Examples: "
                        "'SELECT name, version, accuracy, is_production FROM model_metadata WHERE is_active=true ORDER BY created_at DESC', "
                        "'SELECT model_name, COUNT(*) as nb FROM predictions WHERE timestamp > NOW()-INTERVAL\\'7 days\\' GROUP BY model_name', "
                        "'SELECT username, role, rate_limit_per_day FROM users WHERE is_active=true'."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default: 20, max: 100).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "call_api",
        "description": (
            "Makes any HTTP request to the PredictML API on behalf of the logged-in user "
            "(Bearer token from their session). "
            "You have access to ALL documented endpoints — consult the DOC: API_REFERENCE "
            "section in your context for the exact endpoint, method, path parameters, "
            "query params and expected JSON body. "
            "Available endpoints (non-exhaustive): "
            "GET /models, GET /models/{name}/drift, GET /models/{name}/performance, "
            "GET /models/leaderboard, GET /predictions, GET /predictions/stats, "
            "GET /models/{name}/ab-compare, GET /models/{name}/feature-importance, "
            "GET /users, GET /users/me, GET /health, "
            "POST /predict, POST /models/{name}/{version}/retrain, "
            "POST /models/{name}/{version}/validate-input, "
            "PATCH /models/{name}/{version}, PATCH /models/{name}/policy, "
            "PATCH /models/{name}/{version}/schedule, PATCH /users/{id}, "
            "DELETE /predictions/purge, DELETE /models/{name}/{version}. "
            "For destructive calls (DELETE, PATCH modifying production), "
            "announce the action to the user before executing it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "description": (
                        "HTTP method. Consult the API_REFERENCE documentation for the correct "
                        "method for each endpoint. GET = read, POST = create/execute, "
                        "PATCH = partial update, PUT = replace, DELETE = delete."
                    ),
                },
                "endpoint": {
                    "type": "string",
                    "description": (
                        "Full endpoint path with resolved path parameters "
                        "(without the base URL). Examples: '/models', '/models/iris/drift', "
                        "'/models/iris/1.0.0/retrain', '/predictions/purge', '/users/42'. "
                        "Replace {name}, {version}, {id} with real values."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Query string parameters as a dict. "
                        "Ex: {\"days\": 7, \"model_name\": \"iris\", \"dry_run\": true}."
                    ),
                    "additionalProperties": True,
                },
                "body": {
                    "type": "object",
                    "description": (
                        "JSON body for POST/PUT/PATCH. "
                        "Ex: {\"features\": {\"sepal_length\": 5.1}, \"model_name\": \"iris\"}. "
                        "Consult the API_REFERENCE doc for the exact expected schema."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["method", "endpoint"],
        },
    },
]


# ── PostgreSQL connection ─────────────────────────────────────────────────────


def _get_db_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "sklearn_api"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        connect_timeout=5,
    )


# ── Tool implementations ──────────────────────────────────────────────────────


def execute_sql(query: str, limit: int = 20) -> dict[str, Any]:
    """Executes a SELECT query and returns columns + rows."""
    limit = min(max(1, int(limit)), 100)

    clean = query.strip()
    if not clean.upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed for security reasons."}

    # Add LIMIT if absent
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
        return {"error": f"SQL error: {e.pgerror or str(e)}"}
    except Exception as e:
        return {"error": f"PostgreSQL connection error: {str(e)}"}


def execute_api_call(
    method: str,
    endpoint: str,
    api_url: str,
    token: str,
    params: dict | None = None,
    body: dict | None = None,
) -> dict[str, Any]:
    """Makes an HTTP call to the PredictML API."""
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
        return {"error": "Timeout: the API did not respond within 15 seconds."}
    except requests.ConnectionError:
        return {"error": f"Unable to reach the API at {api_url}. Make sure the service is running."}
    except Exception as e:
        return {"error": str(e)}


def execute_tool(
    tool_name: str,
    tool_input: dict,
    api_url: str,
    token: str,
) -> dict[str, Any]:
    """Dispatches execution to the appropriate tool."""
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
    return {"error": f"Unknown tool: {tool_name}"}


# ── Rendering helpers ─────────────────────────────────────────────────────────


def tool_expander_label(tool_name: str, tool_input: dict) -> str:
    """Generates a short label for the expander title."""
    if tool_name == "query_database":
        # Extract the first tables mentioned
        q = tool_input.get("query", "")
        short = q.replace("\n", " ").strip()[:80]
        return f"🗄️ SQL — `{short}…`" if len(q) > 80 else f"🗄️ SQL — `{short}`"
    if tool_name == "call_api":
        m = tool_input.get("method", "GET")
        ep = tool_input.get("endpoint", "")
        return f"🌐 API — `{m} {ep}`"
    return f"🔧 {tool_name}"


def render_tool_input(tool_name: str, tool_input: dict) -> None:
    """Displays the parameters of a tool call."""
    import streamlit as st

    if tool_name == "query_database":
        st.code(tool_input.get("query", ""), language="sql")
        if tool_input.get("limit"):
            st.caption(f"Limit: {tool_input['limit']} rows")
    elif tool_name == "call_api":
        method = tool_input.get("method", "GET")
        endpoint = tool_input.get("endpoint", "")
        st.code(f"{method} {endpoint}", language="http")
        if tool_input.get("params"):
            st.caption("Parameters:")
            st.json(tool_input["params"])
        if tool_input.get("body"):
            st.caption("Body:")
            st.json(tool_input["body"])
    else:
        st.json(tool_input)


def render_tool_result(tool_name: str, result: dict) -> None:
    """Displays the result of a tool call in a readable format."""
    import pandas as pd
    import streamlit as st

    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    if tool_name == "query_database":
        row_count = result.get("row_count", 0)
        st.caption(f"{row_count} row(s) returned")
        if row_count > 0:
            try:
                df = pd.DataFrame(result["rows"])
                st.dataframe(df, width='stretch', hide_index=True)
            except Exception:
                st.json(result["rows"])
        else:
            st.info("No rows returned.")

    elif tool_name == "call_api":
        status = result.get("status_code", "?")
        ok = result.get("ok", False)
        color = "green" if ok else "red"
        st.markdown(f"**Status:** :{color}[{status}]")
        data = result.get("data")
        if isinstance(data, (dict, list)):
            # Truncate large responses
            raw = json.dumps(data, ensure_ascii=False, default=str)
            if len(raw) > 3000:
                st.json(json.loads(raw[:3000]))
                st.caption(f"… response truncated ({len(raw)} characters total)")
            else:
                st.json(data)
        else:
            st.text(str(data)[:1000])

    else:
        st.json(result)


def build_tool_summary(tool_name: str, tool_input: dict, result: dict) -> dict:
    """Builds a compact summary for the conversation history."""
    if tool_name == "query_database":
        q = tool_input.get("query", "")
        rows = result.get("row_count", "?") if "error" not in result else "error"
        return {
            "type": "sql",
            "label": f"🗄️ SQL ({rows} rows)",
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
