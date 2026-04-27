"""
Gestion des utilisateurs — admin only
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from utils.auth import get_client, require_admin, require_auth


def show_quota_progress(used: int, limit: int) -> None:
    if limit <= 0:
        st.info("Quota illimité.")
        return
    pct = used / limit
    pct_clamped = min(pct * 100, 100)
    remaining = max(limit - used, 0)

    if pct < 0.70:
        bar_color = "#21c354"
    elif pct < 0.90:
        bar_color = "#ff9f0a"
    else:
        bar_color = "#ff3b30"

    tooltip = f"{used} / {limit} prédictions utilisées aujourd'hui — reset à minuit UTC"
    st.markdown(
        f"""<div title="{tooltip}" style="margin-bottom:6px">
  <div style="background:#e0e0e0;border-radius:6px;height:18px;width:100%;overflow:hidden">
    <div style="background:{bar_color};width:{pct_clamped:.1f}%;height:100%;border-radius:6px;transition:width .3s"></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )
    caption = f"{used} / {limit} prédictions utilisées aujourd'hui — {remaining} restante{'s' if remaining != 1 else ''} — reset à minuit UTC"
    if pct >= 0.90:
        st.error(caption)
    elif pct >= 0.70:
        st.warning(caption)
    else:
        st.success(caption)


def show_token_with_copy(token: str) -> None:
    """Display a token with a one-click copy button."""
    st.code(token, language=None)
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText('{token}').then(() => {{
            this.innerText = '✅ Copié !';
            setTimeout(() => this.innerText = '📋 Copier le token', 2000);
        }})" style="
            background:#4CAF50; color:white; border:none; padding:6px 14px;
            border-radius:4px; cursor:pointer; font-size:14px;
        ">📋 Copier le token</button>
        """,
        height=42,
    )


st.set_page_config(page_title="Users — PredictML", page_icon="👥", layout="wide")

require_auth()

if not st.session_state.get("is_admin"):
    client = get_client()
    with st.expander("Mon profil", expanded=True):
        try:
            me = client.get_me()
            quota = client.get_my_quota()
            st.markdown(f"**Utilisateur :** {me['username']}  |  **Rôle :** {me['role']}")
            st.markdown(f"**Email :** {me['email']}")

            st.markdown("**Token API :**")
            token_placeholder = st.empty()
            token_placeholder.text("••••••••••••••••••••••••")
            if st.button("Révéler le token"):
                show_token_with_copy(me["api_token"])

            st.markdown("**Quota journalier :**")
            used = quota["used_today"]
            limit = quota["rate_limit_per_day"]
            show_quota_progress(used, limit)
        except Exception as e:
            st.error(f"Impossible de charger le profil : {e}")
    st.stop()

require_admin()

st.title("👥 Gestion des utilisateurs")

client = get_client()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_users(api_url, token):
    c = get_client()
    return c.list_users()


def reload():
    st.cache_data.clear()
    st.rerun()


# --- Créer un utilisateur ---
with st.expander("➕ Créer un nouvel utilisateur", expanded=False):
    with st.form("create_user_form"):
        col1, col2 = st.columns(2)
        username = col1.text_input("Nom d'utilisateur", placeholder="john_doe")
        email = col2.text_input("Email", placeholder="john@example.com")
        col3, col4 = st.columns(2)
        role = col3.selectbox("Rôle", ["user", "admin", "readonly"])
        rate_limit = col4.number_input(
            "Quota journalier", min_value=1, max_value=100000, value=1000
        )
        submitted = st.form_submit_button("Créer", use_container_width=True, type="primary")

    if submitted:
        if not username or not email:
            st.error("Nom d'utilisateur et email sont requis.")
        else:
            try:
                result = client.create_user(
                    {
                        "username": username,
                        "email": email,
                        "role": role,
                        "rate_limit": rate_limit,
                    }
                )
                st.toast(f"Utilisateur {result['username']} créé.", icon="✅")
                st.info("Conservez ce token — il ne sera plus affiché !")
                show_token_with_copy(result["api_token"])
                reload()
            except Exception as e:
                st.toast(f"Erreur : {e}", icon="❌")

st.divider()

# --- Liste des utilisateurs ---
try:
    users = fetch_users(st.session_state.get("api_url"), st.session_state.get("api_token"))
except Exception as e:
    st.error(f"Impossible de charger les utilisateurs : {e}")
    st.stop()

if not users:
    st.info("Aucun utilisateur trouvé.")
    st.stop()

df = pd.DataFrame(users)[
    [
        "id",
        "username",
        "email",
        "role",
        "is_active",
        "rate_limit_per_day",
        "last_login",
        "created_at",
    ]
]
df["last_login"] = pd.to_datetime(df["last_login"]).dt.strftime("%Y-%m-%d %H:%M").fillna("—")
df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d")
df["is_active"] = df["is_active"].map({True: "✅ Actif", False: "❌ Inactif"})

st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Actions")

user_options = {f"{u['username']} (id:{u['id']})": u for u in users}
selected_label = st.selectbox("Sélectionner un utilisateur", list(user_options.keys()))
selected = user_options[selected_label]

col_a, col_b, col_c, col_d = st.columns(4)

# Toggle actif/inactif
current_active = selected["is_active"]
toggle_label = "Désactiver" if current_active else "Activer"
if col_a.button(f"{'🔴' if current_active else '🟢'} {toggle_label}", use_container_width=True):
    try:
        client.update_user(selected["id"], {"is_active": not current_active})
        st.toast(f"Utilisateur {'désactivé' if current_active else 'activé'}.", icon="✅")
        reload()
    except Exception as e:
        st.toast(f"Erreur : {e}", icon="❌")

# Renouveler token
if col_b.button("🔄 Renouveler token", use_container_width=True):
    try:
        result = client.update_user(selected["id"], {"regenerate_token": True})
        st.toast("Nouveau token généré.", icon="✅")
        show_token_with_copy(result["api_token"])
        reload()
    except Exception as e:
        st.toast(f"Erreur : {e}", icon="❌")

# Modifier rôle
new_roles = [r for r in ["user", "admin", "readonly"] if r != selected["role"]]
new_role = col_c.selectbox("Changer rôle", new_roles, key="role_select")
if col_c.button("✏️ Appliquer rôle", use_container_width=True):
    try:
        client.update_user(selected["id"], {"role": new_role})
        st.toast(f"Rôle mis à jour → {new_role}", icon="✅")
        reload()
    except Exception as e:
        st.toast(f"Erreur : {e}", icon="❌")

# Supprimer
with col_d:
    if st.button("🗑️ Supprimer", use_container_width=True, type="secondary"):
        st.session_state["confirm_delete_user"] = selected["id"]

if st.session_state.get("confirm_delete_user") == selected["id"]:
    st.warning(
        f"Confirmer la suppression de **{selected['username']}** et toutes ses prédictions ?"
    )
    c1, c2 = st.columns(2)
    if c1.button("Oui, supprimer", type="primary"):
        try:
            client.delete_user(selected["id"])
            st.toast("Utilisateur supprimé.", icon="✅")
            st.session_state.pop("confirm_delete_user", None)
            reload()
        except Exception as e:
            st.toast(f"Erreur : {e}", icon="❌")
    if c2.button("Annuler"):
        st.session_state.pop("confirm_delete_user", None)
        st.rerun()

# --- Analytics d'usage ---
st.divider()
with st.expander(
    f"📊 Analytics d'usage — {selected['username']} (30 derniers jours)", expanded=True
):
    try:
        usage = client.get_user_usage(selected["id"], days=30)
        daily_limit = selected["rate_limit_per_day"]

        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        today_calls = next((d["calls"] for d in usage["by_day"] if str(d["date"]) == today_str), 0)
        pct = today_calls / daily_limit * 100 if daily_limit > 0 else 0

        st.metric(
            label="Consommation quota (aujourd'hui)",
            value=f"{pct:.1f}%",
            delta=f"{today_calls} / {daily_limit} appels",
            delta_color="off",
        )
        show_quota_progress(today_calls, daily_limit)

        col_left, col_right = st.columns(2)

        with col_left:
            if usage["by_model"]:
                df_model = pd.DataFrame(usage["by_model"])
                fig_model = px.bar(
                    df_model,
                    x="model_name",
                    y="calls",
                    title="Prédictions par modèle",
                    labels={"model_name": "Modèle", "calls": "Appels"},
                    color_discrete_sequence=["#4C78A8"],
                )
                fig_model.update_layout(
                    xaxis_title="Modèle",
                    yaxis_title="Appels",
                    showlegend=False,
                    margin={"t": 40},
                )
                st.plotly_chart(fig_model, use_container_width=True)
            else:
                st.info("Aucune prédiction par modèle sur la période.")

        with col_right:
            if usage["by_day"]:
                df_day = pd.DataFrame(usage["by_day"])
                df_day["date"] = pd.to_datetime(df_day["date"])
                df_day = df_day.sort_values("date")

                fig_day = go.Figure()
                fig_day.add_trace(
                    go.Scatter(
                        x=df_day["date"],
                        y=df_day["calls"],
                        mode="lines+markers",
                        name="Appels/jour",
                        line={"color": "#4C78A8"},
                    )
                )
                fig_day.add_hline(
                    y=daily_limit,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Quota ({daily_limit}/jour)",
                    annotation_position="top right",
                )
                fig_day.update_layout(
                    title="Volume par jour",
                    xaxis_title="Date",
                    yaxis_title="Appels",
                    showlegend=False,
                    margin={"t": 40},
                )
                st.plotly_chart(fig_day, use_container_width=True)
            else:
                st.info("Aucune donnée journalière sur la période.")

    except Exception as e:
        st.error(f"Impossible de charger les analytics : {e}")
