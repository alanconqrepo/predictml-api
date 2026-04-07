"""
Gestion des utilisateurs — admin only
"""
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from utils.auth import require_admin, get_client

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
        rate_limit = col4.number_input("Quota journalier", min_value=1, max_value=100000, value=1000)
        submitted = st.form_submit_button("Créer", use_container_width=True, type="primary")

    if submitted:
        if not username or not email:
            st.error("Nom d'utilisateur et email sont requis.")
        else:
            try:
                result = client.create_user({
                    "username": username,
                    "email": email,
                    "role": role,
                    "rate_limit": rate_limit,
                })
                st.success(f"Utilisateur **{result['username']}** créé avec succès.")
                st.info("Conservez ce token — il ne sera plus affiché !")
                show_token_with_copy(result["api_token"])
                reload()
            except Exception as e:
                st.error(f"Erreur : {e}")

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

df = pd.DataFrame(users)[["id", "username", "email", "role", "is_active", "rate_limit_per_day", "last_login", "created_at"]]
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
        st.success(f"Utilisateur {'désactivé' if current_active else 'activé'}.")
        reload()
    except Exception as e:
        st.error(f"Erreur : {e}")

# Renouveler token
if col_b.button("🔄 Renouveler token", use_container_width=True):
    try:
        result = client.update_user(selected["id"], {"regenerate_token": True})
        st.success("Nouveau token généré. Conservez-le !")
        show_token_with_copy(result["api_token"])
        reload()
    except Exception as e:
        st.error(f"Erreur : {e}")

# Modifier rôle
new_roles = [r for r in ["user", "admin", "readonly"] if r != selected["role"]]
new_role = col_c.selectbox("Changer rôle", new_roles, key="role_select")
if col_c.button("✏️ Appliquer rôle", use_container_width=True):
    try:
        client.update_user(selected["id"], {"role": new_role})
        st.success(f"Rôle mis à jour → {new_role}")
        reload()
    except Exception as e:
        st.error(f"Erreur : {e}")

# Supprimer
with col_d:
    if st.button("🗑️ Supprimer", use_container_width=True, type="secondary"):
        st.session_state["confirm_delete_user"] = selected["id"]

if st.session_state.get("confirm_delete_user") == selected["id"]:
    st.warning(f"Confirmer la suppression de **{selected['username']}** et toutes ses prédictions ?")
    c1, c2 = st.columns(2)
    if c1.button("Oui, supprimer", type="primary"):
        try:
            client.delete_user(selected["id"])
            st.success("Utilisateur supprimé.")
            st.session_state.pop("confirm_delete_user", None)
            reload()
        except Exception as e:
            st.error(f"Erreur : {e}")
    if c2.button("Annuler"):
        st.session_state.pop("confirm_delete_user", None)
        st.rerun()
