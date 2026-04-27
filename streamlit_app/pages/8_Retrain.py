"""
Gestion centralisée des ré-entraînements et plannings cron
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from utils.auth import get_client, require_auth

st.set_page_config(page_title="Retrain — PredictML", page_icon="🔄", layout="wide")
require_auth()

is_admin = st.session_state.get("is_admin", False)
if not is_admin:
    st.error("⛔ Accès réservé aux administrateurs.")
    st.stop()

st.title("🔄 Gestion des ré-entraînements")

client = get_client()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_models(api_url, token):
    c = get_client()
    return c.list_models()


def reload():
    st.cache_data.clear()
    st.rerun()


try:
    models = fetch_models(st.session_state.get("api_url"), st.session_state.get("api_token"))
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()

if not models:
    st.info("Aucun modèle disponible.")
    st.stop()

tab_overview, tab_manual, tab_schedule, tab_policy, tab_history = st.tabs(
    ["📅 Schedules", "🚀 Retrain manuel", "⏰ Planning cron", "🏆 Auto-promotion", "📜 Historique"]
)

# ─── Onglet 1 — Vue d'ensemble des schedules ──────────────────────────────

with tab_overview:
    st.subheader("Vue d'ensemble des schedules")

    rows = []
    for m in models:
        sched = m.get("retrain_schedule") or {}
        has_script = bool(m.get("train_script_object_key"))

        if sched:
            cron = sched.get("cron") or "—"
            last_run = sched.get("last_run_at")
            next_run = sched.get("next_run_at")
            enabled = sched.get("enabled", True)
            badge = "🟢 Actif" if enabled else "🔴 Désactivé"
        else:
            cron = "—"
            last_run = None
            next_run = None
            badge = "⚫ Aucun schedule"

        rows.append(
            {
                "Modèle": m.get("name", ""),
                "Version": m.get("version", ""),
                "Script train.py": "✅" if has_script else "❌",
                "Cron": cron,
                "Dernier retrain": (
                    pd.to_datetime(last_run).strftime("%Y-%m-%d %H:%M") if last_run else "—"
                ),
                "Prochain retrain": (
                    pd.to_datetime(next_run).strftime("%Y-%m-%d %H:%M") if next_run else "—"
                ),
                "Statut": badge,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    active = sum(
        1
        for m in models
        if (m.get("retrain_schedule") or {}).get("enabled")
        and (m.get("retrain_schedule") or {}).get("cron")
    )
    trainable_count = sum(1 for m in models if m.get("train_script_object_key"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Schedules actifs", active)
    col2.metric("Modèles avec train.py", trainable_count)
    col3.metric("Total modèles", len(models))

# ─── Onglet 2 — Retrain manuel ────────────────────────────────────────────

with tab_manual:
    st.subheader("Déclencher un ré-entraînement manuel")

    trainable = [m for m in models if m.get("train_script_object_key")]
    if not trainable:
        st.warning(
            "Aucun modèle ne possède de script `train.py`. "
            "Uploadez-en un via `POST /models` avec le paramètre `train_file`."
        )
    else:
        model_opts = {f"{m['name']} v{m['version']}": m for m in trainable}
        retrain_search = st.text_input("Filtrer par nom", key="retrain_search", placeholder="Rechercher un modèle…")
        retrain_keys = [k for k in model_opts if retrain_search.lower() in k.lower()] if retrain_search else list(model_opts.keys())
        selected_label = st.selectbox(
            "Modèle à ré-entraîner", retrain_keys or list(model_opts.keys()), key="retrain_select"
        )
        sel = model_opts[selected_label]

        with st.form("manual_retrain_form"):
            col_s, col_e = st.columns(2)
            with col_s:
                start_date = st.date_input(
                    "Date de début",
                    value=date.today() - timedelta(days=30),
                    key="manual_start",
                )
            with col_e:
                end_date = st.date_input(
                    "Date de fin",
                    value=date.today(),
                    key="manual_end",
                )
            new_version_input = st.text_input(
                "Nouvelle version (laisser vide = auto-généré)",
                value="",
                placeholder=f"{sel['version']}-retrain-YYYYMMDDHHMMSS",
                key="manual_new_version",
            )
            set_prod = st.checkbox(
                "Mettre en production après entraînement",
                value=False,
                key="manual_set_prod",
            )
            submitted = st.form_submit_button("🚀 Lancer le ré-entraînement", type="primary")

        if submitted:
            if start_date > end_date:
                st.error("La date de début doit être antérieure à la date de fin.")
            else:
                with st.spinner("Ré-entraînement en cours… (peut prendre jusqu'à 10 minutes)"):
                    try:
                        result = client.retrain_model(
                            name=sel["name"],
                            version=sel["version"],
                            start_date=str(start_date),
                            end_date=str(end_date),
                            new_version=new_version_input.strip() or None,
                            set_production=set_prod,
                        )
                        if result.get("success"):
                            st.toast(
                                f"Ré-entraînement réussi ! "
                                f"Nouvelle version : {result['new_version']}",
                                icon="✅",
                            )
                            auto_promoted = result.get("auto_promoted")
                            if auto_promoted is True:
                                st.info(
                                    "✅ Auto-promotion effectuée — "
                                    "la nouvelle version est en production."
                                )
                            elif auto_promoted is False:
                                st.warning(
                                    f"⚠️ Auto-promotion évaluée mais non effectuée : "
                                    f"{result.get('auto_promote_reason') or '—'}"
                                )
                        else:
                            st.toast(
                                f"Échec du ré-entraînement : "
                                f"{result.get('error', 'Erreur inconnue')}",
                                icon="❌",
                            )
                        with st.expander("📋 Logs stdout", expanded=not result.get("success")):
                            st.code(result.get("stdout", "(vide)"), language="text")
                        with st.expander("⚠️ Logs stderr", expanded=not result.get("success")):
                            st.code(result.get("stderr", "(vide)"), language="text")
                        if result.get("success"):
                            reload()
                    except Exception as e:
                        st.toast(f"Erreur lors du ré-entraînement : {e}", icon="❌")

# ─── Onglet 3 — Planning cron ─────────────────────────────────────────────

with tab_schedule:
    st.subheader("Configurer un planning cron")

    trainable_sched = [m for m in models if m.get("train_script_object_key")]
    if not trainable_sched:
        st.warning(
            "Aucun modèle ne possède de script `train.py`. "
            "Uploadez-en un via `POST /models` avec le paramètre `train_file`."
        )
    else:
        with st.expander("ℹ️ Aide — expressions cron (UTC)", expanded=False):
            st.markdown("""
| Expression | Déclenchement |
|---|---|
| `0 3 * * 1` | Chaque lundi à 03h00 |
| `0 2 * * *` | Chaque jour à 02h00 |
| `0 4 1 * *` | Le 1er de chaque mois à 04h00 |
| `0 6 * * 1,3,5` | Lundi, mercredi, vendredi à 06h00 |

Format : `minute heure jour-du-mois mois jour-de-la-semaine`
""")

        sched_opts = {f"{m['name']} v{m['version']}": m for m in trainable_sched}
        sched_search = st.text_input("Filtrer par nom", key="sched_search", placeholder="Rechercher un modèle…")
        sched_keys = [k for k in sched_opts if sched_search.lower() in k.lower()] if sched_search else list(sched_opts.keys())
        sched_label = st.selectbox(
            "Modèle à planifier", sched_keys or list(sched_opts.keys()), key="sched_select"
        )
        sched_sel = sched_opts[sched_label]
        existing_sched = sched_sel.get("retrain_schedule") or {}

        if existing_sched:
            st.markdown("**Planning actuel :**")
            st.json(existing_sched)

        with st.form("schedule_form"):
            cron_val = st.text_input(
                "Expression cron (5 champs, UTC)",
                value=existing_sched.get("cron") or "",
                placeholder="0 3 * * 1",
                help="Laisser vide pour désactiver sans effacer l'expression",
            )
            lookback = st.slider(
                "Fenêtre d'historique lookback_days",
                min_value=1,
                max_value=365,
                value=int(existing_sched.get("lookback_days") or 30),
                help="Nombre de jours transmis via TRAIN_START_DATE / TRAIN_END_DATE",
            )
            col_ap, col_en = st.columns(2)
            with col_ap:
                auto_promote_sched = st.checkbox(
                    "Auto-promotion après retrain",
                    value=bool(existing_sched.get("auto_promote", False)),
                    key="sched_auto_promote",
                    help="Évalue la promotion_policy après chaque exécution planifiée",
                )
            with col_en:
                enabled_sched = st.checkbox(
                    "Activer le planning",
                    value=bool(existing_sched.get("enabled", True)),
                    key="sched_enabled",
                    help="Décocher pour suspendre sans effacer la configuration",
                )
            save_sched = st.form_submit_button("💾 Sauvegarder le planning", type="primary")

        if save_sched:
            try:
                result = client.set_schedule(
                    name=sched_sel["name"],
                    version=sched_sel["version"],
                    cron=cron_val.strip() or None,
                    lookback_days=lookback,
                    auto_promote=auto_promote_sched,
                    enabled=enabled_sched,
                )
                st.toast(
                    f"Planning sauvegardé pour {sched_sel['name']} v{sched_sel['version']}.",
                    icon="✅",
                )
                saved_sched = result.get("retrain_schedule") or {}
                if saved_sched.get("next_run_at"):
                    st.info(f"Prochain déclenchement prévu : `{saved_sched['next_run_at']}`")
                reload()
            except Exception as e:
                st.toast(f"Erreur lors de la sauvegarde du planning : {e}", icon="❌")

# ─── Onglet 4 — Politique d'auto-promotion ───────────────────────────────

with tab_policy:
    st.subheader("Politique d'auto-promotion post-retrain")
    st.caption("La politique s'applique à toutes les versions actives du modèle sélectionné.")

    model_names = sorted({m["name"] for m in models})
    policy_search = st.text_input("Filtrer par nom", key="policy_model_search", placeholder="Rechercher un modèle…")
    policy_filtered = [n for n in model_names if policy_search.lower() in n.lower()] if policy_search else model_names
    policy_name = st.selectbox("Modèle", policy_filtered or model_names, key="policy_model_select")

    matching = [m for m in models if m["name"] == policy_name]
    current_policy: dict = {}
    for m in matching:
        if m.get("promotion_policy"):
            current_policy = m["promotion_policy"]
            break

    if current_policy:
        st.markdown("**Politique actuelle :**")
        st.json(current_policy)
    else:
        st.info("Aucune politique d'auto-promotion définie pour ce modèle.")

    with st.form("policy_form"):
        col_l, col_r = st.columns(2)

        with col_l:
            min_acc_enabled = st.checkbox(
                "Activer min_accuracy",
                value=current_policy.get("min_accuracy") is not None,
            )
            min_acc = st.slider(
                "min_accuracy",
                min_value=0.0,
                max_value=1.0,
                value=float(current_policy.get("min_accuracy") or 0.9),
                step=0.01,
                help="Précision minimale requise sur les paires (prédiction, résultat observé)",
            )

            max_latency_enabled = st.checkbox(
                "Activer max_latency_p95_ms",
                value=current_policy.get("max_latency_p95_ms") is not None,
            )
            max_latency = st.number_input(
                "max_latency_p95_ms",
                min_value=1.0,
                value=float(current_policy.get("max_latency_p95_ms") or 200.0),
                step=10.0,
                help="Latence P95 maximale autorisée en ms",
            )

        with col_r:
            max_mae_enabled = st.checkbox(
                "Activer max_mae",
                value=current_policy.get("max_mae") is not None,
            )
            max_mae = st.number_input(
                "max_mae",
                min_value=0.0,
                value=float(current_policy.get("max_mae") or 0.1),
                step=0.01,
                format="%.3f",
                help="Erreur absolue moyenne maximale autorisée (modèles de régression)",
            )

            min_samples = st.number_input(
                "min_sample_validation",
                min_value=1,
                value=int(current_policy.get("min_sample_validation") or 10),
                step=1,
                help="Nombre minimal de paires (prédiction, résultat observé) pour évaluer la politique",
            )
            auto_promote_policy = st.checkbox(
                "Activer auto_promote",
                value=bool(current_policy.get("auto_promote", False)),
                help="Si désactivé, la politique est stockée mais jamais évaluée automatiquement",
            )

        save_policy = st.form_submit_button("💾 Enregistrer la politique", type="primary")

    if save_policy:
        try:
            result = client.set_policy(
                name=policy_name,
                min_accuracy=min_acc if min_acc_enabled else None,
                max_mae=max_mae if max_mae_enabled else None,
                max_latency_p95_ms=max_latency if max_latency_enabled else None,
                min_sample_validation=int(min_samples),
                auto_promote=auto_promote_policy,
            )
            updated = result.get("updated_versions", 0)
            st.toast(
                f"Politique enregistrée pour {policy_name} "
                f"({updated} version(s) mise(s) à jour).",
                icon="✅",
            )
            reload()
        except Exception as e:
            st.toast(f"Erreur lors de la sauvegarde de la politique : {e}", icon="❌")

# ─── Onglet 5 — Historique des retrains ──────────────────────────────────

with tab_history:
    st.subheader("Historique des ré-entraînements")

    model_names_hist = sorted({m["name"] for m in models})
    hist_search = st.text_input("Filtrer par nom", key="hist_model_search", placeholder="Rechercher un modèle…")
    hist_filtered = [n for n in model_names_hist if hist_search.lower() in n.lower()] if hist_search else model_names_hist
    hist_model_name = st.selectbox("Modèle", hist_filtered or model_names_hist, key="hist_model_select")

    try:
        raw_history = client.get_model_history(name=hist_model_name, limit=200)
    except Exception as e:
        st.error(f"Impossible de charger l'historique : {e}")
        raw_history = {"entries": [], "total": 0}

    all_entries = raw_history.get("entries", [])

    # Versions promues en production (SET_PRODUCTION avec is_production=True)
    promoted_versions: set = {
        e["model_version"]
        for e in all_entries
        if e["action"] == "set_production" and e["snapshot"].get("is_production")
    }

    # Entrées de retrain : action=created + snapshot.parent_version non nul
    retrain_entries = [
        e
        for e in all_entries
        if e["action"] == "created" and e["snapshot"].get("parent_version")
    ]

    if not retrain_entries:
        st.info(
            "Aucun ré-entraînement enregistré pour ce modèle. "
            "Les retrains apparaissent ici après leur première exécution."
        )
    else:
        rows = []
        for e in retrain_entries:
            snap = e["snapshot"]
            trained_by = snap.get("trained_by") or e.get("changed_by_username") or "—"
            accuracy = snap.get("accuracy")
            f1 = snap.get("f1_score")
            promoted = e["model_version"] in promoted_versions
            rows.append(
                {
                    "Date": pd.to_datetime(e["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                    "Version créée": e["model_version"],
                    "Trained by": trained_by,
                    "Version source": snap.get("parent_version") or "—",
                    "Accuracy": round(accuracy, 4) if accuracy is not None else None,
                    "F1 Score": round(f1, 4) if f1 is not None else None,
                    "En production": "✅" if promoted else "❌",
                }
            )

        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        st.caption(f"{len(retrain_entries)} ré-entraînement(s) au total.")

        # Graphique de progression de l'accuracy
        chart_df = df_hist[df_hist["Accuracy"].notna()].copy()
        if not chart_df.empty:
            st.markdown("#### Progression de l'accuracy")
            chart_df = chart_df.sort_values("Date")
            chart_df = chart_df.rename(columns={"Date": "index"}).set_index("index")
            st.line_chart(chart_df[["Accuracy", "F1 Score"]].dropna(how="all"))
