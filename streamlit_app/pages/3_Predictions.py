"""
Historique des prédictions avec filtres
"""

import io
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth

_SCRIPTS_DIR = Path(__file__).parent.parent / "documentation" / "Scripts"

_EXAMPLE_SCRIPTS = [
    (
        "send_predictions_iris.py",
        "Envoie 5 observations Iris en mode unitaire puis en lot (`POST /predict` + `POST /predict-batch`)",
    ),
    (
        "send_ground_truth.py",
        "Envoie les labels réels (ground truth) pour calculer la performance réelle du modèle (`POST /observed-results`)",
    ),
]


def _read_script(filename: str) -> str:
    try:
        return (_SCRIPTS_DIR / filename).read_text(encoding="utf-8")
    except Exception:
        return f"# Fichier introuvable : {filename}"


@st.dialog("Aperçu du script", width="large")
def _view_script_dialog(filename: str) -> None:
    st.code(_read_script(filename), language="python", line_numbers=True)
    st.download_button(
        "⬇️ Télécharger",
        data=_read_script(filename),
        file_name=filename,
        mime="text/x-python",
        key=f"dl_dialog_{filename}",
        use_container_width=True,
    )


st.set_page_config(page_title="Predictions — PredictML", page_icon="📊", layout="wide")
require_auth()

col_title, col_refresh = st.columns([8, 1])
col_title.title("📊 Prédictions")
if col_refresh.button("🔄 Rafraîchir", key="pred_refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

client = get_client()

with st.expander("📋 Scripts d'exemple — Iris", expanded=False):
    st.caption(
        "Scripts de référence pour envoyer des prédictions et des résultats observés "
        "vers votre modèle Iris. À exécuter localement après avoir uploadé le modèle."
    )
    for _script_name, _script_desc in _EXAMPLE_SCRIPTS:
        _col_desc, _col_view, _col_dl = st.columns([5, 1.5, 1.5])
        _col_desc.markdown(f"**`{_script_name}`**  \n{_script_desc}")
        if _col_view.button("👁 Visualiser", key=f"view_{_script_name}", use_container_width=True):
            _view_script_dialog(_script_name)
        _col_dl.download_button(
            "⬇️ Télécharger",
            data=_read_script(_script_name),
            file_name=_script_name,
            mime="text/x-python",
            key=f"dl_{_script_name}",
            use_container_width=True,
        )

tab_history, tab_batch = st.tabs(["📋 Historique", "📦 Prédictions batch"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Historique
# ───────────────────────────────────────────────────────────────────────────────
with tab_history:
    # --- Couverture du ground truth ---
    with st.expander("🏷️ Couverture du ground truth", expanded=True):
        try:
            coverage_data = client.get_observed_results_stats()
            total_pred = coverage_data.get("total_predictions", 0)
            labeled = coverage_data.get("labeled_count", 0)
            rate = coverage_data.get("coverage_rate", 0.0)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Prédictions totales", f"{total_pred:,}")
            col_b.metric("Labellisées", f"{labeled:,}")
            col_c.metric("Couverture", f"{rate * 100:.1f} %")

            by_model = coverage_data.get("by_model") or []
            if by_model:
                st.markdown("**Par modèle :**")
                for m in by_model:
                    cov = m.get("coverage", 0.0)
                    st.progress(
                        cov,
                        text=f"{m['model_name']} — {m['labeled']}/{m['predictions']} ({cov * 100:.1f} %)",
                    )
        except Exception:
            st.caption("Données de couverture non disponibles.")

    # --- Filtres ---
    with st.expander("🔍 Filtres", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)

        # Liste des modèles disponibles
        try:
            models = get_models_cached(
                st.session_state.get("api_url"), st.session_state.get("api_token")
            )
            model_names = sorted({m["name"] for m in models})
        except Exception:
            models = []
            model_names = []

        with col1:
            hist_search = st.text_input(
                "Filtrer par nom", key="hist_model_search", placeholder="Rechercher…"
            )
            hist_filtered = (
                [n for n in model_names if hist_search.lower() in n.lower()]
                if hist_search
                else model_names
            )
            model_name = st.selectbox("Modèle", ["(tous)"] + (hist_filtered or model_names))
        if model_name == "(tous)":
            model_name = model_names[0] if model_names else None

        today = date.today()
        start_date = col2.date_input("Date début", value=today - timedelta(days=7))
        end_date = col3.date_input("Date fin", value=today)
        status_filter = col4.selectbox("Statut", ["Tous", "success", "error"])
        limit = col5.selectbox("Limite", [50, 100, 500], index=1)

        # Confidence sliders — désactivés pour les régresseurs (pas de classes)
        is_classifier = any(
            m.get("name") == model_name and m.get("classes")
            for m in models
        ) if model_name else False

        col_conf1, col_conf2, col_conf3 = st.columns(3)
        conf_min = col_conf1.slider(
            "Confiance min",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="hist_conf_min",
            disabled=not is_classifier,
            help="Filtrer les prédictions avec une confiance ≥ à ce seuil (classifieurs uniquement)",
        )
        conf_max = col_conf2.slider(
            "Confiance max",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            key="hist_conf_max",
            disabled=not is_classifier,
            help="Filtrer les prédictions avec une confiance ≤ à ce seuil (classifieurs uniquement)",
        )
        filter_mismatch_only = col_conf3.checkbox(
            "Prédictions incorrectes uniquement",
            key="hist_mismatch_only",
            help="N'afficher que les lignes où la prédiction diffère du ground truth",
        )
        # N'envoyer les filtres que si le modèle est un classifieur et les valeurs non-défaut
        filter_min_conf = conf_min if (is_classifier and conf_min > 0.0) else None
        filter_max_conf = conf_max if (is_classifier and conf_max < 1.0) else None

    if not model_name:
        st.warning("Aucun modèle disponible. Créez d'abord un modèle via l'API.")
    elif start_date > end_date:
        st.error("La date de début doit être avant la date de fin.")
    else:
        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat()

        # Fetch
        try:
            data = client.get_predictions(
                model_name=model_name,
                start=start_iso,
                end=end_iso,
                limit=limit,
                offset=0,
                min_confidence=filter_min_conf,
                max_confidence=filter_max_conf,
            )
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            st.stop()

        total = data.get("total", 0)
        predictions = data.get("predictions", [])

        # Filtre statut côté client (l'API ne supporte pas le filtre statut)
        if status_filter != "Tous":
            predictions = [p for p in predictions if p.get("status") == status_filter]

        st.caption(f"**{total}** prédictions trouvées — {len(predictions)} affichées")

        if not predictions:
            st.info("Aucune prédiction pour ces critères.")
        else:
            # Fetch ground truth for visible id_obs values
            gt_lookup: dict = {}
            id_obs_list = [p["id_obs"] for p in predictions if p.get("id_obs")]
            if id_obs_list:
                try:
                    obs_data = client.get_observed_results(
                        model_name=model_name or None,
                        limit=len(id_obs_list) + 50,
                    )
                    for obs in obs_data.get("results", obs_data if isinstance(obs_data, list) else []):
                        if obs.get("id_obs"):
                            gt_lookup[obs["id_obs"]] = str(obs.get("observed_result", ""))
                except Exception:
                    pass

            rows = []
            for p in predictions:
                mc = p.get("max_confidence")
                id_obs_val = p.get("id_obs")
                gt_val = gt_lookup.get(id_obs_val, "—") if id_obs_val else "—"
                pred_val = str(p.get("prediction_result", ""))
                mismatch = gt_val != "—" and pred_val != gt_val
                rows.append(
                    {
                        "ID": p.get("id"),
                        "Timestamp": (
                            pd.to_datetime(p.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
                            if p.get("timestamp")
                            else "—"
                        ),
                        "Modèle": p.get("model_name", ""),
                        "Version": p.get("model_version") or "—",
                        "id_obs": id_obs_val or "—",
                        "Résultat": pred_val,
                        "Ground Truth": gt_val,
                        "Confiance": f"{mc:.2%}" if mc is not None else "—",
                        "Temps (ms)": (
                            f"{p['response_time_ms']:.1f}"
                            if p.get("response_time_ms") is not None
                            else "—"
                        ),
                        "Statut": "✅" if p.get("status") == "success" else "❌",
                        "Shadow": "🔮" if p.get("is_shadow") else "—",
                        "Utilisateur": p.get("username") or "—",
                        "_mismatch": mismatch,
                    }
                )

            df = pd.DataFrame(rows)

            if filter_mismatch_only:
                df = df[df["_mismatch"]].reset_index(drop=True)
                if df.empty:
                    st.info("Aucune prédiction incorrecte sur cette page.")

            mismatch_flags = df["_mismatch"].to_numpy()
            df_display = df.drop(columns=["_mismatch"])

            def _highlight_mismatch(row):
                return (
                    ["background-color: #ffcccc"] * len(row)
                    if mismatch_flags[row.name]
                    else [""] * len(row)
                )

            styled = df_display.style.apply(_highlight_mismatch, axis=1)
            sel = st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            # ── Panneau détail (ligne sélectionnée) ───────────────────────────
            selected_rows = sel.selection.rows if sel.selection else []
            if selected_rows:
                import plotly.graph_objects as go

                row_idx = selected_rows[0]
                pred_id_at_row = df_display.iloc[row_idx]["ID"]
                p = next((x for x in predictions if x.get("id") == pred_id_at_row), None)
                if p is None:
                    st.info("Prédiction introuvable.")
                    st.stop()
                pred_id = pred_id_at_row
                st.divider()
                st.markdown(f"#### 🔍 Prédiction #{pred_id}")

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("**Features d'entrée :**")
                    st.json(p.get("input_features", {}))
                with col_r:
                    st.markdown("**Résultat :**")
                    st.json({
                        "prediction": p.get("prediction_result"),
                        "probabilities": p.get("probabilities"),
                    })
                    if p.get("error_message"):
                        st.error(f"Erreur : {p['error_message']}")

                # Ground truth
                st.divider()
                st.markdown("**── Résultat observé ──**")
                id_obs_val = p.get("id_obs")
                if not id_obs_val:
                    st.caption("Pas d'`id_obs` — impossible de soumettre un résultat observé.")
                else:
                    obs_cache_key = f"obs_result_{pred_id}"
                    if obs_cache_key not in st.session_state:
                        try:
                            resp = client.get_observed_results(
                                model_name=p.get("model_name"), id_obs=id_obs_val, limit=1
                            )
                            results = resp.get("results", [])
                            st.session_state[obs_cache_key] = results[0] if results else None
                        except Exception:
                            st.session_state[obs_cache_key] = None

                    existing = st.session_state.get(obs_cache_key)
                    if existing is not None:
                        st.success(f"✅ Résultat enregistré : **{existing['observed_result']}**")
                    else:
                        obs_input_val = st.text_input(
                            "Valeur observée",
                            key=f"obs_input_{pred_id}",
                            placeholder="Ex: 0, 1.5, setosa…",
                        )
                        if st.button("Enregistrer le résultat réel", key=f"obs_btn_{pred_id}"):
                            if not obs_input_val.strip():
                                st.warning("Veuillez saisir une valeur.")
                            else:
                                try:
                                    parsed_val = int(obs_input_val)
                                except ValueError:
                                    try:
                                        parsed_val = float(obs_input_val)
                                    except ValueError:
                                        parsed_val = obs_input_val
                                try:
                                    client.submit_observed_result(
                                        id_obs=id_obs_val,
                                        model_name=p.get("model_name"),
                                        observed_result=parsed_val,
                                    )
                                    st.session_state[obs_cache_key] = {"observed_result": parsed_val}
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Erreur : {exc}")

                # SHAP
                if p.get("status") == "success":
                    st.divider()
                    st.markdown("**── Explication SHAP ──**")
                    if st.button("🧠 Calculer l'explication SHAP", key=f"shap_btn_{pred_id}"):
                        with st.spinner("Calcul SHAP en cours…"):
                            try:
                                st.session_state[f"shap_{pred_id}"] = client.explain_prediction(pred_id)
                            except Exception as exc:
                                st.error(f"Impossible de calculer : {exc}")

                    shap_data = st.session_state.get(f"shap_{pred_id}")
                    if shap_data:
                        shap_values: dict = shap_data.get("shap_values", {})
                        base_value: float = shap_data.get("base_value", 0.0)
                        model_type: str = shap_data.get("model_type", "")
                        col_s1, col_s2, col_s3 = st.columns(3)
                        col_s1.metric("E[f(X)]", f"{base_value:.4f}")
                        col_s2.metric("Prédiction", str(shap_data.get("prediction")))
                        col_s3.metric("Type", model_type)
                        if shap_values:
                            sorted_features = sorted(
                                shap_values.items(), key=lambda x: abs(x[1]), reverse=True
                            )[:10]
                            feat_names = [f for f, _ in sorted_features]
                            shap_vals = [v for _, v in sorted_features]
                            fig = go.Figure(go.Bar(
                                x=shap_vals, y=feat_names, orientation="h",
                                marker_color=["#e05252" if v >= 0 else "#5282e0" for v in shap_vals],
                                text=[f"{v:+.4f}" for v in shap_vals],
                                textposition="outside",
                            ))
                            fig.update_layout(
                                title="Contributions SHAP (top 10)",
                                xaxis_title="Contribution SHAP",
                                yaxis={"autorange": "reversed"},
                                height=max(300, len(sorted_features) * 40 + 100),
                                margin={"l": 20, "r": 60, "t": 50, "b": 40},
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            if len(shap_values) > 10:
                                with st.expander("Voir toutes les features"):
                                    st.dataframe(
                                        pd.DataFrame(
                                            sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True),
                                            columns=["Feature", "SHAP"],
                                        ),
                                        use_container_width=True, hide_index=True,
                                    )
                        else:
                            st.info("Aucune valeur SHAP retournée.")

                # Suppression
                if st.session_state.get("is_admin", False):
                    st.divider()
                    if st.button(
                        f"🗑️ Supprimer la prédiction #{pred_id}",
                        key=f"del_pred_{pred_id}",
                        type="primary",
                    ):
                        try:
                            client.delete_prediction(pred_id)
                            st.toast(f"Prédiction #{pred_id} supprimée.", icon="✅")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Erreur : {exc}")

            with st.expander("⬇️ Exporter toutes les prédictions (serveur)", expanded=False):
                st.caption(
                    "L'export serveur inclut **toutes** les prédictions correspondant aux filtres, "
                    "pas seulement la page courante."
                )
                col_exp1, col_exp2 = st.columns(2)
                export_fmt = col_exp1.selectbox(
                    "Format", ["csv", "jsonl", "parquet"], key="pred_export_fmt"
                )
                export_status = col_exp2.selectbox(
                    "Statut", ["(tous)", "success", "error"], key="pred_export_status"
                )
                if st.button("Préparer l'export", key="pred_export_btn"):
                    with st.spinner("Préparation de l'export en cours…"):
                        try:
                            content = client.export_predictions(
                                start=start_iso,
                                end=end_iso,
                                model_name=model_name,
                                export_format=export_fmt,
                                status=None if export_status == "(tous)" else export_status,
                            )
                            mime_map = {
                                "csv": "text/csv",
                                "jsonl": "application/x-ndjson",
                                "parquet": "application/octet-stream",
                            }
                            st.download_button(
                                label=f"⬇️ Télécharger predictions_export.{export_fmt}",
                                data=content,
                                file_name=f"predictions_export.{export_fmt}",
                                mime=mime_map[export_fmt],
                                key="pred_export_download",
                            )
                        except Exception as exc:
                            st.error(f"Erreur lors de l'export : {exc}")

        # --- Import / Export résultats observés ---
        CSV_TEMPLATE = "id_obs,model_name,observed_result,date_time\n"

        with st.expander("📤 Importer des résultats observés (CSV)"):
            st.download_button(
                "⬇️ Télécharger un template CSV",
                data=CSV_TEMPLATE,
                file_name="template_observed_results.csv",
                mime="text/csv",
            )
            uploaded_file = st.file_uploader("Fichier CSV", type=["csv"], key="csv_obs_upload")
            model_name_override = st.text_input(
                "Modèle (override colonne CSV — optionnel)",
                key="csv_obs_model_override",
            )
            if uploaded_file is not None and st.button("Importer", key="csv_obs_submit"):
                try:
                    result = client.upload_observed_results_csv(
                        file_bytes=uploaded_file.read(),
                        filename=uploaded_file.name,
                        model_name=model_name_override.strip() or None,
                    )
                    st.toast(
                        f"{result['upserted']} résultats importés depuis {result['filename']}.",
                        icon="✅",
                    )
                    if result.get("skipped_rows", 0) > 0:
                        st.warning(f"{result['skipped_rows']} ligne(s) ignorée(s)")
                        errors = result.get("parse_errors", [])
                        if errors:
                            st.dataframe(
                                pd.DataFrame(errors),
                                use_container_width=True,
                                hide_index=True,
                            )
                except Exception as exc:
                    st.error(f"Erreur lors de l'import : {exc}")

        with st.expander("📥 Exporter les résultats observés (ground truth)"):
            col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
            ex_model = col_ex1.text_input("Modèle (optionnel)", key="ex_obs_model")
            ex_start = col_ex2.date_input(
                "Date début", value=date.today() - timedelta(days=30), key="ex_obs_start"
            )
            ex_end = col_ex3.date_input("Date fin", value=date.today(), key="ex_obs_end")
            ex_format = col_ex4.selectbox("Format", ["csv", "jsonl"], key="ex_obs_format")

            if st.button("Préparer l'export", key="ex_obs_btn"):
                if ex_start > ex_end:
                    st.error("La date de début doit être avant la date de fin.")
                else:
                    try:
                        content = client.export_observed_results(
                            start=datetime.combine(ex_start, datetime.min.time()).isoformat(),
                            end=datetime.combine(ex_end, datetime.max.time()).isoformat(),
                            model_name=ex_model.strip() or None,
                            export_format=ex_format,
                        )
                        mime = "text/csv" if ex_format == "csv" else "application/x-ndjson"
                        st.download_button(
                            label=f"⬇️ Télécharger observed_results_export.{ex_format}",
                            data=content,
                            file_name=f"observed_results_export.{ex_format}",
                            mime=mime,
                            key="ex_obs_download",
                        )
                    except Exception as exc:
                        st.error(f"Erreur lors de l'export : {exc}")

        # --- Maintenance RGPD (admin uniquement) ---
        if st.session_state.get("is_admin", False):
            with st.expander("🗑️ Maintenance RGPD — Purge des prédictions"):
                st.caption(
                    "Supprime définitivement les prédictions anciennes. "
                    "Utilisez **Simuler** avant de confirmer."
                )

                col_m1, col_m2 = st.columns(2)
                purge_days = col_m1.slider(
                    "Purger les prédictions antérieures à",
                    min_value=7,
                    max_value=365,
                    value=90,
                    format="%d jours",
                    key="purge_days_slider",
                )
                purge_model_sel = col_m2.selectbox(
                    "Filtrer par modèle (optionnel)",
                    ["(tous)"] + (model_names if model_names else []),
                    key="purge_model_sel",
                )
                purge_model_name = None if purge_model_sel == "(tous)" else purge_model_sel

                col_sim, col_purge = st.columns(2)

                if col_sim.button(
                    "🔍 Simuler (dry_run)", key="purge_simulate", use_container_width=True
                ):
                    try:
                        result = client.purge_predictions(
                            older_than_days=purge_days,
                            model_name=purge_model_name,
                            dry_run=True,
                        )
                        st.info(
                            f"Simulation : **{result['deleted_count']}** prédiction(s) seraient supprimées."
                        )
                        if result.get("oldest_remaining"):
                            st.caption(
                                f"Prédiction la plus ancienne restante : {result['oldest_remaining']}"
                            )
                        if result.get("models_affected"):
                            st.caption(f"Modèles affectés : {', '.join(result['models_affected'])}")
                        if result.get("linked_observed_results_count", 0) > 0:
                            st.warning(
                                f"⚠️ {result['linked_observed_results_count']} résultat(s) observé(s) "
                                "lié(s) seraient perdus (perte de données de performance historiques)."
                            )
                    except Exception as exc:
                        st.error(f"Erreur lors de la simulation : {exc}")

                @st.dialog("⚠️ Confirmer la purge définitive")
                def _confirm_purge_dialog():
                    st.warning(
                        f"Vous allez **supprimer définitivement** toutes les prédictions "
                        f"antérieures à **{purge_days} jours**."
                    )
                    if purge_model_name:
                        st.info(f"Modèle ciblé : **{purge_model_name}**")
                    else:
                        st.info("Tous les modèles sont ciblés.")
                    st.markdown("Cette action est **irréversible**.")
                    if st.button(
                        "Confirmer la suppression", type="primary", key="purge_dialog_confirm"
                    ):
                        try:
                            result = client.purge_predictions(
                                older_than_days=purge_days,
                                model_name=purge_model_name,
                                dry_run=False,
                            )
                            st.toast(
                                f"{result['deleted_count']} prédiction(s) supprimée(s).", icon="✅"
                            )
                            if result.get("linked_observed_results_count", 0) > 0:
                                st.warning(
                                    f"{result['linked_observed_results_count']} résultat(s) observé(s) "
                                    "liés ont été perdus."
                                )
                            st.rerun()
                        except Exception as exc:
                            st.toast(f"Erreur lors de la purge : {exc}", icon="❌")

                if col_purge.button(
                    "⚠️ Confirmer la purge",
                    key="purge_open_dialog",
                    type="primary",
                    use_container_width=True,
                ):
                    _confirm_purge_dialog()


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Prédictions batch
# ───────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📦 Scoring en lot — CSV ou Parquet")
    st.caption(
        "Importez un fichier contenant vos observations, sélectionnez un modèle "
        "et téléchargez le CSV enrichi avec les prédictions."
    )

    # --- Format attendu ---
    with st.expander("ℹ️ Format attendu du fichier", expanded=False):
        st.markdown("""
Le fichier doit contenir **une colonne par feature** attendue par le modèle.
Les colonnes doivent correspondre exactement aux noms des features du modèle (casse incluse).

Une colonne `id_obs` optionnelle permet de tracer chaque ligne dans l'historique.

**Exemple pour un modèle Iris :**
""")
        example_df = pd.DataFrame(
            {
                "id_obs": ["obs-001", "obs-002", "obs-003"],
                "sepal_length": [5.1, 6.3, 4.7],
                "sepal_width": [3.5, 2.9, 3.2],
                "petal_length": [1.4, 5.6, 1.3],
                "petal_width": [0.2, 1.8, 0.2],
            }
        )
        st.dataframe(example_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Télécharger cet exemple (CSV)",
            data=example_df.to_csv(index=False),
            file_name="exemple_batch_iris.csv",
            mime="text/csv",
            key="batch_example_download",
        )

    # --- Sélection modèle + version ---
    try:
        all_models = get_models_cached(
            st.session_state.get("api_url"), st.session_state.get("api_token")
        )
        model_names_batch = sorted({m["name"] for m in all_models})
    except Exception:
        all_models = []
        model_names_batch = []

    if not model_names_batch:
        st.warning("Aucun modèle disponible. Créez d'abord un modèle via l'API.")
        st.stop()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        batch_search = st.text_input(
            "Filtrer par nom", key="batch_model_search", placeholder="Rechercher…"
        )
        batch_filtered = (
            [n for n in model_names_batch if batch_search.lower() in n.lower()]
            if batch_search
            else model_names_batch
        )
        batch_model = st.selectbox(
            "Modèle cible", batch_filtered or model_names_batch, key="batch_model_sel"
        )

    # Versions disponibles pour le modèle sélectionné
    batch_versions = ["(production / auto)"] + sorted(
        {m["version"] for m in all_models if m["name"] == batch_model},
        reverse=True,
    )
    batch_version_sel = col_b2.selectbox("Version", batch_versions, key="batch_version_sel")
    batch_version = None if batch_version_sel == "(production / auto)" else batch_version_sel

    # --- Upload fichier ---
    batch_file = st.file_uploader(
        "Fichier CSV ou Parquet",
        type=["csv", "parquet"],
        key="batch_file_uploader",
        help="CSV (.csv) ou Parquet (.parquet). La première ligne du CSV doit être l'en-tête.",
    )

    if batch_file is not None:
        # Lire le fichier selon son extension
        try:
            fname = batch_file.name.lower()
            if fname.endswith(".parquet"):
                df_input = pd.read_parquet(io.BytesIO(batch_file.read()))
            else:
                df_input = pd.read_csv(io.BytesIO(batch_file.read()))
        except Exception as exc:
            st.error(f"Impossible de lire le fichier : {exc}")
            st.stop()

        st.caption(
            f"Fichier chargé : **{batch_file.name}** — {len(df_input):,} lignes, {len(df_input.columns)} colonnes"
        )
        st.dataframe(df_input.head(10), use_container_width=True, hide_index=True)

        if st.button("🚀 Lancer le scoring", type="primary", key="batch_run"):
            # Extraire id_obs si présente, puis construire les features
            id_obs_col = None
            if "id_obs" in df_input.columns:
                id_obs_col = df_input["id_obs"].astype(str).tolist()
                feature_df = df_input.drop(columns=["id_obs"])
            else:
                feature_df = df_input

            rows_payload = feature_df.to_dict(orient="records")

            # Enrichir avec id_obs si disponible
            if id_obs_col is not None:
                inputs = [
                    {"features": row, "id_obs": obs_id}
                    for row, obs_id in zip(rows_payload, id_obs_col)
                ]
            else:
                inputs = [{"features": row} for row in rows_payload]

            with st.spinner(f"Scoring de {len(rows_payload):,} observations en cours…"):
                try:
                    import requests as _requests

                    result = client.predict_batch_from_df(
                        model_name=batch_model,
                        rows=[inp["features"] for inp in inputs],
                        model_version=batch_version,
                    )
                except _requests.exceptions.Timeout:
                    st.error(
                        "Le scoring a dépassé le délai de 120 secondes. "
                        "Essayez avec un fichier plus petit ou contactez l'administrateur."
                    )
                    st.stop()
                except Exception as exc:
                    detail = str(exc)
                    try:
                        import json as _json

                        body = _json.loads(str(exc).split(" - ", 1)[-1])
                        detail = body.get("detail", detail)
                    except Exception:
                        pass
                    st.error(f"Erreur lors du scoring : {detail}")
                    st.stop()

            predictions_out = result.get("predictions", [])
            used_version = result.get("model_version", "—")

            st.toast(
                f"{len(predictions_out):,} prédictions générées — "
                f"modèle {batch_model} v{used_version}",
                icon="✅",
            )

            # Construire le DataFrame résultat : colonnes originales + prediction + probabilities
            df_result = df_input.copy()
            df_result["prediction"] = [p.get("prediction") for p in predictions_out]

            # Déplier les probabilités par classe si disponibles
            first_proba = next(
                (p.get("probability") for p in predictions_out if p.get("probability")), None
            )
            if first_proba is not None:
                n_classes = len(first_proba)
                for i in range(n_classes):
                    df_result[f"proba_class_{i}"] = [
                        (p.get("probability") or [None] * n_classes)[i] for p in predictions_out
                    ]

            # Colonne low_confidence si disponible
            if any(p.get("low_confidence") is not None for p in predictions_out):
                df_result["low_confidence"] = [p.get("low_confidence") for p in predictions_out]

            # Prévisualisation (50 premières lignes)
            st.markdown("**Prévisualisation des résultats (50 premières lignes) :**")
            st.dataframe(df_result.head(50), use_container_width=True, hide_index=True)

            # Téléchargement CSV complet
            csv_out = df_result.to_csv(index=False)
            st.download_button(
                label=f"⬇️ Télécharger le CSV complet ({len(df_result):,} lignes)",
                data=csv_out,
                file_name=f"predictions_batch_{batch_model}.csv",
                mime="text/csv",
                key="batch_download_btn",
            )

            # Résumé statistique
            with st.expander("📊 Résumé des prédictions"):
                pred_series = df_result["prediction"]
                st.markdown(f"- **Nombre de prédictions :** {len(pred_series):,}")
                try:
                    numeric_preds = pred_series.astype(float)
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("Moyenne", f"{numeric_preds.mean():.4f}")
                    col_s2.metric("Médiane", f"{numeric_preds.median():.4f}")
                    col_s3.metric("Min", f"{numeric_preds.min():.4f}")
                    col_s4.metric("Max", f"{numeric_preds.max():.4f}")
                except (ValueError, TypeError):
                    # Classification — afficher la distribution des classes
                    dist = pred_series.value_counts()
                    st.dataframe(
                        dist.rename_axis("Classe").reset_index(name="Count"),
                        use_container_width=True,
                        hide_index=True,
                    )
