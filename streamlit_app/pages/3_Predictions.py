"""
Historique des prédictions avec filtres
"""

import io
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from utils.auth import get_client, require_auth

st.set_page_config(page_title="Predictions — PredictML", page_icon="📊", layout="wide")
require_auth()

st.title("📊 Prédictions")

client = get_client()

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

    # --- Import CSV résultats observés ---
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
                st.success(f"{result['upserted']} résultats importés depuis **{result['filename']}**")
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

    # --- Export résultats observés ---
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

    # --- Filtres ---
    with st.expander("🔍 Filtres", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)

        # Liste des modèles disponibles
        try:
            models = client.list_models()
            model_names = sorted({m["name"] for m in models})
        except Exception:
            model_names = []

        with col1:
            hist_search = st.text_input("Filtrer par nom", key="hist_model_search", placeholder="Rechercher…")
            hist_filtered = [n for n in model_names if hist_search.lower() in n.lower()] if hist_search else model_names
            model_name = st.selectbox("Modèle", ["(tous)"] + (hist_filtered or model_names))
        if model_name == "(tous)":
            model_name = model_names[0] if model_names else None

        today = date.today()
        start_date = col2.date_input("Date début", value=today - timedelta(days=7))
        end_date = col3.date_input("Date fin", value=today)
        status_filter = col4.selectbox("Statut", ["Tous", "success", "error"])
        limit = col5.selectbox("Limite", [50, 100, 500], index=1)

    if not model_name:
        st.warning("Aucun modèle disponible. Créez d'abord un modèle via l'API.")
    elif start_date > end_date:
        st.error("La date de début doit être avant la date de fin.")
    else:
        # Pagination via session state
        if "pred_offset" not in st.session_state:
            st.session_state["pred_offset"] = 0

        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat()
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat()

        # Fetch
        try:
            data = client.get_predictions(
                model_name=model_name,
                start=start_iso,
                end=end_iso,
                limit=limit,
                offset=st.session_state["pred_offset"],
            )
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            st.stop()

        total = data.get("total", 0)
        predictions = data.get("predictions", [])

        # Filtre statut côté client (l'API ne supporte pas le filtre statut)
        if status_filter != "Tous":
            predictions = [p for p in predictions if p.get("status") == status_filter]

        st.caption(
            f"**{total}** prédictions trouvées — affichage {st.session_state['pred_offset'] + 1}–{min(st.session_state['pred_offset'] + limit, total)}"
        )

        if not predictions:
            st.info("Aucune prédiction pour ces critères.")
        else:
            rows = []
            for p in predictions:
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
                        "id_obs": p.get("id_obs") or "—",
                        "Résultat": str(p.get("prediction_result", "")),
                        "Temps (ms)": (
                            f"{p['response_time_ms']:.1f}"
                            if p.get("response_time_ms") is not None
                            else "—"
                        ),
                        "Statut": "✅" if p.get("status") == "success" else "❌",
                        "Shadow": "🔮" if p.get("is_shadow") else "—",
                        "Utilisateur": p.get("username") or "—",
                    }
                )

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.download_button(
                label="⬇️ Télécharger en CSV",
                data=df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )

            # Détail features
            with st.expander("🔍 Voir les features d'une prédiction"):
                pred_ids = {str(p["id"]): p for p in predictions}
                selected_id = st.selectbox("Prédiction ID", list(pred_ids.keys()))
                if selected_id:
                    p = pred_ids[selected_id]
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.markdown("**Features d'entrée :**")
                        st.json(p.get("input_features", {}))
                    with col_r:
                        st.markdown("**Résultat :**")
                        st.json(
                            {
                                "prediction": p.get("prediction_result"),
                                "probabilities": p.get("probabilities"),
                            }
                        )
                        if p.get("error_message"):
                            st.error(f"Erreur : {p['error_message']}")

            # Explication SHAP par prédiction
            with st.expander("🧠 Explication SHAP d'une prédiction"):
                import plotly.graph_objects as go

                shap_pred_ids = {str(p["id"]): p for p in predictions if p.get("status") == "success"}
                if not shap_pred_ids:
                    st.info("Aucune prédiction réussie sur cette page — sélectionnez une autre plage de dates.")
                else:
                    shap_sel_id = st.selectbox(
                        "Prédiction ID",
                        list(shap_pred_ids.keys()),
                        key="shap_pred_sel",
                    )
                    if st.button("🔍 Expliquer", key="shap_explain_btn"):
                        with st.spinner("Calcul de l'explication SHAP en cours…"):
                            try:
                                explanation = client.explain_prediction(int(shap_sel_id))
                            except Exception as exc:
                                st.error(f"Impossible de calculer l'explication : {exc}")
                                explanation = None

                        if explanation:
                            shap_values: dict = explanation.get("shap_values", {})
                            base_value: float = explanation.get("base_value", 0.0)
                            prediction = explanation.get("prediction")
                            model_type: str = explanation.get("model_type", "")

                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("Valeur de base E[f(X)]", f"{base_value:.4f}")
                            col_m2.metric("Prédiction finale", str(prediction))
                            col_m3.metric("Type de modèle", model_type)

                            if shap_values:
                                # Top 10 features par valeur absolue
                                sorted_features = sorted(
                                    shap_values.items(), key=lambda x: abs(x[1]), reverse=True
                                )[:10]
                                features_names = [f for f, _ in sorted_features]
                                shap_vals = [v for _, v in sorted_features]
                                colors = ["#e05252" if v >= 0 else "#5282e0" for v in shap_vals]

                                fig = go.Figure(
                                    go.Bar(
                                        x=shap_vals,
                                        y=features_names,
                                        orientation="h",
                                        marker_color=colors,
                                        text=[f"{v:+.4f}" for v in shap_vals],
                                        textposition="outside",
                                    )
                                )
                                fig.update_layout(
                                    title="Contributions SHAP (top 10 features)",
                                    xaxis_title="Contribution SHAP",
                                    yaxis={"autorange": "reversed"},
                                    height=max(300, len(sorted_features) * 40 + 100),
                                    margin={"l": 20, "r": 60, "t": 50, "b": 40},
                                    showlegend=False,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Tableau complet si plus de 10 features
                                if len(shap_values) > 10:
                                    with st.expander("Voir toutes les features"):
                                        all_sorted = sorted(
                                            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
                                        )
                                        st.dataframe(
                                            pd.DataFrame(all_sorted, columns=["Feature", "SHAP"]),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                            else:
                                st.info("Aucune valeur SHAP retournée pour cette prédiction.")

        # --- Pagination ---
        st.divider()
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.session_state["pred_offset"] > 0:
                if st.button("← Précédent", use_container_width=True):
                    st.session_state["pred_offset"] = max(
                        0, st.session_state["pred_offset"] - limit
                    )
                    st.rerun()
        with col_info:
            current_page = st.session_state["pred_offset"] // limit + 1
            total_pages = max(1, (total + limit - 1) // limit)
            st.caption(f"Page {current_page} / {total_pages}")
        with col_next:
            if st.session_state["pred_offset"] + limit < total:
                if st.button("Suivant →", use_container_width=True):
                    st.session_state["pred_offset"] += limit
                    st.rerun()

    # --- Maintenance RGPD (admin uniquement) ---
    if st.session_state.get("is_admin", False):
        st.divider()
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
                        st.success(f"✅ {result['deleted_count']} prédiction(s) supprimée(s).")
                        if result.get("linked_observed_results_count", 0) > 0:
                            st.warning(
                                f"{result['linked_observed_results_count']} résultat(s) observé(s) "
                                "liés ont été perdus."
                            )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Erreur lors de la purge : {exc}")

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
        st.markdown(
            """
Le fichier doit contenir **une colonne par feature** attendue par le modèle.
Les colonnes doivent correspondre exactement aux noms des features du modèle (casse incluse).

Une colonne `id_obs` optionnelle permet de tracer chaque ligne dans l'historique.

**Exemple pour un modèle Iris :**
"""
        )
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
        all_models = client.list_models()
        model_names_batch = sorted({m["name"] for m in all_models})
    except Exception:
        all_models = []
        model_names_batch = []

    if not model_names_batch:
        st.warning("Aucun modèle disponible. Créez d'abord un modèle via l'API.")
        st.stop()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        batch_search = st.text_input("Filtrer par nom", key="batch_model_search", placeholder="Rechercher…")
        batch_filtered = [n for n in model_names_batch if batch_search.lower() in n.lower()] if batch_search else model_names_batch
        batch_model = st.selectbox("Modèle cible", batch_filtered or model_names_batch, key="batch_model_sel")

    # Versions disponibles pour le modèle sélectionné
    batch_versions = ["(production / auto)"] + sorted(
        {m["version"] for m in all_models if m["name"] == batch_model},
        reverse=True,
    )
    batch_version_sel = col_b2.selectbox(
        "Version", batch_versions, key="batch_version_sel"
    )
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

        st.caption(f"Fichier chargé : **{batch_file.name}** — {len(df_input):,} lignes, {len(df_input.columns)} colonnes")
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

            st.success(
                f"✅ {len(predictions_out):,} prédictions générées — "
                f"modèle **{batch_model}** v{used_version}"
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
                        (p.get("probability") or [None] * n_classes)[i]
                        for p in predictions_out
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
