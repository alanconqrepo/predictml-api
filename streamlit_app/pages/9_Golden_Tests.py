"""
Validation des cas de référence (Golden Test Set) pour les modèles ML
"""

import json

import pandas as pd
import streamlit as st
from utils.api_client import get_golden_tests as get_golden_tests_cached
from utils.api_client import get_models as get_models_cached
from utils.auth import get_client, require_auth

st.set_page_config(page_title="Golden Tests — PredictML", page_icon="🧪", layout="wide")
require_auth()

is_admin = st.session_state.get("is_admin", False)

st.title("🧪 Golden Tests")
st.caption(
    "Validez qu'un modèle produit toujours les sorties attendues sur des cas de référence. "
    "Particulièrement utile après un ré-entraînement."
)

client = get_client()


def reload():
    st.cache_data.clear()
    st.rerun()


# ─── Chargement des modèles ───────────────────────────────────────────────────

try:
    models = get_models_cached(
        st.session_state.get("api_url"), st.session_state.get("api_token")
    )
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()

if not models:
    st.info("Aucun modèle disponible.")
    st.stop()

# ─── Section 1 — Sélection du modèle ─────────────────────────────────────────

st.subheader("1. Sélection du modèle")

model_names = sorted({m["name"] for m in models})
col_name, col_version = st.columns([2, 1])

with col_name:
    selected_name = st.selectbox("Modèle", model_names, key="gt_model_name")

versions = [m["version"] for m in models if m["name"] == selected_name]
with col_version:
    selected_version = st.selectbox("Version à tester", versions, key="gt_model_version")

# ─── Section 2 — Cas de tests existants ──────────────────────────────────────

st.markdown("---")
st.subheader("2. Cas de tests enregistrés")

try:
    golden_tests = get_golden_tests_cached(
        st.session_state.get("api_url"),
        st.session_state.get("api_token"),
        selected_name,
    )
except Exception as e:
    st.error(f"Impossible de charger les golden tests : {e}")
    golden_tests = []

if not golden_tests:
    st.info(
        f"Aucun cas de test enregistré pour **{selected_name}**. "
        "Utilisez le formulaire ci-dessous pour en ajouter."
    )
else:
    rows = []
    for t in golden_tests:
        rows.append(
            {
                "ID": t.get("id"),
                "Description": t.get("description") or "—",
                "Input features": json.dumps(t.get("input_features", {})),
                "Expected output": str(t.get("expected_output", "—")),
                "Date": (
                    pd.to_datetime(t["created_at"]).strftime("%Y-%m-%d %H:%M")
                    if t.get("created_at")
                    else "—"
                ),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"{len(golden_tests)} cas de test(s) enregistré(s).")

    col_run, col_spacer = st.columns([1, 3])
    with col_run:
        run_clicked = st.button(
            f"▶ Lancer tous les tests sur v{selected_version}",
            type="primary",
            key="run_tests_btn",
        )

    if is_admin:
        with st.expander("🗑️ Supprimer un cas de test", expanded=False):
            test_opts = {
                f"#{t['id']} — {t.get('description') or t.get('expected_output', '')}": t["id"]
                for t in golden_tests
            }
            to_delete_label = st.selectbox(
                "Cas de test à supprimer", list(test_opts.keys()), key="gt_delete_select"
            )
            if st.button("Supprimer", type="secondary", key="gt_delete_btn"):
                test_id = test_opts[to_delete_label]
                try:
                    client.delete_golden_test(selected_name, test_id)
                    st.toast(f"Cas de test #{test_id} supprimé.", icon="✅")
                    reload()
                except Exception as e:
                    st.toast(f"Erreur lors de la suppression : {e}", icon="❌")

    if run_clicked:
        with st.spinner(
            f"Exécution de {len(golden_tests)} test(s) sur {selected_name} v{selected_version}…"
        ):
            try:
                run_result = client.run_golden_tests(selected_name, selected_version)

                total = run_result.get("total_tests", len(golden_tests))
                passed = run_result.get("passed", 0)
                failed = run_result.get("failed", 0)
                pass_rate = run_result.get("pass_rate", 0.0)
                details = run_result.get("details", [])

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Total", total)
                col_m2.metric("✅ Passés", passed)
                col_m3.metric(
                    "❌ Échoués",
                    failed,
                    delta=f"-{failed}" if failed else None,
                    delta_color="inverse",
                )
                col_m4.metric("Taux de réussite", f"{pass_rate * 100:.1f}%")

                if not details:
                    st.info("Aucun détail retourné par l'API.")
                else:
                    st.markdown("#### Résultats détaillés")
                    for r in details:
                        test_passed = r.get("passed", False)
                        icon = "✅" if test_passed else "❌"
                        desc = r.get("description") or f"Test #{r.get('test_id')}"
                        label = f"{icon} {desc}"

                        with st.expander(label, expanded=not test_passed):
                            col_exp, col_rec = st.columns(2)
                            with col_exp:
                                st.markdown("**Attendu**")
                                st.code(str(r.get("expected", "—")))
                            with col_rec:
                                st.markdown("**Reçu**")
                                actual = str(r.get("actual", "—"))
                                if not test_passed:
                                    st.error(actual)
                                else:
                                    st.code(actual)

                            if not test_passed:
                                expected_val = str(r.get("expected", ""))
                                actual_val = str(r.get("actual", ""))
                                if expected_val != actual_val:
                                    st.markdown("**Diff**")
                                    st.markdown(
                                        f"`attendu` → **{expected_val}** | "
                                        f"`reçu` → **{actual_val}**"
                                    )

                            st.markdown("**Input utilisé**")
                            st.json(r.get("input", {}))

            except Exception as e:
                st.error(f"Erreur lors de l'exécution des tests : {e}")

# ─── Section 3 — Ajouter un cas de test (admin uniquement) ───────────────────

st.markdown("---")
st.subheader("3. Ajouter un cas de test")

if not is_admin:
    st.info("Section réservée aux administrateurs.")
else:
    with st.expander("➕ Nouveau cas de test", expanded=False):
        with st.form("add_golden_test_form"):
            description = st.text_input(
                "Description",
                placeholder="ex: iris setosa typique — toutes features nominales",
                key="gt_description",
            )

            features_json = st.text_area(
                "Features d'entrée (JSON)",
                value='{\n  "feature1": 1.0,\n  "feature2": 2.0\n}',
                height=130,
                key="gt_features",
                help="Objet JSON clé/valeur des features du modèle",
            )

            expected_output = st.text_input(
                "Sortie attendue",
                placeholder="ex: setosa  |  0  |  1.23",
                key="gt_expected",
                help="Valeur de sortie attendue du modèle pour ces features (texte libre)",
            )

            submitted = st.form_submit_button("💾 Enregistrer", type="primary")

        if submitted:
            try:
                features = json.loads(features_json)
            except json.JSONDecodeError as e:
                st.error(f"JSON invalide dans les features : {e}")
            else:
                if not expected_output.strip():
                    st.error("La sortie attendue est obligatoire.")
                elif not features:
                    st.error("Les features d'entrée ne peuvent pas être vides.")
                else:
                    try:
                        payload = {
                            "input_features": features,
                            "expected_output": expected_output.strip(),
                            "description": description.strip() or None,
                        }
                        client.create_golden_test(selected_name, payload)
                        st.toast("Cas de test enregistré avec succès.", icon="✅")
                        reload()
                    except Exception as e:
                        st.toast(f"Erreur lors de l'enregistrement : {e}", icon="❌")

    with st.expander("📥 Import CSV (lot de tests)", expanded=False):
        st.markdown("""
Importez plusieurs cas de test à la fois depuis un fichier CSV.

**Format attendu :**
```
description,input_features,expected_output
"iris setosa","{""sepal_length"": 5.1, ""sepal_width"": 3.5, ""petal_length"": 1.4, ""petal_width"": 0.2}",setosa
```

Colonnes : `description` (optionnel), `input_features` (JSON stringifié), `expected_output`.
""")
        uploaded_csv = st.file_uploader(
            "Fichier CSV", type=["csv"], key="gt_csv_upload"
        )
        if uploaded_csv is not None:
            if st.button("📤 Importer", type="primary", key="gt_csv_import_btn"):
                try:
                    result = client.upload_golden_tests_csv(
                        selected_name,
                        uploaded_csv.read(),
                        uploaded_csv.name,
                    )
                    imported = result.get("imported", result.get("count", "?"))
                    st.toast(f"{imported} cas de test importé(s).", icon="✅")
                    reload()
                except Exception as e:
                    st.toast(f"Erreur lors de l'import : {e}", icon="❌")
