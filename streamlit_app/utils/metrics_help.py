# NOTE: This file is kept for backward compatibility with 10_Aide.py, which uses
# METRIC_HELP to build a KPIs section for the AI assistant system prompt (sent to
# an LLM — intentionally in French regardless of UI language).
#
# Pages 4_Stats, 6_AB_Testing and 7_Supervision should use t("metrics.<key>")
# directly instead of METRIC_HELP["<key>"], as those values now live in
# translations/fr.yaml and translations/en.yaml.

METRIC_HELP = {
    # Global supervision — KPIs
    "predictions_prod": (
        "Production predictions returned to clients (is_shadow=False) over the period. "
        "Does not include shadow executions."
    ),
    "predictions_shadow": (
        "Silent predictions computed in the background (is_shadow=True), "
        "not returned to the client.\n\n"
        "Used to compare new versions against production "
        "without impacting real traffic."
    ),
    "modeles_actifs": (
        "Number of distinct models with at least one active version (is_active=True) "
        "with predictions in the selected period.\n\n"
        "A model is 'active' as soon as one version is loadable, "
        "regardless of its deployment mode (Production, A/B, Shadow or no routing).\n\n"
        "Not to be confused with 'in production': an active model may only be "
        "in shadow or uploaded without being exposed to traffic."
    ),
    "alertes_sante": (
        "Number of models in alert status over the period.\n\n"
        "Each model's health status is the worst of 4 indicators:\n\n"
        "• Execution error rate\n"
        "  🟡 ≥ 5 %  ·  🔴 ≥ 10 %\n\n"
        "• Feature drift (Z-score + PSI + null rate)\n"
        "  Z-score 🟡 ≥ 2  · 🔴 ≥ 3\n"
        "  PSI 🟡 ≥ 0.1  ·  🔴 ≥ 0.2\n"
        "  Null rate 🟡 gap ≥ 5 pts  ·  🔴 ≥ 15 pts or > 30 %\n\n"
        "• Performance drift (accuracy 1st vs 2nd half of period)\n"
        "  🟡 drop ≥ 5 pts  ·  🔴 drop ≥ 10 pts\n\n"
        "• Output drift (PSI of prediction distribution)\n"
        "  🟡 PSI ≥ 0.1  ·  🔴 PSI ≥ 0.2\n\n"
        "The final status is the most severe among these four.\n"
        "⚪ no_data / no_baseline does not degrade the overall status."
    ),
    # Error / performance
    "taux_erreur": (
        "Proportion of requests that failed execution over the period.\n\n"
        "⚠️ This is NOT an ML quality indicator: a prediction can be "
        "correctly computed and returned to the client, yet be wrong from a business standpoint.\n\n"
        "An execution error means the API could not produce a prediction "
        "(server exception, model not loaded, timeout, invalid input…). "
        "Computed on production predictions only (shadow excluded).\n\n"
        "🟡 Warning: ≥ 5 %  ·  🔴 Critical: ≥ 10 %"
    ),
    "accuracy": "Proportion of correct predictions. 1.0 = 100 % correct predictions.",
    "auc": (
        "AUC-ROC (Area Under the ROC Curve): area under the Receiver Operating Characteristic curve.\n\n"
        "Measures the model's ability to distinguish classes, independently of the decision threshold.\n\n"
        "• 1.0 = perfect discriminator\n"
        "• 0.5 = no better than random (random classifier)\n"
        "• < 0.5 = worse than random\n\n"
        "Interpretation:\n"
        "• 🟢 ≥ 0.90 → excellent\n"
        "• 🟡 0.70–0.90 → good to acceptable\n"
        "• 🔴 < 0.70 → needs improvement\n\n"
        "Requires predicted probabilities (the model must output confidence scores, not just a class).\n"
        "Binary classification: standard AUC. Multiclass: OvR (One vs Rest) weighted."
    ),
    "mae": "Mean Absolute Error: average absolute difference between the prediction and the actual value. Lower is better.",
    "rmse": "Root Mean Square Error: root mean squared error. Penalises large errors more than MAE.",
    "r2": "Coefficient of determination: proportion of variance explained by the model. 1.0 = perfect model, 0 = null model, negative = worse than the mean.",
    "precision": "Of all positive predictions, the proportion that are truly positive. Evaluates the reliability of positive detections. (w.) = weighted average by class.",
    "recall": "Of all actual positive cases, the proportion correctly detected. Evaluates the ability not to miss positive cases. (w.) = weighted average by class.",
    "f1": "Harmonic mean of precision and recall. Useful when classes are imbalanced. (w.) = weighted average by class.",
    # Latency
    "latence_mediane": "Median response time: half of requests have a time below this value.",
    "latence_avg": "Average response time across all requests.",
    "latence_p95": "95th percentile of latency: 95 % of requests are processed in less than this time. Key production performance indicator.",
    # A/B Testing
    "concordance_shadow": "Agreement rate between the shadow model and the production model: proportion of cases where both models give the same prediction.",
    "p_value": "Probability of observing a difference this large if the two versions were equivalent. Below the threshold (e.g. 0.05) → statistically significant difference.",
    "niveau_confiance": "Confidence level of the test. 95 % means: we accept a 5 % risk of wrongly concluding there is a difference when there is none.",
    "test_statistique": "Statistical test used to compare versions. Chi-squared for error rates (proportions), Mann-Whitney U for latency distributions (continuous).",
    "metrique_analysee": "Metric on which the statistical test is performed: error rate or response time.",
    # Calibration — classification
    "brier_score": "Brier score: measures the accuracy of predicted probabilities. 0 = perfect (exact probabilities), 1 = worst case. Below 0.25 is generally acceptable.",
    "gap_confiance": "Gap between the average predicted confidence and the actual observed accuracy. Positive = overconfidence (model overestimates its certainty), negative = underconfidence. Ideally close to 0.",
    "statut_calibration": "Model calibration status: OK if the gap is small, Overconfident if the model overestimates its probabilities, Underconfident if it underestimates them.",
    # Calibration — regression
    "calib_mae": (
        "MAE (Mean Absolute Error): average absolute error between predicted ŷ and actual y.\n\n"
        "MAE = mean(|ŷ − y|)\n\n"
        "Lower is better. "
        "Same unit as the target variable."
    ),
    "calib_rmse": (
        "RMSE (Root Mean Square Error): square root of the mean squared error.\n\n"
        "RMSE = √(mean((ŷ − y)²))\n\n"
        "Penalises large errors more than MAE. "
        "Same unit as the target variable. Lower is better."
    ),
    "calib_r2": (
        "R² (coefficient of determination): proportion of variance explained by the model.\n\n"
        "R² = 1 − SS_res / SS_tot\n"
        "SS_res = Σ(ŷ − y)²,  SS_tot = Σ(y − ȳ)²\n\n"
        "1.0 = perfect model · 0 = no better than the mean · negative = worse than the mean. "
        "Ideally > 0.80."
    ),
    "calib_biais": (
        "Systematic model bias: mean of residuals (ŷ − y).\n\n"
        "Bias = mean(ŷ − y)\n\n"
        "Positive → the model overestimates on average.\n"
        "Negative → the model underestimates on average.\n"
        "Close to 0 → no systematic bias (good calibration).\n\n"
        "Status: 🟢 OK if |relative bias| < 10 % of the standard deviation of observed y, "
        "🟡 Overestimation or Underestimation otherwise."
    ),
    # Confidence
    "confiance_moyenne": "Average maximum probability returned by the model for its predictions. Indicates how 'confident' the model is on average.",
    "p25_confiance": "25th percentile of confidence: 25 % of predictions have a confidence below this value.",
    "p75_confiance": "75th percentile of confidence: 75 % of predictions have a confidence below this value.",
    # Trend
    "tendance_performance": "Comparison of performance between the 1st and 2nd half of the selected period. A drop indicates potential model drift.",
}
