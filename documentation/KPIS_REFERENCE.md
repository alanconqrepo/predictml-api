# KPIs and Metrics Reference — PredictML

This document explains each indicator displayed in the Streamlit dashboard and returned by the API.

---

## Performance Metrics (Classification)

### Accuracy (Overall Accuracy)
**What it is**: the proportion of correct predictions out of the total.

```
accuracy = (true positives + true negatives) / total
```

**Interpretation**:
- `1.0` = 100% correct predictions (perfect)
- `0.9` = 90% correct predictions
- Naive baseline (always predicting the majority class): proportion of the majority class

**When to act**: a drop of more than 5% compared to the baseline → investigate. Drop > 10% → retrain.

**Limitations**: misleading on imbalanced datasets (e.g. 98% "non-fraud" → 98% accuracy by always predicting "non-fraud").

---

### F1 Score
**What it is**: harmonic mean of precision and recall. Robust to class imbalance.

```
F1 = 2 × (precision × recall) / (precision + recall)
```

**Interpretation**:
- `1.0` = perfect precision and recall
- `0.8+` = good for most use cases
- More useful than accuracy when classes are imbalanced

**`f1_weighted`**: weighted average by class support — recommended as the primary metric.

---

### Precision
**What it is**: among positive predictions, what proportion is actually positive.

```
precision = true positives / (true positives + false positives)
```

**When to prefer**: when a false positive is costly (e.g. a fire alarm triggered incorrectly).

---

### Recall (Sensitivity)
**What it is**: among actual positive cases, what proportion is detected.

```
recall = true positives / (true positives + false negatives)
```

**When to prefer**: when a false negative is costly (e.g. undetected cancer).

---

### Error Rate
**What it is**: proportion of incorrect predictions.

```
error_rate = 1 - accuracy
```

**Interpretation**: `0.05` = 5% of predictions are incorrect.

---

## Performance Metrics (Regression)

### MAE (Mean Absolute Error)
**What it is**: average absolute difference between the prediction and the actual value.

```
MAE = mean(|y_pred - y_real|)
```

**Interpretation**: in the target variable's unit. An MAE of 500 on an apartment price = average error of 500.

---

### RMSE (Root Mean Square Error)
**What it is**: root mean square error — penalises large errors more heavily.

```
RMSE = sqrt(mean((y_pred - y_real)²))
```

**Interpretation**: always greater than or equal to MAE. If RMSE >> MAE, there are predictions that are very far off.

---

### R² (Coefficient of Determination)
**What it is**: proportion of the target variable's variance explained by the model.

```
R² = 1 - (SS_residual / SS_total)
```

**Interpretation**:
- `1.0` = perfect model (explains 100% of the variance)
- `0.0` = model equivalent to predicting the mean
- Negative = the model is worse than predicting the mean

---

## Latency Metrics

### Average Latency
**What it is**: average API response time for a prediction.

**Typical values**: 10–50 ms for a simple sklearn model. Higher if the model is large (RandomForest with 500 trees) or if Redis is cold.

---

### P95 Latency (95th Percentile)
**What it is**: 95% of predictions are processed in less than this duration.

**Why P95 rather than the mean**: the mean hides outliers. If P95 = 200ms, the slowest 5% of requests take more than 200ms.

**Recommended threshold**: set `max_latency_p95_ms` in your auto-promotion policy (e.g. 500ms).

---

### Median Latency (P50)
**What it is**: 50% of predictions are processed in less than this duration.

---

## Drift Metrics

### Z-Score (per feature)
**What it is**: the difference between the production mean value and the baseline, expressed in number of standard deviations.

```
Z-score = (mean_production - mean_baseline) / std_baseline
```

**Thresholds**:
- `|Z| < 2` → `ok`
- `2 ≤ |Z| < 3` → `warning` (moderate drift)
- `|Z| ≥ 3` → `critical` (strong drift)

**Interpretation**: a Z-score of 3 means the production distribution has shifted by 3 standard deviations — statistically very unlikely without a real change.

---

### PSI (Population Stability Index)
**What it is**: measures the distribution shift between two populations (training vs production).

```
PSI = Σ (% prod - % train) × ln(% prod / % train)
```

**Standard thresholds**:
- `PSI < 0.1` → `ok` (stable distribution)
- `0.1 ≤ PSI < 0.2` → `warning` (slight drift)
- `PSI ≥ 0.2` → `critical` (significant drift — retraining recommended)

---

### Null Rate (missing value rate)
**What it is**: proportion of recent predictions where this feature is `null` or absent.

**Alert threshold**: configurable per model in `alert_thresholds`. Default: `warning` if null rate > 5%, `critical` if > 10%.

---

### Output Drift (Label Shift)
**What it is**: drift in the distribution of predicted classes compared to the reference distribution (from `training_stats.label_distribution`).

**Calculation**: PSI applied to class distributions.

**Interpretation**: if your model was predicting 33% of each Iris class and is now predicting 80% "setosa", the production population has changed (or the model is biased).

---

## A/B Testing Metrics

### P-value
**What it is**: probability of observing as large a difference between versions if they were actually equivalent.

**Interpretation**:
- `p < 0.05` → statistically significant difference (95% confidence)
- `p < 0.01` → very strong evidence (99% confidence)
- `p ≥ 0.05` → not enough data or no real difference

**Warning**: do not promote a version simply because it appears better. Wait until `p < confidence_level`.

---

### Confidence Level
**What it is**: the chosen statistical significance threshold (default: 95%).

**Meaning**: with 95% confidence, we accept a 5% risk of incorrectly concluding there is a difference.

---

### Statistical Test Used
| Condition | Test | Metric tested |
|---|---|---|
| ≥ 1 error in either group | Chi-² (contingency table) | Error rate |
| 0 errors + response times available | Mann-Whitney U | Response time (ms) |

---

### Winner
**What it is**: the version with the best metric (lower error rate or lower latency), but only if `significant: true`.

**Never promote if `significant: false`** — more data needs to be accumulated.

---

### min_samples_needed
**What it is**: estimated number of observations needed per version to have 80% statistical power at the configured threshold.

**Interpretation**: if you have 150 observations but `min_samples_needed: 400`, wait before drawing conclusions.

---

### Concordance (shadow)
**What it is**: proportion of cases where the shadow model and the production model give the same prediction.

**Interpretation**:
- `> 95%` → the two models are very similar
- `80–95%` → notable differences, in-depth study recommended
- `< 80%` → very different behaviours

---

## Calibration Metrics

### Brier Score
**What it is**: measures the quality of predicted probabilities.

```
Brier = mean((p_pred - y_real)²)
```

**Interpretation**:
- `0` = perfect probabilities
- `0.25` = baseline (predicting 50% for everything)
- `< 0.1` = very well calibrated
- `0.1–0.25` = acceptable
- `> 0.25` = poor calibration

---

### Confidence Gap
**What it is**: the difference between the average predicted confidence and the actual observed accuracy.

```
gap = mean_predicted_confidence - actual_accuracy
```

**Interpretation**:
- `> 0` = **overconfidence**: the model overestimates its certainty (predicts 90% confidence but is only right 75% of the time)
- `< 0` = **underconfidence**: the model is too cautious
- `≈ 0` = well calibrated

---

### Calibration Status
| Status | Condition |
|---|---|
| `OK` | `|gap| < 0.05` |
| `Overconfident` | `gap > 0.05` |
| `Underconfident` | `gap < -0.05` |

---

## Confidence Metrics

### Average Confidence
**What it is**: mean `max(probabilities)` across recent predictions — how "sure" the model is on average.

---

### Confidence Distribution
**What it is**: histogram of the confidence level (`max(probabilities)`) of predictions.

**Use**: identify the proportion of uncertain predictions. If many predictions have confidence < 60%, refine your `confidence_threshold`.

---

## Trend Metrics

### Performance Trend
**What it is**: comparison of accuracy between the first and second half of the selected period.

**Interpretation**: a drop indicates potential model drift or a change in the population.

---

## Operational KPIs

### Prediction Volume
**What it is**: number of predictions made over a given period.

**Monitoring**: an unusual spike can indicate abuse (scraping, infinite loop); a drop can indicate an upstream outage.

---

### Daily Quota
**What it is**: number of predictions allowed per day per user (`rate_limit_per_day`).

**Behaviour**: if the quota is reached, the API returns HTTP 429. The quota resets at midnight UTC.

---

### Ground Truth Coverage
**What it is**: proportion of predictions that have an associated observed result.

```
coverage = nb_pairs(prediction, observed_result) / total_predictions
```

**Interpretation**: 100% coverage means you can calculate real performance across all predictions. Below 20%, real performance metrics are not reliable.

---

## Quick Reading Guide

| Metric | Good | Warning | Alert |
|---|---|---|---|
| Accuracy | > 0.9 | 0.7–0.9 | < 0.7 |
| F1 Score | > 0.85 | 0.6–0.85 | < 0.6 |
| Error Rate | < 0.05 | 0.05–0.15 | > 0.15 |
| P95 Latency | < 100ms | 100–500ms | > 500ms |
| Z-score drift | < 2 | 2–3 | > 3 |
| PSI | < 0.1 | 0.1–0.2 | > 0.2 |
| Brier Score | < 0.1 | 0.1–0.25 | > 0.25 |
| p-value A/B | < 0.05 = significant | — | > 0.05 = inconclusive |
