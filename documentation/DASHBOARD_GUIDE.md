# Streamlit Dashboard Guide — PredictML Admin

The dashboard is accessible at **http://localhost:8501**. Log in with your Bearer token.

---

## Login

1. Open http://localhost:8501
2. In the **API URL** field, leave `http://localhost:8000` (or replace with your server URL)
3. Paste your **Bearer token** (e.g. the default admin token `<ADMIN_TOKEN>`)
4. Click **Sign in**

The dashboard automatically detects whether your token is admin or regular user.

---

## Home Page

System status overview:
- **API Health**: status, latency, number of cached models
- **Key Metrics**: number of active models, predictions, users
- **Quick links** to all services (API Swagger, MLflow, MinIO, Grafana)

---

## 👥 Page 1 — Users (`/1_Users`)

**Accessible to admins only.**

### What you can do
- **Create a user**: fill in username, email, role (`admin`, `user`, `readonly`) and daily quota
- **View token**: "Show token" button with one-click copy
- **Renew a token**: "Regenerate" button (the old token is immediately invalidated)
- **Deactivate / reactivate** an account
- **View quotas**: daily consumption gauge per user
- **Usage statistics**: expander with predictions by model and by day

### Creation form fields
| Field | Description |
|---|---|
| Username | Unique identifier (no spaces) |
| Email | Unique email address |
| Role | `admin` = full access, `user` = predictions, `readonly` = read-only |
| Quota/day | Max number of predictions per day (default: 1000) |

---

## 🤖 Page 2 — Models (`/2_Models`)

### Available Tabs

#### Details
- Table of all versions with accuracy, F1, status (production/active/deprecated)
- **Set to production** button: select the version, confirm
- Download the `.joblib` file directly from MinIO
- Link to the associated MLflow run

#### Upload a Model
Multipart form to upload a `.joblib`:
- `.joblib` file (required)
- `train.py` (optional — enables retraining)
- Metadata: name, version, description, algorithm, accuracy, f1_score, features_count, classes

#### What-If Explorer
- Sliders for each known feature of the model
- Real-time prediction on each change
- History of tested combinations

#### Feature Importance (SHAP)
- Global feature importance over the N most recent predictions
- Bar chart of `mean(|SHAP|)` per feature

#### Schema Validation
- Test a features JSON against the expected schema of the model
- Returns: missing features, unexpected features, type coercions

#### A/B Comparison
- Side-by-side performance table for each version
- Statistical test result (Chi-² or Mann-Whitney U, p-value, winner)

#### Golden Tests
- View test cases associated with the model
- Run golden tests on a specific version (PASS/FAIL per case)

---

## 📊 Page 3 — Predictions (`/3_Predictions`)

### History Tab
- Filters: model, start/end date, status, version
- Paginated table of predictions with detailed features
- Click a row to submit an observed result (feedback)
- CSV/JSONL/Parquet export of filtered predictions

### Batch Tab
- Submit multiple predictions as JSON
- Results displayed with probabilities
- Import/export of observed results as CSV

### GDPR Purge
- Simulate (`dry_run=true`) then confirm deletion of old predictions
- Parameters: number of retention days, target model
- Shows how many rows would be deleted before confirmation

---

## 📈 Page 4 — Stats (`/4_Stats`)

### Global Metrics
- **Prediction volume** by hour/day (time series chart)
- **Error rate** with alert if above threshold
- **Average and P95 latency** by model

### Leaderboard
Ranking of production models by:
- Accuracy, F1 Score, P95 latency, prediction volume

### Scatter Plot
Accuracy vs P95 Latency to identify performance/speed trade-offs.

### Prediction Distribution
- Histogram by predicted class
- Confidence distribution (useful for adjusting `confidence_threshold`)

---

## 💡 Page 5 — Code Example (`/5_Code_Example`)

Dynamically generated code examples with your session URL and token:
- **Python**: train with MLflow, upload, predict, observed results
- **curl / bash**: upload, predict, history, observed result
- **JavaScript**: same workflow with `fetch()`

---

## 🔬 Page 6 — A/B Testing (`/6_AB_Testing`)

### Configuration
- Select a model and its versions
- Define the `deployment_mode`: `production`, `ab_test`, `shadow`
- Adjust the `traffic_weight` (0–100% for A/B test versions)

### Results
- Comparative table: predictions, errors, latency, concordance rate (shadow)
- **Statistical test**: p-value, significance threshold, winner
- **Promote to production** button if the winner is identified

### Deployment Modes
| Mode | Behaviour |
|---|---|
| `production` | Receives 100% of traffic (default) |
| `ab_test` | Receives `traffic_weight`% of real traffic |
| `shadow` | Receives all requests in the background, result not returned |

---

## 🔍 Page 7 — Supervision (`/7_Supervision`)

### Global Dashboard
- Health status of each production model
- Active alerts (drift, error rate, latency)

### Drift Detection
For each feature:
- **Z-score**: deviation in number of standard deviations from the baseline
- **PSI** (Population Stability Index): measures distribution shift
- **Null rate**: rate of missing values
- Status: `ok`, `warning`, `critical`, `no_baseline`

### Output Drift (Label Shift)
- Compares the distribution of predicted classes to the reference distribution
- Useful for detecting a semantic shift in the population

### Threshold Configuration
Alert thresholds configurable per model (overrides global values).

### Export
Report exportable as CSV or Markdown to share with the team.

---

## 🔄 Page 8 — Retrain (`/8_Retrain`)

### Manual Retraining
1. Select the model and source version
2. Enter the training date range
3. Enter the new version number
4. Optional: check "Set to production automatically"
5. Click **Launch retrain**
6. stdout/stderr logs are displayed in real time

### Automatic Schedule (cron)
- Configure a cron expression (e.g. `0 3 * * 1` = every Monday at 3:00 UTC)
- `lookback_days`: time window for training data
- `auto_promote`: promote automatically if the auto-promotion policy is defined
- Enable/disable without losing the configuration

### Auto-Promotion Policy
Define the criteria for a new version to be automatically promoted:
- `min_accuracy`: minimum required accuracy
- `max_latency_p95_ms`: maximum P95 latency
- `min_sample_validation`: minimum number of validation observations
- `min_golden_test_pass_rate`: minimum golden test pass rate

### Retrain History
Chronological table: source version → new version, before/after metrics, auto-promoted or not.

---

## 🧪 Page 9 — Golden Tests (`/9_Golden_Tests`)

### Manage Test Cases
- **Create**: add a case with input features + expected output + description
- **CSV Import**: columns `input_features` (JSON), `expected_output`, `description`
- **Delete** existing cases

### Run Tests
1. Select the model and version to test
2. Click **Run tests**
3. Each case is marked **PASS** or **FAIL** with expected/received diff

### Integration with Auto-Promotion
Configure `min_golden_test_pass_rate` in the auto-promotion policy to block promotion if too many tests fail.

---

## 💬 Page 10 — Help & AI Assistant (`/10_Aide`)

This page. Ask your questions to the Claude chatbot specialised in PredictML.

---

## Dashboard Usage Tips

### Quick Navigation
- The Streamlit sidebar lists all pages
- The home page shows a summary table with direct links

### Data Refresh
- Much of the data is cached (TTL 30s)
- Click **🔄 Refresh** on pages that offer it to force a reload

### Common Errors
| Message | Cause | Solution |
|---|---|---|
| "Access restricted to administrators" | Non-admin token | Use the admin token |
| "API connection error" | API not started | `docker-compose up -d api` |
| "No model available" | Empty DB | Upload a model first |
| "Invalid token" | Expired or incorrect token | Log in again |
