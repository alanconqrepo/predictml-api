# Grafana & OpenTelemetry — Usage Guide

## Observability Architecture

PredictML uses the all-in-one **Grafana LGTM** stack (`grafana/otel-lgtm`) which includes:

| Component | Role | Internal Port |
|---|---|---|
| **Prometheus** | Scrapes API `/metrics` | 9090 |
| **Loki** | Structured log storage | 3100 |
| **Tempo** | Distributed trace storage | 3200 |
| **Grafana** | Visualisation — accessible UI | **3000** |
| **OTel Collector** | OTLP reception (traces + metrics + logs) | 4317 (gRPC), 4318 (HTTP) |

```
FastAPI API
  │
  ├─► /metrics (Prometheus scrape)  ──────────────► Prometheus ─► Grafana
  │
  └─► OTLP gRPC :4317 (if ENABLE_OTEL=true)
        ├─► Traces  ──► Tempo  ──► Grafana
        ├─► Metrics ──► Prometheus ─► Grafana
        └─► Logs ────► Loki ───► Grafana
```

## Accessing Grafana

```bash
# Start the stack
docker-compose up -d

# Grafana is available at
open http://localhost:3000

# Default credentials
# Login    : admin  (configurable via GRAFANA_ADMIN_USER in .env)
# Password : value of GRAFANA_ADMIN_PASSWORD in .env
```

## Pre-configured Dashboards

Both dashboards appear automatically at startup, in the **General** folder.

### 1. PredictML — API Overview

**File**: `monitoring/grafana/dashboards/api-overview.json`

| Panel | Metric | Description |
|---|---|---|
| Throughput (req/s) | `http_requests_total` | Number of requests per second (all routes) |
| 5xx error rate | `http_requests_total{status_code=~"5.."}` | Share of server errors |
| P95 latency | `http_request_duration_seconds_bucket` | 95th percentile of duration |
| Active routes | `http_requests_total` | Number of routes with traffic |
| Throughput by route | by `handler` | Time series per endpoint |
| P50/P95/P99 latency | histogram | Global latency percentiles |
| HTTP errors | 4xx + 5xx by route | Time-based error evolution |
| HTTP code distribution | donut | Share of each status code |
| Top endpoints | table | Most requested routes + latencies |
| RSS memory | `process_resident_memory_bytes` | Worker memory consumption |
| CPU | `process_cpu_seconds_total` | Worker CPU consumption |

> This dashboard works **immediately** — it does not require `ENABLE_OTEL=true`.

---

### 2. PredictML — Model Performance

**File**: `monitoring/grafana/dashboards/model-performance.json`

| Panel | Source | Description |
|---|---|---|
| /predict volume (req/s) | Prometheus | Prediction rate per second |
| /predict errors (%) | Prometheus | Share of 4xx/5xx on the predict endpoint |
| P95 /predict latency | Prometheus | ML inference latency |
| Prediction volume (chart) | Prometheus | Success vs client errors vs server errors |
| /predict P50/P95/P99 latency | Prometheus | Inference duration percentiles |
| /models errors | Prometheus | Errors on model management routes |
| Retrain events | **Loki** | Retrain logs (OTEL required) |
| Drift alerts | **Loki** | Drift detection logs (OTEL required) |
| Recent error logs | **Loki** | ERROR/WARNING lines (OTEL required) |

> **Loki panels** (logs) remain empty without `ENABLE_OTEL=true`.

---

## Enabling OpenTelemetry (traces + logs → Loki/Tempo)

Add to the `.env` file:

```bash
ENABLE_OTEL=true
OTEL_SERVICE_NAME=predictml-api          # service label in Loki/Tempo
OTEL_EXPORTER_OTLP_ENDPOINT=http://grafana:4317  # already default in Docker
```

Then restart the API:

```bash
docker-compose restart api
```

Once enabled:
- **Python logs** (structlog) are bridged to Loki via the OTLP Collector
- **Traces** from each FastAPI request and SQL query are sent to Tempo
- **OTEL metrics** are also exported (in addition to Prometheus)

---

## Reading Logs in Loki (Explore)

1. Go to **Grafana → Explore**
2. Select the **Loki** datasource
3. Example LogQL queries:

```logql
# All API logs
{service_name="predictml-api"}

# Error logs only
{service_name="predictml-api"} | json | level =~ "(?i)(error|critical)"

# Retrain events
{service_name="predictml-api"} |= "retrain" | json

# Predictions on a specific model
{service_name="predictml-api"} |= "predict" | json | model_name="iris"

# Drift detection
{service_name="predictml-api"} |= "drift" | json

# Supervision alerts
{service_name="predictml-api"} |= "alert" | json

# Slow requests (latency > 1 s)
{service_name="predictml-api"} | json | response_time > 1000
```

---

## Reading Traces in Tempo (Explore)

1. Go to **Grafana → Explore**
2. Select the **Tempo** datasource
3. Search by:
   - **Service**: `predictml-api`
   - **Span name**: `POST /predict`, `GET /models`, etc.
   - **Tag**: `http.status_code=500` to filter errors

FastAPI traces are automatically instrumented by `FastAPIInstrumentor`.  
Each request creates a root span with the tags `http.method`, `http.route`, `http.status_code`.  
SQL queries (SQLAlchemy) create child spans traced by `SQLAlchemyInstrumentor`.

### Traces → Logs Link

In Tempo, clicking a span opens a **"Logs for this span"** link that filters Loki  
on the corresponding `trace_id` — automatic request / logs correlation.

---

## Available Prometheus Metrics

### HTTP Metrics (prometheus-fastapi-instrumentator)

| Metric | Type | Labels | Description |
|---|---|---|---|
| `http_requests_total` | Counter | `handler`, `method`, `status_code` | Total number of requests |
| `http_request_duration_seconds` | Histogram | `handler`, `method`, `status_code` | Request duration |
| `http_request_duration_highr_seconds` | Histogram | `handler`, `method`, `status_code` | Duration (high resolution) |
| `http_request_size_bytes` | Histogram | `handler`, `method` | Request body size |
| `http_response_size_bytes` | Histogram | `handler`, `method` | Response body size |

### Python Process Metrics

| Metric | Description |
|---|---|
| `process_cpu_seconds_total` | CPU time consumed |
| `process_resident_memory_bytes` | RSS memory |
| `process_virtual_memory_bytes` | Virtual memory |
| `process_open_fds` | Open file descriptors |
| `process_start_time_seconds` | Startup timestamp |

### Useful PromQL Query Examples

```promql
# Global throughput in req/s
sum(rate(http_requests_total[5m]))

# 5xx error rate in %
100 * sum(rate(http_requests_total{status_code=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))

# Global P95 latency (ms)
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
) * 1000

# P95 latency for /predict only
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket{handler=~"/predict.*"}[5m])) by (le)
) * 1000

# Volume by endpoint
sum(rate(http_requests_total[5m])) by (handler)

# Errors by route
sum(rate(http_requests_total{status_code=~"[45].."}[5m])) by (handler, status_code)
```

---

## Securing the /metrics Endpoint

By default in development, `/metrics` is accessible without a token.  
In production, define `METRICS_TOKEN` in the `.env`:

```bash
METRICS_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Then update `monitoring/prometheus.yml` so Prometheus can authenticate:

```yaml
scrape_configs:
  - job_name: predictml-api
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
    authorization:
      credentials: "<value of METRICS_TOKEN>"
```

---

## Modifying or Adding a Dashboard

1. Edit a dashboard in the Grafana interface
2. Export via **Dashboard → Share → Export → Save to file**
3. Replace the corresponding `.json` file in `monitoring/grafana/dashboards/`
4. Grafana automatically reloads files every 30 seconds (configurable via `updateIntervalSeconds` in `dashboards.yaml`)

To create a new dashboard from scratch and make it persistent:

```bash
# 1. Create in Grafana UI, export as JSON
# 2. Place in monitoring/grafana/dashboards/my-dashboard.json
# 3. No restart needed — the provider scans the directory
```

---

## Troubleshooting

### Dashboards don't appear

```bash
# Check that the files are properly mounted in the container
docker exec predictml-grafana ls /etc/grafana/provisioning/dashboards/

# Check Grafana logs
docker-compose logs grafana | grep -i "dashboard\|provision\|error"
```

### Loki is empty

Check that `ENABLE_OTEL=true` is in the `.env` and that the API has restarted:

```bash
docker-compose logs api | grep -i "otel\|telemetry"
# Should display: "OpenTelemetry enabled — endpoint: http://grafana:4317"
```

### Prometheus is not scraping metrics

```bash
# Check that the API exposes /metrics
curl http://localhost:8000/metrics

# If METRICS_TOKEN is set
curl -H "Authorization: Bearer <METRICS_TOKEN>" http://localhost:8000/metrics

# Check in Grafana → Explore → Prometheus → Metrics browser
# Search for: http_requests_total
```

### Tempo is not receiving traces

```bash
# Check OTLP connectivity
docker-compose logs api | grep -i "otlp\|span\|trace"

# Test OTLP HTTP send
curl -v http://localhost:4318/v1/traces
```
