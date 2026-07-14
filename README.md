# 🚀 InsightForge-AI

**Enterprise-grade AI Data Analytics Agent Platform.**

Upload _any_ dataset — CSV, Excel, JSON, Parquet, SQLite, PostgreSQL, or MySQL — and get
instant, agent-powered analytics: automatic schema detection, data quality scoring, statistics,
correlation & anomaly analysis, an auto-built dashboard, forecasts, natural-language querying,
and one-click exports to PDF / Excel / Markdown / PowerPoint. No predefined schema. No column-name
assumptions. Works on data it has never seen before.

```
Health Score: 🟢 96.4/100     Rows: 1,510     Columns: 11     LLM: deterministic-fallback (offline mode)
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Installation (from an empty computer)](#4-installation-from-an-empty-computer)
5. [Running the Backend](#5-running-the-backend)
6. [Running the Streamlit Frontend](#6-running-the-streamlit-frontend)
7. [Running the Tests](#7-running-the-tests)
8. [Docker](#8-docker)
9. [Environment Variables](#9-environment-variables)
10. [Using Your Own Data](#10-using-your-own-data)
11. [Screenshots](#11-screenshots)
12. [Extending the Platform](#12-extending-the-platform)
    - [Adding a New Agent](#adding-a-new-agent)
    - [Adding a New Chart Type](#adding-a-new-chart-type)
    - [Adding a New LLM Provider](#adding-a-new-llm-provider)
13. [Deployment](#13-deployment)
14. [Troubleshooting & Common Errors](#14-troubleshooting--common-errors)
15. [FAQ](#15-faq)
16. [Performance Optimization](#16-performance-optimization)
17. [Future Roadmap](#17-future-roadmap)
18. [Contributing](#18-contributing)
19. [License](#19-license)

---

## 1. Project Overview

InsightForge-AI is a **data-independent** analytics platform built around a pipeline of small,
single-responsibility **agents** (24 of them — see below), each doing one job well: loading data,
detecting schema, cleaning, profiling, computing statistics, finding correlations and anomalies,
recommending and rendering charts, forecasting, generating insights and recommendations, and
exporting reports.

**Nothing in the codebase depends on specific column names.** Every agent works off the
_semantic type_ of a column (numeric, categorical, datetime, currency, email, phone, identifier,
location, boolean, text) rather than what it happens to be called, so the exact same pipeline
runs unmodified on a sales ledger, an HR roster, or a scientific dataset.

**LLMs are optional, not required.** Every agent that would benefit from an LLM (insights,
recommendations, executive summaries, natural-language querying) has a **deterministic,
rule-based fallback** that runs with zero configuration. If you plug in an API key, those same
agents produce richer, LLM-generated narratives layered _on top of_ — not instead of — the
deterministic output, so results stay reproducible either way.

**Key capabilities:**

| Capability                | Details                                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| Data sources              | CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, SQLite, PostgreSQL, MySQL                             |
| Schema detection          | dtype + 10 semantic types, all without hard-coded column names                                     |
| Data quality              | Health score, missing values, duplicates, constant/high-cardinality columns                        |
| Statistics                | Descriptive stats, distribution shape, normality testing                                           |
| Correlation               | Pearson correlation matrix + ranked strongest relationships                                        |
| Outliers & anomalies      | IQR/Z-score (per-column) + Isolation Forest (multivariate)                                         |
| Charts                    | 17 Plotly chart types, auto-recommended from column semantics                                      |
| Forecasting               | Holt-Winters seasonal / linear trend, auto seasonality detection                                   |
| Natural language querying | Safe, whitelisted intent execution — never `eval()`s LLM output                                    |
| Reports                   | PDF, Excel, Markdown, PowerPoint, CSV — one click each                                             |
| LLM providers             | OpenAI, Azure OpenAI, Anthropic, Gemini, Ollama, LM Studio, OpenRouter, DeepSeek, Mistral, or none |
| Frontend                  | 100% Streamlit — no React/Vue/Next.js/Flask templates anywhere                                     |
| Backend                   | FastAPI REST API (optional companion to the Streamlit UI)                                          |

---

## 2. Architecture

```
                              ┌──────────────────────────────┐
                              │        Data Sources          │
                              │  CSV · Excel · JSON · Parquet │
                              │  SQLite · PostgreSQL · MySQL  │
                              └───────────────┬───────────────┘
                                              │
                                   DataLoaderAgent
                                              │
                                              ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                         app/agents/orchestrator.py                │
        │            (sequential pipeline — LangGraph-portable nodes)       │
        │                                                                    │
        │  SchemaDetection → DataQuality → DataCleaning → Profiling         │
        │        → Statistics → Correlation → OutlierDetection              │
        │        → AnomalyDetection → ChartRecommendation → KPI             │
        │        → Insight → Recommendation → BusinessAnalyst               │
        │        → ExecutiveSummary → DashboardBuilder → Report             │
        └───────────────────────────────┬────────────────────────────────────┘
                                         │
                     ┌───────────────────┼────────────────────┐
                     ▼                                        ▼
        ┌────────────────────────┐              ┌───────────────────────────┐
        │   Streamlit Frontend    │              │     FastAPI Backend       │
        │  (frontend/streamlit_   │              │     (backend/main.py)     │
        │   app.py) — PRIMARY UI  │              │  Optional REST API layer  │
        │  Uses app/ directly —   │              │  for external/programmatic│
        │  works fully standalone │              │  integrations             │
        └────────────────────────┘              └───────────────────────────┘
                     │                                        │
                     └───────────────────┬────────────────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │   app/llm/llm_factory.py   │
                          │  LLMRouter — tries the      │
                          │  configured provider, NEVER │
                          │  raises; returns None on    │
                          │  any failure so agents fall │
                          │  back to deterministic logic│
                          └───────────────────────────┘
```

### Why the frontend doesn't call the backend over HTTP

The Streamlit app imports `app/agents/orchestrator.py` directly rather than calling the FastAPI
backend. This is an intentional design choice: it means **the primary UI works with zero extra
moving parts** — no need to have a backend process running in another terminal just to use the
app. The FastAPI backend shares the exact same `app/` core library and exists as an **optional**
integration surface for teams that want to call InsightForge-AI's analytics from another service,
a script, or a CI job. Both layers are always in sync because they import the same agents.

### The 24 Agents

| #   | Agent                | File                            | Responsibility                                                      |
| --- | -------------------- | ------------------------------- | ------------------------------------------------------------------- |
| 1   | Data Loader          | `data_loader_agent.py`          | Reads CSV/Excel/JSON/Parquet/SQLite/Postgres/MySQL into a DataFrame |
| 2   | Schema Detection     | `schema_detection_agent.py`     | Infers dtype + semantic type per column                             |
| 3   | Data Quality         | `data_quality_agent.py`         | Health score, missing values, duplicates, warnings                  |
| 4   | Data Cleaning        | `data_cleaning_agent.py`        | Trims strings, dedups, imputes missing values                       |
| 5   | Profiling            | `profiling_agent.py`            | Per-column counts, top values, numeric summaries                    |
| 6   | Statistics           | `statistics_agent.py`           | Descriptive stats, skew/kurtosis, normality                         |
| 7   | Correlation          | `correlation_agent.py`          | Correlation matrix + ranked strongest pairs                         |
| 8   | Outlier Detection    | `outlier_detection_agent.py`    | Per-column IQR/Z-score outliers                                     |
| 9   | Anomaly Detection    | `anomaly_detection_agent.py`    | Multivariate anomalies via Isolation Forest                         |
| 10  | Feature Engineering  | `feature_engineering_agent.py`  | Suggests/generates derived features                                 |
| 11  | Chart Recommendation | `chart_recommendation_agent.py` | Decides which chart types fit the data                              |
| 12  | Visualization        | `visualization_agent.py`        | Renders 17 Plotly chart types                                       |
| 13  | KPI                  | `kpi_agent.py`                  | Headline totals/averages/growth cards                               |
| 14  | Forecast             | `forecast_agent.py`             | Time-series forecasting with seasonality                            |
| 15  | Insight              | `insight_agent.py`              | Plain-English insights (rule-based + optional LLM)                  |
| 16  | Recommendation       | `recommendation_agent.py`       | Prioritized, actionable recommendations                             |
| 17  | Root Cause Analysis  | `root_cause_agent.py`           | Ranks candidate drivers behind a target metric                      |
| 18  | Business Analyst     | `business_analyst_agent.py`     | Business-readable narrative synthesis                               |
| 19  | Executive Summary    | `executive_summary_agent.py`    | 4-5 bullet C-suite summary                                          |
| 20  | Query                | `query_agent.py`                | Natural-language querying (safe intent executor)                    |
| 21  | Report               | `report_agent.py`               | Assembles all outputs into one report structure                     |
| 22  | Export               | `export_agent.py`               | PDF / Excel / Markdown / PPTX / CSV generation                      |
| 23  | Dashboard Builder    | `dashboard_builder_agent.py`    | Lays out KPIs + charts into a grid                                  |
| 24  | Orchestrator         | `orchestrator.py`               | Runs every agent above in sequence, logs each step                  |

Every agent extends `BaseAgent` (`app/agents/base_agent.py`), which gives it, for free:

- A uniform `.run(**kwargs) -> AgentResult` interface
- Automatic timing and structured logging
- Exception containment (one agent failing never crashes the pipeline — see the **Agent Log**
  tab in the dashboard)

---

## 3. Folder Structure

```
InsightForge-AI/
├── app/                          # Shared core library (used by BOTH backend and frontend)
│   ├── agents/                   # All 24 agents + orchestrator + optional LangGraph adapter
│   ├── llm/                      # Pluggable LLM providers (OpenAI, Anthropic, Gemini, Ollama...)
│   ├── models/                   # Pydantic schemas shared across the API boundary
│   ├── utils/                    # Logging, file I/O, database connection helpers
│   └── config.py                 # Central environment-driven settings
├── backend/                      # Optional FastAPI REST API layer
│   ├── api/routes.py             # /upload, /analyze, /charts, /query, /export, etc.
│   ├── core/                     # Dataset store, config/logging re-exports
│   └── main.py                   # FastAPI app entry point
├── frontend/                     # Streamlit dashboard (the PRIMARY user interface)
│   ├── components/               # One file per tab (Overview, Quality, Forecast, ...)
│   ├── assets/style.css          # Enterprise dashboard theme
│   └── streamlit_app.py          # Entry point — `streamlit run frontend/streamlit_app.py`
├── data/
│   ├── samples/                  # Ready-to-use demo datasets (sales, employees)
│   └── cache/                    # Runtime cache (gitignored)
├── docker/                       # Dockerfile.backend, Dockerfile.frontend, docker-compose.yml
├── scripts/                      # run_backend.sh, run_frontend.sh, run_tests.sh, data generator
├── tests/                        # pytest unit + integration tests (48+ tests)
├── .github/workflows/ci.yml      # CI: install → generate samples → test → docker build
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md                     # You are here
```

---

## 4. Installation (from an empty computer)

These steps assume a fresh machine with nothing installed beyond a terminal.

### Step 1 — Install Python 3.10 or 3.11

InsightForge-AI is tested on **Python 3.10 and 3.11** (3.12 also works). Check what you have:

```bash
python3 --version
```

If you don't have Python, install it:

- **macOS**: `brew install python@3.11`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip`
- **Windows**: download from [python.org/downloads](https://www.python.org/downloads/) and check
  "Add Python to PATH" during install.

### Step 2 — Clone the repository

```bash
git clone https://github.com/avviiiral/InsightForge_AI.git
cd insightforge-ai
```

(If you received this project as a ZIP instead of via git, just unzip it and `cd` into the
extracted folder.)

### Step 3 — Create and activate a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

Your terminal prompt should now be prefixed with `(venv)`.

### Step 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit, FastAPI, pandas, scikit-learn, Plotly, statsmodels, reportlab,
python-pptx, and every other dependency (~35 packages). It typically takes 1-3 minutes.

### Step 5 — Create your `.env` file (optional)

```bash
# macOS / Linux
cp .env.example .env

# Windows
copy .env.example .env
```

You do **not** need to edit `.env` to run the platform — it works fully offline with
deterministic analytics out of the box. Only edit it if you want to plug in an LLM (see
[Environment Variables](#9-environment-variables)).

### Step 6 — Generate the sample datasets

```bash
python scripts/generate_sample_data.py
```

This writes `data/samples/sales_sample.csv` (1,500 rows) and `data/samples/employees_sample.csv`
(600 rows) — both with realistic, deliberately-imperfect data (some missing values, duplicates,
and outliers) so you can see every feature in action immediately.

### Step 7 — Run the app

```bash
streamlit run frontend/streamlit_app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) — open it in your browser.

### Step 8 — Verify installation

In the sidebar: choose **Sample Dataset** → pick `sales_sample.csv` → click **▶ Run Full
Analysis**. Within a few seconds you should see a health score, KPI cards, and an auto-built
dashboard. If you see that, installation succeeded. 🎉

---

## 5. Running the Backend

The FastAPI backend is an **optional** companion for programmatic access (see
[Architecture](#2-architecture) for why it's optional).

```bash
# Easiest:
./scripts/run_backend.sh

# Or directly:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then visit:

- **API docs (Swagger UI)**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/api/v1/health

Example end-to-end flow with `curl`:

```bash
# 1. Upload a dataset
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@data/samples/sales_sample.csv"
# → {"dataset_id": "...", "filename": "sales_sample.csv", "n_rows": 1510, ...}

# 2. Run the full analysis pipeline
curl -X POST "http://localhost:8000/api/v1/datasets/<dataset_id>/analyze"

# 3. Fetch results
curl "http://localhost:8000/api/v1/datasets/<dataset_id>/quality"
curl "http://localhost:8000/api/v1/datasets/<dataset_id>/kpis"
curl "http://localhost:8000/api/v1/datasets/<dataset_id>/charts"

# 4. Ask a question
curl -X POST "http://localhost:8000/api/v1/datasets/<dataset_id>/query" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "<dataset_id>", "question": "top 5 by revenue"}'

# 5. Export a report
curl -X POST "http://localhost:8000/api/v1/datasets/<dataset_id>/export?export_format=pdf" \
  --output report.pdf
```

---

## 6. Running the Streamlit Frontend

```bash
# Easiest:
./scripts/run_frontend.sh

# Or directly:
streamlit run frontend/streamlit_app.py
```

The dashboard has 10 tabs:

1. **📌 Overview** — KPIs + the auto-built dashboard
2. **🩺 Data Quality** — health score, missing values, duplicates, cleaning log
3. **🔬 Profiling** — expandable per-column deep dive
4. **📊 Visualizations** — the full recommended chart gallery + a manual chart builder
5. **🔗 Correlation & Anomalies** — heatmap, top relationships, outliers, Isolation Forest anomalies, root-cause explorer
6. **📈 Forecast** — pick a date + value column, get a Holt-Winters/linear forecast
7. **💡 Insights & Recommendations** — executive summary, business narrative, insight feed, recommendations
8. **💬 Ask Your Data** — chat-style natural-language querying
9. **📤 Export** — one-click PDF / Excel / Markdown / PowerPoint / CSV downloads
10. **🧠 Agent Log** — full transparency into every agent's success/failure and timing

---

## 7. Running the Tests

```bash
# Easiest:
./scripts/run_tests.sh

# Or directly:
pytest tests/ -v

# With coverage:
pytest tests/ -v --cov=app --cov=backend --cov-report=term-missing
```

The suite includes 48+ tests covering every agent individually, the full orchestrated pipeline
end-to-end, every export format, every Plotly chart type, and the FastAPI backend (via
`TestClient`, no live server required). All tests use synthetic in-memory fixtures — no network
access or external services required, and none of them depend on an LLM being configured.

---

## 8. Docker

### Build and run both services

```bash
docker compose -f docker/docker-compose.yml up --build
```

This builds and starts:

- **Backend** at http://localhost:8000
- **Frontend** at http://localhost:8501

### Build/run individually

```bash
# Backend
docker build -f docker/Dockerfile.backend -t insightforge-backend .
docker run -p 8000:8000 --env-file .env insightforge-backend

# Frontend
docker build -f docker/Dockerfile.frontend -t insightforge-frontend .
docker run -p 8501:8501 --env-file .env insightforge-frontend
```

Both images are based on `python:3.11-slim`, install only the system libraries needed for
scipy/pandas/reportlab wheels, and include a `HEALTHCHECK` directive.

---

## 9. Environment Variables

All configuration lives in `.env` (copy from `.env.example`). **Every variable is optional** —
the platform runs fully offline with zero configuration.

| Variable                                  | Default                    | Purpose                                                                                          |
| ----------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------ |
| `APP_ENV`                                 | `development`              | `development` / `production`                                                                     |
| `LOG_LEVEL`                               | `INFO`                     | Loguru log level                                                                                 |
| `BACKEND_HOST` / `BACKEND_PORT`           | `0.0.0.0` / `8000`         | FastAPI bind address                                                                             |
| `LLM_PROVIDER`                            | `none`                     | `none, openai, azure_openai, anthropic, gemini, ollama, lmstudio, openrouter, deepseek, mistral` |
| `OPENAI_API_KEY` / `OPENAI_MODEL`         | —                          | OpenAI credentials                                                                               |
| `AZURE_OPENAI_*`                          | —                          | Azure OpenAI endpoint, deployment, API version                                                   |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL`   | —                          | Anthropic (Claude) credentials                                                                   |
| `GEMINI_API_KEY` / `GEMINI_MODEL`         | —                          | Google Gemini credentials                                                                        |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL`        | `http://localhost:11434`   | Local Ollama daemon                                                                              |
| `LMSTUDIO_BASE_URL` / `LMSTUDIO_MODEL`    | `http://localhost:1234/v1` | Local LM Studio server                                                                           |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | —                          | OpenRouter credentials                                                                           |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL`     | —                          | DeepSeek credentials                                                                             |
| `MISTRAL_API_KEY` / `MISTRAL_MODEL`       | —                          | Mistral credentials                                                                              |
| `POSTGRES_URI` / `MYSQL_URI`              | —                          | Example connection strings for the DB source tab                                                 |

**If `LLM_PROVIDER=none` (or is left unset), or if the configured provider fails for any
reason** (missing key, network error, rate limit), every LLM-touching agent transparently falls
back to its deterministic logic. You will never see a crash from a missing API key.

---

## 10. Using Your Own Data

From the sidebar, choose one of three sources:

- **Upload File** — CSV, TSV, Excel (multi-sheet supported), JSON, Parquet, or SQLite
  (multi-table supported — you'll be prompted to pick a sheet/table if there's more than one).
- **Sample Dataset** — the two bundled demo datasets.
- **Database Connection** — enter a PostgreSQL or MySQL SQLAlchemy URI
  (e.g. `postgresql+psycopg2://user:pass@host:5432/db`), then either pick a table or run a
  custom **read-only** `SELECT` query. Non-`SELECT` statements are rejected before they ever
  reach the database.

There is **no schema configuration step** — upload anything and click **Run Full Analysis**.

---

## 11. Screenshots

> Screenshots are not bundled in this repository to keep it lightweight. After running the app
> locally (`streamlit run frontend/streamlit_app.py`), you can generate your own by taking a
> browser screenshot of each tab and dropping the images here:
>
> - `docs/screenshots/overview.png`
> - `docs/screenshots/data_quality.png`
> - `docs/screenshots/visualizations.png`
> - `docs/screenshots/forecast.png`
> - `docs/screenshots/ask_your_data.png`
>
> Then reference them in this section, e.g. `![Overview](docs/screenshots/overview.png)`.

---

## 12. Extending the Platform

### Adding a New Agent

1. Create `app/agents/my_new_agent.py`:

```python
from typing import Any
import pandas as pd
from app.agents.base_agent import BaseAgent


class MyNewAgent(BaseAgent):
    name = "my_new_agent"
    description = "One sentence describing what this agent does."

    def _execute(self, dataframe: pd.DataFrame, **kwargs) -> dict[str, Any]:
        # Your logic here — return a JSON-serializable dict.
        return {"result": "..."}
```

2. Wire it into the pipeline in `app/agents/orchestrator.py` (add it to `__init__` and add a
   `self._step(...)` call in `run_full_pipeline`), **or** call it standalone from a Streamlit tab
   or a new FastAPI route — every agent works independently via `agent.run(**kwargs)`.
3. Add a test in `tests/test_my_new_agent.py` following the pattern of the existing agent tests.

Because every agent extends `BaseAgent`, you get timing, logging, and exception containment for
free — a bug in your new agent will show up as a red row in the **Agent Log** tab instead of
crashing the whole dashboard.

### Adding a New Chart Type

1. Add a `build_<name>()` method to `VisualizationAgent` in `app/agents/visualization_agent.py`
   that returns a `plotly.graph_objects.Figure`.
2. Register it in the `DISPATCH` dict at the top of the class.
3. Optionally teach `ChartRecommendationAgent` when to recommend it (in
   `app/agents/chart_recommendation_agent.py`).
4. Add it to `_CHART_OPTIONS` in `frontend/components/visualization_tab.py` so it's selectable in
   the manual chart builder.

### Adding a New LLM Provider

1. Create `app/llm/my_provider.py` implementing the `LLMProvider` interface
   (`app/llm/base_provider.py`) — just one method, `complete(system_prompt, user_prompt,
max_tokens)`, that returns a string or raises `LLMProviderError`.
2. Register it in `_PROVIDER_BUILDERS` in `app/llm/llm_factory.py`.
3. Add its config fields to `Settings` in `app/config.py` and to `.env.example`.

No agent code needs to change — every agent depends only on `LLMRouter.generate()`, which is
provider-agnostic.

---

## 13. Deployment

### Single VM / bare metal

Use `scripts/run_backend.sh` and `scripts/run_frontend.sh` behind a process manager
(systemd, supervisor, or `pm2`) and put Nginx or Caddy in front for TLS termination.

### Docker / container platforms

The provided `docker/Dockerfile.backend` and `docker/Dockerfile.frontend` work as-is on any
container platform (AWS ECS/Fargate, Google Cloud Run, Azure Container Apps, Kubernetes,
Railway, Render, Fly.io). Use `docker/docker-compose.yml` as a reference for wiring the two
services together, and mount `/app/data` to persistent storage if you want sample/cache data to
survive restarts.

### Streamlit Community Cloud

You can deploy `frontend/streamlit_app.py` directly to Streamlit Community Cloud — set
`requirements.txt` as the dependency file and add any LLM secrets via the platform's "Secrets"
UI (they're read the same way as `.env` variables through `pydantic-settings`).

### Scaling notes

- The FastAPI backend's `DatasetStore` (`backend/core/dataset_store.py`) is an in-memory,
  single-process store by design — swap it for Redis or a database-backed store before running
  multiple backend replicas behind a load balancer.
- `AnomalyDetectionAgent` (Isolation Forest) is the most CPU-intensive agent on large datasets;
  see [Performance Optimization](#16-performance-optimization) below.

---

## 14. Troubleshooting & Common Errors

| Symptom                                                                               | Cause                                                                               | Fix                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'app'`                                          | Streamlit launched from the wrong directory, or PYTHONPATH not set                  | Always run `streamlit run frontend/streamlit_app.py` from the project root; the entry point auto-adds the root to `sys.path`, but the command itself must be run from there                                            |
| `pip install` fails on `psycopg2-binary`                                              | Missing PostgreSQL system headers                                                   | Linux: `sudo apt install libpq-dev`; macOS: `brew install postgresql`                                                                                                                                                  |
| `ResolutionImpossible` during `pip install`                                           | A dependency version conflicts with something already installed in your environment | Use a clean virtual environment (Step 3 in Installation) — don't install into a shared/global Python                                                                                                                   |
| Excel upload fails with a sheet error                                                 | Multi-sheet workbook                                                                | The sidebar will prompt you to pick a sheet — if it doesn't, verify the file isn't password-protected                                                                                                                  |
| SQLite upload fails with a table error                                                | Multi-table database file                                                           | The sidebar will prompt you to pick a table                                                                                                                                                                            |
| "Need at least 2 numeric columns" on Correlation tab                                  | Dataset has 0-1 numeric columns                                                     | Expected behavior — correlation requires ≥2 numeric columns                                                                                                                                                            |
| Forecast tab shows "No datetime column was detected"                                  | `SchemaDetectionAgent` didn't classify any column as `datetime`                     | Ensure dates are in a standard, parseable format (`YYYY-MM-DD`, `MM/DD/YYYY`, ISO 8601, etc.)                                                                                                                          |
| LLM features aren't producing AI narratives                                           | `LLM_PROVIDER` is `none`/unset, or the credentials are invalid                      | This is expected/by-design fallback behavior — check the sidebar's LLM status badge and your `.env`                                                                                                                    |
| `psycopg2.OperationalError` / `pymysql.err.OperationalError` connecting to a database | Wrong URI, firewall, or the DB isn't reachable from where the app is running        | Verify the connection URI works with a plain DB client first; if running in Docker, `localhost` inside the container is NOT your host machine — use `host.docker.internal` (Mac/Windows) or the service name (Compose) |
| `streamlit run` opens but the page is blank                                           | Browser extension blocking WebSockets, or a stale cache                             | Try an incognito window, or run with `streamlit run frontend/streamlit_app.py --server.headless true` and open the printed URL manually                                                                                |
| Docker build fails installing `scipy`/`pandas` wheels                                 | Missing `build-essential` in a customized base image                                | Use the provided Dockerfiles as-is — they already include the required system packages                                                                                                                                 |

---

## 15. FAQ

**Q: Do I need an OpenAI/Anthropic API key to use this?**
No. Every feature works fully offline using deterministic, rule-based analytics. LLM keys are
purely optional enhancements for richer natural-language narratives.

**Q: Does it work on datasets with columns it's never seen before?**
Yes — that's the core design goal. Nothing depends on specific column names; every agent reasons
from detected _semantic type_ (numeric, categorical, datetime, etc.), not naming conventions.

**Q: What's the largest dataset this can handle?**
It's been tested comfortably into the low millions of rows for most agents; `AnomalyDetectionAgent`
(Isolation Forest) and chart rendering are the first to slow down on very large data — see
[Performance Optimization](#16-performance-optimization).

**Q: Can I use this with my company's internal LLM gateway?**
Yes — if it exposes an OpenAI-compatible `/v1/chat/completions` endpoint, point `LLM_PROVIDER=
lmstudio` (or `openrouter`) at it via `LMSTUDIO_BASE_URL`/`OPENROUTER_*`, or add a small new
provider as described in [Adding a New LLM Provider](#adding-a-new-llm-provider).

**Q: Is the natural-language query feature safe? Does it execute LLM-generated code?**
No. `QueryAgent` never calls `eval()`/`exec()`. The LLM (or, without one, a keyword parser) only
ever produces a small, schema-validated JSON "intent" object; a fixed, whitelisted executor then
performs one of a handful of safe pandas operations. See `app/agents/query_agent.py`.

**Q: Why Streamlit only, and not React/Next.js?**
Per this project's design constraints — the entire frontend is intentionally Streamlit, styled
with custom CSS for a non-default, enterprise look, with zero other frontend frameworks.

**Q: Can I run the backend without the frontend, or vice versa?**
Yes — they're fully decoupled (see [Architecture](#2-architecture)). The Streamlit app never
requires the FastAPI backend to be running.

---

## 16. Performance Optimization

- **Large files**: prefer Parquet over CSV for repeated loads — it's both smaller and faster to
  parse. `DataLoaderAgent` supports it natively.
- **Anomaly detection cost**: `AnomalyDetectionAgent` runs Isolation Forest with `n_jobs=-1`
  (uses all CPU cores). For very large datasets, consider sampling before running the full
  pipeline, or raising `contamination` cautiously to reduce estimator count needs.
- **Chart rendering**: `pair_plot` and `parallel_coordinates` scale poorly past ~6 numeric
  columns and a few thousand rows; `ChartRecommendationAgent` already caps these, but if you're
  building custom charts, sample the DataFrame first (`df.sample(5000)`).
- **Repeated analysis**: the Streamlit app caches the full pipeline result in
  `st.session_state["context"]` — it only re-runs when you click **Run Full Analysis** again, not
  on every UI interaction.
- **FastAPI backend at scale**: swap the in-memory `DatasetStore` for Redis/a database (see
  [Deployment](#13-deployment)) and put the backend behind multiple `uvicorn` workers
  (`uvicorn backend.main:app --workers 4`).
- **LLM latency**: LLM calls (when configured) are synchronous and add latency to insight/summary
  generation; the deterministic fallback path has zero network latency by construction.

---

## 17. Future Roadmap

- [ ] Streaming/chunked ingestion for datasets that don't fit in memory
- [ ] Native LangGraph execution mode (adapter already scaffolded in
      `app/agents/langgraph_adapter.py`) with branching/retry pipelines
- [ ] Role-based access control and multi-tenant dataset isolation in the FastAPI backend
- [ ] Persistent, database-backed `DatasetStore` (Postgres/Redis) for horizontal scaling
- [ ] Scheduled/recurring analysis jobs with email or Slack report delivery
- [ ] Additional semantic types (IP address, URL, postal code formats by country)
- [ ] A `polars`-backed execution path for very large datasets
- [ ] Pluggable custom insight rules via a YAML rules file

Contributions toward any of these are very welcome — see below.

---

## 18. Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/my-feature`.
2. Follow the existing code style: PEP 8, type hints on all public functions, docstrings on
   every class and non-trivial method.
3. Add or update tests for any behavior change — `pytest tests/ -v` must pass.
4. Keep agents single-responsibility: if your change does two unrelated things, it's probably two
   agents.
5. Open a pull request describing the change and, if it's user-facing, include a before/after
   description or screenshot.

Please open an issue first for larger changes (new data sources, new LLM providers, architectural
changes) so we can discuss the approach before you invest significant time.

---

## 19. License

Released under the [MIT License](LICENSE) — see the `LICENSE` file for the full text. You are
free to use, modify, and distribute this project, including commercially, provided the copyright
notice is retained.

---

<p align="center">Built with 🧠 agents, 🐼 pandas, and ☕ — no predefined schema required.</p>
