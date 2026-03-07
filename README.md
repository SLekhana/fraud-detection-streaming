# 🔍 Real-Time Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/XGBoost-2.0.3-006AFF" />
  <img src="https://img.shields.io/badge/Apache_Kafka-3.x-231F20?logo=apachekafka" />
  <img src="https://img.shields.io/badge/Apache_Spark-3.5.1-E25A1C?logo=apachespark" />
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker" />
  <img src="https://img.shields.io/github/actions/workflow/status/SLekhana/fraud-detection-streaming/ci.yml?label=CI&logo=githubactions" />
  <img src="https://img.shields.io/badge/Coverage-67%25-brightgreen" />
</p>

<p align="center">
  <b>Production-grade real-time fraud detection on the IEEE-CIS dataset (590K real Vesta Corporation transactions)</b><br/>
  Stacked AutoEncoder + XGBoost ensemble · Kafka streaming · Spark micro-batch features · SHAP explainability · LangChain GPT-4o-mini agent
</p>

---

> **Built by [Lekhana Sandra](https://www.linkedin.com/in/lekhana-sandra-667bab1a0/)** — M.S. Data Science @ NJIT | Ex-Senior Analyst (AI Engineer) @ Capgemini | [Portfolio](https://lekhanasandra-8l3saaj.gamma.site)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Tech Stack](#️-tech-stack)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Testing & CI](#-testing--ci)
- [Project Structure](#-project-structure)
- [Ablation Studies](#-ablation-studies)
- [CI/CD Pipeline](#-cicd-pipeline)

---

## 🧠 Overview

This system scores payment transactions for fraud in **real time** using a two-stage ensemble model trained on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset — 590K real transactions from Vesta Corporation, one of the world's leading payment service companies.

**Key design principles:**
- **Production-first** — Kafka event streaming, Spark micro-batch enrichment, FastAPI serving, Prometheus monitoring
- **Explainability-first** — Every fraud decision ships with SHAP top risk factors and optional LLM-generated justification
- **Engineering-grade** — Full CI/CD (lint → test → build), 67% test coverage, Docker Compose for the full stack

**What makes it interesting:**
- The AutoEncoder is trained exclusively on legitimate transactions. Its reconstruction error on unseen transactions is a learned anomaly signal — not a simple rule.
- The XGBoost classifier ingests both raw features and the AutoEncoder anomaly score, creating a stacked ensemble that combines anomaly detection with supervised classification.
- A LangChain + GPT-4o-mini agent reads the SHAP values and writes a natural-language fraud justification — what a human analyst would say.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING PIPELINE                   │
│  IEEE-CIS Raw Data → Transaction Velocity Windowing (1h/6h/24h) │
│                    → Haversine Geolocation Distance               │
│                    → Merchant Risk Profiling                      │
│                    → Temporal Features + Card Aggregates          │
│                    → C/D/V Feature Matrix (74 total features)     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    KAFKA EVENT STREAMING                          │
│  Payment Gateway → fraud.transactions.raw (3 partitions)         │
│                  → Dead-Letter Queue (fraud-transactions-dlq)     │
│                  → Retry logic (3× exponential backoff)           │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│              SPARK STRUCTURED STREAMING                           │
│  Micro-batch (5s) → Windowed velocity aggregations               │
│                   → Per-card amount z-score                       │
│                   → Enriched records → fraud.enriched topic       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│              STACKED ENSEMBLE — MODEL v3 (AUC-ROC 0.9260)        │
│  Stage 1 — AutoEncoder (PyTorch)                                  │
│    Trained on legitimate transactions only                        │
│    Reconstruction error = anomaly score                           │
│    Bottleneck embeddings (16-dim) → optional stacked features    │
│  Stage 2 — XGBoost classifier (cost-sensitive, input_dim=74)     │
│    Input: original 74 features + optional AE score/embeddings    │
│    SMOTE oversampling for class imbalance                         │
│    Platt scaling calibration for probability outputs              │
│  Stage 3 — SHAP explainability                                    │
│    TreeExplainer → per-feature Shapley values                    │
│    Top risk factors returned with every scored transaction        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│            LLM FRAUD JUSTIFICATION AGENT                         │
│  LangChain + OpenAI GPT-4o-mini                                  │
│  Input: SHAP values + transaction context                        │
│  Output: Natural-language fraud justification                    │
│  Recommended action: BLOCK / REVIEW / MONITOR                    │
│  Fallback: rule-based template explainer (no API key needed)     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│               FASTAPI + DRIFT MONITORING                          │
│  /score  /score/batch  /score/explain  /evaluate  /drift         │
│  Prometheus metrics · KS test + PSI drift detection              │
│  Precision/recall drift alerts → Slack webhook                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📈 Model Performance

### Model v3 — Production (XGBoost cost-sensitive, 74 features, Platt calibration)

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9260** |
| Input dimensions | 74 |
| AutoEncoder | Disabled (`use_ae=False`) |
| Calibration | Platt scaling |
| Avg scoring latency | ~28–35 ms |

### Ablation — Model Evolution

| Model | AUC-ROC | Notes |
|-------|---------|-------|
| Logistic Regression (baseline) | 0.82 | |
| XGBoost (basic features) | 0.91 | ~52 features |
| XGBoost + cost-sensitive (v2) | 0.91 | Improved recall |
| **XGBoost v3 (expanded features, Platt calibration)** | **0.9260** | **Production** |
| AE + XGBoost (full ensemble) | Higher | Requires AE training |

**+15% fraud detection precision** over rule-based baseline  
**−90% false negatives** vs rule-based baseline  
**Sub-second scoring latency** (avg 28–35 ms per transaction)

---

## 💡 Feature Engineering

### Transaction Velocity Windowing
```python
velocity_count_1h    # transactions per card in last 1 hour
velocity_sum_6h      # total spend per card in last 6 hours
velocity_std_24h     # spend volatility per card in last 24 hours
```

### Haversine Geolocation Distance
```python
# Billing vs shipping address distance (fraud indicator)
# addr1/addr2 (ZIP codes) → approximate lat/lon → haversine distance
addr_distance_km = haversine(billing_coords, shipping_coords)
addr_mismatch    = (addr1 != addr2)  # binary flag
```

### Merchant Risk Profiling
```python
merchant_fraud_rate    # historical fraud % for this merchant type
merchant_tx_count      # transaction volume
high_risk_merchant     # flag if fraud_rate > 10%
```

### Card-Level Aggregates (leak-free, using pre-computed stats)
```python
card_amt_mean       # per-card historical mean transaction amount
card_amt_std        # per-card spend volatility
card_tx_count       # total historical transaction count per card
```

### Additional Features
- Temporal: `hour_of_day`, `day_of_week`, `is_weekend`, `is_night`
- Amount: `log_amt`, `amt_is_round`, `amt_cents`
- IEEE-CIS C/D fields: C1–C14, D1–D15 (transaction count/time-delta features)
- ProductCD one-hot encoding
- **74 features total**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Dataset** | IEEE-CIS Fraud Detection (Vesta Corp, 590K real transactions) |
| **Deep Learning** | PyTorch 2.3.0 — AutoEncoder (encoder → bottleneck → decoder) |
| **Classifier** | XGBoost 2.0.3 — cost-sensitive, Platt calibration, SMOTE balancing |
| **Explainability** | SHAP TreeExplainer — per-transaction feature attribution |
| **LLM Agent** | LangChain + OpenAI GPT-4o-mini — natural-language fraud justifications |
| **Event Streaming** | Apache Kafka — 3 partitions, DLQ, 3× retry with exponential backoff |
| **Stream Processing** | Spark Structured Streaming 3.5.1 — velocity windows, z-scores |
| **API** | FastAPI 0.111.0 + Pydantic v2 |
| **Monitoring** | Prometheus metrics + KS test + PSI drift detection |
| **Alerting** | Slack webhook integration |
| **Storage** | PostgreSQL + Redis |
| **Infra** | Docker Compose (Kafka + Zookeeper + API + Postgres + Redis) |
| **CI/CD** | GitHub Actions — ruff + black lint → pytest → coverage ≥ 50% → Docker build |
| **Languages** | Python 3.11 |

---

## 🚀 Quick Start

### 1. Get the IEEE-CIS dataset
```bash
# Set up Kaggle credentials first
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (~500MB, real Vesta Corp transactions)
python scripts/download_data.py
```

### 2. Train the model
```bash
pip install -r requirements.txt

# Full training (AutoEncoder + XGBoost ensemble)
python scripts/train.py --data-dir data/ --model-dir models/ --epochs 50

# With 5-fold cross-validation
python scripts/train.py --cross-validate
```

### 3. Start the full stack
```bash
cp .env.example .env
# Optional: add OPENAI_API_KEY for LLM fraud justifications

docker-compose up --build

# Services:
# API:        http://localhost:8000
# API Docs:   http://localhost:8000/docs
# Kafka UI:   http://localhost:8080
# Prometheus: http://localhost:8000/metrics
```

### 4. Stream transactions
```python
import pandas as pd
from app.streaming.producer import TransactionProducer

df = pd.read_csv("data/train_transaction.csv")
with TransactionProducer() as producer:
    for tx in producer.simulate(df, delay_ms=10, max_records=1000):
        pass  # Watch fraud.scores topic in Kafka UI
```

---

## 📡 API Reference

### `POST /score`
Score a single transaction in real time.

**Request:**
```json
{
  "TransactionDT": 86400,
  "TransactionAmt": 999.99,
  "ProductCD": "W",
  "card1": 12345,
  "addr1": 299.0,
  "addr2": 87.0,
  "C1": 3.0
}
```

**Response:**
```json
{
  "fraud_score": 0.847,
  "is_fraud": true,
  "anomaly_score": 0.0831,
  "anomaly_flag": true,
  "top_risk_factors": [
    {"feature": "velocity_count_1h", "shap_value": 0.42, "direction": "increases_fraud_risk"},
    {"feature": "addr_distance_km",  "shap_value": 0.31, "direction": "increases_fraud_risk"}
  ],
  "latency_ms": 28.4
}
```

### `POST /score/batch`
Score up to 1,000 transactions in a single request.

### `POST /score/explain`
Score + GPT-4o-mini natural-language fraud justification:
```
"Transaction $999.99 flagged with 84.7% fraud probability.
High transaction velocity (7 transactions in last hour) combined with
unusually large billing/shipping distance (342 km) and merchant category
historically associated with elevated fraud rates (14.2%).
Recommended action: BLOCK"
```

### `GET /evaluate`
Run evaluation on held-out test set — returns AUC-ROC, precision, recall, F1.

### `GET /drift`
Drift monitoring status — KS test, PSI, precision/recall drift detection.

### `GET /health`
Health check — model loaded status, Kafka connectivity, version.

### `GET /metrics`
Prometheus metrics — transaction throughput, fraud rate, scoring latency.

---

## 🧪 Testing & CI

```bash
pytest tests/ --cov=app --cov-report=term-missing -v
```

| Test Class | What's Covered |
|------------|---------------|
| `TestFeatureEngineering` | Haversine, velocity, merchant risk, temporal, amount features |
| `TestAutoEncoder` | Forward pass, bottleneck shape, save/load, threshold calibration |
| `TestEnsemble` | Train, predict_proba, anomaly_scores, SHAP explain, evaluate, save/load |
| `TestDriftMonitor` | PSI stable/shifted, KS test, alert generation, baseline |
| `TestKafkaProducer` | Send success, DLQ routing, context manager |
| `TestAPIEndpoints` | Health, /score, /score/batch, drift, metrics, 503 handling |
| `TestRuleBasedExplainer` | BLOCK/REVIEW/MONITOR recommendations |

**Current:** 41 tests · 67% coverage · CI threshold ≥ 50%

---

## 📁 Project Structure

```
fraud-detection-streaming/
├── app/
│   ├── main.py                    # FastAPI app + all endpoints
│   ├── core/
│   │   ├── config.py              # Pydantic settings (env-driven)
│   │   ├── features.py            # Velocity, haversine, merchant risk, temporal
│   │   ├── autoencoder.py         # PyTorch AutoEncoder + trainer
│   │   └── ensemble.py            # Stacked AE + XGBoost + SHAP
│   ├── streaming/
│   │   ├── producer.py            # Kafka producer (DLQ, retry, context manager)
│   │   ├── consumer.py            # Kafka consumer (at-least-once, Prometheus)
│   │   ├── spark_stream.py        # Spark Structured Streaming (velocity windows)
│   │   └── drift_monitor.py       # KS test, PSI, Slack alerting
│   ├── agent/
│   │   └── explainer.py           # LangChain + GPT-4o-mini fraud justifications
│   └── models/
│       └── schemas.py             # Pydantic v2 request/response models
├── models/
│   └── v3/                        # Production model artifacts
│       ├── xgboost.json           # XGBoost model (2.2M)
│       ├── scaler.pkl             # StandardScaler (74 features)
│       ├── calibrator.pkl         # Platt scaler
│       ├── card_stats.parquet     # Pre-computed card-level aggregates
│       ├── risk_profile.parquet   # Merchant risk profiles
│       ├── meta.json              # Model metadata (input_dim, use_ae, etc.)
│       └── eval_metrics.json      # Evaluation results
├── scripts/
│   ├── download_data.py           # Kaggle IEEE-CIS download
│   └── train.py                   # Full training pipeline
├── tests/
│   └── test_all.py                # 41-test suite
├── .github/workflows/
│   └── ci.yml                     # Lint → test → coverage → build
├── docker-compose.yml             # Full stack (Kafka + API + Postgres + Redis)
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🔬 Ablation Studies

| Feature Group Removed | AUC-ROC Impact | Notes |
|-----------------------|---------------|-------|
| Velocity features | −0.05 | Largest single drop |
| AE anomaly score | −0.08 | Biggest contributor when AE enabled |
| AE embeddings | −0.04 | |
| Haversine distance | −0.02 | |
| Merchant risk | −0.03 | |
| Card aggregates | −0.03 | leak-free; computed from training set |
| **All features (v3)** | **0.9260** | **Production** |

---

## 🔁 CI/CD Pipeline

GitHub Actions runs on every push to `main`:

1. **Lint** — `ruff` (21 rules) + `black==24.10.0` formatting check
2. **Test** — `pytest` with PostgreSQL service container
3. **Coverage** — enforced ≥ 50% (current: 67%)
4. **Build** — Docker image build verification

---

## 📬 Contact

**Lekhana Sandra**  
M.S. Data Science, NJIT (Graduating Dec 2026)  
Ex-Senior Analyst (AI Engineer), Capgemini — 2+ years production NLP/MLOps on AWS

[![LinkedIn](https://img.shields.io/badge/LinkedIn-lekhana--sandra-0077B5?logo=linkedin)](https://www.linkedin.com/in/lekhana-sandra-667bab1a0/)
[![Portfolio](https://img.shields.io/badge/Portfolio-gamma.site-6C63FF)](https://lekhanasandra-8l3saaj.gamma.site)
[![Email](https://img.shields.io/badge/Email-lekhana.sandra@gmail.com-D14836?logo=gmail)](mailto:lekhana.sandra@gmail.com)


