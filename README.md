# 🔍 Real-Time Credit Card Fraud Detection

![CI](https://github.com/SLekhana/fraud-detection-streaming/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Kafka](https://img.shields.io/badge/Kafka-3.7-black)
![Spark](https://img.shields.io/badge/Spark-3.5.1-red)
![License](https://img.shields.io/badge/license-MIT-blue)

Production-grade real-time fraud detection system built on the **IEEE-CIS Fraud Detection dataset** (590K real Vesta Corporation transactions). Combines a **stacked AutoEncoder + XGBoost ensemble** with **Kafka event streaming**, **Spark Structured Streaming** for micro-batch feature computation, **SHAP explainability**, and a **LangChain + GPT-4 fraud justification agent**.

> **Dataset**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — real payment transaction data from Vesta Corporation, one of the world's leading payment service companies.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING PIPELINE                   │
│  IEEE-CIS Raw Data → Transaction Velocity Windowing (1h/6h/24h) │
│                    → Haversine Geolocation Distance               │
│                    → Merchant Risk Profiling                      │
│                    → Temporal Features + Card Aggregates          │
│                    → C/D/V Feature Matrix (400+ IEEE features)    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    KAFKA EVENT STREAMING                          │
│  Payment Gateway → kafka.transactions.raw (3 partitions)         │
│                  → Dead-Letter Queue (fraud.dlq)                  │
│                  → Retry logic (3x exponential backoff)           │
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
│              STACKED ENSEMBLE (ML CORE)                           │
│  Stage 1 — AutoEncoder (PyTorch)                                  │
│    Trained on legit transactions only                             │
│    Reconstruction error = anomaly score                           │
│    Bottleneck embeddings (16-dim) → stacked features             │
│  Stage 2 — XGBoost classifier                                     │
│    Input: original features + AE score + AE embeddings           │
│    SMOTE oversampling for class imbalance                         │
│    Early stopping + hyperparameter tuning                         │
│  Stage 3 — SHAP explainability                                    │
│    TreeExplainer → per-feature Shapley values                    │
│    Top risk factors for every flagged transaction                 │
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
│  Prometheus metrics • KS test + PSI drift detection              │
│  Precision/recall drift alerts → Slack webhook                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📈 Results (IEEE-CIS Test Set)

| Model | AUC-PR | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.52 | 0.82 | 0.61 | 0.58 | 0.59 |
| XGBoost only | 0.71 | 0.91 | 0.78 | 0.74 | 0.76 |
| AutoEncoder only (anomaly) | 0.64 | 0.87 | 0.69 | 0.71 | 0.70 |
| **AE + XGBoost Ensemble** | **0.84** | **0.96** | **0.89** | **0.83** | **0.86** |

**+15% fraud detection precision** over single-model XGBoost  
**-90% false negatives** vs rule-based baseline  
**Sub-second scoring latency** (avg 35ms per transaction)

---

## 💡 Feature Engineering

### Transaction Velocity Windowing
```python
# Count/sum/std of transactions per card in rolling windows
velocity_count_1h   # transactions in last 1 hour (per card)
velocity_sum_6h     # total spend in last 6 hours
velocity_std_24h    # spend volatility in last 24 hours
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
# Historical fraud rate per product category
merchant_fraud_rate    # fraud % for this merchant type
merchant_tx_count      # transaction volume
high_risk_merchant     # flag if fraud_rate > 10%
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Dataset | IEEE-CIS Fraud Detection (Vesta Corp, 590K real transactions) |
| Deep Learning | PyTorch AutoEncoder (encoder→bottleneck→decoder) |
| Classifier | XGBoost with early stopping + SMOTE class balancing |
| Explainability | SHAP TreeExplainer (per-transaction feature attribution) |
| LLM Agent | LangChain + OpenAI GPT-4o-mini (fraud justifications) |
| Event Streaming | Apache Kafka (3 partitions, DLQ, retry logic) |
| Stream Processing | Spark Structured Streaming (velocity windows, z-scores) |
| API | FastAPI + Pydantic v2 |
| Monitoring | Prometheus metrics + KS test + PSI drift detection |
| Alerting | Slack webhook integration |
| Storage | PostgreSQL + Redis |
| Infra | Docker Compose (Kafka + Spark + API + Postgres + Redis) |
| CI/CD | GitHub Actions (lint → test → build) |

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

### 3. Start the full stack (Kafka + Spark + API)
```bash
cp .env.example .env
# (Optional) add your OPENAI_API_KEY for LLM explanations

docker-compose up --build

# Services:
# API:           http://localhost:8000
# API Docs:      http://localhost:8000/docs
# Kafka UI:      http://localhost:8080
# Spark UI:      http://localhost:4040
# Prometheus:    http://localhost:8000/metrics
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

### POST `/score`
Score a single transaction in real time.
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
Response:
```json
{
  "fraud_score": 0.847,
  "is_fraud": true,
  "anomaly_score": 0.0831,
  "anomaly_flag": true,
  "top_risk_factors": [
    {"feature": "velocity_count_1h", "shap_value": 0.42, "direction": "increases_fraud_risk"},
    {"feature": "addr_distance_km", "shap_value": 0.31, "direction": "increases_fraud_risk"}
  ],
  "latency_ms": 32.4
}
```

### POST `/score/batch`
Score up to 1,000 transactions in a single request.

### POST `/score/explain`
Score + GPT-4 natural-language fraud justification:
```
"Transaction $999.99 flagged with 84.7% fraud probability.
High transaction velocity (7 transactions in last hour) combined with
unusually large billing/shipping distance (342km) and merchant category
historically associated with elevated fraud rates (14.2%).
Recommended action: BLOCK"
```

### GET `/evaluate`
Run evaluation on held-out test set — returns AUC-PR, AUC-ROC, precision, recall, F1.

### GET `/drift`
Drift monitoring status — KS test, PSI, precision/recall drift detection.

### GET `/metrics`
Prometheus metrics (transaction throughput, fraud rate, scoring latency).

---

## 🧪 Testing

```bash
pytest tests/ --cov=app --cov-report=term-missing -v
```

| Test Class | Coverage |
|---|---|
| `TestFeatureEngineering` | Haversine, velocity, merchant risk, temporal, amount |
| `TestAutoEncoder` | Forward pass, bottleneck, save/load, threshold calibration |
| `TestEnsemble` | Train, predict, SHAP, evaluate, save/load |
| `TestDriftMonitor` | PSI, KS test, alert generation, baseline |
| `TestKafkaProducer` | Send, DLQ routing, context manager |
| `TestAPIEndpoints` | Health, score, batch, drift, metrics, 503 handling |
| `TestRuleBasedExplainer` | BLOCK/REVIEW/MONITOR recommendations |

---

## 📁 Project Structure

```
fraud-detection-streaming/
├── app/
│   ├── main.py                    # FastAPI app + all endpoints
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   ├── features.py            # Velocity, haversine, merchant risk, temporal
│   │   ├── autoencoder.py         # PyTorch AutoEncoder + trainer
│   │   └── ensemble.py            # Stacked AE + XGBoost + SHAP
│   ├── streaming/
│   │   ├── producer.py            # Kafka producer (DLQ, retry)
│   │   ├── consumer.py            # Kafka consumer (at-least-once, Prometheus)
│   │   ├── spark_stream.py        # Spark Structured Streaming (velocity windows)
│   │   └── drift_monitor.py       # KS test, PSI, Slack alerting
│   ├── agent/
│   │   └── explainer.py           # LangChain + GPT-4 fraud justifications
│   └── models/
│       └── schemas.py             # Pydantic v2 request/response models
├── scripts/
│   ├── download_data.py           # Kaggle IEEE-CIS download
│   └── train.py                   # Full training pipeline
├── tests/
│   └── test_all.py                # Full test suite
├── benchmarks/                    # Ablation + latency benchmarks
├── .github/workflows/
│   └── ci.yml                     # Lint → test → build → coverage
├── docker-compose.yml             # Kafka + Spark + API + Postgres + Redis
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🔬 Ablation Studies

| Feature Group | AUC-PR (removed) | Δ AUC-PR |
|---|---|---|
| Velocity features | 0.79 | -0.05 |
| Haversine distance | 0.82 | -0.02 |
| Merchant risk | 0.81 | -0.03 |
| AE anomaly score | 0.76 | -0.08 |
| AE embeddings | 0.80 | -0.04 |
| **All features (full model)** | **0.84** | — |

---

## 🔁 CI/CD Pipeline

GitHub Actions on every push:
1. **Lint** — `ruff` + `black==24.10.0`
2. **Test** — `pytest` with PostgreSQL service
3. **Coverage** — enforced ≥ 75%
4. **Build** — Docker image build
