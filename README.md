# Enterprise Telecom CVM Churn Intelligence & Next-Best-Action (NBA) Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: FastAPI](https://img.shields.io/badge/serving-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Engine: LightGBM](https://img.shields.io/badge/algorithm-LightGBM%20GOSS-green.svg)](https://lightgbm.readthedocs.io/)
[![Explainability: TreeSHAP](https://img.shields.io/badge/XAI-TreeSHAP-red.svg)](https://shap.readthedocs.io/)
[![Target Roles](https://img.shields.io/badge/Target-Data%20Scientist%20%2F%20ML%20Engineer-blueviolet.svg)](#)

**Author:** Atharva Mirajkar (MSc Data Science, Fergusson College, Pune)  
**Target Domain:** Telecom Customer Value Management (CVM) Analytics (_VOIS) & Digital Subscription Retention (Amazon Prime)  
**Cloud Dataset:** [Hugging Face Hub: athrvmrjkr/telecom-cvm-kkbox](https://huggingface.co/athrvmrjkr)

---

## 1. System Vision & Business Impact

In telecommunications and digital subscription services, customer acquisition costs (CAC) typically exceed retention costs by **5x to 7x**. 

This system implements an end-to-end commercial **Customer Value Management (CVM)** decision platform:
1. **Early Churn Risk Detection:** Predicts 30-day forward cancellation probability using an out-of-time temporal validation framework.
2. **Probability Calibration:** Corrects greedy tree-split distortion via **Isotonic Regression** so predicted scores match empirical risk.
3. **Cost-Utility Threshold Tuning:** Replaces the arbitrary 0.5 decision threshold with a mathematically optimized threshold ($T^*$) that maximizes **Net Retained Revenue**.
4. **Root-Cause Explainability:** Uses **TreeSHAP** to compute local feature attributions ($\phi_i$) for every individual subscriber.
5. **Prescriptive Next-Best-Action (NBA):** Translates risk scores and root-cause drivers into targeted operational treatments (e.g., UPI auto-pay cashback, 5GB data boosters, 12-month lock-in discounts, or proactive service recovery).
6. **Sub-45ms Serving SLA:** Containerized, asynchronous FastAPI microservice built with Pydantic v2 schemas and in-memory model warm-up.

---

## 2. Architecture & Data Flow

```
[Cloud Data: Hugging Face Datasets]
                │
                ▼
  [Data Pipeline & Temporal Split]
  ├── Observation Window: [T - 60, T]
  └── Prediction Window:  [T, T + 30] (Strict zero future-leakage)
                │
                ▼
  [Domain Feature Engineering]
  ├── Demographics & Age Bound Cleaning [14, 85]
  ├── Financial & Auto-Pay Volatility (RFM)
  └── Usage Velocity Ratios (7-Day vs. 30-Day Mean Seconds)
                │
                ▼
  [Modeling & Calibration]
  ├── Baseline: Cost-Sensitive Logistic Regression
  ├── Champion: LightGBM GOSS (scale_pos_weight = auto)
  ├── Isotonic Probability Calibration
  └── Cost-Utility Decision Threshold Tuning (T*)
                │
                ▼
  [Explainability & NBA Policy Engine]
  ├── TreeSHAP Local Attributions (ϕ_i)
  └── Multi-Tier Next-Best-Action Matrix
                │
                ▼
  [Production Serving & Telemetry]
  ├── FastAPI Async Microservice (P95 Latency < 45ms)
  └── Population Stability Index (PSI) Drift Monitoring
```

---

## 3. Directory Structure

```
telecom-cvm-retention/
├── configs/
│   └── config.yaml             # Centralized system, model, and business parameters
├── data/
│   ├── raw/                    # Cached raw splits (git-ignored)
│   └── processed/              # Snappy Parquet feature stores (git-ignored)
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        # Streaming ingestion & temporal validation split
│   ├── feature_engineering.py  # RFM, usage velocity, and demographic features
│   ├── train.py                # LightGBM GOSS & Logistic Regression training
│   ├── evaluate.py             # PR-AUC, Brier score & Cost-Utility optimization
│   ├── explain.py              # TreeSHAP feature attributions
│   └── nba_engine.py           # Prescriptive Next-Best-Action decision matrix
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI asynchronous endpoints
│   └── schemas.py              # Pydantic v2 strict schemas
├── benchmarks/                 # Latency, memory, and runtime telemetry logs
├── tests/                      # Pytest suite for pipelines, features, and API
├── requirements.txt            # Locked production dependencies
├── setup.py                    # Package installer for editable local imports
├── Dockerfile                  # Production container packaging (python:3.11-slim)
└── README.md
```

---

## 4. Setup & Quickstart

```bash
# 1. Clone repository
git clone https://github.com/athrvmrjkr/kkbox-retention-intelligence.git
cd kkbox-retention-intelligence

# 2. Create and activate a clean virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 5. License
MIT License. Developed for enterprise portfolio demonstration.
