# AI-Assisted Signal Regeneration in Critical CAN Bus Interruptions

**Course:** Özyeğin University 2025–2026 Spring · DS570 Term Project
**Author:** Şakir Buğra Aksu

An end-to-end data science and MLOps pipeline: data processing, feature analysis, an interactive telemetry dashboard, predictive modeling, model evaluation under failure scenarios, and containerized deployment.

---

## Problem Statement

In automotive E/E architectures, sensor signals on vehicle networks (CAN Bus, CAN-FD, LIN) can drop out due to hardware failures, wiring faults, or attacks such as spoofing and DoS. When the vehicle speed sensor fails, dependent ADAS and chassis modules lose critical telemetry.

**Objective:** a real-time ML fallback that reconstructs the lost `Vehicle_Speed` (`Signal_Y`) from the remaining active signals — turning a *fail-silent* outage into a *fail-operational* degraded mode.

---

## Dataset

Real CAN Bus logs from vehicle field tests: **104,840 frames** on a 10 ms grid. For NDA compliance, proprietary CAN IDs and signal names are removed (features generalized as `Signal_X1…X11`) and all channels are MinMax-scaled to **[0, 1]**; metrics are therefore reported in the scaled domain. The dataset ships with the repository and loads automatically at container start — no manual download or account required.

---

## Repository Structure

```text
DS570_Project/
├── .dockerignore                      # Keeps caches/venv out of the build context
├── .gitignore                         # NOTE: model artifacts are intentionally tracked
├── Dockerfile                         # python-slim based, single-command build
├── README.md
├── requirements.txt                   # Pinned dependencies, consistent with the Dockerfile
├── app/
│   └── dashboard.py                   # Streamlit frontend (EDA + ablation + live inference)
├── data/
│   └── Finalized_Dataset.csv          # Anonymized, scaled telemetry (104,840 rows)
├── models/
│   ├── linear_model.joblib            # Baseline production weights
│   ├── advanced_model.joblib          # Random Forest production weights
│   ├── linear_model_drop_x7.joblib    # ┐
│   ├── linear_model_drop_x8.joblib    # │ Ablation-study artifacts:
│   ├── linear_model_drop_x7x8.joblib  # │ retrained per sensor-loss
│   ├── advanced_model_drop_x7.joblib  # │ scenario (see FuSa section)
│   ├── advanced_model_drop_x8.joblib  # │
│   └── advanced_model_drop_x7x8.joblib# ┘
├── notebooks/
│   └── Project_Test_Workspace.ipynb   # initial tests
│   └── DS570_Model_Experiments.ipynb  # Reproducible analysis: EDA → training →
│                                      # bias-variance → temporal CV → ablation → export
└── src/
    ├── __init__.py
    └── preprocessing.py               # Ingestion, locale cleaning, chronological split
```

---

## Models

1. **Baseline — Linear Regression.** Vehicle speed is physically linear in its proxies (V = r·ω); a linear model is the principled, interpretable reference.
2. **Advanced — Random Forest** (`n_estimators=50, max_depth=12, random_state=42`). Captures the non-linear couplings between redundant channels and the target — the property that matters under sensor loss.

Both models independently concentrate on `Signal_X7` (wheel speed) and `Signal_X8` (inverter RPM), the two physical proxies of vehicle speed — consistent with vehicle dynamics.

---

## Evaluation

The train/test split is strictly **chronological** (`shuffle=False`, 80/20): frames 10 ms apart are nearly identical, so a shuffled split would leak the future into training. Stability was additionally confirmed with expanding-window time-series cross-validation.

| Model | R² | MAE (×10⁻³, scaled) | RMSE (×10⁻³, scaled) |
| :--- | :---: | :---: | :---: |
| Baseline (Linear Regression) | **0.9998** | 1.54 | 2.69 |
| Advanced (Random Forest) | **0.9998** | **1.33** | **2.41** |

### Sensor-Loss Ablation Study (FuSa)

Both models retrained under four sensor-availability scenarios:

| Scenario | LR — Test R² | RF — Test R² |
| :--- | :---: | :---: |
| Baseline (all 11) | 0.9998 | 0.9998 |
| Drop X7 | 0.9998 | 0.9997 |
| Drop X8 | 0.9997 | 0.9996 |
| **Drop X7 + X8 (crisis)** | **0.04** | **0.97** |

Single-sensor loss is harmless — X7 and X8 are mutual physical proxies. When **both** are lost (hardware failure / DoS), Linear Regression collapses to noise level, while Random Forest degrades gracefully: its splits rediscover the conditional physics of the redundant channels — importance redistributes onto `Signal_X11` (regen torque, **95.8%** in the crisis model), a signal that can only exist while the motor spins. This is the FuSa value of the pipeline: architectural redundancy for **limp-home** safety modes.

The per-scenario artifacts live in `models/*_drop_*.joblib`; the dashboard evaluates them live (see below), and the notebook regenerates them deterministically.

---

## Dashboard

* **Tab 1 — EDA & Feature Weights:** dataset overview, target distribution, LR coefficients vs. RF importances (read live from the model artifacts), and the ablation-study panel with metrics computed at runtime.
* **Tab 2 — Real-Time Inference:** replays the unseen test region frame by frame, with runtime model switching, what-if telemetry sliders, an actual-vs-prediction chart, and a residual-distribution panel.

---

##  Deployment & Local Setup Guide via Docker

This system is completely containerized via Docker to guarantee absolute environment replication across different operating systems, eliminating local python configuration dependencies.

###  Step-by-Step Execution Instructions

#### 1. Clone the Repository

Open a terminal (Git Bash, PowerShell, or Command Prompt) and pull the project workspace:

```bash
git clone https://github.com/bugraaksu1/DS570_Project.git
cd DS570_Project

```

#### 2. Build the Docker Image

Compile the isolated environment, including the Python runtime layer and dependencies, by executing:

```bash
docker build -t vehicle-speed-dashboard .

```

#### 3. Run the Container

Spin up the interactive telemetry simulation server:

```bash
docker run -p 8501:8501 vehicle-speed-dashboard

```

#### 4. Access the Live Dashboard

Once initialized, open any modern web browser and navigate to:
 **[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)**

---

## Limitations & Future Work

* **Temporal blindness:** the regressors evaluate each frame independently; error propagation is possible during prolonged blackouts. *Path forward:* RNN/LSTM architectures with sliding-window context.
* **Edge-case volatility:** extreme transients (e.g., ABS triggering on ice) can momentarily distort convergence. *Path forward:* Kalman-filter pre-filtering + anomaly detection on the inputs.
* **Single-vehicle scope:** data originates from one vehicle's field tests; cross-vehicle and cross-condition validation is required before any production claim.
