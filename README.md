# AI-Assisted Signal Regeneration in Critical CAN Bus Interruptions

**Course:** Özyeğin University 2025–2026 Spring · DS570 Term Project
**Author:** Şakir Buğra Aksu

An end-to-end data science and MLOps pipeline: data processing, feature analysis, an interactive telemetry dashboard, predictive modeling, model evaluation under failure scenarios, and containerized deployment.

---

## Project Overview & Problem Statement

In modern automotive Electrical/Electronic (E/E) architectures, critical sensor signals transmitted over vehicle networks (CAN Bus, CAN-FD, LIN) are vulnerable to temporary or permanent interruptions — physical hardware failures, wiring harness short circuits, or malicious interventions such as spoofing and Denial-of-Service attacks.

When a primary sensor such as the vehicle speed sensor fails, dependent ADAS modules and chassis control units lose critical telemetry, compromising passenger safety.

**Objective:** Implement an intelligent, real-time signal fallback mechanism. By exploiting correlations among the remaining active network signals, machine learning estimators reconstruct the lost `Vehicle_Speed` (`Signal_Y`) telemetry at runtime — turning a *fail-silent* sensor outage into a *fail-operational* degraded mode.

---

## Dataset & Proprietary Safety (NDA Compliance)

The data is derived from real CAN Bus logs captured during vehicle field tests: **104,840 frames** resampled on a 10 ms grid. To adhere to corporate Non-Disclosure Agreements, rigorous sanitization is enforced:

* **Signal masking (DBC independence):** proprietary CAN IDs, message frames, and database signal names are removed. Features are generalized as `Signal_X1` … `Signal_X11`; the target velocity is `Signal_Y`.
* **Mathematical anonymization:** all channels are MinMax-scaled to **[0, 1]**, concealing physical operating envelopes. Error metrics are reported in the scaled domain; a conservative physical bound is given below.
* **Automated ingestion:** the anonymized dataset ships with the repository and loads automatically when the container starts — no manual download, no account, no user intervention. The loader is locale-robust (handles semicolon-delimited, decimal-comma CSV exports) and falls back to the public GitHub anchor if the local file is absent.

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

## Methodology & Model Architecture

Two model classes are compared deliberately — one as a physically principled baseline, one for fault tolerance:

1. **Baseline — Linear Regression.** Vehicle speed is physically linear in its proxies (V = r·ω), so a 12-parameter linear model is the principled reference point: interpretable, deterministic, trains in seconds.
2. **Advanced — Random Forest Regressor** (`n_estimators=50, max_depth=12, bootstrap=True, random_state=42`). The ensemble's threshold-based splits capture the **non-linear, conditional couplings** between the redundant channels and the target — the property that matters when the primary proxies disappear.

**The Physical Proxy Principle.** Trained independently, both architectures concentrate on the same two channels — `Signal_X7` (wheel speed) and `Signal_X8` (inverter RPM), the direct physical proxies of vehicle speed (RF importance: 86.3% / 13.7%). ML output agreeing with domain physics validates that the models learned mechanism, not spurious correlation.

---

## Evaluation

### Chronological split — no temporal leakage

Frames 10 ms apart are nearly identical; a shuffled split would scatter near-duplicates across train and test, leaking the future into training and inflating scores. The split is therefore strictly chronological (`shuffle=False`): the first 80% (83,872 frames) trains the models, the final 20% (20,968 frames — the *unseen future*) tests them, mirroring the production scenario.

### Hold-out test performance

| Model | R² | MAE (×10⁻³, scaled) | RMSE (×10⁻³, scaled) |
| :--- | :---: | :---: | :---: |
| Baseline (Linear Regression) | **0.9998** | 1.54 | 2.69 |
| Advanced (Random Forest) | **0.9998** | **1.33** | **2.41** |

*Physical scale:* the inverse MinMax transform bounds both models at **MAE ≤ 0.16 km/h and RMSE ≤ 0.27 km/h** — below typical speedometer display resolution. (Exact speed range withheld per NDA.)

### Bias–variance diagnostics

LR is structurally stable (test/train MAE ratio 1.31×, ΔR² ≈ 0.0002). RF's saturated depth-12 trees partially memorize the train set (ratio 2.96×), but bootstrap averaging across 50 decorrelated trees cancels the variance — ΔR² ≈ 0 and the absolute test error remains the lowest. Optimistic training error, no harmful overfitting.

### Temporal cross-validation (expanding window)

Classic k-fold is invalid for time series (temporal leakage), so a 5-fold `TimeSeriesSplit` expanding-window validation is used instead: train on the past, test on the immediate future, grow the window, repeat. Across all non-degenerate folds both models stay at R² = 0.999+, MAE 1.3–2.3×10⁻³ — the hold-out result is not a lucky split. One fold lands on a stationary segment (`Signal_Y ≡ 0`, zero target variance) where R² is undefined while MAE is near-perfect; this metric edge case is documented in the notebook (Section 6b).

---

## Functional Safety (ISO 26262) — Sensor-Loss Ablation Study

Feature importance says the 9 secondary channels contribute ~0% — but **importance ≠ capability**. Both models were retrained from scratch under four sensor-availability scenarios:

| Scenario | LR — Test R² | RF — Test R² | LR — MAE mult. | RF — MAE mult. |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (all 11) | 0.9998 | 0.9998 | 1.0× | 1.0× |
| Drop X7 | 0.9998 | 0.9997 | 0.9× | 1.1× |
| Drop X8 | 0.9997 | 0.9996 | 1.2× | 1.4× |
| **Drop X7 + X8 (crisis)** | **0.04** | **0.97** | **72.2×** | **13.4×** |

Single-sensor loss is harmless — X7 and X8 are mutual physical proxies. When **both** are lost (hardware failure / DoS), Linear Regression collapses to noise level, while Random Forest degrades gracefully: its splits rediscover the conditional physics of the redundant channels — importance redistributes onto `Signal_X11` (regen torque, **95.8%** in the crisis model), a signal that can only exist while the motor spins. This is the FuSa value of the pipeline: architectural redundancy for **limp-home** safety modes.

The per-scenario artifacts live in `models/*_drop_*.joblib`; the dashboard evaluates them live (see below), and the notebook regenerates them deterministically.

---

## Interactive Dashboard

The Streamlit app integrates exploratory views and model results:

* **Tab 1 — Exploratory Data Analysis & Feature Weights:**
  * Dataset overview (row counts, chronological split sizes, sampling grid) and target distribution.
  * Side-by-side LR coefficients vs. RF feature importances — **extracted live from the `.joblib` artifacts, no hardcoded values.**
  * **Sensor-Loss Ablation panel:** the four-scenario Test-R² chart, the MAE degradation table, and the crisis-model importance redistribution — all metrics computed at runtime from the per-scenario artifacts on the chronological test split. If the ablation artifacts are absent, the panel degrades to an informative message instead of crashing.
* **Tab 2 — Real-Time Model Inference:** replays the *unseen* chronological test region frame by frame (`Next Frame` / `Reload Frame`), with runtime model switching (Baseline ↔ Advanced), live-computed test metrics, 11 what-if telemetry sliders, an actual-vs-prediction verification chart, and a live residual-distribution panel (unbiasedness check).

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


### Reproducibility

All results are deterministic: fixed `random_state=42`, chronological split, pinned dependencies (`requirements.txt`, scikit-learn 1.8.0). Re-running `notebooks/DS570_Model_Experiments.ipynb` end-to-end regenerates **all eight model artifacts** and every number quoted above, bit-for-bit.

---

## Limitations & Future Work

* **Temporal blindness:** the regressors evaluate each frame independently; error propagation is possible during prolonged blackouts. *Path forward:* RNN/LSTM architectures with sliding-window context.
* **Edge-case volatility:** extreme transients (e.g., ABS triggering on ice) can momentarily distort convergence. *Path forward:* Kalman-filter pre-filtering + anomaly detection on the inputs.
* **Single-vehicle scope:** data originates from one vehicle's field tests; cross-vehicle and cross-condition validation is required before any production claim.
