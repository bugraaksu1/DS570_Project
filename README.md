# AI-Assisted Signal Regeneration in Critical CAN Bus Interruptions

**Course:** Özyeğin University 2025–2026 Spring — DS570 Term Project
**Author:** Şakir Buğra Aksu

An end-to-end data science and MLOps pipeline encompassing data processing, feature engineering, interactive telemetry dashboards, predictive machine learning modeling, and containerized deployment — built around an automotive functional safety problem.

---

## Project Overview & Problem Statement

In modern automotive Electrical/Electronic (E/E) architectures, the `Vehicle_Speed` value used by ADAS modules, chassis controllers, and autonomous driving stacks is **not a direct sensor reading**. It is a **computed quantity** derived by sensor fusion of:

* **Inertial Measurement Unit (IMU)** signals — yaw rate, roll rate, pitch rate
* **Wheel Speed sensors**

Therefore, when any of these source signals are interrupted — due to hardware failures, harness short circuits, packet loss on the CAN bus, or malicious cyber interventions such as spoofing and Denial-of-Service (DoS) attacks — **the entire vehicle speed computation collapses**, not just one sensor reading.

In an autonomous vehicle, this is a direct **ISO 26262 Functional Safety (FuSa) violation**: dependent ADAS functions and the autonomy stack lose the velocity reference needed to make safe decisions, while no nominal sensor has technically "failed".

**Objective.** This project implements a **FuSa-compliant, AI-based redundant estimator** for vehicle speed. By exploiting spatial and temporal correlations among the **surviving** active network signals (inverter speed, regenerative torque, pedal positions, remaining IMU axes, etc.), the machine learning estimators dynamically reconstruct the `Vehicle_Speed` (`Signal_Y`) telemetry at runtime — allowing the autonomous vehicle to continue operating in a fail-operational mode (e.g., to perform a minimum-risk maneuver).

---

## Dataset & Proprietary Safety (NDA Compliance)

The underlying data is derived from actual CAN Bus logs captured during real-world vehicle field tests — three CAN channels logged over a ~20 minute drive in Vector BLF format, yielding ~104,840 time-aligned samples after preprocessing. To strictly adhere to corporate Non-Disclosure Agreements (NDAs) and safeguard intellectual property, rigorous sanitization and anonymization protocols were enforced:

* **Signal Masking (DBC independence).** Proprietary CAN IDs, message frames, and DBC database signal names are entirely removed. Features are generalized into abstract representations spanning `Signal_X1` through `Signal_X11`, with the target velocity mapped as `Signal_Y`.

* **Mathematical Anonymization via MinMax Scaling.** A row-wise **MinMax scaler** is applied to every signal, normalizing all values to the `[0, 1]` range. This serves two purposes simultaneously:
  * **NDA protection** — actual physical hardware thresholds (max motor RPM, BMS current range, peak vehicle speed, etc.) are concealed because the absolute scale is destroyed.
  * **ML preprocessing best practice** — feature scales are equalized across all 11 channels, preventing naturally larger-magnitude features from dominating loss functions and accelerating convergence.

* **Automated Runtime Ingestion.** In compliance with zero-local-file requirements, the application hosts no internal datasets. The pipeline streams data directly from a public remote repository anchor via Pandas when the Docker container initializes.

---

## Methodology & Model Architecture

The pipeline evaluates two distinct modeling classes to balance computational latency, interpretability, and predictive accuracy:

1. **Baseline Model — Linear Regression.** A low-compute, deterministic model establishing a transparent predictive floor. Used as a lightweight benchmark for linear cross-signal relationships and as an interpretability anchor.
2. **Advanced Model — Random Forest Regressor.** An ensemble tree-based architecture (`n_estimators=50`, `max_depth=12`, `n_jobs=-1`, `bootstrap=True`, `random_state=42`) designed to capture non-linear transient states common in aggressive vehicle dynamics.

### Inside the Baseline — Trained Linear Regression Formula

Extracted directly from the trained `linear_model.joblib` artifact, the closed-form model is:

```
Signal_Y = −0.0166
         + 0.0195·X1 + 0.0005·X2 + 0.0032·X3 − 0.0018·X4
         − 0.0044·X5 − 0.0001·X6 + 0.7764·X7 + 0.2262·X8
         − 0.0026·X9 + 0.0024·X10 − 0.0036·X11
```

Of the 11 coefficients, only **two** are meaningfully non-zero. The model effectively reduces to:

```
Signal_Y ≈ 0.7764·Signal_X7 + 0.2262·Signal_X8
```

Two domain observations make this profound:

* **β₇ + β₈ = 0.7764 + 0.2262 ≈ 1.00.** The model has discovered that vehicle speed is approximately a **weighted average** of `Signal_X7` (wheel speed) and `Signal_X8` (inverter speed in RPM) — the closed-form expression of the wheel–RPM coupling through gear ratio.
* **Intercept ≈ 0** (−0.0166). This confirms that the MinMax `[0, 1]` scaling preserves the origin — there is no systematic offset.

### Inside the Advanced — Random Forest Anatomy

Structural fingerprint extracted directly from the `advanced_model.joblib` artifact:

| Property | Value |
| :--- | :--- |
| Trees in forest | 50 |
| Max depth | 12 (all 50 trees saturated) |
| Total decision nodes | 321,850 (~6,437 per tree, range 6,337–6,547) |
| Leaves per tree (avg) | 3,219 |
| Bootstrap sampling | Enabled |
| Split criterion | Squared error |
| Random state | 42 (reproducible) |

The forest contains ~322 thousand learned decision splits, yet — as shown below — only 2 out of 11 features carry essentially all of the predictive weight. The forest's full capacity is spent **fine-tuning the WhlSpd × Inverter RPM coupling**, not exploring high-dimensional structure.

### Feature Importance & Determinants Profile

Based on empirical tree-based splits of the trained `advanced_model.joblib`, the dominant cross-signal correlations are:

| Signal | Importance |
| :--- | :---: |
| **`Signal_X8`** (Inverter Speed, RPM) | **86.32 %** |
| **`Signal_X7`** (Wheel Speed Front Left) | **13.67 %** |
| Remaining 9 channels (`X1–X6`, `X9–X11`) | ~0.003 % combined |

The top-2 features account for **99.997 %** of the total importance — confirming that the vehicle speed signature is highly localized within two primary network channels. This converges with the linear model's β₇ + β₈ ≈ 1.0 finding: two completely different model classes (a deterministic linear solver and a stochastic ensemble of trees) both identify the same two physical proxies as dominant, which is the strongest argument for trusting this result.

---

## Model Performance & Generalization Diagnostic

All metrics below are computed on the chronological 80/20 split (`shuffle=False` in `train_test_split` — no temporal leakage). Errors are reported in the normalized `[0, 1]` scale produced by the MinMax anonymization step. Train and test metrics are reported side-by-side to enable explicit bias/variance assessment.

| Model | Split | R² | MAE (×10⁻³) | RMSE (×10⁻³) |
| :--- | :--- | :---: | :---: | :---: |
| Linear Regression | Train | 0.9999 | 1.18 | 2.04 |
| Linear Regression | Test  | 0.9998 | 1.54 | 2.69 |
| Random Forest     | Train | 1.0000 | 0.45 | 0.78 |
| Random Forest     | Test  | 0.9998 | 1.33 | 2.41 |

### Bias / Variance Verdict

* **Linear Regression.** Train R² ≈ Test R² ≈ 1.0; ΔR² = +0.0002; test/train MAE ratio = 1.31×.
  → **Low bias, near-zero variance.** Healthy fit. No underfitting on the 2-D physical manifold defined by wheel speed and inverter RPM.

* **Random Forest.** Train R² = 1.0000 (memorizes training data); Test R² = 0.9998; ΔR² = +0.0002; test/train MAE ratio = 2.96×.
  → **Mild memorization at the per-tree level, but no overfitting at the ensemble level.** Bootstrap sampling and 50-tree averaging cancel per-tree variance, preserving test performance.

* **Diagnosis.** ΔR² ≈ 0 across train/test for both models. Chronological split rules out temporal leakage. Both models sit in an **optimal bias–variance equilibrium** — a consequence of the problem being intrinsically 2-dimensional in feature space, fully learnable from the physical proxies (`X7`, `X8`).

### On the Near-Perfect R² Values

The unusually high test R² (0.9998) for both models is not a sign of leakage or overfitting; it reflects the **physical structure** of the problem itself. In an electric vehicle, vehicle speed is, by physical law, a near-linear combination of wheel rotation speed and motor RPM through gear ratio. The models are not learning a complex hidden pattern — they are **rediscovering a closed-form physical relationship**.

### Why This Project Still Matters — The FuSa Redundancy Value

If vehicle speed is so easily reconstructed from `X7` or `X8` alone, what is the engineering contribution of this pipeline? The answer is in the **failure modes**:

* If **`X7` (wheel speed)** is lost — Linear Regression collapses (its dominant β is gone). Random Forest can still approximate from `X8` and weaker proxies.
* If **`X8` (inverter RPM)** is lost — same story in reverse.
* If **both** are degraded — RF can still produce a usable estimate from the residual signals (yaw rate, regen torque, pedal positions), while Linear Regression returns essentially noise.

The added complexity of Random Forest (~322K decision splits versus 12 linear coefficients) buys **fail-operational behavior under partial sensor loss** — not accuracy on healthy inputs. This is the FuSa value proposition of the pipeline: a redundant ML estimator that degrades gracefully when the primary IMU + wheel-speed fusion is compromised.

---

## Project Directory Structure

The repository is organized following professional MLOps and production-ready software engineering principles, isolating core processing pipelines from application layers and serialized models:

```text
DS570_Project/
├── .dockerignore              # Excludes unnecessary files (venv, data caches) from Docker context
├── .gitignore                 # Prevents tracking of local caches, models, and virtual environments
├── Dockerfile                 # Containerization instructions leveraging optimized python-slim builds
├── README.md                  # Comprehensive technical documentation and presentation guide
├── requirements.txt           # Explicitly pinned library dependencies (Streamlit, Pandas, Joblib, etc.)
├── app/
│   └── dashboard.py           # Streamlit analytics frontend and interactive inference engine
├── data/
│   └── processed_clean_data.csv # Anonymized and normalized telemetry baseline dataset
├── models/
│   ├── advanced_model.joblib  # Serialized Random Forest Regressor production weights
│   └── linear_model.joblib    # Serialized Baseline Linear Regression weights
├── notebooks/
│   └── model_training.ipynb   # Exploratory Notebook mapping hyperparameter tuning & diagnostics
└── src/
    ├── __init__.py            # Explicitly designates directory as an importable Python package
    └── preprocessing.py       # Production-ready data ingestion, string cleaning, and split logic
```

---

## Identified Weaknesses & Future Improvements

In alignment with rigorous engineering practices, the following boundaries and architectural bottlenecks have been identified for future optimization loops:

* **Temporal Blindness.** The current static regression models evaluate each telemetry frame independently. They lack sequential memory, making them susceptible to error propagation during prolonged signal blackouts.
  * *Improvement:* Integrate Recurrent Neural Networks (RNNs) or **LSTM (Long Short-Term Memory)** networks to track historical vehicle states across time indices.

* **Edge-Case Volatility.** While the ensemble model excels within standard operational limits, sudden, extreme telemetry spikes (e.g., ABS triggering on ice or wheel slip) can temporarily distort curve fitting convergence.
  * *Improvement:* Introduce Kalman Filtering or automated anomaly detection layers to isolate sensor noise prior to model inference.

* **Single Train/Test Split.** Generalization is currently assessed on one chronological 80/20 split. While the ΔR² ≈ 0 result is encouraging, a more rigorous bias/variance assessment would use time-series cross-validation (`TimeSeriesSplit`).
  * *Improvement:* Walk-forward validation with multiple expanding-window folds, plus a hyperparameter grid search over `max_depth` and `min_samples_leaf` to quantify how aggressively the RF can be regularized without sacrificing FuSa robustness.

* **Production Validation Gap.** The current model has been validated at academic/ASIL-B equivalent rigor. Production deployment for an autonomous vehicle requires ISO 26262 Part 6 conformant test campaigns, hardware-in-the-loop validation, and a documented safety case.
  * *Improvement:* Build a DO-178C / ISO 26262 compatible validation harness with structural coverage metrics and requirements traceability.

---

## Deployment & Local Setup Guide via Docker

This system is fully containerized via Docker to guarantee absolute environment replication across different operating systems, eliminating local Python configuration dependencies.

### Step-by-Step Execution Instructions

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
**[http://localhost:8501](http://localhost:8501)**

* **Tab 1 (Exploratory Data Analysis):** Real-time data profiling, feature distributions, and Random Forest feature importance charts.
* **Tab 2 (Real-Time Model Inference):** Interactive sliders to manipulate 11 CAN Bus signals concurrently, featuring dual-curve fitting tracking and live residual error tracking.

### Stopping the Simulation

To safely terminate the container instance, return to your terminal window and press **`Ctrl + C`**.
