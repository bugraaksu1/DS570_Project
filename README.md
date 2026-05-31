# AI-Assisted Signal Regeneration in Critical CAN Bus Interruptions

**Course:** Özyeğin University 2025-2026 Spring - DS570 Term Project  
**Author:** Şakir Buğra Aksu  

An end-to-end data science and MLOps pipeline encompassing data processing, feature engineering, interactive telemetry dashboards, predictive machine learning modeling, and containerized deployment.

---

## Project Overview & Problem Statement
In modern automotive Electrical/Electronic (E/E) architectures, critical sensor signals transmitted over vehicle networks (CAN Bus, CAN-FD, LIN) are vulnerable to temporary or permanent interruptions. These dropouts can occur due to physical hardware failures, wiring harness short circuits, or malicious cyber interventions such as spoofing and Denial-of-Service (DoS) attacks.

When a primary sensor—such as the vehicle speed sensor—fails, dependent ADAS modules and chassis control units lose critical telemetry, compromising passenger safety. 

**Objective:** This project implements an intelligent, real-time signal fallback mechanism. By exploiting spatial and temporal correlations among other active network signals (e.g., Wheel Speeds, Engine RPM, Throttle Position), machine learning estimators dynamically reconstruct and regenerate the lost `Vehicle_Speed` (`Signal_Y`) telemetry directly at runtime.

---

## Dataset & Proprietary Safety (NDA Compliance)
The underlying data is derived from actual CAN Bus logs captured during real-world vehicle field tests. To strictly adhere to corporate Non-Disclosure Agreements (NDAs) and safeguard intellectual property, rigorous sanitization and anonymization protocols were enforced:

* **Signal Masking (DBC Independence):** Proprietary CAN IDs, message frames, and database signal names are entirely removed. Features are generalized into abstract representations spanning `Signal_X1` through `Signal_X11`, with the target velocity mapped as `Signal_Y`.
* **Mathematical Anonymization:** To conceal exact physical hardware performance thresholds, values are scaled and normalized based on feature distribution envelopes.
* **Automated Runtime Ingestion:** In compliance with zero-local-file requirements, the application hosts no internal datasets. Instead, the isolated pipeline streams the data directly from a public remote repository anchor via Pandas when the Docker container initializes.

---

## Methodology & Model Architecture
The pipeline evaluates two distinct modeling classes to balance computational latency against predictive accuracy:

1. **Baseline Model (Linear Regression):** A low-compute, deterministic model used to establish a predictive floor. It serves as a lightweight benchmark for linear cross-signal relationships.
2. **Advanced Model (Random Forest Regressor):** An ensemble tree-based architecture designed to capture high-frequency, non-linear transient states common in aggressive vehicle dynamics.

### Feature Importance & Determinants Profile
Based on empirical tree-based splits, the system identifies the dominant cross-signal correlations natively:
* **`Signal_X8`:** Accountable for **83.82%** of the total predictive weight.
* **`Signal_X7`:** Accountable for **16.17%** of the total predictive weight.
* The remaining channels (`Signal_X1` to `Signal_X6`, `Signal_X9` to `Signal_X11`) supply residual noise filtering, confirming that the vehicle speed signature is highly localized within two primary network channels.

---

## 📈 Model Performance & Evaluation Results
The estimators achieved the following cross-validation diagnostics under baseline telemetry configurations:

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Baseline (Linear Regression)** | `0.842` | `3.12 km/h` | `4.05 km/h` |
| **Advanced (Random Forest)** | **`0.968`** | **`1.08 km/h`** | **`1.42 km/h`** |

### Curve Fitting & Residual Analysis
* **Convergence:** The Random Forest model demonstrates tight curve fitting against the actual ground truth, capturing rapid accelerations and deceleration transients seamlessly.
* **Error Distribution:** Residual diagnostics indicate that the estimation errors are normally distributed and strictly clustered around the zero-error line, validating the absence of systematic model bias.

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
    ├── __init__.py            # Explicitly designates directory as a importable Python package
    └── preprocessing.py       # Production-ready data ingestion, string cleaning, and split logic

## Deployment & Local Setup Guide via Docker


This system is completely containerized via Docker to guarantee absolute environment replication across different operating systems, eliminating local python configuration dependencies.



### Step-by-Step Execution Instructions
#### 1. Clone the Repository

Open a terminal (Git Bash, PowerShell, or Command Prompt) and pull the project workspace:



    git clone [https://github.com/bugraaksu1/DS570_Project.git](https://github.com/bugraaksu1/DS570_Project.git)

  

    cd DS570_Project


#### 2. Build the Docker Image

    docker build -t vehicle-speed-dashboard .


#### 3. Run the Container

    docker run -p 8501:8501 vehicle-speed-dashboard


#### 4. Access the Live Dashboard

    http://localhost:8501
