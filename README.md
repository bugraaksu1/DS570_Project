# AI-Assisted Signal Regeneration in Critical CAN Bus Interruptions

**Course:** Özyeğin University 2025-2026 Spring - DS570 Term Project  
**Author:** Şakir Buğra Aksu  

An end-to-end data science and MLOps pipeline encompassing data processing, feature engineering, interactive telemetry dashboards, predictive machine learning modeling, and containerized deployment.

---

##  Project Overview & Problem Statement
In modern automotive Electrical/Electronic (E/E) architectures, critical sensor signals transmitted over vehicle networks (CAN Bus, CAN-FD, LIN) are vulnerable to temporary or permanent interruptions. These dropouts can occur due to physical hardware failures, wiring harness short circuits, or malicious cyber interventions such as spoofing and Denial-of-Service (DoS) attacks.

When a primary sensor—such as the vehicle speed sensor—fails, dependent ADAS modules and chassis control units lose critical telemetry, compromising passenger safety. 

**Objective:** This project implements an intelligent, real-time signal fallback mechanism. By exploiting spatial and temporal correlations among other active network signals, machine learning estimators dynamically reconstruct and regenerate the lost `Vehicle_Speed` (`Signal_Y`) telemetry directly at runtime.

---

##  Dataset & Proprietary Safety (NDA Compliance)
The underlying data is derived from actual CAN Bus logs captured during real-world vehicle field tests. To strictly adhere to corporate Non-Disclosure Agreements (NDAs) and safeguard intellectual property, rigorous sanitization and anonymization protocols were enforced:

* **Signal Masking (DBC Independence):** Proprietary CAN IDs, message frames, and database signal names are entirely removed. Features are generalized into abstract representations spanning `Signal_X1` through `Signal_X11`, with the target velocity mapped as `Signal_Y`.
* **Mathematical Anonymization:** To conceal exact physical hardware performance thresholds, values are scaled and normalized based on feature distribution envelopes.
* **Automated Runtime Ingestion:** In compliance with zero-local-file requirements, the application hosts no internal datasets. Instead, the isolated pipeline streams the data directly from a public remote repository anchor via Pandas when the Docker container initializes.

---

##  Methodology & Model Architecture
The pipeline evaluates two distinct modeling classes to balance computational latency against predictive accuracy and fault tolerance:

1. **Baseline Model (Linear Regression):** A low-compute, deterministic model used to establish a predictive baseline and deconstruct linear mathematical proxies.
2. **Advanced Model (Random Forest Regressor):** An ensemble tree-based architecture designed to capture high-frequency, non-linear transient states and provide architectural redundancy.

### The Physical Proxy Principle & Intrinsic Linearity
Empirical analysis of the serialized models indicates that the underlying target (`Signal_Y`) is strongly tied to physical proxies (e.g., Wheel Speed and Inverter RPM). The machine learning models natively discover the near-closed-form mathematical equations of automotive dynamics, yielding near-perfect fits based on highly localized network channels without any data leakage.

---

##  Model Performance & Evaluation Results
The estimators achieved the following cross-validation diagnostics under baseline telemetry configurations. Training and test splits were rigorously enforced chronologically using `shuffle=False` to preserve time-series integrity and eliminate verification bias:

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Baseline (Linear Regression)** | **`0.9998`** | `0.0015 (Scaled)` | `0.0026 (Scaled)` |
| **Advanced (Random Forest)** | **`0.9998`** | `0.0013 (Scaled)` | `0.0024 (Scaled)` |

###  Functional Safety (ISO 26262) & Cyber-Physical Robustness
While both models display stellar $R^2 \approx 1.0$ diagnostics due to the underlying physical linearity of the proxies, their structural resilience diverges in edge-case failure scenarios:
* **Linear Regression Vulnerability:** Strictly relies on fixed coefficient matrices. If primary physical proxies are severed (0.0) due to a sensor short circuit or DoS attack, the linear formula collapses and yields hazardous output.
* **Random Forest Fault Tolerance:** The 100-tree ensemble architecture leverages diverse feature subsets (Bootstrap aggregation). If primary channels drop to zero, the remaining trees successfully reconstruct velocity metrics using alternative signal cross-correlations (e.g., pedal positions, battery current, torque commands), providing crucial architectural redundancy for **Limp-Home** safety modes.

---

##  Project Directory Structure

The repository is organized following professional MLOps and production-ready software engineering principles, isolating core processing pipelines from application layers and serialized models:

```text
DS570_Project/
├── .dockerignore              # Excludes unnecessary files (venv, data caches) from Docker context
├── .gitignore                 # Prevents tracking of local caches, models, and virtual environments
├── Dockerfile                 # Containerization instructions leveraging optimized python-slim builds
├── README.md                  # Comprehensive technical documentation and presentation guide
├── requirements.txt           # Explicitly pinned library dependencies (Streamlit, Pandas, Joblib, etc.)
├── app/
│   └── dashboard.py           # Streamlit analytics frontend with integrated FuSa Stress Testing
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

##  Deployment & Local Setup Guide via Docker

This system is completely containerized via Docker to guarantee absolute environment replication across different operating systems, eliminating local python configuration dependencies.

###  Step-by-Step Execution Instructions

#### 1. Clone the Repository

Open a terminal (Git Bash, PowerShell, or Command Prompt) and pull the project workspace:

```bash
git clone [https://github.com/bugraaksu1/DS570_Project.git](https://github.com/bugraaksu1/DS570_Project.git)
cd DS570_Project

```

#### 2. Build the Docker Image

Compile the isolated environment, including the Python runtime layer and dependencies, by executing:

```bash
docker build -t vehicle-speed-dashboard .

```

*(Ensure the trailing dot `.` is included so Docker detects the local context).*

#### 3. Run the Container

Spin up the interactive telemetry simulation server:

```bash
docker run -p 8501:8501 vehicle-speed-dashboard

```

#### 4. Access the Live Dashboard

Once initialized, open any modern web browser and navigate to:
 **[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)**

* **Tab 1 (Real-Time Inference Simulator):** Features interactive sliders to manipulate 11 CAN Bus signals concurrently. Includes an active **Functional Safety Stress Test checkbox** to simulate real-time cyber attacks or physical sensor drops, demonstrating Random Forest's real-time signal reconstruction resilience.
* **Tab 2 (Generalization & Diagnostics):** Displays verified train/test model parameters, bias-variance metrics, and the de-serialized kumeled mathematical equation extracted straight from production weights.

###  Stopping the Simulation

To safely terminate the container instance, return to your terminal window and press **`Ctrl + C`**.
