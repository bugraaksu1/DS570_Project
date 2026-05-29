# DS570_Project
Özyeğin Univ. 2025-2026 Spring DS570 Term Project by Şakir Buğra Aksu. An end-to-end data science pipeline including data processing, visualization, interactive dashboards, predictive ML modeling &amp; evaluation, Git version control, and Docker.

## AI-Assisted Signal Regeneration in Critical CAN Bus Interruptions
### Project Overview
In modern in-vehicle communication networks (CAN Bus, CAN-FD), the interruption of signals from critical sensors due to hardware failure, short circuits, or cyber interventions (spoofing) directly endangers driving safety.

The main objective of this project is to use correlated signals on the network (such as Wheel Speeds, Engine RPM, Throttle Position, etc.) to regenerate the lost signal in real-time using machine learning algorithms during a critical scenario where the main Vehicle_Speed signal from the primary speed sensor is interrupted.

### Data Set & Privacy Policy (Anonymization & NDA Compliance)
The dataset used in this project is based on real CAN Bus logs obtained from real-world E/E (Electrical/Electronic) system architecture field tests. However, to protect trade secrets and strictly comply with NDA (Non-Disclosure Agreement) rules, rigorous anonymization (masking) procedures have been applied to the dataset:

### DBC Independence: 
No real CAN IDs or proprietary message/signal names were used. Column names have been generalized to formats like Signal_1, Signal_2, and Target_Velocity.

### Mathematical Normalization: 
To conceal physical hardware characteristics, all physical values have been scaled between 0 and 1 using MinMaxScaler.

### Automated Data Ingestion (Runtime): 
The dataset is not stored as a local file. Instead, it is automatically fetched without authentication from a public cloud URL (GitHub Raw) via Pandas when the application boots up on Docker.

## Project Planned Timeline
The development of this project is structured over a 4-week agile sprint:
### Week 1: Data Preparation & Ingestion Pipeline
Data collection from CAN Bus logs.
Data preprocessing, cleaning, and strict anonymization (masking).
Creating the final dataset and uploading it to a public GitHub repository for automated runtime ingestion.

### Week 2: Model Training & Evaluation
Establishing the baseline model (Zero-Order Hold).
Training advanced machine learning models (Linear Regression, Tree-Based Algorithms).
Performance measurement and hyperparameter tuning.

### Week 3: Interactive Dashboard Development
Building the user interface using Streamlit.
Integrating the trained models to visualize signal loss and real-time regeneration.

### Week 4: Deployment & Containerization
Writing the Dockerfile and requirements.txt.
Containerizing the entire application for seamless, dependency-free execution in any environment.

### Deployment & Local Setup Guide via Docker
###  Step-by-Step Execution Instructions

#### 1. Clone the Repository
Open your terminal (Git Bash, PowerShell, or Command Prompt), clone the project to your local machine, and navigate into the root directory:

    git clone [https://github.com/bugraaksu1/DS570_Project.git](https://github.com/bugraaksu1/DS570_Project.git)
    cd DS570_Project

#### 2. Build the Docker Image
    docker build -t vehicle-speed-dashboard .

#### 3. Running the Container
    docker run -p 8501:8501 vehicle-speed-dashboard

#### 4. Accessing the Live Dashboard
As soon as the container initializes, Streamlit's network outputs will appear in your terminal. Open any web browser (Chrome, Edge, Safari, Firefox, etc.) and head over to the following address to test the real-time telemetry model inference:
    
    👉 http://localhost:8501


