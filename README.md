# DS570_Project
Özyeğin Univ. 2025-2026 Spring DS570 Term Project by Şakir Buğra Aksu. An end-to-end data science pipeline including data processing, visualization, interactive dashboards, predictive ML modeling &amp; evaluation, Git version control, and Docker.

### Data Architecture & Future Roadmap

This project implements an end-to-end machine learning pipeline to dynamically forecast gold prices. Currently, the primary data ingestion module utilizes the `yfinance` library to automatically fetch historical and near real-time market data (e.g., Open, Close, High, Low, Volume) without requiring user authentication or manual downloads. This ensures seamless reproducibility and rapid containerized deployment.

**Model Extensibility and Feature Integration**
The current architecture is intentionally designed to be modular and highly extensible. While the baseline model relies on core financial metrics and technical indicators derived from Yahoo Finance, the data pipeline can be easily scaled. 

Future iterations of this project plan to integrate additional data ingestion pipelines from alternative, non-Yahoo Finance platforms. By incorporating external APIs (such as FRED for macroeconomic indicators or alternative data sources for global market sentiment), we can introduce diverse input variables into the model. These external datasets will be engineered into advanced features—such as inflation rates, central bank interest rate decisions, or geopolitical risk indices—to further enhance the model's predictive accuracy, robustness, and ability to capture complex market dynamics.
