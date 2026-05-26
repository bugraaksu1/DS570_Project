# 1. Use a lightweight, stable, and modern Python 3.12 image
FROM python:3.12-slim

# 2. Create the working directory inside the container
WORKDIR /workspace

# 3. Update Linux system dependencies (tools required for Plotly and GCC)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements.txt first and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all project source and asset folders into the container
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

# 6. Expose the default port Streamlit uses to communicate with the outside world
EXPOSE 8501

# 7. Command that automatically launches the dashboard when the container starts
ENTRYPOINT ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]