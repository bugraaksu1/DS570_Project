import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import load_and_clean_data

# Page layout setup for broad telemetry visibility
st.set_page_config(page_title="Vehicle Speed ML Dashboard", layout="wide")

# Main interface entry point
st.title("🚗 Vehicle Speed Prediction & Telemetry Analytics")
st.markdown("This dashboard integrates exploratory data analysis with real-time machine learning inference for CAN Bus signal validation.")

# Cached data loader to optimize dashboard responsiveness
@st.cache_data
def get_dashboard_data():
    return load_and_clean_data()

# Load runtime dataframe
df = get_dashboard_data()

# Two-tabbed layout to separate descriptive analytics from live ML inference
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis (Descriptive)", "🤖 Real-Time Model Inference"])

# ==========================================
# TAB 1: DESCRIPTIVE & EXPLORATORY STATE
# ==========================================
with tab1:
    st.subheader("Dataset Overview & Descriptive Statistics")
    
    # Top-level asset KPIs for quick health check
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Telemetry Rows", f"{len(df):,}")
    col2.metric("Max Vehicle Speed", f"{df['Signal_Y'].max():.2f} km/h")
    col3.metric("Avg Vehicle Speed", f"{df['Signal_Y'].mean():.2f} km/h")
    
    st.markdown("---")
    
    # Target distribution plot using strict engineering nomenclature
    st.write("### Target Variable Distribution")
    fig_hist = px.histogram(
        df, x="Signal_Y", 
        nbins=50, 
        title="<b>Distribution of Vehicle Speed (Signal_Y) in Historical Dataset</b>",
        labels={"Signal_Y": "Vehicle Speed (km/h)", "count": "Frequency / Row Count"},
        color_discrete_sequence=['#1f77b4']
    )
    fig_hist.update_layout(hovermode="x", title_x=0.5)
    st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# TAB 2: PRODUCTION MODEL RUNTIME SIMULATOR
# ==========================================
with tab2:
    st.subheader("Live CAN Bus Telemetry Inference Simulator")
    
    # Resource caching to prevent heavy disk I/O on every slider interaction
    @st.cache_resource
    def load_ml_models():
        linear = joblib.load("models/linear_model.joblib")
        advanced = joblib.load("models/advanced_model.joblib")
        return linear, advanced

    linear_model, advanced_model = load_ml_models()
    
    st.info("Use the sidebar controllers to simulate real-time CAN Bus frame values and evaluate model prediction curves.")
    
    # Dynamic architecture toggle for quick cross-validation comparisons
    selected_model_type = st.radio("Select Active Model Architecture:", ["Baseline (Linear Regression)", "Advanced (Random Forest)"])
    
    # TODO: Connect the local telemetry sliders configuration block here 
    # and link the live dictionary payload to the active joblib estimator object.