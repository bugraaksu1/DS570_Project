import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from src.preprocessing import load_and_clean_data

# Page layout setup for broad telemetry visibility
st.set_page_config(page_title="Vehicle Speed ML Dashboard", layout="wide")

# ------------------------------------------------------------------
# 🔥 INJECT CUSTOM CSS TO HIDE AI/STREAMLIT DEVELOPER MENUS
# ------------------------------------------------------------------
# This removes the "Deploy", "Theme mode", hamburger menu, and "Made with Streamlit" footer
# to ensure the application looks like a professionally custom-built corporate tool.
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none !important;}
    </style>
    """, unsafe_allow_html=True)

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
    
    counts, bins = np.histogram(df['Signal_Y'], bins=50)
    fig_hist = go.Figure(data=[go.Bar(x=bins[:-1], y=counts, marker_color='#1f77b4')])
    
    fig_hist.update_layout(
        title="<b>Distribution of Vehicle Speed (Signal_Y) in Historical Dataset</b>",
        xaxis_title="Vehicle Speed (km/h)",
        yaxis_title="Frequency / Row Count",
        title_x=0.5,
        hovermode="x"
    )
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
    
    # Layout split: Left side for telemetry controllers, Right side for live plot
    col_controls, col_plot = st.columns([1, 2])
    
    with col_controls:
        st.write("### 🎛️ CAN Bus Inputs")
        selected_model_type = st.radio("Active Model Architecture:", ["Baseline (Linear Regression)", "Advanced (Random Forest)"])
        
        st.markdown("---")
        # Generating the 11 telemetry input sliders dynamically based on dataset scales
        input_data = {}
        for i in range(1, 12):
            col_name = f"Signal_X{i}"
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            mean_val = float(df[col_name].mean())
            
            input_data[col_name] = st.slider(
                label=f"{col_name} (Telemetry Input)",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100.0
            )
            
    with col_plot:
        st.write("### 📈 Live Model Inference Result")
        
        # Structure the payload into a DataFrame for the estimator object
        input_df = pd.DataFrame([input_data])
        
        # Execute prediction based on selected architecture toggle
        if selected_model_type == "Baseline (Linear Regression)":
            prediction = linear_model.predict(input_df)[0]
            model_label = "Linear Regression"
        else:
            prediction = advanced_model.predict(input_df)[0]
            model_label = "Random Forest (Advanced)"
            
        # Display the crisp numeric result via a massive metric block
        st.metric(label=f"Predicted Vehicle Speed ({model_label})", value=f"{prediction:.2f} km/h")
        
        # Comparative Gauge/Bar visualization to demonstrate signal validation bounds
        fig_res = go.Figure()
        
        # Historical range reference bars
        fig_res.add_trace(go.Bar(
            name="Dataset Max Speed",
            x=["Speed Comparison"],
            y=[df['Signal_Y'].max()],
            marker_color='rgba(200, 200, 200, 0.5)'
        ))
        
        # Live inference indicator bar
        fig_res.add_trace(go.Bar(
            name="Current Live Prediction",
            x=["Speed Comparison"],
            y=[prediction],
            marker_color='#2ca02c'
        ))
        
        fig_res.update_layout(
            barmode='overlay',
            title="<b>Current Prediction vs. Historical Upper Limit</b>",
            yaxis_title="Vehicle Speed (km/h)",
            title_x=0.5,
            height=450
        )
        
        st.plotly_chart(fig_res, use_container_width=True)