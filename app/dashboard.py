"""
Vehicle Speed Prediction & Telemetry Analytics Dashboard
DS570 Term Project — AI-Assisted Signal Regeneration in Critical CAN Bus Interruptions
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_LOCAL_PATH = Path("data/Finalized_Dataset.csv")
DATA_GITHUB_URL = (
    "https://raw.githubusercontent.com/bugraaksu1/DS570_Project/main/"
    "data/Finalized_Dataset.csv"
)  # remote fallback if the local file is missing

LINEAR_MODEL_PATH = "models/linear_model.joblib"
ADVANCED_MODEL_PATH = "models/advanced_model.joblib"

FEATURE_COLS = [f"Signal_X{i}" for i in range(1, 12)]
TARGET_COL = "Signal_Y"
TEST_SIZE = 0.2  # same split ratio as training (chronological, shuffle=False)
HISTORY_WINDOW = 30

st.set_page_config(page_title="Vehicle Speed ML Dashboard", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Vehicle Speed Prediction & Telemetry Analytics")
st.markdown(
    "This dashboard integrates exploratory data analysis with real-time machine "
    "learning inference for CAN Bus signal validation. All signals are anonymized "
    "(`Signal_X1…X11`) and MinMax-scaled to **[0, 1]** for NDA compliance."
)


# ------------------------------------------------------------------
# Data & model loading
# ------------------------------------------------------------------
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the real telemetry dataset.

    Tries the local file first (Docker image ships it), then falls back to the
    public GitHub anchor. `sep=None` + python engine auto-detects comma vs
    semicolon delimiters, mirroring src/preprocessing.py.
    """
    source = DATA_LOCAL_PATH if DATA_LOCAL_PATH.exists() else DATA_GITHUB_URL
    df = pd.read_csv(source, sep=None, engine="python")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # CSV is exported with decimal commas (e.g. "0,8423") -> convert to float
    expected = FEATURE_COLS + [TARGET_COL]
    for col in expected:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = int(df[expected].isna().any(axis=1).sum())
    if bad_rows:
        df = df.dropna(subset=expected)
        if len(df) == 0:
            raise ValueError(
                "All rows became NaN after numeric coercion — check the CSV's "
                "delimiter/decimal format."
            )
    return df.reset_index(drop=True)


@st.cache_resource
def load_ml_models():
    linear = joblib.load(LINEAR_MODEL_PATH)
    advanced = joblib.load(ADVANCED_MODEL_PATH)
    return linear, advanced


@st.cache_data
def compute_test_metrics(_model, X: pd.DataFrame, y: pd.Series, cache_key: str):
    """R2, MAE, RMSE of a model on the chronological test split."""
    preds = _model.predict(X)
    residuals = y.to_numpy() - preds
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return r2, mae, rmse


try:
    df = load_dataset()
except Exception as exc:
    st.error(
        "Dataset could not be loaded. Place `Finalized_Dataset.csv` under "
        f"`{DATA_LOCAL_PATH}` or check the GitHub anchor URL.\n\nDetails: {exc}"
    )
    st.stop()

try:
    linear_model, advanced_model = load_ml_models()
except Exception as exc:
    st.error(
        "Model artifacts could not be loaded. Expected "
        f"`{LINEAR_MODEL_PATH}` and `{ADVANCED_MODEL_PATH}`.\n\nDetails: {exc}"
    )
    st.stop()

# Chronological split — identical to training (shuffle=False)
split_idx = int(len(df) * (1 - TEST_SIZE))
test_df = df.iloc[split_idx:].reset_index(drop=True)
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]


# Button callbacks: callbacks run before widgets render, so writing
# slider session_state here is safe.
def _load_frame_into_sliders(frame_index: int) -> None:
    """Copy one test-region frame into the slider states."""
    frame_index = int(min(max(frame_index, 0), len(test_df) - 1))
    for col_name in FEATURE_COLS:
        st.session_state[f"slider_{col_name}"] = float(
            np.clip(test_df.loc[frame_index, col_name], 0.0, 1.0)
        )


def _advance_frame(n_frames: int) -> None:
    """Move to the next test frame and load its telemetry."""
    st.session_state.timestamp_counter = min(
        st.session_state.timestamp_counter + 1, n_frames - 1
    )
    _load_frame_into_sliders(st.session_state.timestamp_counter)

# ------------------------------------------------------------------
# Session state for the live simulator
# ------------------------------------------------------------------
if "timestamp_counter" not in st.session_state:
    st.session_state.timestamp_counter = 0
if "actual_history" not in st.session_state:
    st.session_state.actual_history = []
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

tab1, tab2 = st.tabs(
    ["📊 Exploratory Data Analysis & Feature Weights", "🤖 Real-Time Model Inference"]
)

# ==========================================
# TAB 1: DESCRIPTIVE & FEATURE WEIGHTS
# ==========================================
with tab1:
    st.subheader("Dataset Overview & Descriptive Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Telemetry Rows", f"{len(df):,}")
    col1.metric("Train / Test Split", f"{split_idx:,} / {len(test_df):,} (chronological)")
    col2.metric("Max Vehicle Speed (scaled)", f"{df[TARGET_COL].max():.4f}")
    col2.metric("Avg Vehicle Speed (scaled)", f"{df[TARGET_COL].mean():.4f}")
    col3.metric("Telemetry Channels", f"{len(FEATURE_COLS)}")
    col3.metric("Sampling Grid", "10 ms (resampled)")

    st.markdown("---")

    col_dist, col_lr_imp, col_rf_imp = st.columns(3)

    with col_dist:
        st.write("### Target Variable Distribution")
        counts, bins = np.histogram(df[TARGET_COL], bins=50)
        fig_hist = go.Figure(data=[go.Bar(x=bins[:-1], y=counts, marker_color="#1f77b4")])
        fig_hist.update_layout(
            xaxis_title="Vehicle Speed (scaled, [0,1])",
            yaxis_title="Frequency / Row Count",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_lr_imp:
        st.write("### Linear Regression Coefficients")
        st.caption("Extracted live from `linear_model.joblib` — no hardcoded values.")

        lr_df = (
            pd.DataFrame(
                {"Signal": FEATURE_COLS, "Coefficient": linear_model.coef_}
            ).sort_values(by="Coefficient", ascending=True)
        )
        fig_lr = go.Figure(
            data=[
                go.Bar(
                    x=lr_df["Coefficient"],
                    y=lr_df["Signal"],
                    orientation="h",
                    marker_color="#d62728",
                )
            ]
        )
        fig_lr.update_layout(
            xaxis_title="Coefficient Value (β)",
            yaxis_title="CAN Bus Input Signals",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_lr, use_container_width=True)
        st.caption(f"Intercept β₀ = {linear_model.intercept_:.4f}")

    with col_rf_imp:
        st.write("### Random Forest Feature Importance")
        st.caption("Extracted live from `advanced_model.joblib` — no hardcoded values.")

        rf_imp_df = (
            pd.DataFrame(
                {"Signal": FEATURE_COLS, "Importance": advanced_model.feature_importances_}
            ).sort_values(by="Importance", ascending=True)
        )
        fig_rf = go.Figure(
            data=[
                go.Bar(
                    x=rf_imp_df["Importance"],
                    y=rf_imp_df["Signal"],
                    orientation="h",
                    marker_color="#1c1b35",
                )
            ]
        )
        fig_rf.update_layout(
            xaxis_title="Importance Score (Gini)",
            yaxis_title="CAN Bus Input Signals",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    st.markdown(
        """
        💡 **Engineering Insight:** Both architectures concentrate their statistical
        weight on `Signal_X7` (WhlSpd) and `Signal_X8` (Inverter RPM) — the two direct
        physical proxies of vehicle speed. The remaining nine channels carry redundant
        information that becomes critical under sensor loss — **importance ≠ capability**,
        as the ablation study below demonstrates.
        """
    )

    # Ablation study: evaluate the per-scenario retrained models live
    st.markdown("---")
    st.subheader("Sensor-Loss Ablation Study (FuSa Redundancy)")

    ABLATION_SCENARIOS = [
        ("Drop X7", ["Signal_X7"], "drop_x7"),
        ("Drop X8", ["Signal_X8"], "drop_x8"),
        ("Drop X7 + X8", ["Signal_X7", "Signal_X8"], "drop_x7x8"),
    ]

    @st.cache_data
    def compute_ablation_results():
        """Evaluate baseline + retrained ablation models on the test split."""
        rows = []
        for label, mdl in [("Linear Regression", linear_model),
                           ("Random Forest", advanced_model)]:
            p = mdl.predict(X_test)
            res = y_test.to_numpy() - p
            rows.append(dict(
                Scenario="Baseline (all 11)", Model=label,
                R2=1.0 - float(np.sum(res**2)) / float(np.sum((y_test - y_test.mean())**2)),
                MAE=float(np.mean(np.abs(res))) * 1e3,
            ))
        try:
            crisis_imp = None
            for name, drop, tag in ABLATION_SCENARIOS:
                feats = [f for f in FEATURE_COLS if f not in drop]
                for label, prefix in [("Linear Regression", "linear_model"),
                                      ("Random Forest", "advanced_model")]:
                    mdl = joblib.load(f"models/{prefix}_{tag}.joblib")
                    p = mdl.predict(X_test[feats])
                    res = y_test.to_numpy() - p
                    rows.append(dict(
                        Scenario=name, Model=label,
                        R2=1.0 - float(np.sum(res**2)) / float(np.sum((y_test - y_test.mean())**2)),
                        MAE=float(np.mean(np.abs(res))) * 1e3,
                    ))
                    if tag == "drop_x7x8" and prefix == "advanced_model":
                        crisis_imp = pd.Series(
                            mdl.feature_importances_, index=list(mdl.feature_names_in_)
                        ).sort_values(ascending=False)
        except FileNotFoundError:
            return None, None
        out = pd.DataFrame(rows)
        base = out[out.Scenario == "Baseline (all 11)"].set_index("Model")["MAE"]
        out["MAE multiplier"] = out.apply(lambda r: r["MAE"] / base[r["Model"]], axis=1)
        return out, crisis_imp

    ablation_df, crisis_imp = compute_ablation_results()

    if ablation_df is None:
        st.info(
            "Ablation artifacts (`models/*_drop_*.joblib`) not found. Run "
            "`notebooks/DS570_Model_Experiments.ipynb` to generate them."
        )
    else:
        st.caption(
            "Both models **retrained from scratch** under four sensor-availability "
            "scenarios; all metrics computed live on the chronological test split."
        )
        col_abl_chart, col_abl_detail = st.columns([3, 2])

        scenario_order = ["Baseline (all 11)"] + [s[0] for s in ABLATION_SCENARIOS]
        with col_abl_chart:
            fig_abl = go.Figure()
            for label, color in [("Linear Regression", "#8A94A6"),
                                 ("Random Forest", "#06D6A0")]:
                sub = ablation_df[ablation_df.Model == label].set_index("Scenario")
                sub = sub.reindex(scenario_order)
                fig_abl.add_trace(go.Bar(
                    x=scenario_order, y=sub["R2"], name=label, marker_color=color,
                    text=[f"{v:.2f}" for v in sub["R2"]], textposition="outside",
                ))
            fig_abl.update_layout(
                title="<b>Test R² Under Sensor-Loss Conditions</b>",
                yaxis_title="Test R²", yaxis_range=[0, 1.12],
                barmode="group", title_x=0.5, height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_abl, use_container_width=True)

        with col_abl_detail:
            st.write("**Degradation table** (MAE multiplier vs. baseline)")
            piv = ablation_df.pivot(index="Scenario", columns="Model",
                                    values="MAE multiplier").reindex(scenario_order)
            st.dataframe(piv.style.format("{:.1f}×"), use_container_width=True)
            if crisis_imp is not None:
                st.write("**Crisis-model importance redistribution** (no X7/X8)")
                st.caption(
                    f"`{crisis_imp.index[0]}` takes over **{crisis_imp.iloc[0]:.1%}** "
                    "of the importance — a channel that was ~0% under normal "
                    "conditions becomes the backbone of the backup model."
                )

        crisis = ablation_df[ablation_df.Scenario == "Drop X7 + X8"].set_index("Model")
        st.markdown(
            f"""
            🛡️ **FuSa Verdict:** Single-sensor loss is harmless — X7/X8 are mutual
            physical proxies. When **both** fail, Linear Regression collapses to noise
            (R² = {crisis.loc['Linear Regression', 'R2']:.2f}) while Random Forest
            degrades gracefully (R² = {crisis.loc['Random Forest', 'R2']:.2f}):
            *fail-operational* instead of *fail-silent*.
            """
        )

# ==========================================
# TAB 2: RUNTIME INFERENCE SIMULATOR
# ==========================================
with tab2:
    st.subheader("Live CAN Bus Telemetry Inference Simulator")
    st.caption(
        "The simulator replays the **chronological test region** — telemetry frames the "
        "models never saw during training — faithfully reproducing the production scenario."
    )

    col_controls, col_plots = st.columns([1, 2])

    with col_controls:
        st.write("### 🎛️ Runtime Parameters")
        selected_model_type = st.radio(
            "Active Model Architecture:",
            ["Baseline (Linear Regression)", "Advanced (Random Forest)"],
        )
        active_model = (
            linear_model
            if selected_model_type == "Baseline (Linear Regression)"
            else advanced_model
        )

        if st.button("🔄 Clear Simulation Timeline"):
            st.session_state.actual_history = []
            st.session_state.pred_history = []
            st.session_state.timestamp_counter = 0
            st.session_state.last_recorded_frame = None
            _load_frame_into_sliders(0)
            st.rerun()

        st.markdown("---")

        st.write("### 📐 Model Performance Diagnostics")
        st.caption(f"Computed live on the {len(test_df):,}-sample chronological test split.")
        r2, mae, rmse = compute_test_metrics(
            active_model, X_test, y_test, cache_key=selected_model_type
        )
        st.markdown(
            f"""
            * **R² Score:** `{r2:.4f}`
            * **MAE:** `{mae * 1e3:.2f} ×10⁻³ (scaled)`
            * **RMSE:** `{rmse * 1e3:.2f} ×10⁻³ (scaled)`
            """
        )

        st.markdown("---")

        # Sliders use fixed keys and no value= param: if the default value
        # changed every rerun, Streamlit would treat it as a new widget and
        # reset the handle. Frames advance only via the Next Frame button.
        current_index = min(st.session_state.timestamp_counter, len(test_df) - 1)

        # Initialize slider states once (first run) from frame 0
        if f"slider_{FEATURE_COLS[0]}" not in st.session_state:
            _load_frame_into_sliders(current_index)

        st.write("### ⏱️ Test-Region Playback")
        col_b1, col_b2 = st.columns(2)
        col_b1.button(
            "⏭️ Next Frame",
            help="Advance to the next unseen test-region frame and load its telemetry into the sliders.",
            on_click=_advance_frame,
            args=(len(test_df),),
            use_container_width=True,
        )
        col_b2.button(
            "↩️ Reload Frame",
            help="Discard manual what-if edits and reload the current frame's real telemetry.",
            on_click=_load_frame_into_sliders,
            args=(current_index,),
            use_container_width=True,
        )

        input_data = {}
        for col_name in FEATURE_COLS:
            input_data[col_name] = st.slider(
                label=f"{col_name} (Telemetry Input)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=f"slider_{col_name}",
            )

    with col_plots:
        st.write("### 📈 Live Model Inference & Signal Verification")

        input_df = pd.DataFrame([input_data], columns=FEATURE_COLS)
        prediction = float(active_model.predict(input_df)[0])
        actual_value = float(test_df.loc[current_index, TARGET_COL])

        # One history point per frame; slider tweaks update the last point.
        if st.session_state.get("last_recorded_frame") != current_index:
            st.session_state.actual_history.append(actual_value)
            st.session_state.pred_history.append(prediction)
            st.session_state.last_recorded_frame = current_index
        else:
            st.session_state.pred_history[-1] = prediction
        if len(st.session_state.actual_history) > HISTORY_WINDOW:
            st.session_state.actual_history.pop(0)
            st.session_state.pred_history.pop(0)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Actual Ground Truth (scaled)", f"{actual_value:.4f}")
        col_m2.metric(
            "Predicted Inference (scaled)",
            f"{prediction:.4f}",
            delta=f"{prediction - actual_value:+.4f}",
        )
        col_m3.metric("Test-Region Frame", f"{current_index:,} / {len(test_df) - 1:,}")

        fig_curve = go.Figure()
        timeline_x = list(range(len(st.session_state.actual_history)))
        fig_curve.add_trace(
            go.Scatter(
                x=timeline_x,
                y=st.session_state.actual_history,
                mode="lines+markers",
                name="Actual Ground Truth",
                line=dict(color="#1f77b4", width=3, dash="dash"),
            )
        )
        fig_curve.add_trace(
            go.Scatter(
                x=timeline_x,
                y=st.session_state.pred_history,
                mode="lines+markers",
                name="Model Prediction",
                line=dict(color="#ff7f0e", width=3),
            )
        )
        fig_curve.update_layout(
            title="<b>Signal Verification: Actual vs. Prediction (Unseen Test Region)</b>",
            xaxis_title="Sequential Simulation Index",
            yaxis_title="Vehicle Speed (scaled)",
            title_x=0.5,
            height=350,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        st.write("### 🎯 Residual Error Distribution")
        residuals = np.array(st.session_state.pred_history) - np.array(
            st.session_state.actual_history
        )
        res_counts, res_bins = np.histogram(residuals, bins=15)
        fig_res = go.Figure(
            data=[go.Bar(x=res_bins[:-1], y=res_counts, marker_color="#2ca02c", opacity=0.75)]
        )
        fig_res.update_layout(
            title="<b>Error Deviation (Residual Distribution Around Zero-Line)</b>",
            xaxis_title="Error Margin (Predicted − Actual, scaled)",
            yaxis_title="Count / Interaction Hit",
            title_x=0.5,
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_res, use_container_width=True)

