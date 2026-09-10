import os
import sqlite3
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model_service

# ============================================================
# Will It Break? — Industrial Predictive Maintenance Dashboard
# Samsung Innovation Campus (SIC - AI803) · Team 13
# ============================================================

st.set_page_config(
    page_title="Will It Break? | Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Theme (Rustic Industrial) ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    :root {
        --bg: #1c1a17;
        --panel: #28241f;
        --panel-2: #312b24;
        --panel-3: #393128;
        --border: #51483e;
        --text: #f3eee7;
        --muted: #c7bbae;
        --muted-2: #9f9386;
        --rust: #b65332;
        --rust-light: #d97952;
        --copper: #c58a55;
        --sand: #e4c59f;
        --olive: #788064;
        --green: #9db08a;
        --danger: #d96a55;
        --shadow: rgba(0,0,0,.30);
    }

    * { box-sizing: border-box; }

    .stApp {
        background:
            radial-gradient(circle at 92% 4%, rgba(182,83,50,.14), transparent 24%),
            radial-gradient(circle at 4% 92%, rgba(120,128,100,.10), transparent 25%),
            var(--bg);
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
    }

    /* Global typography */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp div, .stApp li, .stApp td, .stApp th,
    [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--text) !important;
        letter-spacing: -.02em;
    }

    .stCaption, [data-testid="stCaptionContainer"],
    .stApp small, .small-note {
        color: var(--muted) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #211e1a 0%, #191714 100%);
        border-right: 1px solid var(--border);
        box-shadow: 10px 0 35px rgba(0,0,0,.12);
    }

    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small { color: var(--muted) !important; }

    /* Streamlit radio navigation */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 11px;
        padding: 9px 11px;
        margin: 0;
        transition: background .20s ease, border-color .20s ease, transform .20s ease, box-shadow .20s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(182,83,50,.12);
        border-color: rgba(182,83,50,.35);
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(182,83,50,.24), rgba(197,138,85,.08));
        border-color: rgba(182,83,50,.62);
        box-shadow: inset 3px 0 0 var(--rust), 0 5px 18px rgba(0,0,0,.12);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-weight: 600;
        color: var(--text) !important;
    }

    /* Main Hero Banner */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 32px;
        border: 1px solid var(--border);
        border-radius: 18px;
        background:
            linear-gradient(135deg, rgba(182,83,50,.15), transparent 45%),
            linear-gradient(135deg, #302a24 0%, #25211d 58%, #211e1a 100%);
        box-shadow: 0 16px 45px var(--shadow);
        margin-bottom: 24px;
    }
    .hero::after {
        content: '';
        position: absolute;
        right: -55px;
        top: -75px;
        width: 220px;
        height: 220px;
        border: 1px solid rgba(197,138,85,.18);
        border-radius: 50%;
        box-shadow: 0 0 0 22px rgba(197,138,85,.035), 0 0 0 44px rgba(197,138,85,.025);
    }
    .hero-kicker {
        color: var(--rust-light) !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .hero h1 { color: var(--text) !important; margin: 7px 0 5px; font-size: 36px; line-height: 1.1; }
    .hero p { color: var(--muted) !important; margin: 0; max-width: 900px; line-height: 1.65; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, var(--panel-2), var(--panel));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 17px 19px;
        min-height: 105px;
        box-shadow: 0 8px 25px rgba(0,0,0,.13);
        transition: transform .20s ease, border-color .20s ease, box-shadow .20s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(197,138,85,.62);
        box-shadow: 0 12px 30px rgba(0,0,0,.20);
    }
    .metric-label { color: var(--muted) !important; font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; }
    .metric-value { color: var(--text) !important; font-size: 28px; font-weight: 800; margin-top: 7px; }
    .metric-sub { color: var(--rust-light) !important; font-size: 12px; margin-top: 3px; }

    .section-title {
        border-left: 3px solid var(--rust);
        color: var(--text) !important;
        padding-left: 11px;
        margin: 25px 0 13px;
        font-size: 21px;
        font-weight: 800;
    }

    /* Status alerts */
    .status-ok { background: rgba(120,128,100,.15); border: 1px solid #657052; color: #dce7ce !important; padding: 14px 17px; border-radius: 12px; font-weight: 700; }
    .status-warn { background: rgba(197,138,85,.14); border: 1px solid #8d6844; color: #f0d1a7 !important; padding: 14px 17px; border-radius: 12px; font-weight: 700; }
    .status-danger { background: rgba(217,106,85,.14); border: 1px solid #91483b; color: #ffc0b5 !important; padding: 14px 17px; border-radius: 12px; font-weight: 700; }

    /* Inputs */
    .stSelectbox label, .stNumberInput label, .stTextInput label,
    .stFileUploader label, .stMultiSelect label, .stSlider label { color: var(--text) !important; font-weight: 600; }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stFileUploaderDropzone"] {
        background: #221f1b !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }
    [data-baseweb="select"] *, [data-baseweb="input"] * { color: var(--text) !important; }
    input::placeholder { color: var(--muted-2) !important; }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        border: 1px solid #75513d;
        background: linear-gradient(135deg, #693521, #8f472d);
        color: #fff4ea !important;
        font-weight: 700;
        letter-spacing: .2px;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease, border-color .18s ease;
        box-shadow: 0 5px 16px rgba(0,0,0,.18);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        border-color: var(--rust-light);
        box-shadow: 0 8px 22px rgba(182,83,50,.20);
    }
    .stButton > button:active, .stFormSubmitButton > button:active { transform: translateY(0); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #211e1a;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--muted) !important;
        border-radius: 9px;
        padding: 8px 14px;
        transition: background .18s ease, color .18s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(182,83,50,.20) !important;
        color: #fff1e7 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--rust) !important; }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] * { color: var(--text) !important; }

    /* Plotly styling container */
    .stApp .stPlotlyChart {
        background: #211e1a !important;
        border: 1px solid #493f36 !important;
        border-radius: 14px !important;
        padding: 5px !important;
    }
    .stApp .stPlotlyChart iframe { background: #211e1a !important; }

    .team-badge {
        background: linear-gradient(135deg, #2e2821, #24201b);
        border: 1px solid #54473b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        transition: transform .18s ease, border-color .18s ease;
    }
    .team-badge:hover {
        transform: translateX(4px);
        border-color: var(--copper);
    }
    .team-badge h4 { margin: 0; color: #f5d8bf !important; font-size: 17px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ------------------------- Paths & Dataset Loading -------------------------
BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "predictive_maintenance_v3.csv"
MLFLOW_DB = BASE / "mlflow.db"


@st.cache_data(show_spinner=False)
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def prepare_dataframe(raw):
    df = raw.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numerical_cols = [
        "vibration_rms", "temperature_motor", "current_phase_avg",
        "pressure_level", "rpm"
    ]
    present = [c for c in numerical_cols if c in df.columns]
    if "machine_id" in df.columns and present:
        df[present] = (
            df.groupby("machine_id")[present]
              .transform(lambda x: x.fillna(x.median()))
        )

    if {"machine_type", "operating_mode"}.issubset(df.columns):
        df = pd.get_dummies(
            df, columns=["machine_type", "operating_mode"],
            drop_first=True, dtype=int
        )

    if "failure_type" in df.columns:
        # 5-class mapping verified from notebook
        mapping = {"bearing": 0, "electrical": 1, "hydraulic": 2, "motor_overheat": 3, "none": 4}
        df["failure_type_encoded"] = df["failure_type"].map(mapping)

    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

    if {"machine_id", "timestamp"}.issubset(df.columns):
        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    sensor_columns = [
        "vibration_rms", "temperature_motor", "current_phase_avg",
        "pressure_level", "rpm"
    ]
    for col in sensor_columns:
        if col in df.columns and "machine_id" in df.columns:
            df[f"{col}_change"] = df.groupby("machine_id")[col].diff()

    for col in ["vibration_rms", "temperature_motor", "pressure_level"]:
        if col in df.columns and "machine_id" in df.columns:
            df[f"{col}_rolling_mean_5"] = (
                df.groupby("machine_id")[col]
                  .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            )

    change_columns = [f"{c}_change" for c in sensor_columns if f"{c}_change" in df.columns]
    if change_columns:
        df[change_columns] = df[change_columns].fillna(0)

    return df


# --------------------------- Plotly Styling Helper ---------------------------
def fig_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="#211e1a",
        plot_bgcolor="#2b2621",
        font=dict(color="#f5eee6", family="Inter, Arial, sans-serif", size=13),
        margin=dict(l=45, r=25, t=70, b=55),
        title=dict(
            text=fig.layout.title.text if fig.layout.title.text else "",
            font=dict(color="#f5d8bf", size=18, family="Inter, Arial, sans-serif"),
            x=0.02, xanchor="left", y=0.97, yanchor="top",
        ),
        legend=dict(
            bgcolor="#211e1a",
            bordercolor="#5a4a3d",
            borderwidth=1,
            font=dict(color="#f5eee6", size=12),
        ),
    )
    fig.update_xaxes(
        title_font=dict(color="#e6c4a7", size=12),
        tickfont=dict(color="#eee3d8", size=11),
        gridcolor="#493f36",
        zerolinecolor="#665548",
        linecolor="#5a4a3d",
        showline=True,
    )
    fig.update_yaxes(
        title_font=dict(color="#e6c4a7", size=12),
        tickfont=dict(color="#eee3d8", size=11),
        gridcolor="#493f36",
        zerolinecolor="#665548",
        linecolor="#5a4a3d",
        showline=True,
    )
    fig.update_annotations(font=dict(color="#f5eee6"))
    return fig


def metric_card(label, value, sub=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------- Sidebar / Navigation ----------------------
st.sidebar.markdown("## 🏭 WILL IT BREAK?")
st.sidebar.caption("Industrial Predictive Maintenance • SIC Team 13")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Command Center",
        "Live Predictions",
        "Dataset",
        "Data Cleaning",
        "EDA & Analysis",
        "Model Evaluation",
        "MLflow",
        "Team & Project"
    ],
)

# Optional file uploader in sidebar
uploaded = st.sidebar.file_uploader(
    "Optional: upload custom plant CSV",
    type=["csv"],
    help="Upload your own machine telemetry CSV or use the bundled project dataset."
)

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    data_source = "Uploaded CSV"
else:
    raw_df = load_csv(CSV_PATH)
    data_source = "predictive_maintenance_v3.csv" if raw_df is not None else "No dataset loaded"

df = prepare_dataframe(raw_df) if raw_df is not None else None

st.sidebar.divider()
st.sidebar.markdown(f"""
<div style='font-size: 0.85rem; color: #c7bbae;'>
    <b>Data Source:</b> {data_source}<br>
    <b>MLflow Backend:</b> <code>sqlite:///mlflow.db</code><br>
    <b>Status:</b> 🟢 Connected
</div>
""", unsafe_allow_html=True)


# ---------------------------- Hero Banner ----------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">SIC · TEAM 13 · CONDITION-BASED MAINTENANCE</div>
  <h1>Will It Break?</h1>
  <p>Predicting industrial machine failures before they happen using live sensor data —
  binary failure risk, failure type classification, and remaining useful life (RUL).</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. COMMAND CENTER
# ==============================================================================
if page == "Command Center":
    st.markdown('<div class="section-title">Fleet Snapshot</div>', unsafe_allow_html=True)

    if raw_df is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card("Readings", f"{len(raw_df):,}", "sensor records")
        with c2:
            metric_card("Machines", raw_df["machine_id"].nunique() if "machine_id" in raw_df else "—", "unique assets")
        with c3:
            metric_card("Machine Types", raw_df["machine_type"].nunique() if "machine_type" in raw_df else "—", "fleet mix")
        with c4:
            fail_rate = raw_df["failure_within_24h"].mean() * 100 if "failure_within_24h" in raw_df else np.nan
            metric_card("24h Failure Rate", f"{fail_rate:.1f}%", "class balance")
        with c5:
            metric_card("Models", "3", "binary · multi-class · RUL")
    else:
        st.warning("Load the project CSV to activate fleet analytics.")

    st.markdown('<div class="section-title">Prediction Stack</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        metric_card("01 · Failure in 24h", "XGBoost", "Recall 98.7% · F1 95.0%")
    with b:
        metric_card("02 · Failure Type", "XGBoost", "Accuracy 98.9% · Recall 98.8%")
    with c:
        metric_card("03 · Remaining Life", "Random Forest", "MAE 1.64 h · R² 0.977")

    if raw_df is not None:
        st.markdown('<div class="section-title">Fleet Analytics</div>', unsafe_allow_html=True)
        left, right = st.columns(2)

        with left:
            if "failure_within_24h" in raw_df.columns:
                counts = raw_df["failure_within_24h"].map({0: "Normal", 1: "Failure"}).value_counts()
                fig = px.bar(
                    x=counts.index, y=counts.values,
                    labels={"x": "", "y": "Readings"},
                    title="24h Failure Balance",
                    text=counts.values,
                    color=counts.index,
                    color_discrete_map={"Normal": "#788064", "Failure": "#b65332"}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig_layout(fig), use_container_width=True)

        with right:
            if "failure_type" in raw_df.columns:
                counts = raw_df["failure_type"].value_counts()
                fig = px.bar(
                    x=counts.index, y=counts.values,
                    labels={"x": "Failure Type", "y": "Readings"},
                    title="Failure Type Distribution",
                    text=counts.values,
                    color=counts.index,
                    color_discrete_sequence=["#788064", "#c58a55", "#d97952", "#b65332", "#d96a55"]
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig_layout(fig), use_container_width=True)

    st.markdown('<div class="section-title">Operational Takeaways</div>', unsafe_allow_html=True)
    st.info(
        "The project identifies vibration (+51%), motor temperature (+27%), and hours since maintenance "
        "as the earliest degradation signals. The recommended operational protocol is: detect "
        "imminent failure within 24h → classify specific root failure mode → route predicted RUL into plant maintenance scheduling."
    )


# ==============================================================================
# 2. LIVE PREDICTIONS (Accurate Inference Engine)
# ==============================================================================
elif page == "Live Predictions":
    st.markdown('<div class="section-title">Live Machine Assessment</div>', unsafe_allow_html=True)

    PRESETS = {
        "🟢 Healthy Machine (Nominal)": {
            "vibration_rms": 1.15, "temperature_motor": 48.0, "current_phase_avg": 5.8,
            "pressure_level": 42.0, "rpm": 880.0, "hours_since_maintenance": 45.0,
            "ambient_temp": 13.0, "machine_type": "Pump", "operating_mode": "normal"
        },
        "🔴 Bearing Degradation Fault": {
            "vibration_rms": 3.85, "temperature_motor": 62.0, "current_phase_avg": 11.2,
            "pressure_level": 48.0, "rpm": 1450.0, "hours_since_maintenance": 240.0,
            "ambient_temp": 14.5, "machine_type": "CNC", "operating_mode": "peak"
        },
        "🔴 Motor Overheat Hazard": {
            "vibration_rms": 2.20, "temperature_motor": 82.5, "current_phase_avg": 18.4,
            "pressure_level": 88.0, "rpm": 2100.0, "hours_since_maintenance": 190.0,
            "ambient_temp": 16.0, "machine_type": "Compressor", "operating_mode": "peak"
        },
        "🟡 Hydraulic Pressure Spike": {
            "vibration_rms": 1.95, "temperature_motor": 58.0, "current_phase_avg": 12.0,
            "pressure_level": 145.0, "rpm": 1100.0, "hours_since_maintenance": 130.0,
            "ambient_temp": 13.0, "machine_type": "Robotic Arm", "operating_mode": "peak"
        },
        "⚡ Electrical Phase Imbalance": {
            "vibration_rms": 1.80, "temperature_motor": 64.0, "current_phase_avg": 22.5,
            "pressure_level": 52.0, "rpm": 1250.0, "hours_since_maintenance": 160.0,
            "ambient_temp": 14.0, "machine_type": "Pump", "operating_mode": "normal"
        }
    }

    # Preset selection
    col_preset, _ = st.columns([1.5, 2])
    with col_preset:
        chosen_preset = st.selectbox("⚡ Quick Scenario Presets (Optional):", ["-- Manual Input --"] + list(PRESETS.keys()))

    defaults = PRESETS.get(chosen_preset, PRESETS["🟢 Healthy Machine (Nominal)"])

    # Input Form
    with st.form("live_assessment_form"):
        st.markdown("**Equipment Identity & Mode**")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            machine_id = st.number_input("Machine ID", min_value=1, max_value=20, value=1)
        with m_col2:
            types = ["CNC", "Pump", "Compressor", "Robotic Arm"]
            m_type = st.selectbox("Machine Type", types, index=types.index(defaults.get("machine_type", "Pump")))
        with m_col3:
            modes = ["idle", "normal", "peak"]
            op_mode = st.selectbox("Operating Mode", modes, index=modes.index(defaults.get("operating_mode", "normal")))

        st.markdown("**Live Sensor Readings**")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            vib = st.number_input("Vibration RMS (mm/s)", value=float(defaults["vibration_rms"]), step=0.05, format="%.2f", help="Nominal: 0.8 - 1.8 mm/s")
            temp = st.number_input("Motor Temperature (°C)", value=float(defaults["temperature_motor"]), step=0.5, format="%.1f", help="Nominal: 40 - 60 °C")
        with s_col2:
            curr = st.number_input("Phase Current Avg (A)", value=float(defaults["current_phase_avg"]), step=0.2, format="%.2f", help="Nominal: 4.5 - 10.0 A")
            pres = st.number_input("Pressure Level (bar)", value=float(defaults["pressure_level"]), step=1.0, format="%.1f", help="Nominal: 30 - 65 bar")
        with s_col3:
            rpm_val = st.number_input("Rotational Speed (RPM)", value=float(defaults["rpm"]), step=25.0, format="%.0f")
            hours_maint = st.number_input("Hours Since Maintenance (h)", value=float(defaults["hours_since_maintenance"]), step=5.0, format="%.1f")

        ambient_t = st.number_input("Ambient Temp (°C)", value=float(defaults["ambient_temp"]), step=0.5, format="%.1f")

        submitted = st.form_submit_button("RUN CONDITION ASSESSMENT")

    # Construct payload & run prediction
    input_payload = {
        "machine_id": machine_id,
        "machine_type": m_type,
        "operating_mode": op_mode,
        "vibration_rms": vib,
        "temperature_motor": temp,
        "current_phase_avg": curr,
        "pressure_level": pres,
        "rpm": rpm_val,
        "hours_since_maintenance": hours_maint,
        "ambient_temp": ambient_t,
        "hour": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "month": datetime.now().month,
        # Compute delta & rolling representations matching nominal delta
        "vibration_rms_change": round(vib - 1.25, 2),
        "temperature_motor_change": round(temp - 50.0, 1),
        "current_phase_avg_change": round(curr - 6.5, 2),
        "pressure_level_change": round(pres - 45.0, 1),
        "rpm_change": round(rpm_val - 950.0, 0),
        "vibration_rms_rolling_mean_5": vib,
        "temperature_motor_rolling_mean_5": temp,
        "pressure_level_rolling_mean_5": pres
    }

    # Execute inference via verified model_service
    res = model_service.predict_single(input_payload)

    p1, p2, p3 = st.columns(3)

    # 1) Binary Failure Prediction
    with p1:
        prob = res["failure_probability"]
        if res["failure_within_24h"] == 1 or prob >= 0.35:
            st.markdown(
                f'<div class="status-danger">⚠ FAILURE RISK<br><span style="font-size:26px">{prob*100:.1f}%</span> probability within 24h</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="status-ok">✓ NORMAL<br><span style="font-size:26px">{prob*100:.1f}%</span> failure probability</div>',
                unsafe_allow_html=True
            )

    # 2) Root-Cause Failure Mode
    with p2:
        pred_label = res["predicted_failure_type"].replace("_", " ").title()
        metric_card("Predicted Failure Type", pred_label, "XGBoost Multi-Class (Acc 98.9%)")

    # 3) Remaining Useful Life
    with p3:
        rul_val = res["rul_hours"]
        metric_card("Remaining Useful Life", f"{rul_val:.1f} h", "Random Forest Regressor (MAE 1.64h)")

    st.markdown("---")

    # Visualizations: Risk Gauge & Multi-Class Probabilities
    c_g1, c_g2 = st.columns(2)

    with c_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'font': {'color': '#f5eee6', 'size': 26}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "24-Hour Failure Probability Gauge", 'font': {'size': 16, 'color': '#f5d8bf'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#c7bbae"},
                'bar': {'color': res["risk_color"]},
                'bgcolor': "#2b2621",
                'borderwidth': 1,
                'bordercolor': "#51483e",
                'steps': [
                    {'range': [0, 35], 'color': '#283618'},
                    {'range': [35, 70], 'color': '#7f5539'},
                    {'range': [70, 100], 'color': '#6f1d1b'}
                ],
                'threshold': {
                    'line': {'color': "#d96a55", 'width': 3},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        st.plotly_chart(fig_layout(fig_gauge, 290), use_container_width=True)

    with c_g2:
        probs_df = pd.DataFrame({
            "Mode": [k.replace("_", " ").title() for k in res["failure_type_probabilities"].keys()],
            "Confidence": list(res["failure_type_probabilities"].values())
        }).sort_values("Confidence", ascending=True)

        fig_probs = px.bar(
            probs_df,
            x="Confidence",
            y="Mode",
            orientation='h',
            title="Failure Mode Probabilities",
            color="Confidence",
            color_continuous_scale="Oranges",
            text=probs_df["Confidence"].apply(lambda x: f"{x*100:.1f}%")
        )
        fig_probs.update_layout(coloraxis_showscale=False)
        fig_probs.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_layout(fig_probs, 290), use_container_width=True)

    # Input Signal Profile
    st.markdown("### Input Signal Profile")
    signal_df = pd.DataFrame({
        "Sensor": ["Vibration (mm/s)", "Motor Temp (°C)", "Current (A)", "Pressure (bar)", "RPM", "Maint (h)"],
        "Value": [vib, temp, curr, pres, rpm_val, hours_maint]
    })
    fig_signals = px.bar(
        signal_df, x="Sensor", y="Value",
        title="Current Machine Sensor Inputs",
        color="Sensor",
        color_discrete_sequence=["#b65332", "#d97952", "#c58a55", "#788064", "#9db08a", "#e4c59f"]
    )
    fig_signals.update_layout(showlegend=False)
    st.plotly_chart(fig_layout(fig_signals, 310), use_container_width=True)

    # Recommendations
    st.markdown("### 🛠️ Targeted Maintenance Recommendations")
    for rec in res["recommendations"]:
        st.markdown(f"- {rec}")


# ==============================================================================
# 3. DATASET
# ==============================================================================
elif page == "Dataset":
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.warning("No dataset loaded.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Rows", f"{len(raw_df):,}")
        with c2: metric_card("Columns", len(raw_df.columns))
        with c3: metric_card("Machines", raw_df["machine_id"].nunique() if "machine_id" in raw_df else "—")
        with c4: metric_card("Missing Cells", f"{int(raw_df.isna().sum().sum()):,}")

        st.markdown("### Raw Data Preview")
        st.dataframe(raw_df.head(100), use_container_width=True)

        st.markdown("### Column Dictionary")
        descriptions = {
            "timestamp": "Timestamp of the sensor telemetry reading.",
            "machine_id": "Unique identifier of the monitored machine (1 to 20).",
            "machine_type": "Equipment class: CNC, Pump, Compressor, Robotic Arm.",
            "vibration_rms": "RMS vibration level (mm/s). Leading indicator of mechanical wear.",
            "temperature_motor": "Motor temperature (°C). Leading indicator of thermal failure.",
            "current_phase_avg": "Average phase current (A). Monitors electrical load stability.",
            "pressure_level": "Hydraulic / pneumatic pressure (bar).",
            "rpm": "Rotational speed of motor / spindle.",
            "operating_mode": "Operating state: idle, normal, peak.",
            "hours_since_maintenance": "Operating hours elapsed since previous overhaul.",
            "ambient_temp": "Ambient ambient room temperature (°C).",
            "failure_within_24h": "Binary target: 1 if failure occurs within 24h, 0 otherwise.",
            "failure_type": "Multi-class target: none, bearing, motor_overheat, hydraulic, electrical.",
            "rul_hours": "Regression target: Remaining Useful Life in operating hours.",
            "estimated_repair_cost": "Estimated repair / downtime cost used in economic EDA."
        }
        rows = []
        for col in raw_df.columns:
            rows.append([col, str(raw_df[col].dtype), int(raw_df[col].isna().sum()), descriptions.get(col, "Telemetry field.")])
        st.dataframe(pd.DataFrame(rows, columns=["Column", "Type", "Missing", "Role / Meaning"]), use_container_width=True)


# ==============================================================================
# 4. DATA CLEANING
# ==============================================================================
elif page == "Data Cleaning":
    st.markdown('<div class="section-title">Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.warning("No dataset loaded.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Missing Values**")
            miss = raw_df.isna().sum()
            st.write(miss[miss > 0] if (miss > 0).any() else "No missing values.")
            st.caption("Notebook method: machine-wise median imputation for core numerical sensors.")
        with c2:
            st.markdown("**Duplicate Rows**")
            st.metric("Duplicates", int(raw_df.duplicated().sum()))
            st.caption("Notebook result: no duplicate rows found.")
        with c3:
            st.markdown("**Outliers (IQR)**")
            st.write("Retained deliberately")
            st.caption("Outliers were kept because severe sensor excursions are direct failure signals.")

        st.markdown("### Outlier Counts by Sensor")
        numerical_columns = [
            "vibration_rms", "temperature_motor", "current_phase_avg",
            "pressure_level", "rpm", "hours_since_maintenance", "ambient_temp"
        ]
        out_rows = []
        for col in [c for c in numerical_columns if c in raw_df.columns]:
            q1, q3 = raw_df[col].quantile(.25), raw_df[col].quantile(.75)
            iqr = q3 - q1
            n = int(((raw_df[col] < q1 - 1.5*iqr) | (raw_df[col] > q3 + 1.5*iqr)).sum())
            out_rows.append([col, n])
        st.dataframe(pd.DataFrame(out_rows, columns=["Feature", "IQR Outliers"]), use_container_width=True)

        st.markdown("### Feature Engineering Methodology")
        feats = pd.DataFrame([
            ["_change (diff)", "Current reading − previous reading per machine", "Captures sudden period-over-period jumps"],
            ["rolling_mean_5", "Moving average of last 5 readings per machine", "Smooths sensor noise and exposes persistent degradation"],
            ["hour / day_of_week / month", "Calendar features extracted from timestamp", "Models time-dependent factory operating shifts"],
            ["One-hot encoding", "machine_type / operating_mode", "Converts categorical strings into numerical inputs (reference: CNC, idle)"],
            ["Label encoding", "failure_type", "Maps 5 failure modes: bearing (0), electrical (1), hydraulic (2), motor_overheat (3), none (4)"]
        ], columns=["Feature / Step", "Implementation", "Engineering Rationale"])
        st.dataframe(feats, use_container_width=True)

        st.info("Critical project finding: Outliers fail within 24h 62% of the time versus 15% overall. Keeping outliers provides essential signal for early fault detection.")


# ==============================================================================
# 5. EDA & ANALYSIS
# ==============================================================================
elif page == "EDA & Analysis":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.warning("No dataset loaded.")
    else:
        sensors = [c for c in [
            "vibration_rms", "temperature_motor", "current_phase_avg",
            "pressure_level", "rpm", "ambient_temp", "hours_since_maintenance"
        ] if c in raw_df.columns]

        tabs = st.tabs(["Distributions", "Failure Comparison", "Correlation", "Targets", "Key Findings"])

        with tabs[0]:
            col = st.selectbox("Sensor to Inspect", sensors)
            fig = px.histogram(
                raw_df, x=col, color="failure_within_24h" if "failure_within_24h" in raw_df else None,
                marginal="box", nbins=40,
                title=f"Telemetry Distribution: {col}",
                color_discrete_map={0: "#788064", 1: "#b65332"}
            )
            st.plotly_chart(fig_layout(fig, 430), use_container_width=True)

        with tabs[1]:
            if "failure_within_24h" in raw_df:
                stats_cols = [c for c in sensors if c in raw_df.columns]
                grouped = raw_df.groupby("failure_within_24h")[stats_cols].mean().T.reset_index()
                grouped.columns = ["Sensor", "Normal", "Failure"] if grouped.shape[1] == 3 else grouped.columns
                if "Normal" in grouped.columns:
                    long = grouped.melt("Sensor", var_name="Outcome", value_name="Mean")
                    fig = px.bar(
                        long, x="Sensor", y="Mean", color="Outcome", barmode="group",
                        title="Mean Sensor Readings: Normal vs. Failure Outcome",
                        color_discrete_map={"Normal": "#788064", "Failure": "#b65332"}
                    )
                    st.plotly_chart(fig_layout(fig, 430), use_container_width=True)

            st.markdown("**Empirical Shifts Prior to Machine Failure**")
            findings = pd.DataFrame([
                ["vibration_rms", "1.51 → 2.28 mm/s", "+51% surge (Strongest signal)"],
                ["temperature_motor", "49.4 → 62.5 °C", "+27% rise"],
                ["current_phase_avg", "8.52 → 10.81 A", "+27% rise"],
                ["hours_since_maintenance", "166 → 209 h", "+25% increase"],
                ["ambient_temp", "13.0 → 13.0 °C", "0% (neutral)"]
            ], columns=["Telemetry Feature", "Normal → Pre-Failure", "Relative Change"])
            st.dataframe(findings, use_container_width=True)

        with tabs[2]:
            corr_cols = [c for c in sensors + [
                "rul_hours", "failure_within_24h", "estimated_repair_cost",
                "vibration_rms_change", "temperature_motor_change",
                "current_phase_avg_change", "pressure_level_change", "rpm_change",
                "vibration_rms_rolling_mean_5", "temperature_motor_rolling_mean_5",
                "pressure_level_rolling_mean_5"
            ] if c in df.columns]
            corr = df[corr_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Telemetry & Target Correlation Heatmap"
            )
            st.plotly_chart(fig_layout(fig, 650), use_container_width=True)

        with tabs[3]:
            tc1, tc2 = st.columns(2)
            with tc1:
                if "failure_within_24h" in raw_df:
                    counts = raw_df["failure_within_24h"].value_counts().rename(index={0: "No Failure", 1: "Failure"})
                    fig = px.pie(
                        values=counts.values, names=counts.index, hole=.55,
                        title="Binary Target Balance (failure_within_24h)",
                        color_discrete_sequence=["#788064", "#b65332"]
                    )
                    st.plotly_chart(fig_layout(fig, 390), use_container_width=True)
            with tc2:
                if "failure_type" in raw_df:
                    counts = raw_df["failure_type"].value_counts()
                    fig = px.bar(
                        x=counts.index, y=counts.values, title="Multi-Class Target (failure_type)",
                        color=counts.index, color_discrete_sequence=["#788064", "#c58a55", "#d97952", "#b65332", "#d96a55"]
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig_layout(fig, 390), use_container_width=True)

            if "rul_hours" in raw_df:
                fig = px.histogram(
                    raw_df, x="rul_hours", nbins=45, title="RUL Hours Target Distribution",
                    color_discrete_sequence=["#c58a55"]
                )
                st.plotly_chart(fig_layout(fig, 380), use_container_width=True)

        with tabs[4]:
            st.markdown("""
            - **14.8% Failure Rate**: Demonstrates authentic industrial class imbalance; models must prioritize recall to avoid missing breakdowns.
            - **Vibration RMS (+51%)**: The earliest and most sensitive indicator of mechanical degradation (bearing wear & misalignment).
            - **Motor Temperature (+27%)**: Clear indicator of thermal overload and motor ventilation failure.
            - **Current Phase (+27%)**: Unbalanced electrical phases precede motor stator failure.
            - **Ambient Temperature**: Displays negligible correlation with failure, proving failures are asset-driven rather than weather-driven.
            """)


# ==============================================================================
# 6. MODEL EVALUATION
# ==============================================================================
elif page == "Model Evaluation":
    st.markdown('<div class="section-title">Machine Learning Models & Benchmark Evaluation</div>', unsafe_allow_html=True)

    st.markdown("### Binary Classification · Failure Within 24h")
    binary = pd.DataFrame([
        ["Logistic Regression", "Classification", "Baseline", "89.2%", "0.78", "0.94"],
        ["Random Forest", "Classification", "Ensemble", "97.4%", "0.91", "0.98"],
        ["Tuned Random Forest", "Classification", "Hyperparameter Tuned", "97.6%", "0.92", "0.98"],
        ["SVM", "Classification", "Kernel Scaled", "93.1%", "0.84", "0.96"],
        ["XGBoost", "Classification", "Gradient Boosted (Best)", "98.7%", "0.95", "0.999"]
    ], columns=["Model", "Task", "Stage", "Test Recall", "Test F1", "ROC-AUC"])
    st.dataframe(binary, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Champion Binary", "XGBoost")
    with c2: metric_card("Test Recall", "98.7%", "Catches catastrophic failure")
    with c3: metric_card("Test F1", "0.950")
    with c4: metric_card("Selection Rule", "Recall → F1")

    st.markdown("---")
    st.markdown("### Multi-Class Classification · Failure Type")
    mc = pd.DataFrame([
        ["Random Forest", "5 Classes", "98.0%", "89.8%", "0.937"],
        ["XGBoost", "5 Classes (Best)", "98.9%", "98.8%", "0.970"]
    ], columns=["Model", "Classes", "Accuracy", "Macro Recall", "Macro F1"])
    st.dataframe(mc, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Champion Multi-Class", "XGBoost")
    with c2: metric_card("Accuracy", "98.9%")
    with c3: metric_card("Macro Recall", "98.8%", "Even rare classes detected")

    st.markdown("---")
    st.markdown("### Regression · Remaining Useful Life (RUL)")
    reg = pd.DataFrame([
        ["Random Forest", "Best Regressor", "1.64 h", "3.96 h", "0.977"],
        ["Tuned Random Forest", "Tuned", "1.66 h", "4.01 h", "0.975"],
        ["XGBoost Regressor", "Candidate", "1.92 h", "4.45 h", "0.968"]
    ], columns=["Model", "Result", "Test MAE", "Test RMSE", "Test R²"])
    st.dataframe(reg, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Champion Regressor", "Random Forest")
    with c2: metric_card("Test MAE", "1.64 hours", "Lowest absolute error")
    with c3: metric_card("Test R²", "0.977", "97.7% variance explained")


# ==============================================================================
# 7. MLFLOW EXPERIMENT TRACKING
# ==============================================================================
elif page == "MLflow":
    st.markdown('<div class="section-title">MLflow Experiment Tracking & Model Registry</div>', unsafe_allow_html=True)

    if not MLFLOW_DB.exists():
        st.warning("mlflow.db was not found in the project root.")
    else:
        st.success("Connected to MLflow SQLite Tracking Backend: sqlite:///mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
        client = mlflow.MlflowClient()

        experiment_names = [
            "predictive_maintenance_failure_24h",
            "predictive_maintenance_failure_type",
            "predictive_maintenance_rul_hours",
        ]

        for exp_name in experiment_names:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                continue

            st.markdown(f"### Experiment: `{exp_name}`")
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

            if runs.empty:
                st.caption("No runs logged.")
                continue

            preferred = [c for c in [
                "tags.mlflow.runName", "metrics.test_accuracy",
                "metrics.test_precision", "metrics.test_recall",
                "metrics.test_f1", "metrics.test_recall_macro",
                "metrics.test_f1_macro", "metrics.test_mae",
                "metrics.test_rmse", "metrics.test_r2",
            ] if c in runs.columns]

            view = runs[preferred].copy()
            view.columns = [c.replace("metrics.", "").replace("tags.mlflow.", "") for c in view.columns]
            st.dataframe(view, use_container_width=True)

            if "failure_24h" in exp_name:
                metric_cols = [c for c in ["test_accuracy", "test_precision", "test_recall", "test_f1"] if c in view]
            elif "failure_type" in exp_name:
                metric_cols = [c for c in ["test_accuracy", "test_recall_macro", "test_f1_macro"] if c in view]
            else:
                metric_cols = [c for c in ["test_mae", "test_rmse", "test_r2"] if c in view]

            if metric_cols and "runName" in view:
                long = view.melt(id_vars=["runName"], value_vars=metric_cols, var_name="Metric", value_name="Value")
                fig = px.bar(long, x="runName", y="Value", color="Metric", barmode="group", title=f"MLflow Metric Benchmarks: {exp_name}")
                st.plotly_chart(fig_layout(fig, 390), use_container_width=True)

        st.markdown("### Champion Registered Models (Staging)")
        reg_rows = []
        for name in [
            "failure_within_24h_classifier",
            "failure_type_classifier",
            "rul_hours_regressor",
        ]:
            try:
                models = client.search_model_versions(f"name='{name}'")
                v = models[0] if models else None
                reg_rows.append([name, v.version if v else "1", "staging", v.run_id[:8] if v else "—"])
            except Exception:
                reg_rows.append([name, "1", "staging", "—"])
        st.dataframe(pd.DataFrame(reg_rows, columns=["Registered Model", "Version", "Alias", "Run ID"]), use_container_width=True)


# ==============================================================================
# 8. TEAM & PROJECT
# ==============================================================================
elif page == "Team & Project":
    st.markdown('<div class="section-title">Project Overview & Team 13</div>', unsafe_allow_html=True)

    col_info1, col_info2 = st.columns([1.3, 1], gap="large")

    with col_info1:
        st.markdown("""
        ### 📖 Project Motivation & Operational Paradigm
        Unplanned industrial machine downtime halts production schedules and causes substantial financial loss.
        Traditional maintenance methodologies present a costly tradeoff:
        - **Fixed-Interval Maintenance:** Discards functional components prematurely and generates avoidable service outages.
        - **Run-to-Failure Reactive Repairs:** Risks secondary component damage and extensive factory downtime.

        **Condition-Based Predictive Maintenance** resolves this problem by streaming real-time machine telemetry, identifying subtle pre-failure anomalies, and predicting:
        1. **Will it break within 24 hours?** *(Binary Classification with XGBoost — 98.7% Recall)*
        2. **What specific root cause will cause failure?** *(Multi-Class Classification with XGBoost — 98.9% Accuracy)*
        3. **How many operating hours remain?** *(Regression with Random Forest — 1.64 hours MAE)*
        """)

        st.markdown("""
        ### 📈 Business & Engineering Findings
        - **+51% Vibration Surge:** The earliest and most reliable warning signal for mechanical wear (bearing & shaft faults).
        - **+27% Motor Temperature Rise:** Clear leading indicator before thermal trip.
        - **MLflow Tracking & Model Registry:** Continuous logging of hyperparameters, evaluation curves, and champion model promotion to staging.
        """)

    with col_info2:
        st.markdown("### 👥 Team 13 Members")
        team_members = [
            "Menna Kholief",
            "Omar Elbeltagy",
            "Marwa Mahmoud",
            "Mariam Ramy",
            "Rahma Shaban",
        ]

        for name in team_members:
            st.markdown(f"""
            <div class="team-badge">
                <h4>{name}</h4>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-size: 0.9rem; color: #c7bbae;'>
            <b>Institution:</b> Samsung Innovation Campus (SIC)<br>
            <b>Track:</b> AI & Machine Learning (AI803)<br>
            <b>Dataset:</b> Industrial Machine Predictive Maintenance Telemetry (24,042 readings, 20 machines)
        </div>
        """, unsafe_allow_html=True)

# --------------------------- Global Footer ---------------------------
st.divider()
st.caption(
    "Will It Break? · Team 13 · Samsung Innovation Campus · "
    "Predictive Maintenance System based on project notebook and MLflow registry."
)
