
import os
import sqlite3
import tempfile
from pathlib import Path

import joblib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Will It Break? — Industrial Predictive Maintenance Dashboard
# Based directly on Team 13's notebook + presentation.
# ============================================================

st.set_page_config(
    page_title="Will It Break? | Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Theme ----------------------------
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

    /* ---------- Global readable typography ---------- */
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

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #211e1a 0%, #191714 100%);
        border-right: 1px solid var(--border);
        box-shadow: 10px 0 35px rgba(0,0,0,.12);
    }

    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small { color: var(--muted) !important; }

    /* Streamlit radio navigation → compact professional menu */
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

    /* ---------- Main hero ---------- */
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

    /* ---------- Cards ---------- */
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

    /* ---------- Status / alerts ---------- */
    .status-ok { background: rgba(120,128,100,.15); border: 1px solid #657052; color: #dce7ce !important; padding: 14px 17px; border-radius: 12px; font-weight: 700; }
    .status-warn { background: rgba(197,138,85,.14); border: 1px solid #8d6844; color: #f0d1a7 !important; padding: 14px 17px; border-radius: 12px; font-weight: 700; }
    .status-danger { background: rgba(217,106,85,.14); border: 1px solid #91483b; color: #ffc0b5 !important; padding: 14px 17px; border-radius: 12px; font-weight: 700; }

    /* ---------- Inputs ---------- */
    .stSelectbox label, .stNumberInput label, .stTextInput label,
    .stFileUploader label, .stMultiSelect label { color: var(--text) !important; font-weight: 600; }
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

    /* ---------- Buttons ---------- */
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

    /* ---------- Tabs ---------- */
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

    /* ---------- Dataframes ---------- */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] * { color: var(--text) !important; }

    /* ---------- Expanders / alerts / containers ---------- */
    [data-testid="stExpander"] {
        background: var(--panel);
        border-color: var(--border);
        border-radius: 12px;
    }
    [data-testid="stAlert"] {
        background: #2b2722;
        color: var(--text) !important;
        border-color: var(--border);
    }
    [data-testid="stAlert"] * { color: var(--text) !important; }

    /* Make transitions feel like an app without hiding Streamlit's rerun behavior. */
    .block-container { animation: pageIn .22s ease-out; }
    @keyframes pageIn { from { opacity: .35; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* ---------- Streamlit / Plotly contrast hardening ---------- */
    .stApp [data-testid="stHeader"] { background: transparent !important; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3,
    .stApp [data-testid="stMarkdownContainer"] h4,
    .stApp [data-testid="stMarkdownContainer"] h5,
    .stApp [data-testid="stMarkdownContainer"] h6 {
        color: #f5eee6 !important;
    }
    .stApp .stPlotlyChart {
        background: #211e1a !important;
        border: 1px solid #493f36 !important;
        border-radius: 14px !important;
        padding: 5px !important;
    }
    .stApp .stPlotlyChart iframe { background: #211e1a !important; }
    .stApp [data-testid="stMetricLabel"] p,
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricDelta"] { color: #f5eee6 !important; }
    .stApp [data-testid="stMarkdownContainer"] strong { color: #f5d8bf !important; }
    .stApp [data-testid="stCaptionContainer"] p { color: #cdbdaf !important; }
    .stApp [data-testid="stSelectbox"] label,
    .stApp [data-testid="stNumberInput"] label,
    .stApp [data-testid="stFileUploader"] label,
    .stApp [data-testid="stTextInput"] label { color: #eadfd5 !important; }
    .stApp [data-baseweb="select"] input,
    .stApp [data-baseweb="input"] input { color: #f5eee6 !important; -webkit-text-fill-color: #f5eee6 !important; }
    .stApp [data-baseweb="select"] [role="option"] { color: #f5eee6 !important; background: #2b2621 !important; }
    .stApp [data-baseweb="popover"] * { color: #f5eee6 !important; }
    .stApp [data-testid="stDataFrame"] { background: #211e1a !important; }

    hr { border-color: var(--border) !important; }
    ::selection { background: rgba(182,83,50,.35); color: white; }
</style>
""", unsafe_allow_html=True)


# ------------------------- Paths / loading -------------------------
BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE / "predictive_maintenance_v3.csv",
    BASE / "predictive_maintenance_v3 (1).csv",
    BASE / "data" / "predictive_maintenance_v3.csv",
    BASE / "dataset" / "predictive_maintenance_v3.csv",
]
MLFLOW_DB = BASE / "mlflow_test.db"


@st.cache_data(show_spinner=False)
def load_csv(path):
    raw = pd.read_csv(path)
    return raw


def find_local_dataset():
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def prepare_dataframe(raw):
    df = raw.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Same machine-wise median imputation used in the notebook.
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

    # Same categorical encoding logic.
    if {"machine_type", "operating_mode"}.issubset(df.columns):
        df = pd.get_dummies(
            df, columns=["machine_type", "operating_mode"],
            drop_first=True, dtype=int
        )

    if "failure_type" in df.columns:
        # Preserve the notebook's five-class labels when possible.
        classes = sorted(df["failure_type"].dropna().astype(str).unique().tolist())
        mapping = {name: i for i, name in enumerate(classes)}
        df["failure_type_encoded"] = df["failure_type"].astype(str).map(mapping)

    # Same feature engineering.
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


def locate_model_db():
    return MLFLOW_DB if MLFLOW_DB.exists() else None


def repair_mlflow_paths(db_path):
    """Make the bundled MLflow registry portable after moving the project folder."""
    if not db_path or not Path(db_path).exists():
        return

    root = Path(db_path).resolve().parent
    mlruns = root / "mlruns"
    mapping = {
        ("failure_within_24h_classifier", 1): (1, "293c17ca33d54751809fda0d0499a230", "6328aa65dcfe47a09b7ad0f75f01e005"),
        ("failure_within_24h_classifier", 2): (1, "f511d12e08744e078b3109ea10e1e335", "563489b3c39c441abd86c5acff82ce94"),
        ("rul_hours_regressor", 1): (2, "e28dfb3078d24b958c7489058a61d911", "6ac275f4e51f4149986a74b8cc9109b3"),
        ("failure_type_classifier", 1): (3, "01aad14449724d55899586a58f3cbbf5", "62e648d820ad4c409a82c95ab9e1911b"),
    }

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        for eid in (0, 1, 2, 3):
            exp_dir = (mlruns / str(eid)).resolve()
            if exp_dir.exists():
                cur.execute(
                    "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                    (exp_dir.as_uri(), eid),
                )

        for (name, version), (eid, model_id, run_id) in mapping.items():
            model_dir = (mlruns / str(eid) / "models" / f"m-{model_id}" / "artifacts").resolve()
            if not (model_dir / "MLmodel").exists():
                continue
            source = model_dir.as_uri()
            cur.execute(
                "UPDATE model_versions SET source=?, storage_location=?, run_id=?, status='READY', status_message=NULL WHERE name=? AND version=?",
                (source, source, run_id, name, version),
            )

        for eid, run_id in cur.execute("SELECT experiment_id, run_uuid FROM runs").fetchall():
            run_dir = (mlruns / str(eid) / run_id / "artifacts").resolve()
            if run_dir.exists():
                cur.execute("UPDATE runs SET artifact_uri=? WHERE run_uuid=?", (run_dir.as_uri(), run_id))
        con.commit()
    finally:
        con.close()


def _load_bundled_models():
    """
    Deployment fallback: Streamlit Cloud does not have the local MLflow SQLite
    registry, so load the same registered model versions from compact joblib files.
    Local runs still use the MLflow registry below when mlflow_test.db exists.
    """
    model_dir = BASE / "models"

    binary_path = model_dir / "failure_within_24h_classifier.joblib"
    mc_path = model_dir / "failure_type_classifier.joblib"

    # The RUL model is ~54 MB even after joblib compression, so it is split into
    # GitHub-uploadable chunks and reconstructed into the temporary directory.
    rul_path = Path(tempfile.gettempdir()) / "will_it_break_rul_hours_regressor.joblib"
    rul_chunks = sorted(model_dir.glob("rul_hours_regressor.joblib.part*"))

    if not binary_path.exists() or not mc_path.exists() or not rul_chunks:
        return None, None, None, {}

    try:
        if not rul_path.exists() or rul_path.stat().st_size != sum(p.stat().st_size for p in rul_chunks):
            with open(rul_path, "wb") as out:
                for chunk in rul_chunks:
                    with open(chunk, "rb") as src:
                        while True:
                            block = src.read(1024 * 1024)
                            if not block:
                                break
                            out.write(block)

        binary = joblib.load(binary_path)
        multiclass = joblib.load(mc_path)
        rul = joblib.load(rul_path)

        return binary, multiclass, rul, {
            "binary": "2 (bundled)",
            "multiclass": "1 (bundled)",
            "rul": "1 (bundled)",
        }
    except Exception as e:
        st.warning(f"Could not load bundled prediction models: {e}")
        return None, None, None, {}


@st.cache_resource(show_spinner=False)
def load_registered_models(db_path):
    # Local development: use the MLflow registry exactly as before.
    if db_path:
        try:
            # The DB shipped with the project contains paths from the training PC.
            # Rewrite those paths to the current app folder before MLflow resolves the aliases.
            repair_mlflow_paths(db_path)
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            client = mlflow.MlflowClient()

            names = {
                "binary": "failure_within_24h_classifier",
                "multiclass": "failure_type_classifier",
                "rul": "rul_hours_regressor",
            }

            models = {}
            versions = {}

            for key, name in names.items():
                try:
                    registered_versions = client.search_model_versions(f"name='{name}'")

                    if not registered_versions:
                        models[key] = None
                        versions[key] = None
                        continue

                    # Use the latest registered version.
                    latest_version = max(
                        registered_versions,
                        key=lambda v: int(v.version)
                    )

                    version = latest_version.version

                    models[key] = mlflow.sklearn.load_model(
                        f"models:/{name}/{version}"
                    )

                    versions[key] = version

                except Exception as e:
                    models[key] = None
                    versions[key] = None
                    st.warning(f"Could not load {name}: {e}")

            if any(models.values()):
                return models["binary"], models["multiclass"], models["rul"], versions

        except Exception as e:
            st.warning(f"MLflow registry unavailable; using bundled models instead. Details: {e}")

    # Streamlit Cloud: no local SQLite registry is required.
    return _load_bundled_models()


# --------------------------- Charts ---------------------------
def align_prediction_features(df, expected_features):
    """
    Align live input with the exact feature schema expected by the trained model.
    Categorical variables are one-hot encoded using the same column names used
    during training; missing expected columns are filled with 0 and extra
    columns are dropped.
    """
    out = df.copy()

    # Ensure all feature names are plain Python strings.
    out.columns = [str(c) for c in out.columns]
    expected_features = [str(c) for c in expected_features]

    # One-hot categorical values when raw categorical columns are present.
    categorical_cols = [c for c in ["machine_type", "operating_mode"] if c in out.columns]
    if categorical_cols:
        out = pd.get_dummies(out, columns=categorical_cols, dtype=int)

    # Exact model schema: add missing columns and remove unexpected ones.
    out = out.reindex(columns=expected_features, fill_value=0)

    # Ensure numeric matrix and consistent column-name types.
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
    out.columns = [str(c) for c in out.columns]
    return out

def fig_layout(fig, height=380):
    """Apply one consistent high-contrast rustic-industrial Plotly theme."""
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
    # Force readable annotation/text colors too (heatmaps, labels, etc.).
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


# ---------------------- Sidebar / source ----------------------
st.sidebar.markdown("## 🏭 WILL IT BREAK?")
st.sidebar.caption("Industrial Predictive Maintenance")
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
    ],
)

uploaded = st.sidebar.file_uploader(
    "Optional: upload the project CSV",
    type=["csv"],
    help="If no local CSV is placed beside app.py, upload predictive_maintenance_v3.csv here.",
)

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    data_source = "Uploaded CSV"
else:
    data_path = find_local_dataset()
    raw_df = load_csv(data_path) if data_path else None
    data_source = str(data_path.name) if data_path else "No dataset loaded"

df = prepare_dataframe(raw_df) if raw_df is not None else None
db_path = locate_model_db()
binary_model, mc_model, rul_model, model_versions = load_registered_models(db_path)

# ---------------------------- Header ----------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">SIC · TEAM 13 · CONDITION-BASED MAINTENANCE</div>
  <h1>Will It Break?</h1>
  <p>Predicting industrial machine failures before they happen using live sensor data —
  binary failure risk, failure type classification, and remaining useful life (RUL).</p>
</div>
""", unsafe_allow_html=True)

# ======================== COMMAND CENTER ========================
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
        st.warning("Load the project CSV to activate the fleet analytics.")

    st.markdown('<div class="section-title">Prediction Stack</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        metric_card("01 · Failure in 24h", "XGBoost", "Recall 98.9% · F1 96.9%")
    with b:
        metric_card("02 · Failure Type", "XGBoost", "Accuracy 98.9% · Recall 98.6%")
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
                )
                st.plotly_chart(fig_layout(fig), use_container_width=True)

        with right:
            if "failure_type" in raw_df.columns:
                counts = raw_df["failure_type"].value_counts()
                fig = px.bar(
                    x=counts.index, y=counts.values,
                    labels={"x": "Failure Type", "y": "Readings"},
                    title="Failure Type Distribution",
                    text=counts.values,
                )
                st.plotly_chart(fig_layout(fig), use_container_width=True)

    st.markdown('<div class="section-title">Operational Takeaways</div>', unsafe_allow_html=True)
    st.info(
        "The project identifies vibration, motor temperature, and hours since maintenance "
        "as consistent degradation signals. The recommended operational flow is: detect "
        "near-term failure → identify failure type → route predicted RUL into the maintenance queue."
    )

# ======================== LIVE PREDICTIONS ========================
elif page == "Live Predictions":
    st.markdown('<div class="section-title">Live Machine Assessment</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Load the project CSV first. The app uses the latest machine reading to build the engineered features needed by the trained models.")
    else:
        numeric_inputs = [
            "vibration_rms", "temperature_motor", "current_phase_avg",
            "pressure_level", "rpm", "ambient_temp", "hours_since_maintenance"
        ]
        available = [x for x in numeric_inputs if x in raw_df.columns]

        machines = sorted(raw_df["machine_id"].dropna().unique().tolist()) if "machine_id" in raw_df else [0]
        selected_machine = st.selectbox("Machine", machines)

        machine_rows = raw_df[raw_df["machine_id"] == selected_machine].copy()
        if "timestamp" in machine_rows.columns:
            machine_rows["timestamp"] = pd.to_datetime(machine_rows["timestamp"], errors="coerce")
            machine_rows = machine_rows.sort_values("timestamp")

        latest = machine_rows.iloc[-1] if len(machine_rows) else raw_df.iloc[-1]

        with st.form("prediction_form"):
            cols = st.columns(3)
            values = {}
            for i, col in enumerate(available):
                default = float(latest[col]) if pd.notna(latest[col]) else 0.0
                with cols[i % 3]:
                    values[col] = st.number_input(col.replace("_", " ").title(), value=default, format="%.4f")

            # Preserve categorical context if present in raw data.
            cat_cols = [c for c in ["machine_type", "operating_mode"] if c in raw_df.columns]
            cats = {}
            if cat_cols:
                st.markdown("**Operating Context**")
                cc = st.columns(len(cat_cols))
                for i, col in enumerate(cat_cols):
                    opts = sorted(raw_df[col].dropna().astype(str).unique().tolist())
                    default_val = str(latest[col]) if pd.notna(latest[col]) else opts[0]
                    with cc[i]:
                        cats[col] = st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(default_val) if default_val in opts else 0)

            submitted = st.form_submit_button("RUN CONDITION ASSESSMENT")

        if submitted:
            # Build a one-row raw-ish frame, then reproduce feature engineering from the
            # latest machine history. This avoids asking the operator for engineered fields.
            history = machine_rows.copy()
            if "timestamp" in history.columns:
                history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
                history = history.sort_values("timestamp")

            row = latest.copy()
            for k, v in values.items():
                row[k] = v
            for k, v in cats.items():
                row[k] = v
            row["machine_id"] = selected_machine

            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

            # Keep only the fields necessary for feature creation.
            if "timestamp" in history.columns:
                history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
                history = history.sort_values("timestamp").reset_index(drop=True)

            # Compute engineered fields exactly as in the notebook.
            if "timestamp" in history.columns:
                history["hour"] = history["timestamp"].dt.hour.fillna(pd.Timestamp.now().hour)
                history["day_of_week"] = history["timestamp"].dt.dayofweek.fillna(pd.Timestamp.now().dayofweek)
                history["month"] = history["timestamp"].dt.month.fillna(pd.Timestamp.now().month)

            sensors = ["vibration_rms", "temperature_motor", "current_phase_avg", "pressure_level", "rpm"]
            for col in sensors:
                if col in history.columns:
                    history[f"{col}_change"] = history.groupby("machine_id")[col].diff().fillna(0)

            for col in ["vibration_rms", "temperature_motor", "pressure_level"]:
                if col in history.columns:
                    history[f"{col}_rolling_mean_5"] = (
                        history.groupby("machine_id")[col]
                               .transform(lambda x: x.rolling(5, min_periods=1).mean())
                    )

            # Match notebook's one-hot encoding: create columns expected by the model.
            model_features = None
            for m in [binary_model, mc_model, rul_model]:
                if m is not None and hasattr(m, "feature_names_in_"):
                    model_features = list(m.feature_names_in_)
                    break

            if model_features is None:
                st.error("Registered MLflow models were not found. Run the notebook through the MLflow registration section first, then restart the app.")
            else:
                # Encode categoricals without needing the original LabelEncoder object.
                # The trained model's feature names determine the final one-hot columns.
                candidate = history.iloc[[-1]].copy()

                # Align the live row with the exact feature schema used during training.
                # This one-hot encodes categorical context before selecting model columns.
                X_live = align_prediction_features(candidate, model_features)

                p1, p2, p3 = st.columns(3)

                # 1) Binary
                with p1:
                    if binary_model is not None:
                        pred = int(binary_model.predict(X_live)[0])
                        prob = float(binary_model.predict_proba(X_live)[0, 1])
                        if pred == 1:
                            st.markdown(f'<div class="status-danger">⚠ FAILURE RISK<br><span style="font-size:26px">{prob*100:.1f}%</span> probability within 24h</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="status-ok">✓ NORMAL<br><span style="font-size:26px">{prob*100:.1f}%</span> failure probability</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Binary model unavailable.")

                # 2) Failure type
                with p2:
                    if mc_model is not None:
                        mc_pred = mc_model.predict(X_live)[0]
                        try:
                            # Classes are shown in the project presentation.
                            class_names = ["none", "bearing", "motor overheat", "hydraulic", "electrical"]
                            idx = int(mc_pred)
                            label = class_names[idx] if idx < len(class_names) else str(mc_pred)
                        except Exception:
                            label = str(mc_pred)
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Failure Type</div><div class="metric-value">{label.title()}</div><div class="metric-sub">XGBoost multi-class</div></div>', unsafe_allow_html=True)
                    else:
                        st.warning("Multi-class model unavailable.")

                # 3) RUL
                with p3:
                    if rul_model is not None:
                        # RUL model may have the same feature set; use its own names.
                        rf = list(rul_model.feature_names_in_) if hasattr(rul_model, "feature_names_in_") else model_features
                        X_r = candidate.copy()
                        X_r = align_prediction_features(X_r, rf).apply(pd.to_numeric, errors="coerce").fillna(0)
                        # Convert all feature names explicitly to Python string type
                        # to avoid the scikit-learn mixed feature-name error.
                        X_r.columns = [str(col) for col in X_r.columns]
                        # Equivalent alternative: X_r.columns = X_r.columns.astype(str)
                        rul = max(0.0, float(rul_model.predict(X_r)[0]))
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Remaining Useful Life</div><div class="metric-value">{rul:.1f} h</div><div class="metric-sub">Random Forest regression</div></div>', unsafe_allow_html=True)
                    else:
                        st.warning("RUL model unavailable.")

                st.markdown("### Input Signal Profile")
                signal_cols = [c for c in numeric_inputs if c in values]
                chart_df = pd.DataFrame({"Sensor": signal_cols, "Value": [values[c] for c in signal_cols]})
                fig = px.bar(chart_df, x="Sensor", y="Value", title="Current Machine Sensor Inputs")
                st.plotly_chart(fig_layout(fig, 330), use_container_width=True)

# ============================ DATASET ============================
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

        st.markdown("### Raw Data")
        st.dataframe(raw_df.head(100), use_container_width=True)

        st.markdown("### Column Dictionary")
        descriptions = {
            "vibration_rms": "Machine vibration signal.",
            "temperature_motor": "Motor temperature.",
            "current_phase_avg": "Average current phase.",
            "pressure_level": "Pressure sensor reading.",
            "rpm": "Rotational speed.",
            "operating_mode": "Operating context.",
            "hours_since_maintenance": "Hours elapsed since maintenance.",
            "ambient_temp": "Ambient temperature.",
            "failure_within_24h": "Binary target: whether failure occurs within 24 hours.",
            "failure_type": "Multi-class target.",
            "rul_hours": "Regression target: remaining useful life in hours.",
            "estimated_repair_cost": "Estimated repair cost used in the EDA.",
        }
        rows = []
        for col in raw_df.columns:
            rows.append([col, str(raw_df[col].dtype), int(raw_df[col].isna().sum()), descriptions.get(col, "Dataset field.")])
        st.dataframe(pd.DataFrame(rows, columns=["Column", "Type", "Missing", "Role / Meaning"]), use_container_width=True)

# ======================== DATA CLEANING ========================
elif page == "Data Cleaning":
    st.markdown('<div class="section-title">Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)

    if raw_df is None:
        st.warning("No dataset loaded.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Missing values**")
            miss = raw_df.isna().sum()
            st.write(miss[miss > 0] if (miss > 0).any() else "No missing values.")
            st.caption("Notebook method: machine-wise median imputation for core numerical sensors.")
        with c2:
            st.markdown("**Duplicate rows**")
            st.metric("Duplicates", int(raw_df.duplicated().sum()))
            st.caption("Notebook result: no duplicate rows were found.")
        with c3:
            st.markdown("**Outliers**")
            st.write("IQR method")
            st.caption("Outliers were retained because extreme sensor readings may be meaningful failure signals.")

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

        st.markdown("### Feature Engineering")
        feats = pd.DataFrame([
            ["_change", "Current reading − previous reading per machine", "Detect sudden jumps"],
            ["rolling_mean_5", "Mean of last 5 readings", "Smooth noise and expose trends"],
            ["hour / day_of_week / month", "Calendar features from timestamp", "Capture time-related operating patterns"],
            ["One-hot encoding", "machine_type / operating_mode", "Convert categories to model-ready binary columns"],
            ["Label encoding", "failure_type", "Convert multi-class target to numeric labels"],
        ], columns=["Feature", "How it is built", "Why"])
        st.dataframe(feats, use_container_width=True)

        st.info("Important project choice: IQR outliers were detected but kept. The presentation reports that outlier readings fail within 24h 62% of the time versus 15% overall, so they were treated as signal rather than noise.")

# ========================== EDA ==========================
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
            col = st.selectbox("Sensor", sensors)
            fig = px.histogram(
                raw_df, x=col, color="failure_within_24h" if "failure_within_24h" in raw_df else None,
                marginal="box", nbins=40,
                title=f"Distribution of {col}"
            )
            st.plotly_chart(fig_layout(fig, 430), use_container_width=True)

        with tabs[1]:
            if "failure_within_24h" in raw_df:
                stats_cols = [c for c in sensors if c in raw_df.columns]
                grouped = raw_df.groupby("failure_within_24h")[stats_cols].mean().T.reset_index()
                grouped.columns = ["Sensor", "Normal", "Failure"] if grouped.shape[1] == 3 else grouped.columns
                if "Normal" in grouped.columns:
                    long = grouped.melt("Sensor", var_name="Outcome", value_name="Mean")
                    fig = px.bar(long, x="Sensor", y="Mean", color="Outcome", barmode="group", title="Mean Sensor Readings by Failure Outcome")
                    st.plotly_chart(fig_layout(fig, 430), use_container_width=True)

            st.markdown("**Project findings before failure**")
            findings = pd.DataFrame([
                ["temperature_motor", "49.4 → 62.5", "+27%"],
                ["vibration_rms", "1.51 → 2.28", "+51%"],
                ["current_phase_avg", "8.52 → 10.81", "+27%"],
                ["hours_since_maintenance", "166 → 209 h", "+25%"],
                ["ambient_temp", "13.0 → 13.0", "unchanged"],
            ], columns=["Feature", "Normal → Failure", "Change"])
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
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
            st.plotly_chart(fig_layout(fig, 650), use_container_width=True)

            st.caption("The notebook highlights engineered rolling features as useful predictive signals and reports the strongest correlations with each target.")

        with tabs[3]:
            tc1, tc2 = st.columns(2)
            with tc1:
                if "failure_within_24h" in raw_df:
                    counts = raw_df["failure_within_24h"].value_counts().rename(index={0:"No Failure",1:"Failure"})
                    fig = px.pie(values=counts.values, names=counts.index, hole=.55, title="failure_within_24h")
                    st.plotly_chart(fig_layout(fig, 390), use_container_width=True)
            with tc2:
                if "failure_type" in raw_df:
                    counts = raw_df["failure_type"].value_counts()
                    fig = px.bar(x=counts.index, y=counts.values, title="failure_type classes")
                    st.plotly_chart(fig_layout(fig, 390), use_container_width=True)

            if "rul_hours" in raw_df:
                fig = px.histogram(raw_df, x="rul_hours", nbins=45, title="rul_hours Distribution")
                st.plotly_chart(fig_layout(fig, 380), use_container_width=True)

        with tabs[4]:
            st.markdown("""
            - **15%** of records represent failure within 24h → class imbalance matters.
            - **vibration_rms** rises about **51%** before failure.
            - **temperature_motor** rises about **27%** before failure.
            - **current_phase_avg** rises about **27%** before failure.
            - **hours_since_maintenance** rises about **25%** before failure.
            - **ambient_temp** stays at 13.0 in both outcome groups in the project's analysis.
            - The notebook reports **estimated_repair_cost** as highly right-skewed and strongly correlated with near-term failure.
            """)

# ====================== MODEL EVALUATION ======================
elif page == "Model Evaluation":
    st.markdown('<div class="section-title">ML Models & Evaluation</div>', unsafe_allow_html=True)

    st.markdown("### Binary Classification · Failure Within 24h")
    binary = pd.DataFrame([
        ["Logistic Regression", "Classification", "Baseline"],
        ["Random Forest", "Classification", "Baseline"],
        ["Tuned Random Forest", "Classification", "Tuned"],
        ["SVM", "Classification", "Baseline"],
        ["XGBoost", "Classification", "Best"],
    ], columns=["Model", "Task", "Stage"])
    st.dataframe(binary, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Best Model", "XGBoost")
    with c2: metric_card("Recall", "98.9%")
    with c3: metric_card("F1", "96.9%")
    with c4: metric_card("Selection", "Recall → F1")

    st.markdown("### Multi-class Classification · Failure Type")
    mc = pd.DataFrame([
        ["Random Forest", "5 classes"],
        ["XGBoost", "5 classes · Best"],
    ], columns=["Model", "Classes"])
    st.dataframe(mc, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Accuracy", "98.9%")
    with c2: metric_card("Macro Recall", "98.6%")
    with c3: metric_card("Macro F1", "96.9%")

    st.markdown("### Regression · Remaining Useful Life")
    reg = pd.DataFrame([
        ["Random Forest", "Best", "MAE 1.64 h", "RMSE 3.96", "R² 0.977"],
        ["Tuned Random Forest", "Tested", "—", "—", "—"],
        ["XGBoost", "Tested", "—", "—", "—"],
    ], columns=["Model", "Result", "MAE", "RMSE", "R²"])
    st.dataframe(reg, use_container_width=True)

    st.info("Metric logic from the project: recall + F1 decide the binary classifier; macro recall + macro F1 decide the multi-class model; MAE + R² decide the RUL regressor.")

# ============================ MLFLOW ============================
elif page == "MLflow":
    st.markdown('<div class="section-title">MLflow Experiment Tracking</div>', unsafe_allow_html=True)

    if not db_path:
        st.info("Cloud deployment: the registered prediction models are loaded from bundled model files. The local MLflow SQLite tracking backend is available when running the project locally.")
    else:
        st.success(f"Connected to local MLflow backend: {db_path.name}")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
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

            st.markdown(f"### {exp_name}")
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

            if runs.empty:
                st.caption("No runs found.")
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

            # Visualize the most relevant test metric per experiment.
            if "failure_24h" in exp_name:
                metric_cols = [c for c in ["test_accuracy", "test_precision", "test_recall", "test_f1"] if c in view]
            elif "failure_type" in exp_name:
                metric_cols = [c for c in ["test_accuracy", "test_recall_macro", "test_f1_macro"] if c in view]
            else:
                metric_cols = [c for c in ["test_mae", "test_rmse", "test_r2"] if c in view]

            if metric_cols and "runName" in view:
                long = view.melt(id_vars=["runName"], value_vars=metric_cols, var_name="Metric", value_name="Value")
                fig = px.bar(long, x="runName", y="Value", color="Metric", barmode="group", title="MLflow Test Metric Comparison")
                st.plotly_chart(fig_layout(fig, 410), use_container_width=True)

        st.markdown("### Registered Models")
        reg_rows = []
        for name in [
            "failure_within_24h_classifier",
            "failure_type_classifier",
            "rul_hours_regressor",
        ]:
            try:
                v = client.get_model_version_by_alias(name, "staging")
                reg_rows.append([name, v.version, "staging", v.run_id])
            except Exception:
                reg_rows.append([name, "—", "not available", "—"])
        st.dataframe(pd.DataFrame(reg_rows, columns=["Registered Model", "Version", "Alias", "Source Run"]), use_container_width=True)

        st.caption("The project reports 10 MLflow runs across 3 experiments, with hyperparameters, metrics, plots and serialized models logged. Best models are registered under the staging alias.")

# --------------------------- Footer ---------------------------
st.divider()
st.caption(
    "Will It Break? · Team 13 · Samsung Innovation Campus · "
    "Dashboard content follows the submitted project notebook and presentation."
)
