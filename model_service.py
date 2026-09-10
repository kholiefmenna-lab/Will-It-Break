"""
Model Service for Industrial Predictive Maintenance System
Samsung Innovation Campus (SIC - AI803) - Team 13
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
TRACKING_URI = f"sqlite:///{DB_PATH.replace(os.sep, '/')}"

CLASSES = ['bearing', 'electrical', 'hydraulic', 'motor_overheat', 'none']

FEATURE_ORDER = [
    'machine_id',
    'vibration_rms',
    'temperature_motor',
    'current_phase_avg',
    'pressure_level',
    'rpm',
    'hours_since_maintenance',
    'ambient_temp',
    'machine_type_Compressor',
    'machine_type_Pump',
    'machine_type_Robotic Arm',
    'operating_mode_normal',
    'operating_mode_peak',
    'hour',
    'day_of_week',
    'month',
    'vibration_rms_change',
    'temperature_motor_change',
    'current_phase_avg_change',
    'pressure_level_change',
    'rpm_change',
    'vibration_rms_rolling_mean_5',
    'temperature_motor_rolling_mean_5',
    'pressure_level_rolling_mean_5'
]

import joblib

MODELS_DIR = os.path.join(BASE_DIR, "models")
_MODELS_CACHE = {}

def get_models():
    """Load and cache the 3 models (from models/ directory or MLflow SQLite)."""
    global _MODELS_CACHE
    if not _MODELS_CACHE:
        # Check for portable standalone files first (ideal for Streamlit Cloud & GitHub)
        bin_path = os.path.join(MODELS_DIR, "binary_model.joblib")
        rul_path = os.path.join(MODELS_DIR, "rul_model.joblib")
        mc_path = os.path.join(MODELS_DIR, "multiclass_model.joblib")

        # If rul_model.joblib was uploaded in parts to stay under GitHub's 25MB browser limit, reassemble it
        if not os.path.exists(rul_path) and os.path.exists(rul_path + ".part1"):
            with open(rul_path, "wb") as outfile:
                p_num = 1
                while os.path.exists(f"{rul_path}.part{p_num}"):
                    with open(f"{rul_path}.part{p_num}", "rb") as infile:
                        outfile.write(infile.read())
                    p_num += 1

        if os.path.exists(bin_path) and os.path.exists(rul_path) and os.path.exists(mc_path):
            _MODELS_CACHE['binary'] = joblib.load(bin_path)
            _MODELS_CACHE['rul'] = joblib.load(rul_path)
            _MODELS_CACHE['multiclass'] = joblib.load(mc_path)
        else:
            mlflow.set_tracking_uri(TRACKING_URI)
            _MODELS_CACHE['binary'] = mlflow.sklearn.load_model("models:/failure_within_24h_classifier/1")
            _MODELS_CACHE['rul'] = mlflow.sklearn.load_model("models:/rul_hours_regressor/1")
            _MODELS_CACHE['multiclass'] = mlflow.sklearn.load_model("models:/failure_type_classifier/1")
    return _MODELS_CACHE


def prepare_features(input_data: dict) -> pd.DataFrame:
    """Transform raw user/sensor inputs into the 24 model features."""
    m_type = input_data.get('machine_type', 'CNC')
    op_mode = input_data.get('operating_mode', 'normal')
    
    vib = float(input_data.get('vibration_rms', 1.2))
    temp = float(input_data.get('temperature_motor', 50.0))
    curr = float(input_data.get('current_phase_avg', 6.5))
    pres = float(input_data.get('pressure_level', 45.0))
    rpm = float(input_data.get('rpm', 900.0))
    
    features = {
        'machine_id': int(input_data.get('machine_id', 1)),
        'vibration_rms': vib,
        'temperature_motor': temp,
        'current_phase_avg': curr,
        'pressure_level': pres,
        'rpm': rpm,
        'hours_since_maintenance': float(input_data.get('hours_since_maintenance', 50.0)),
        'ambient_temp': float(input_data.get('ambient_temp', 13.0)),
        
        # Categorical dummies (reference baseline: CNC, idle)
        'machine_type_Compressor': 1 if m_type == 'Compressor' else 0,
        'machine_type_Pump': 1 if m_type == 'Pump' else 0,
        'machine_type_Robotic Arm': 1 if m_type == 'Robotic Arm' else 0,
        'operating_mode_normal': 1 if op_mode == 'normal' else 0,
        'operating_mode_peak': 1 if op_mode == 'peak' else 0,
        
        # Datetime features
        'hour': int(input_data.get('hour', 14)),
        'day_of_week': int(input_data.get('day_of_week', 2)),
        'month': int(input_data.get('month', 5)),
        
        # Delta features (default to 0.0 or supplied)
        'vibration_rms_change': float(input_data.get('vibration_rms_change', 0.0)),
        'temperature_motor_change': float(input_data.get('temperature_motor_change', 0.0)),
        'current_phase_avg_change': float(input_data.get('current_phase_avg_change', 0.0)),
        'pressure_level_change': float(input_data.get('pressure_level_change', 0.0)),
        'rpm_change': float(input_data.get('rpm_change', 0.0)),
        
        # Rolling averages (default to current reading if not supplied)
        'vibration_rms_rolling_mean_5': float(input_data.get('vibration_rms_rolling_mean_5', vib)),
        'temperature_motor_rolling_mean_5': float(input_data.get('temperature_motor_rolling_mean_5', temp)),
        'pressure_level_rolling_mean_5': float(input_data.get('pressure_level_rolling_mean_5', pres))
    }
    
    df = pd.DataFrame([features])[FEATURE_ORDER]
    return df


def get_recommendations(failure_type: str, failure_prob: float, rul_hours: float) -> list:
    """Provide domain-driven maintenance recommendations."""
    recs = []
    
    if failure_prob >= 0.70 or rul_hours < 12.0:
        recs.append("🚨 **CRITICAL URGENCY**: Immediate machine shutdown recommended to avoid catastrophic failure.")
    elif failure_prob >= 0.35 or rul_hours < 36.0:
        recs.append("⚠️ **ELEVATED RISK**: Schedule maintenance inspection within the current work shift.")
    else:
        recs.append("✅ **STABLE STATUS**: Machine is operating within acceptable nominal parameters.")
        
    if failure_type == 'bearing':
        recs.append("🔧 **Bearing Wear Detected**: Inspect shaft alignment, check lubrication levels, and listen for acoustic harmonics. Replace worn ball/roller bearings.")
    elif failure_type == 'motor_overheat':
        recs.append("🌡️ **Motor Overheating**: Check cooling fans, inspect thermal paste/fins, verify current load, and clear ventilation blockages.")
    elif failure_type == 'hydraulic':
        recs.append("💧 **Hydraulic System Fault**: Check hydraulic fluid reservoir, inspect high-pressure valves, and test for internal seal leaks.")
    elif failure_type == 'electrical':
        recs.append("⚡ **Electrical Instability**: Measure phase balance across all 3 phases, test insulation resistance (megger test), and check inverter/VFD.")
    elif failure_type == 'none':
        recs.append("📋 **Standard Routine**: Continue regular shift monitoring and schedule routine lubrication at standard intervals.")
        
    return recs


def predict_single(input_data: dict) -> dict:
    """Run full inference on a single machine record."""
    models = get_models()
    X = prepare_features(input_data)
    
    # Binary Classification
    binary_pred = int(models['binary'].predict(X)[0])
    binary_prob = float(models['binary'].predict_proba(X)[0, 1])
    
    # RUL Regression
    rul_pred = float(models['rul'].predict(X)[0])
    rul_pred = max(0.5, round(rul_pred, 2))
    
    # Multi-class Classification
    mc_pred_idx = int(models['multiclass'].predict(X)[0])
    mc_probs = models['multiclass'].predict_proba(X)[0]
    failure_type = CLASSES[mc_pred_idx]
    
    # Risk Level Categorization
    if binary_prob > 0.70 or rul_pred < 12:
        risk_level = "CRITICAL"
        risk_color = "#e63946"
    elif binary_prob > 0.35 or rul_pred < 36:
        risk_level = "WARNING"
        risk_color = "#f4a261"
    else:
        risk_level = "HEALTHY"
        risk_color = "#2a9d8f"
        
    type_prob_dict = {CLASSES[i]: round(float(mc_probs[i]), 4) for i in range(len(CLASSES))}
    
    recommendations = get_recommendations(failure_type, binary_prob, rul_pred)
    
    return {
        "failure_within_24h": binary_pred,
        "failure_probability": round(binary_prob, 4),
        "rul_hours": rul_pred,
        "predicted_failure_type": failure_type,
        "failure_type_probabilities": type_prob_dict,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendations": recommendations,
        "features_used": X.iloc[0].to_dict()
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run batch prediction on an uploaded DataFrame.
    Automatically handles missing columns, encoding, and rolling features.
    """
    models = get_models()
    df_copy = df.copy()
    
    defaults = {
        'machine_id': 1,
        'machine_type': 'CNC',
        'operating_mode': 'normal',
        'vibration_rms': 1.2,
        'temperature_motor': 50.0,
        'current_phase_avg': 6.5,
        'pressure_level': 45.0,
        'rpm': 900.0,
        'hours_since_maintenance': 50.0,
        'ambient_temp': 13.0,
        'hour': 12,
        'day_of_week': 2,
        'month': 5
    }
    
    for col, val in defaults.items():
        if col not in df_copy.columns:
            df_copy[col] = val
            
    if 'vibration_rms_change' not in df_copy.columns:
        if 'machine_id' in df_copy.columns and len(df_copy) > 1:
            sensor_cols = ["vibration_rms", "temperature_motor", "current_phase_avg", "pressure_level", "rpm"]
            for col in sensor_cols:
                df_copy[f"{col}_change"] = df_copy.groupby("machine_id")[col].diff().fillna(0)
            for col in ["vibration_rms", "temperature_motor", "pressure_level"]:
                df_copy[f"{col}_rolling_mean_5"] = (
                    df_copy.groupby("machine_id")[col]
                    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
                )
        else:
            df_copy['vibration_rms_change'] = 0.0
            df_copy['temperature_motor_change'] = 0.0
            df_copy['current_phase_avg_change'] = 0.0
            df_copy['pressure_level_change'] = 0.0
            df_copy['rpm_change'] = 0.0
            df_copy['vibration_rms_rolling_mean_5'] = df_copy['vibration_rms']
            df_copy['temperature_motor_rolling_mean_5'] = df_copy['temperature_motor']
            df_copy['pressure_level_rolling_mean_5'] = df_copy['pressure_level']
            
    df_copy['machine_type_Compressor'] = (df_copy['machine_type'] == 'Compressor').astype(int)
    df_copy['machine_type_Pump'] = (df_copy['machine_type'] == 'Pump').astype(int)
    df_copy['machine_type_Robotic Arm'] = (df_copy['machine_type'] == 'Robotic Arm').astype(int)
    df_copy['operating_mode_normal'] = (df_copy['operating_mode'] == 'normal').astype(int)
    df_copy['operating_mode_peak'] = (df_copy['operating_mode'] == 'peak').astype(int)
    
    X = df_copy[FEATURE_ORDER]
    
    binary_preds = models['binary'].predict(X)
    binary_probs = models['binary'].predict_proba(X)[:, 1]
    rul_preds = models['rul'].predict(X)
    mc_preds = models['multiclass'].predict(X)
    
    df_copy['Pred_Fail_24h'] = binary_preds
    df_copy['Fail_Probability_%'] = (binary_probs * 100).round(1)
    df_copy['Pred_RUL_Hours'] = np.maximum(0.5, rul_preds).round(2)
    df_copy['Pred_Failure_Type'] = [CLASSES[int(i)] for i in mc_preds]
    
    def get_risk(row):
        if row['Fail_Probability_%'] >= 70 or row['Pred_RUL_Hours'] < 12:
            return "CRITICAL"
        elif row['Fail_Probability_%'] >= 35 or row['Pred_RUL_Hours'] < 36:
            return "WARNING"
        return "HEALTHY"
        
    df_copy['Risk_Status'] = df_copy.apply(get_risk, axis=1)
    return df_copy


def load_dataset_samples(n: int = 15) -> pd.DataFrame:
    """Load sample records from the cached dataset or generate clean representative samples."""
    cache_path = os.path.expanduser(
        r"~/.cache/kagglehub/datasets/tatheerabbas/industrial-machine-predictive-maintenance/versions/1/predictive_maintenance_v3.csv"
    )
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            samples = []
            for ft in df['failure_type'].unique():
                sub = df[df['failure_type'] == ft]
                samples.append(sub.sample(min(len(sub), max(2, n // 5)), random_state=42))
            res = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True).head(n)
            return res
        except Exception:
            pass
            
    data = []
    for i in range(n):
        data.append({
            'machine_id': (i % 5) + 1,
            'machine_type': ['CNC', 'Pump', 'Compressor', 'Robotic Arm'][i % 4],
            'operating_mode': ['normal', 'peak', 'idle'][i % 3],
            'vibration_rms': round(np.random.uniform(0.8, 4.5), 2),
            'temperature_motor': round(np.random.uniform(40, 85), 1),
            'current_phase_avg': round(np.random.uniform(4, 20), 1),
            'pressure_level': round(np.random.uniform(30, 110), 1),
            'rpm': round(np.random.uniform(700, 2500), 0),
            'hours_since_maintenance': round(np.random.uniform(10, 350), 1),
            'ambient_temp': 14.0
        })
    return pd.DataFrame(data)
