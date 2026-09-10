# Industrial Machine Predictive Maintenance - Interactive GUI
### Samsung Innovation Campus (SIC - AI803) | Team 13
**Project:** *"Will It Break? Predicting Industrial Machine Failures"*

---

## 🚀 How to Launch the Application

### Option 1: Double-Click Desktop Launcher (Fastest)
Simply double-click the **`run_gui.bat`** file in this folder. It will start the server and automatically launch the dashboard in your default web browser.

### Option 2: Command Line (PowerShell / Command Prompt)
Open your terminal in this directory and run:
```bash
python -m streamlit run app.py
```
Then open your browser to **`http://localhost:8501`**.

---

## 🌟 Application Features

### 1. 🎯 Live Machine Health & Diagnostic Center
- **Equipment Profile**: Select Machine ID (1–20), Machine Type (*CNC, Pump, Compressor, Robotic Arm*), and Operating Mode (*idle, normal, peak*).
- **Interactive Telemetry Sliders**: Adjust live sensor values including:
  - RMS Vibration (`0.3` to `10.0 mm/s`)
  - Motor Temperature (`25` to `100 °C`)
  - Phase Current (`2` to `35 A`)
  - Pressure Level (`10` to `210 bar`)
  - Rotational Speed / RPM (`100` to `4200 RPM`)
  - Hours Since Last Maintenance & Ambient Temperature
- **⚡ Quick Diagnostic Presets**: Single-click testing scenarios in the sidebar:
  - 🟢 *Healthy Machine (Nominal)*
  - 🔴 *Bearing Degradation Fault*
  - 🔴 *Motor Overheat Hazard*
  - 🟡 *Hydraulic Pressure Spike*
  - ⚡ *Electrical Phase Imbalance*
- **Real-Time AI Predictions**:
  - **24-Hour Failure Risk**: Gauge meter + Probability % + Alert Status (*Healthy, Warning, Critical*).
  - **Remaining Useful Life (RUL)**: Continuous countdown in hours.
  - **Root-Cause Failure Mode**: Predicts specific failure type (*Bearing, Motor Overheat, Hydraulic, Electrical, or None*) with multi-class probability bar chart.
  - **Maintenance Action Plan**: Tailored technician checklist and safety protocols.

---

### 2. 🏭 Fleet Monitoring & Batch CSV Diagnosis
- **Batch Processing**: Upload any plant CSV file or click **"Load Fleet Sample Records"** to test 15 machines across different states.
- **Fleet KPI Summary Cards**:
  - Total Monitored Machines
  - Critical 24h Alerts count and percentage
  - Elevated Warnings count
  - Fleet Average Remaining Useful Life
- **Risk Table**: Color-coded table with filters for *CRITICAL*, *WARNING*, and *HEALTHY*.
- **Report Export**: Download full fleet diagnostic predictions as a timestamped CSV report.

---

### 3. 📊 MLflow Registry Snapshot & Model Performance
- Reads the committed `mlflow.db` snapshot directly with SQLite for dashboard display.
- Live inference uses the portable `.joblib` models in `models/`; it does not depend on the MLflow registry at runtime.
- Inspect the 3 champion models registered in MLflow:
  - **Binary Classifier (XGBoost)**: 98.7% Test Recall, 0.950 F1, 0.999 ROC-AUC.
  - **Regression Model (Random Forest)**: 1.64 hours MAE, 3.96 hours RMSE, 0.977 R².
  - **Multi-Class Classifier (XGBoost)**: 98.9% Accuracy, 98.8% Macro Recall.
- Top 10 most predictive telemetry features (Feature Importance chart).

---

### 4. ℹ️ Project Overview & Team 13
- Comprehensive project summary, problem statement, and business value.
- Team 13 Members:
  - **Menna Kholief**
  - **Omar Elbeltagy**
  - **Marwa Mahmoud**
  - **Mariam Ramy**
- Program: Samsung Innovation Campus (SIC - AI803).

---

## 📁 Code Structure

- **`app.py`**: Streamlit graphical dashboard interface and visualizations.
- **`model_service.py`**: MLflow SQLite model loader, feature transformer pipeline (24 features), and inference service.
- **`run_gui.bat`**: One-click Windows launcher.
- **`mlflow.db`**: SQLite database containing all MLflow experiments, metrics, and registered models.
- **`mlruns/`**: Artifact storage for model binaries and evaluation artifacts.
