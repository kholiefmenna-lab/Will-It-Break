# Will It Break? — Rustic Industrial Final GUI

This package includes:
- app.py — redesigned high-contrast rustic-industrial Streamlit app
- predictive_maintenance_v3.csv — project dataset
- mlflow.db — supplied MLflow tracking/registry database
- requirements.txt — dependencies

## Run in VS Code
1. Open this folder in VS Code.
2. Open Terminal → New Terminal.
3. Run `python -m venv venv`
4. Run `venv\Scripts\activate`
5. Run `pip install -r requirements.txt`
6. Run `streamlit run app.py`

## MLflow note
The supplied mlflow.db contains the registered-model metadata and points to model artifact files from the original machine path. The database alone does not contain those artifact files. If the original `mlruns` folder is available, place it beside this project folder (preserving its structure) to enable live model loading. The dashboard and MLflow metadata views can still load from the database.

## Important RUL fix
The Live Predictions RUL path explicitly converts all RUL model feature names to Python strings before prediction to avoid scikit-learn mixed feature-name errors.
