import os
import glob
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional
from datetime import datetime


# Utilities: Artifact Loading
def get_latest_file(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def load_artifacts():
    """Load best model, preprocessor, and optional metadata from deployment/ or models/.
    Returns (model_or_pipeline, preprocessor, metadata_dict_or_none)
    """

    deployment_dir = "deployment"
    models_dir = "models"

    metadata_path = get_latest_file(os.path.join(deployment_dir, "model_metadata_*.pkl"))
    preprocessor_path = get_latest_file(os.path.join(deployment_dir, "preprocessor_*.pkl"))
    best_model_path = get_latest_file(os.path.join(deployment_dir, "best_model_*.pkl"))

    metadata = None
    model = None
    preprocessor = None

    if metadata_path:
        try:
            metadata = joblib.load(metadata_path)
        except Exception:
            metadata = None

    if best_model_path:
        try:
            model = joblib.load(best_model_path)
        except Exception:
            model = None

    if preprocessor_path:
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception:
            preprocessor = None

    if model is None:
        model = joblib.load(get_latest_file(os.path.join(models_dir, "best_*.pkl"))) if get_latest_file(os.path.join(models_dir, "best_*.pkl")) else None

    if preprocessor is None:
        pre_path = get_latest_file(os.path.join(models_dir, "enhanced_preprocessor.pkl"))
        if pre_path is None:
            pre_path = get_latest_file(os.path.join(models_dir, "improved_preprocessor.pkl"))
        preprocessor = joblib.load(pre_path) if pre_path else None

    return model, preprocessor, metadata


# Feature Engineering (must match training)
def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()

    # Ratios and interactions used in training notebook
    processed["complaints_tenure_ratio"] = processed["complaints"] / (processed["tenure_months"] + 1)
    processed["usage_income_ratio"] = processed["usage_gb"] / (processed["income"] / 1000)
    processed["age_tenure_ratio"] = processed["age"] / (processed["tenure_months"] + 1)

    # Bins
    processed["age_group"] = pd.cut(processed["age"], bins=[0, 30, 50, 70, 100], labels=["Young", "Middle", "Senior", "Elderly"], include_lowest=True)
    processed["income_group"] = pd.cut(processed["income"], bins=[0, 30000, 60000, 90000, 100000], labels=["Low", "Medium", "High", "Very_High"], include_lowest=True)
    processed["usage_group"] = pd.cut(processed["usage_gb"], bins=[0, 25, 50, 75, 100], labels=["Low", "Medium", "High", "Very_High"], include_lowest=True)

    # Flags
    processed["is_new_customer"] = (processed["tenure_months"] <= 12).astype(int)
    processed["is_long_term_customer"] = (processed["tenure_months"] >= 36).astype(int)
    processed["high_complaints"] = (processed["complaints"] >= 6).astype(int)
    processed["complaint_frequency"] = processed["complaints"] / (processed["tenure_months"] + 1)

    return processed

# Streamlit UI and Inference
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Telecom Customer Churn Prediction")
st.write("Enter customer details to predict churn probability. The app mirrors the training-time preprocessing and model.")

with st.sidebar:
    st.header("Model Artifacts")
    model, preprocessor, metadata = load_artifacts()
    if metadata:
        st.success(f"Loaded metadata from {metadata.get('timestamp', 'N/A')}")
    if model is None:
        st.error("Could not load a trained model. Please run the training notebook to export artifacts.")
    if preprocessor is None:
        st.warning("Preprocessor not found. If the model is a full pipeline, predictions may still work.")

    # Diagnostics (optional): helps debug missing artifacts on Render/Cloud
    with st.expander("Diagnostics: artifact discovery", expanded=False):
        try:
            cwd = os.getcwd()
            st.text(f"CWD: {cwd}")
        except Exception as e:
            st.text(f"CWD: <error: {e}>")

        dep_dir = "deployment"
        models_dir = "models"

        def safe_listdir(path: str):
            if os.path.isdir(path):
                try:
                    return os.listdir(path)
                except Exception as e:
                    return [f"<error listing {path}: {e}>"]
            return ["<missing>"]

        st.text(f"deployment/: {safe_listdir(dep_dir)}")
        st.text(f"models/: {safe_listdir(models_dir)}")

        dep_metadata = glob.glob(os.path.join(dep_dir, "model_metadata_*.pkl"))
        dep_pre = glob.glob(os.path.join(dep_dir, "preprocessor_*.pkl"))
        dep_model = glob.glob(os.path.join(dep_dir, "best_model_*.pkl"))

        st.text(f"deployment model_metadata_*: {dep_metadata}")
        st.text(f"deployment preprocessor_*: {dep_pre}")
        st.text(f"deployment best_model_*: {dep_model}")

        mod_best = glob.glob(os.path.join(models_dir, "best_*.pkl"))
        mod_pre_candidates = [
            os.path.join(models_dir, "enhanced_preprocessor.pkl"),
            os.path.join(models_dir, "improved_preprocessor.pkl"),
        ]
        mod_pre_exist = [p for p in mod_pre_candidates if os.path.exists(p)]
        st.text(f"models best_*: {mod_best}")
        st.text(f"models preprocessor candidates (exist): {mod_pre_exist}")

st.subheader("Input Features")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (18 - 100)", min_value=18, max_value=100, value=45, step=1)
    income = st.number_input("Annual Income (10000$ - 100000$)", min_value=10000, max_value=100000, value=55000, step=500)
    usage_gb = st.number_input("Monthly Data Usage (1GB - 100GB)", min_value=1, max_value=100, value=50, step=1)
    complaints = st.number_input("Complaints (0 - 10)", min_value=0, max_value=10, value=4, step=1)
with col2:
    tenure_months = st.number_input("Tenure (1 month - 120 months)", min_value=1, max_value=120, value=24, step=1)
    gender = st.selectbox("Gender", options=["female", "male"], index=0)
    plan_type = st.selectbox("Plan Type", options=["Postpaid", "Prepaid"], index=0)

if st.button("Predict Churn"):
    if model is None:
        st.stop()

    input_df = pd.DataFrame([
        {
            "age": age,
            "gender": gender,
            "income": income,
            "usage_gb": usage_gb,
            "complaints": complaints,
            "tenure_months": tenure_months,
            "plan_type": plan_type,
        }
    ])

    input_df_adv = create_advanced_features(input_df)

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df_adv)[0, 1]
            pred = int(proba >= 0.5)
        else:
            raise AttributeError
    except Exception:
        if preprocessor is None:
            st.error("No preprocessor available and model is not a pipeline. Cannot proceed.")
            st.stop()
        X_trans = preprocessor.transform(input_df_adv)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_trans)[0, 1]
            pred = int(proba >= 0.5)
        else:
            pred = int(model.predict(X_trans)[0])
            proba = float(pred)

    st.markdown("---")
    st.subheader("Prediction")
    st.metric(label="Churn Probability", value=f"{proba:.3f}")
    st.metric(label="Predicted Class", value="Churn" if pred == 1 else "Stay")

    st.markdown("### Feature Summary")
    st.dataframe(input_df_adv)

st.markdown("---")
st.caption("Model artifacts are loaded from 'deployment/' (preferred) or 'models/'. Ensure the training notebook has exported the latest artifacts.")


