"""Streamlit dashboard for the Heart Disease Risk Prediction System (educational prototype).

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as cfg  # noqa: E402

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}
    .hero {background: linear-gradient(90deg,#b71c1c,#e53935); color: white;
           padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 1.8rem; color: white;}
    .hero p {margin: .3rem 0 0 0; opacity: .9;}
    .result-box {padding: 1.1rem 1.3rem; border-radius: 12px; font-size: 1.15rem; font-weight: 600;}
    .result-low {background: #e8f5e9; color: #1b5e20; border-left: 6px solid #43a047;}
    .result-high {background: #ffebee; color: #b71c1c; border-left: 6px solid #e53935;}
    @media (max-width: 640px) {.hero h1 {font-size: 1.3rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ loading helpers
@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not cfg.MODEL_PATH.exists():
        return None
    return joblib.load(cfg.MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    if not cfg.METRICS_PATH.exists():
        return None
    with open(cfg.METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(values: dict) -> list:
    """Return a list of problems (empty list = inputs are valid)."""
    problems = []
    missing = [c for c in cfg.FEATURE_COLUMNS if c not in values]
    if missing:
        problems.append(f"Missing inputs: {missing}")
        return problems
    for col, (lo, hi, _) in cfg.NUMERIC_RANGES.items():
        v = values[col]
        if v is None or pd.isna(v) or not (lo <= v <= hi):
            problems.append(f"{col} must be between {lo} and {hi} (got {v}).")
    for col, allowed in cfg.VALID_CATEGORIES.items():
        if values[col] not in allowed:
            problems.append(f"{col} must be one of {allowed} (got {values[col]}).")
    return problems


def build_input_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame with EXACTLY the training columns, in training order."""
    return pd.DataFrame([[values[c] for c in cfg.FEATURE_COLUMNS]], columns=cfg.FEATURE_COLUMNS)


def fmt(options: dict):
    return lambda key: f"{key} - {options[key]}"


pipeline = load_pipeline()
metrics = load_metrics()

# ------------------------------------------------------------------ header + disclaimer
st.markdown(
    """<div class="hero"><h1>❤️ Heart Disease Risk Prediction System</h1>
    <p>Machine-learning classifier trained on the UCI Heart Disease (Cleveland) dataset</p></div>""",
    unsafe_allow_html=True,
)
st.warning("⚠️ **Medical disclaimer:** " + cfg.DISCLAIMER)

if pipeline is None:
    st.error(
        f"Trained model not found at `{cfg.MODEL_PATH}`.\n\n"
        "Run the training step first from the project folder:\n\n"
        "`python src/train.py`\n\nThen refresh this page."
    )
    st.stop()

# Make sure the saved model expects exactly the columns this app sends.
expected = list(getattr(pipeline, "feature_names_in_", cfg.FEATURE_COLUMNS))
if expected != cfg.FEATURE_COLUMNS:
    st.error("Model/feature mismatch. Expected columns: " + str(expected) +
             "\n\nApp columns: " + str(cfg.FEATURE_COLUMNS) + "\n\nRe-run `python src/train.py`.")
    st.stop()

tab_predict, tab_perf, tab_info = st.tabs(["🩺 Prediction", "📊 Model performance", "ℹ️ Project info"])

# ------------------------------------------------------------------ TAB 1: prediction
with tab_predict:
    st.subheader("Enter clinical features")
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        R = cfg.NUMERIC_RANGES
        with c1:
            age = st.number_input("Age (years)", R["age"][0], R["age"][1], R["age"][2], step=1)
            sex = st.selectbox("Sex", cfg.VALID_CATEGORIES["sex"], index=1, format_func=fmt(cfg.SEX_LABELS))
            cp = st.selectbox("Chest pain type", cfg.VALID_CATEGORIES["cp"], index=3, format_func=fmt(cfg.CP_LABELS))
            trestbps = st.number_input("Resting blood pressure (mm Hg)", R["trestbps"][0], R["trestbps"][1],
                                       R["trestbps"][2], step=1)
            chol = st.number_input("Serum cholesterol (mg/dl)", R["chol"][0], R["chol"][1], R["chol"][2], step=1)
        with c2:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", cfg.VALID_CATEGORIES["fbs"],
                               format_func=fmt(cfg.FBS_LABELS))
            restecg = st.selectbox("Resting ECG", cfg.VALID_CATEGORIES["restecg"], format_func=fmt(cfg.RESTECG_LABELS))
            thalach = st.number_input("Max heart rate achieved", R["thalach"][0], R["thalach"][1],
                                      R["thalach"][2], step=1)
            exang = st.selectbox("Exercise-induced angina", cfg.VALID_CATEGORIES["exang"],
                                 format_func=fmt(cfg.EXANG_LABELS))
        with c3:
            oldpeak = st.number_input("ST depression (oldpeak)", float(R["oldpeak"][0]), float(R["oldpeak"][1]),
                                      float(R["oldpeak"][2]), step=0.1, format="%.1f")
            slope = st.selectbox("Slope of peak-exercise ST segment", cfg.VALID_CATEGORIES["slope"], index=1,
                                 format_func=fmt(cfg.SLOPE_LABELS))
            ca = st.selectbox("Major vessels coloured by fluoroscopy (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", cfg.VALID_CATEGORIES["thal"], format_func=fmt(cfg.THAL_LABELS))
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        values = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
                  "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
                  "slope": slope, "ca": ca, "thal": thal}
        problems = validate_inputs(values)
        if problems:
            for p in problems:
                st.error(p)
        else:
            X_new = build_input_frame(values)
            score = float(pipeline.predict_proba(X_new)[0, 1])
            label = int(score >= 0.5)
            st.divider()
            r1, r2 = st.columns([2, 1])
            with r1:
                if label == 1:
                    st.markdown('<div class="result-box result-high">Model output: pattern similar to '
                                '<b>disease-present</b> records (class 1)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-box result-low">Model output: pattern similar to '
                                '<b>no-disease</b> records (class 0)</div>', unsafe_allow_html=True)
                st.caption("This is the model's classification of the dataset label, not a diagnosis.")
            with r2:
                st.metric("Model score for class 1", f"{score:.1%}")
                st.progress(min(max(score, 0.0), 1.0))
            with st.expander("What does the model score mean?"):
                st.write(
                    "The score is the model's predicted probability for class 1 (source target > 0) on a "
                    "0-100% scale. It is **not calibrated**, it is based on only 303 records from one "
                    "hospital, and it is **not a personal risk estimate**. A decision threshold of 50% "
                    "is used to assign the class. Do not use it to make health decisions."
                )
            with st.expander("Input sent to the model"):
                st.dataframe(X_new, use_container_width=True)

# ------------------------------------------------------------------ TAB 2: performance
with tab_perf:
    st.subheader("Held-out test-set evaluation")
    if metrics is None:
        st.info("No metrics file found. Run `python src/train.py` to generate it.")
    else:
        st.write(f"**Selected model:** {metrics['selected_model']}  |  "
                 f"Train rows: {metrics['n_train']}  |  Test rows: {metrics['n_test']}")
        m = st.columns(5)
        m[0].metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
        m[1].metric("Precision", f"{metrics['test_precision']:.3f}")
        m[2].metric("Recall", f"{metrics['test_recall']:.3f}")
        m[3].metric("F1-score", f"{metrics['test_f1']:.3f}")
        m[4].metric("ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
        st.caption("Computed once on a stratified held-out test set. With a test set this small, "
                   "the numbers are noisy; treat them as indicative only.")

        st.markdown("**Cross-validation comparison (training data only)**")
        cv_df = pd.DataFrame(metrics["cv_results"])
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        cm = metrics["confusion_matrix"]
        st.markdown("**Confusion matrix counts:** "
                    f"TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

        g1, g2 = st.columns(2)
        for col, fname in ((g1, "07_confusion_matrix.png"), (g2, "08_roc_curve.png")):
            path = cfg.FIGURES_DIR / fname
            if path.exists():
                col.image(str(path), use_container_width=True)
        fi = cfg.FIGURES_DIR / "09_feature_importance.png"
        if fi.exists():
            st.image(str(fi), use_container_width=True)

# ------------------------------------------------------------------ TAB 3: info
with tab_info:
    st.subheader("About this project")
    st.write(
        "**Goal:** classify records of the UCI Heart Disease (Cleveland) dataset into *no disease* (0) "
        "and *disease present* (1) from 13 clinical features.\n\n"
        "**Target encoding:** the original column `num` takes values 0-4. We set label 0 when `num == 0` "
        "and label 1 when `num > 0`. These are *dataset labels*, not a confirmed clinical diagnosis.\n\n"
        "**Pipeline:** median/most-frequent imputation, standard scaling, one-hot encoding, then "
        "Logistic Regression or Random Forest. All preprocessing is inside one scikit-learn Pipeline "
        "so it is fitted on training folds only (no data leakage).\n\n"
        "**Tech stack:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, joblib."
    )
    st.markdown("**Feature dictionary**")
    st.table(pd.DataFrame({"feature": list(cfg.FEATURE_DESCRIPTIONS),
                           "description": list(cfg.FEATURE_DESCRIPTIONS.values())}))
    st.error("**Disclaimer:** " + cfg.DISCLAIMER)
