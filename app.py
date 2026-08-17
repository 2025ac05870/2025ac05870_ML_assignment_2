"""
Streamlit app - Online Shoppers Purchasing Intention Classifier
Assignment 2 (M.Tech AIML) demo app.

Lets the user upload the test_data.csv (or any CSV with the same schema,
including the true 'Revenue' label), pick one of the 5 trained models,
and see predictions + evaluation metrics + confusion matrix.
"""

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

st.set_page_config(page_title="Online Shoppers Intention Classifier", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_schema():
    with open("model/schema.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics_table():
    return pd.read_csv("model/metrics_comparison.csv", index_col=0)


st.title("🛒 Online Shoppers Purchasing Intention — Classifier Demo")
st.caption(
    "M.Tech AIML — Machine Learning Assignment 2 | "
    "Dataset: UCI Online Shoppers Purchasing Intention"
)

schema = load_schema()
target = schema["target"]

# ---------------------------------------------------------------------
# Sidebar: model selection
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
pipe = load_model(MODEL_FILES[model_name])

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload test data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (must include the 'Revenue' label column)",
    type=["csv"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Tip: use the provided `test_data.csv` from the GitHub repo, "
    "or any CSV with the same columns."
)

# ---------------------------------------------------------------------
# Comparison table of all 6 models (always visible)
# ---------------------------------------------------------------------
with st.expander("📊 Comparison of all models (pre-computed on hold-out test split)", expanded=False):
    st.dataframe(load_metrics_table().style.highlight_max(axis=0, color="#c6f6d5"))

# ---------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to run predictions and see live metrics.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(data.head())

has_labels = target in data.columns

if has_labels:
    y_true = data[target]
    if y_true.dtype == object or y_true.dtype == bool:
        y_true = y_true.astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(y_true).astype(int)
    X = data.drop(columns=[target])
else:
    X = data.copy()
    st.warning(
        "No 'Revenue' column found — showing predictions only "
        "(metrics need the true label column)."
    )

# Align Weekend to int if present as bool/string
if "Weekend" in X.columns and X["Weekend"].dtype != "int64":
    X["Weekend"] = (
        X["Weekend"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(X["Weekend"])
    )
    X["Weekend"] = X["Weekend"].astype(int)

try:
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
except Exception as e:
    st.error(
        f"Prediction failed — make sure the uploaded CSV has the same "
        f"columns as the training data. Error: {e}"
    )
    st.stop()

result_df = X.copy()
result_df["Predicted_Revenue"] = y_pred.astype(bool)
result_df["Purchase_Probability"] = y_proba.round(3)

st.subheader(f"Predictions — {model_name}")
st.dataframe(result_df.head(20))

st.download_button(
    "⬇️ Download predictions as CSV",
    result_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------
# Metrics + confusion matrix (only if ground truth available)
# ---------------------------------------------------------------------
if has_labels:
    st.subheader("📈 Evaluation metrics on uploaded data")

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=["No Purchase", "Purchase"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))

st.markdown("---")
st.caption(
    "Built for BITS Pilani WILP M.Tech (AIML) — Machine Learning Assignment 2."
)
