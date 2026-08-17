"""
train.py
--------
Trains 5 classification models (Logistic Regression, Decision Tree, kNN,
Gaussian Naive Bayes, Random Forest) on the Online Shoppers Purchasing
Intention dataset (UCI). Saves each fitted pipeline (preprocessing + model)
to the model/ folder as a .pkl file, saves a held-out test_data.csv
(with the true label column so the Streamlit app can score it), and
prints/saves a metrics comparison table.

Dataset source: UCI Machine Learning Repository - Online Shoppers Purchasing
Intention Dataset (Sakar & Kastro, 2018).
Target column: Revenue (True/False -> did the visitor make a purchase)
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("online_shoppers_intention.csv")

TARGET = "Revenue"
df[TARGET] = df[TARGET].astype(int)          # True/False -> 1/0
df["Weekend"] = df["Weekend"].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features   :", numeric_features)
print("Categorical features:", categorical_features)

# ---------------------------------------------------------------------
# 2. Train / test split (stratified, since classes are imbalanced ~85/15)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the test split (features + true label) -> this is the file you will
# upload to the Streamlit app / submit as test_data.csv
test_export = X_test.copy()
test_export[TARGET] = y_test.values
test_export.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_export)} rows")

# ---------------------------------------------------------------------
# 3. Preprocessing: scale numeric, one-hot encode categorical
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ---------------------------------------------------------------------
# 4. Define the 5 required models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=RANDOM_STATE
    ),
}

results = []
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)
    fitted_pipelines[name] = pipe

    # save each fitted pipeline (preprocessing + model bundled together)
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(f"model/{safe_name}.pkl", "wb") as f:
        pickle.dump(pipe, f)

    print(f"\n{name}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------------
# 5. Save comparison table (used in README + shown in Streamlit app)
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("Model").round(4)
results_df.to_csv("model/metrics_comparison.csv")
print("\n=== Comparison Table ===")
print(results_df)

# Also store feature lists so the Streamlit app knows the schema
schema = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": TARGET,
}
with open("model/schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("\nAll models trained and saved to model/ folder.")
