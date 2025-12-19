import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -----------------------------
# 1. Load balanced dataset
# -----------------------------
df = pd.read_csv(r"D:\projects\Machine_failure_ai4i2020_dataset\Dataset\ai4i2020_smote.csv")

# -----------------------------
# 2. Encode categorical columns
# -----------------------------
categorical_cols = ["Product_ID", "Type"]

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# -----------------------------
# 3. Split features & target
# -----------------------------
X = df.drop("Machine_failure", axis=1)
y = df["Machine_failure"]

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# 5. Initialize Models
# -----------------------------
models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# -----------------------------
# 6. Create artifacts folder
# -----------------------------
artifacts_dir = r"D:\projects\Machine_failure_ai4i2020_dataset\artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# -----------------------------
# 7. Train, Evaluate & Save Models
# -----------------------------
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    # Save trained model
    model_path = os.path.join(artifacts_dir, f"{name}.joblib")
    joblib.dump(model, model_path)
    print(f"{name} saved to {model_path}")

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

# -----------------------------
# 8. Comparison Table
# -----------------------------
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

print("\n=========== MODEL COMPARISON ===========")
print(results_df)
print("\nBest Model:", results_df.iloc[0]["Model"])
