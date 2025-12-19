# lightgbm_training.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_PATH = r"D:\projects\Machine_failure_ai4i2020_dataset\Dataset\ai4i2020_smote.csv"
ARTIFACTS_DIR = r"D:\projects\Machine_failure_ai4i2020_dataset\artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# DROP NON-SENSOR COLUMNS (CRITICAL)
# -------------------------------------------------
drop_cols = ["UDI", "TWF", "HDF", "PWF", "OSF", "RNF"]
df = df.drop(columns=drop_cols)

# -------------------------------------------------
# Encode categorical columns
# -------------------------------------------------
encoders = {}
for col in ["Product_ID", "Type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, f"{col.lower()}_encoder.joblib"))

# -------------------------------------------------
# Split
# -------------------------------------------------
X = df.drop("Machine_failure", axis=1)
y = df["Machine_failure"]

print("✅ Training features:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------------------------
# Train
# -------------------------------------------------
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate
# -------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -------------------------------------------------
# Save
# -------------------------------------------------
joblib.dump(model, os.path.join(ARTIFACTS_DIR, "lightgbm_model.joblib"))
print("✅ Model saved")
