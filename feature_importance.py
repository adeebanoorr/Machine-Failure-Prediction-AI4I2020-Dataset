import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# BASE PROJECT DIRECTORY
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------

# Backend artifacts
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.joblib")

# React public folder
REACT_PUBLIC_DIR = os.path.join(
    BASE_DIR,
    "myreact",
    "public"
)

# api_2 folder (new requirement)
API2_DIR = os.path.join(
    BASE_DIR,
    "api_2"
)

# Ensure directories exist
os.makedirs(REACT_PUBLIC_DIR, exist_ok=True)
os.makedirs(API2_DIR, exist_ok=True)

# Output paths
REACT_IMAGE_PATH = os.path.join(REACT_PUBLIC_DIR, "feature_importance.png")
API2_IMAGE_PATH = os.path.join(API2_DIR, "feature_importance.png")

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# -------------------------------------------------
# FEATURE NAMES (MUST MATCH TRAINING ORDER)
# -------------------------------------------------

feature_names = [
    "Product_ID",
    "Type",
    "Air_temperature",
    "Process_temperature",
    "Rotational_speed",
    "Torque",
    "Tool_wear"
]

# -------------------------------------------------
# EXTRACT FEATURE IMPORTANCE
# -------------------------------------------------

importances = model.feature_importances_

df_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# -------------------------------------------------
# PLOT FEATURE IMPORTANCE
# -------------------------------------------------

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

ax = sns.barplot(
    x="Importance",
    y="Feature",
    data=df_importance,
    hue="Feature",
    palette="viridis",
    legend=False
)


plt.title(
    "Feature Importance: Machine Failure Prediction",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Importance (Number of Splits)")
plt.ylabel("Sensor Metric")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

# -------------------------------------------------
# SAVE IMAGE (BOTH LOCATIONS)
# -------------------------------------------------

plt.savefig(REACT_IMAGE_PATH, dpi=300)
plt.savefig(API2_IMAGE_PATH, dpi=300)
plt.close()

print("Feature importance image saved successfully at:")
print(REACT_IMAGE_PATH)
print(API2_IMAGE_PATH)
