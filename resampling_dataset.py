import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv(r"D:\projects\Machine_failure_ai4i2020_dataset\Dataset\ai4i2020.csv")

# -----------------------------
# 2. Rename columns
# -----------------------------
df.rename(columns={
    'UDI': 'UDI',
    'Product ID': 'Product_ID',
    'Type': 'Type',
    'Air temperature [K]': 'Air_temperature',
    'Process temperature [K]': 'Process_temperature',
    'Rotational speed [rpm]': 'Rotational_speed',
    'Torque [Nm]': 'Torque',
    'Tool wear [min]': 'Tool_wear',
    'Machine failure': 'Machine_failure',
    'TWF': 'TWF',
    'HDF': 'HDF',
    'PWF': 'PWF',
    'OSF': 'OSF',
    'RNF': 'RNF'
}, inplace=True)

# -----------------------------
# 3. Label encode categorical columns
# -----------------------------
categorical_cols = ['Product_ID', 'Type']

encoders = {}
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# -----------------------------
# 4. Separate features & target
# -----------------------------
X = df.drop("Machine_failure", axis=1)
y = df["Machine_failure"]

# -----------------------------
# 5. Apply SMOTE (no column increase)
# -----------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# -----------------------------
# 6. Combine back into DataFrame
# -----------------------------
df_resampled = pd.DataFrame(X_res, columns=X.columns)
df_resampled["Machine_failure"] = y_res

# -----------------------------
# 7. Reverse Label Encoding (optional)
# -----------------------------
for col in categorical_cols:
    df_resampled[col] = encoders[col].inverse_transform(df_resampled[col].astype(int))
    
# -----------------------------
# 8. Save balanced dataset
# -----------------------------
output_path = r"D:\projects\Machine_failure_ai4i2020_dataset\Dataset\ai4i2020_smote.csv"
df_resampled.to_csv(output_path, index=False)

print(f"Balanced dataset saved as: {output_path}")
print("Original shape:", df.shape)
print("After SMOTE:", df_resampled.shape)
print(df_resampled['Machine_failure'].value_counts())
print(df_resampled['Machine_failure'].value_counts(normalize=True))
