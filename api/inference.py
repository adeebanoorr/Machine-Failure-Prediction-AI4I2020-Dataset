import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Load artifacts saved during training
model = joblib.load(os.path.join(ARTIFACTS_DIR, "lightgbm_model.joblib"))
product_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "product_id_encoder.joblib"))
type_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "type_encoder.joblib"))

# STANDARDIZED FEATURES: Exactly matches resampling_dataset.py output
FEATURES = [
    "Product_ID", "Type", "Air_temperature", "Process_temperature",
    "Rotational_speed", "Torque", "Tool_wear"
]

def predict(req):
    try:
        # Encode categorical strings using saved encoders
        product_id = product_encoder.transform([req.product_id])[0]
        type_ = type_encoder.transform([req.type])[0]
    except ValueError as e:
        raise ValueError(f"Encoding Error: {str(e)}")

    # Construct DataFrame with identical column names and order as training
    df = pd.DataFrame([[
        product_id,
        type_,
        req.air_temperature,
        req.process_temperature,
        req.rotational_speed,
        req.torque,
        req.tool_wear
    ]], columns=FEATURES)

    # Perform prediction
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0, 1])

    return pred, prob