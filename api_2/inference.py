import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Load model and encoders
model = joblib.load(os.path.join(ARTIFACTS_DIR, "lightgbm_model.joblib"))
product_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "product_id_encoder.joblib"))
type_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "type_encoder.joblib"))

FEATURES = [
    "Product_ID",
    "Type",
    "Air_temperature",
    "Process_temperature",
    "Rotational_speed",
    "Torque",
    "Tool_wear",
]

def _build_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # Rename columns to match model expectations
    df = df.rename(columns={
        "product_id": "Product_ID",
        "type": "Type",
        "air_temperature": "Air_temperature",
        "process_temperature": "Process_temperature",
        "rotational_speed": "Rotational_speed",
        "torque": "Torque",
        "tool_wear": "Tool_wear",
    })

    # SAFE ENCODING: Handles unseen labels by checking classes_
    df["Product_ID"] = df["Product_ID"].apply(
        lambda x: product_encoder.transform([x])[0] if x in product_encoder.classes_ else -1
    )
    
    df["Type"] = df["Type"].apply(
        lambda x: type_encoder.transform([x])[0] if x in type_encoder.classes_ else -1
    )

    return df[FEATURES]

def make_prediction(data: dict):
    df = _build_df([data])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return pred, prob

def make_batch_predictions(data: list[dict]):
    df = _build_df(data)
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    return list(zip(preds.astype(int), probs.astype(float)))