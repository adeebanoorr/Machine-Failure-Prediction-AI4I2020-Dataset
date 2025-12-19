import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# 1. Paths to artifacts
# -----------------------------
ARTIFACTS_DIR = r"D:\projects\Machine_failure_ai4i2020_dataset\artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.joblib")
PRODUCT_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "product_id_encoder.joblib")
TYPE_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "type_encoder.joblib")

# -----------------------------
# 2. Load Model & Encoders
# -----------------------------
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        product_encoder = joblib.load(PRODUCT_ENCODER_PATH)
        type_encoder = joblib.load(TYPE_ENCODER_PATH)
        return model, product_encoder, type_encoder
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        st.stop()

model, product_encoder, type_encoder = load_model_and_encoders()

# -----------------------------
# 3. Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Machine Failure Prediction", layout="centered")
st.title("Predictive Maintenance Dashboard")
st.subheader("LightGBM Model for Machine Failure Prediction")

with st.expander("Input Guidelines"):
    st.markdown("""
    Enter the operating parameters to predict machine failure (0 = No Failure, 1 = Failure).  
    - **Product ID** and **Type** must match those used in training.
    """)

# -----------------------------
# 4. User Inputs
# -----------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    product_id_input = st.text_input("Product ID").strip()
    type_input = st.selectbox("Product Type (L, M, H)", ["L", "M", "H"])
    
    air_temp = col1.number_input("Air temperature [K]", 295.0, 310.0, 300.0, 0.1)
    process_temp = col2.number_input("Process temperature [K]", 305.0, 315.0, 310.0, 0.1)
    
    rotational_speed = col1.number_input("Rotational speed [rpm]", 1000.0, 3000.0, 1500.0, 1.0)
    torque = col2.number_input("Torque [Nm]", 20.0, 80.0, 40.0, 0.1)
    tool_wear = st.number_input("Tool wear [min]", 0.0, 300.0, 150.0, 1.0)
    
    submitted = st.form_submit_button("Predict Failure")

# -----------------------------
# 5. Prediction Logic
# -----------------------------
if submitted:
    if not product_id_input:
        st.error("Please enter a Product ID.")
    else:
        # Encode Product ID and Type
        try:
            product_encoded = product_encoder.transform([product_id_input])[0]
        except ValueError:
            st.error(f"Product ID '{product_id_input}' not found in training dataset.")
            st.stop()
            
        try:
            type_encoded = type_encoder.transform([type_input])[0]
        except ValueError:
            st.error(f"Type '{type_input}' not recognized.")
            st.stop()
        
        # Feature engineering (must match training)
        temp_diff = process_temp - air_temp
        power = torque * rotational_speed / 9550
        
        # Create input DataFrame matching training columns
        feature_columns = [
            "Product_ID", "Type",
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            "temp_difference", "power", "TWF", "HDF", "PWF", "OSF"
        ]
        
        # Fill additional columns (TWF, HDF, PWF, OSF) as 0 since unknown
        input_dict = {
            "Product_ID": product_encoded,
            "Type": type_encoded,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "temp_difference": temp_diff,
            "power": power,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0
        }
        
        input_df = pd.DataFrame([input_dict], columns=feature_columns)
        
        # Predict
        pred = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0, 1]
        
        # Display
        if pred == 1:
            st.error(f"**FAILURE PREDICTED!** Probability: {pred_prob:.2f}")
        else:
            st.success(f"**NO FAILURE PREDICTED.** Probability: {pred_prob:.2f}")
        
        # Optional: show calculated metrics
        st.markdown(f"**Calculated Metrics:** Temp Diff: `{temp_diff:.2f} K`, Power: `{power:.2f} kW`")
