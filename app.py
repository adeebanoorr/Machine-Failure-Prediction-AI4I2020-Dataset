# app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Machine Failure Prediction",
    layout="centered"
)

st.title("🔧 Predictive Maintenance Dashboard")
st.subheader("LightGBM Model (FastAPI Backend)")

with st.expander(" Input Guidelines"):
    st.markdown("""
    - Product ID must exist in training data  
    - Type must be one of **L, M, H**  
    - Output:
        - `0` → No Failure  
        - `1` → Failure
    """)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    product_id = st.text_input("Product ID").strip()
    type_ = st.selectbox("Product Type", ["L", "M", "H"])

    air_temp = col1.number_input("Air temperature [K]", 295.0, 310.0, 300.0, 0.1)
    process_temp = col2.number_input("Process temperature [K]", 305.0, 315.0, 310.0, 0.1)

    speed = col1.number_input("Rotational speed [rpm]", 1000.0, 3000.0, 1500.0, 1.0)
    torque = col2.number_input("Torque [Nm]", 20.0, 80.0, 40.0, 0.1)

    tool_wear = st.number_input("Tool wear [min]", 0.0, 300.0, 150.0, 1.0)

    submitted = st.form_submit_button("🔍 Predict Failure")

if submitted:
    if not product_id:
        st.error("❌ Please enter Product ID")
    else:
        payload = {
            "product_id": product_id,
            "type": type_,
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": speed,
            "torque": torque,
            "tool_wear": tool_wear
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()

            if result["prediction"] == 1:
                st.error(f"🚨 FAILURE PREDICTED\n\nProbability: {result['probability']:.2f}")
            else:
                st.success(f"✅ NO FAILURE\n\nProbability: {result['probability']:.2f}")

        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI server is not running")
        except Exception as e:
            st.error(f"Error: {e}")
