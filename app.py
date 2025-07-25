import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("car_price_model.pkl")
fuel_encoder = joblib.load("fuel_encoder.pkl")
seller_encoder = joblib.load("seller_encoder.pkl")
trans_encoder = joblib.load("trans_encoder.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Prediction App")
st.markdown("Enter your car details to estimate its resale value (in lakhs ₹).")

# User input
present_price = st.number_input("🧾 Present Price (in lakhs ₹)", min_value=0.0, step=0.1)
kms_driven = st.number_input("📍 Kilometers Driven", min_value=0, step=500)
owners = st.selectbox("👤 Number of Previous Owners", [0, 1, 2, 3])
car_age = st.slider("📅 Car Age (Years)", 0, 25, step=1)

fuel_type = st.selectbox("⛽ Fuel Type", fuel_encoder.classes_)
seller_type = st.selectbox("🧑‍💼 Seller Type", seller_encoder.classes_)
transmission = st.selectbox("⚙️ Transmission", trans_encoder.classes_)

# Predict button
if st.button("🔍 Predict Price"):
    # Encode categorical values
    fuel_encoded = fuel_encoder.transform([fuel_type])[0]
    seller_encoded = seller_encoder.transform([seller_type])[0]
    trans_encoded = trans_encoder.transform([transmission])[0]

    # Prepare input for prediction
    input_data = np.array([[present_price, kms_driven, owners, fuel_encoded,
                            seller_encoded, trans_encoded, car_age]])

    # Predict
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:.2f} lakhs")
