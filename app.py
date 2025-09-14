import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_rf_model.pkl")

st.set_page_config(page_title="Customer Satisfaction Prediction", layout="wide")

# Title
st.title("📊 Customer Satisfaction Prediction Dashboard")

st.markdown("This app predicts whether a **customer is satisfied or not** based on support ticket details.")

# Sidebar inputs
st.sidebar.header("Enter Ticket Details")

age = st.sidebar.number_input("Customer Age", 18, 100, 30)
gender = st.sidebar.selectbox("Customer Gender", ["Male", "Female", "Other"])
priority = st.sidebar.selectbox("Ticket Priority", ["Low", "Medium", "High", "Critical"])
channel = st.sidebar.selectbox("Ticket Channel", ["Email", "Chat", "Phone", "Social Media"])
ticket_type = st.sidebar.selectbox("Ticket Type", ["Technical issue", "Billing inquiry", "Refund request", "Product inquiry", "Cancellation request"])

# Convert inputs into DataFrame (dummy encoding for demo)
input_data = pd.DataFrame({
    "Customer Age": [age],
    "Customer Gender": [gender],
    "Ticket Priority": [priority],
    "Ticket Channel": [channel],
    "Ticket Type": [ticket_type]
})

# In practice, apply same preprocessing (LabelEncoder, scaling) as training
# Here, we assume model handles encoded features
if st.sidebar.button("Predict"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.success("✅ Prediction: Customer is **Satisfied** 😀")
    else:
        st.error("❌ Prediction: Customer is **Not Satisfied** 😟")

st.markdown("---")

# Show feature importance chart (from training)
st.subheader("🔍 Model Insights")
st.image("feature_importances.png", caption="Top Features Influencing Satisfaction")
