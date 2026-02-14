import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Churn Risk Intelligence",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>📊 Customer Churn Risk Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>AI-powered churn probability scoring system</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("📌 Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)",
     "Credit card (automatic)",
     "Electronic check",
     "Mailed check"]
)

# -----------------------------
# BUILD INPUT DATA
# -----------------------------
input_dict = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
}

# One-hot encoding
input_dict["Contract_One year"] = 1 if contract == "One year" else 0
input_dict["Contract_Two year"] = 1 if contract == "Two year" else 0
input_dict["Internet Service_Fiber optic"] = 1 if internet_service == "Fiber optic" else 0
input_dict["Internet Service_No"] = 1 if internet_service == "No" else 0
input_dict["Payment Method_Credit card (automatic)"] = 1 if payment_method == "Credit card (automatic)" else 0
input_dict["Payment Method_Electronic check"] = 1 if payment_method == "Electronic check" else 0
input_dict["Payment Method_Mailed check"] = 1 if payment_method == "Mailed check" else 0

input_df = pd.DataFrame([input_dict])

# Match training columns
train_columns = pd.read_csv("data/processed_data.csv").drop("Churn", axis=1).columns
input_df = input_df.reindex(columns=train_columns, fill_value=0)

input_scaled = scaler.transform(input_df)

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📌 Customer Summary")
    st.metric("Tenure (Months)", tenure)
    st.metric("Monthly Charges", f"${monthly_charges}")
    st.metric("Contract", contract)
    st.metric("Internet", internet_service)

with col2:
    st.subheader("🎯 Risk Assessment")

    if st.button("🔍 Analyze Churn Risk"):

        probability = model.predict_proba(input_scaled)[0][1]

        st.progress(int(probability * 100))

        st.markdown(f"## Churn Risk Score: **{probability:.2%}**")

        # Risk levels
        if probability > 0.75:
            st.error("🚨 HIGH RISK CUSTOMER")
            st.write("Recommended Action: Immediate retention offer, loyalty discount, proactive support call.")
        elif probability > 0.40:
            st.warning("⚠️ MODERATE RISK CUSTOMER")
            st.write("Recommended Action: Send promotional offers and engagement campaigns.")
        else:
            st.success("✅ LOW RISK CUSTOMER")
            st.write("Recommended Action: Maintain engagement and monitor periodically.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by Aditi Soni</p>", unsafe_allow_html=True)