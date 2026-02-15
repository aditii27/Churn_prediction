import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
model = joblib.load("Models/best_model.pkl")
scaler = joblib.load("Models/scaler.pkl")

# Load training columns
train_columns = pd.read_csv("Data/processed_data.csv").drop("Churn", axis=1).columns

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>📊 Customer Churn Risk Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Predict customer churn probability using behavioral and service features</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("📌 Customer Profile")

# Numeric Inputs
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=840.0)

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

senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])

# -----------------------------
# BUILD INPUT DATAFRAME
# -----------------------------
input_df = pd.DataFrame(columns=train_columns)
input_df.loc[0] = 0

# Numeric Features
if "Tenure Months" in train_columns:
    input_df.at[0, "Tenure Months"] = tenure

if "Monthly Charges" in train_columns:
    input_df.at[0, "Monthly Charges"] = monthly_charges

if "Total Charges" in train_columns:
    input_df.at[0, "Total Charges"] = total_charges

# Contract Encoding
if contract == "One year" and "Contract_One year" in train_columns:
    input_df.at[0, "Contract_One year"] = 1

if contract == "Two year" and "Contract_Two year" in train_columns:
    input_df.at[0, "Contract_Two year"] = 1

# Internet Encoding
if internet_service == "Fiber optic" and "Internet Service_Fiber optic" in train_columns:
    input_df.at[0, "Internet Service_Fiber optic"] = 1

if internet_service == "No" and "Internet Service_No" in train_columns:
    input_df.at[0, "Internet Service_No"] = 1

# Payment Encoding
if payment_method == "Electronic check" and "Payment Method_Electronic check" in train_columns:
    input_df.at[0, "Payment Method_Electronic check"] = 1

if payment_method == "Mailed check" and "Payment Method_Mailed check" in train_columns:
    input_df.at[0, "Payment Method_Mailed check"] = 1

if payment_method == "Credit card (automatic)" and "Payment Method_Credit card (automatic)" in train_columns:
    input_df.at[0, "Payment Method_Credit card (automatic)"] = 1

# Other Features
if senior == "Yes" and "Senior Citizen_Yes" in train_columns:
    input_df.at[0, "Senior Citizen_Yes"] = 1

if dependents == "Yes" and "Dependents_Yes" in train_columns:
    input_df.at[0, "Dependents_Yes"] = 1

if paperless == "Yes" and "Paperless Billing_Yes" in train_columns:
    input_df.at[0, "Paperless Billing_Yes"] = 1

if tech_support == "Yes" and "Tech Support_Yes" in train_columns:
    input_df.at[0, "Tech Support_Yes"] = 1

if online_security == "Yes" and "Online Security_Yes" in train_columns:
    input_df.at[0, "Online Security_Yes"] = 1

# Scale
input_scaled = scaler.transform(input_df)

# -----------------------------
# MAIN DISPLAY
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📌 Customer Summary")
    st.write(f"Tenure: {tenure} months")
    st.write(f"Monthly Charges: ${monthly_charges}")
    st.write(f"Contract: {contract}")
    st.write(f"Internet: {internet_service}")
    st.write(f"Payment: {payment_method}")

with col2:
    st.subheader("🎯 Risk Assessment")

    if st.button("🔍 Analyze Churn Risk"):

        probability = model.predict_proba(input_scaled)[0][1]

        st.progress(int(probability * 100))
        st.markdown(f"## Churn Risk Score: **{probability:.2%}**")

        # Realistic thresholds
        if probability > 0.65:
            st.error("🚨 HIGH RISK CUSTOMER")
            st.write("Recommended: Immediate retention strategy.")
        elif probability > 0.40:
            st.warning("⚠️ MODERATE RISK CUSTOMER")
            st.write("Recommended: Engagement campaign.")
        else:
            st.success("✅ LOW RISK CUSTOMER")
            st.write("Customer likely to stay.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed by Aditi Soni</p>", unsafe_allow_html=True)