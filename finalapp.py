
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

# Load the trained model
with open("my_model_scaler.pkl", "rb") as file:
    bundle = pickle.load(file)

    model = bundle["model"]
    scaler = bundle["scaler"]
     
# Title for the app
# st.title("Loan Approval")
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Loan Approval</b></h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values
Requested_Loan_Amount = st.slider("Loan Amount", min_value=5000.0, max_value=2500000.0, step=1000.0)
col1, col2 = st.columns([4,1])

with col1:
    loan_amount = st.slider("Loan Amount", 1000.0, 250000.0, 1000.0)

with col2:
    st.write(f"`${loan_amount:,.0f}`")
FICO_score = st.slider("FICO Score", min_value=300, max_value=850, value =700, step=10, key="fico_slider")
Monthly_Gross_Income = st.slider("Monthly Income)", min_value=0.0, max_value=20000.0, step=1000.0)
Monthly_Housing_Payment = st.slider("Housing Payment", min_value=0.0, max_value=50000.0, step=100.0)
# Categorical inputs with options
# --- Categorical inputs with prettier labels (model values unchanged) ---

# Employment Status
status_options = ["full_time", "part_time", "unemployed"]

status_labels = {
    "full_time": "Full Time",
    "part_time": "Part Time",
    "unemployed": "Unemployed"
}

Employment_Status = st.selectbox(
    "Employment Status",
    options=status_options,
    format_func=lambda x: status_labels[x]
)

# Employment Sector
sector_options = [
    "financials", "information_technology", "Mgr", "health_care", "industrials",
    "real_estate", "materials", "utilities", "energy", "consumer_staples",
    "consumer_discretionary", "communication_services", "unknown"
]

sector_labels = {
    "financials": "Financials",
    "information_technology": "Information Technology",
    "Mgr": "Manager",
    "health_care": "Health Care",
    "industrials": "Industrials",
    "real_estate": "Real Estate",
    "materials": "Materials",
    "utilities": "Utilities",
    "energy": "Energy",
    "consumer_staples": "Consumer Staples",
    "consumer_discretionary": "Consumer Discretionary",
    "communication_services": "Communication Services",
    "unknown": "Other"
}

Employment_Sector = st.selectbox(
    "Employment Sector",
    options=sector_options,
    format_func=lambda x: sector_labels[x]
)

# Reason for Loan
reason_options = [
    "Home_Improvement",
    "credit_card_refinancing",
    "major_purchase",
    "cover_an_unexpected_cost",
    "debt_conslidation",
    "other"
]

reason_labels = {
    "Home_Improvement": "Home Improvement",
    "credit_card_refinancing": "Credit Card Refinancing",
    "major_purchase": "Major Purchase",
    "cover_an_unexpected_cost": "Cover an Unexpected Cost",
    "debt_conslidation": "Debt Consolidation",
    "other": "Other"
}

Reason = st.selectbox(
    "Reason for Loan",
    options=reason_options,
    format_func=lambda x: reason_labels[x]
)
#Bankrupt
Ever_Bankrupt_or_Foreclose = st.selectbox( "Ever Bankrupt or Foreclose?", [0,1], format_func=lambda x: "No" if x == 0 else "Yes")

# Lender (already fine)
Lender = st.selectbox("Lender", ["A", "B", "C"])


# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Requested_Loan_Amount": [Requested_Loan_Amount],
    "Monthly_Housing_Payment": [Monthly_Housing_Payment],
    "Monthly_Gross_Income": [Monthly_Gross_Income],
    "FICO_score": [FICO_score],
    "Ever_Bankrupt_or_Foreclose": [Ever_Bankrupt_or_Foreclose],
    "Reason": [Reason],
    "Employment_Sector": [Employment_Sector],
    "Employment_Status": [Employment_Status],
    "Lender": [Lender],
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=['Lender', 'Employment_Sector', 'Employment_Status', 'Reason'])

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model_columns = [
    'Requested_Loan_Amount',
    'FICO_score',
    'Monthly_Gross_Income',
    'Monthly_Housing_Payment',
    'Ever_Bankrupt_or_Foreclose',
    'fico_income',
    'income_housing_ratio',
    'request_to_income',
    'Reason_credit_card_refinancing',
    'Reason_debt_conslidation',
    'Reason_home_improvement',
    'Reason_major_purchase',
    'Reason_other',
    'Employment_Status_part_time',
    'Employment_Status_unemployed',
    'Employment_Sector_consumer_discretionary',
    'Employment_Sector_consumer_staples',
    'Employment_Sector_energy',
    'Employment_Sector_financials',
    'Employment_Sector_health_care',
    'Employment_Sector_industrials',
    'Employment_Sector_information_technology',
    'Employment_Sector_materials',
    'Employment_Sector_real_estate',
    'Employment_Sector_unknown',
    'Employment_Sector_utilities',
    'Lender_B',
    'Lender_C'
]
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan"):

    
    # 1. Scale using the SAME scaler as training, on ALL features
    input_scaled = scaler.transform(input_data_encoded)

    # 2. Predict using the loaded model
    prediction = model.predict(input_scaled)[0]


    # Display result
    if prediction == 1:
        st.write("The prediction is: **Approved** ✅")
    else:
        st.write("The prediction is: **Denied** 🚫")

