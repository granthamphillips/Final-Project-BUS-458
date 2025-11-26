
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

# Load the trained model
with open("my_model_scaler.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
# st.title("Loan Approval")
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Home Equity Loan Approval</b></h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values
Requested_Loan_Amount = st.slider("Loan Amount (Requested_Loan_Amount)", min_value=1000, max_value=500000, step=1000)
Monthly_Housing_Payment = st.slider("Housing Payment", min_value=0.0, max_value=1000000.0, step=1000.0)
Monthly_Gross_Income = st.slider("Monthly Income)", min_value=0.0, max_value=1000000.0, step=1000.0)
FICO_score = st.slider("FICO Score", min_value=300, max_value=850, value =700, step=10, key="fico_slider")
Ever_Bankrupt_or_Foreclose = st.selectbox("Ever Bankrupt or Foreclose)", options=list(range(0, 1)))  # Options from 0 to 1


# Categorical inputs with options
Reason = st.selectbox("Reason for Loan (REASON)", ["Home_Improvement", "Credit_card_refinancing", "major_purchase", "cover_an_unexpected_cost", "debt_consolidation", "other"])
Employment_Sector = st.selectbox("Employment Sector", ["financials", "information_technology", "Mgr", "health_care", "industrials", "real_estate", "materials", "utilities", "energy", "consumer_staples", "communication_services", "communication_services", "Unknown"])
Employment_Status = st.selectbox("Employment Status", ["full_time", "part_time", "unemployed"])
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
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]

    # Display result
    if prediction == 1:
        st.write("The prediction is: **Approved** ✅")
    else:
        st.write("The prediction is: **Denied** 🚫")



        """
What happens if the user enters a value not in the training data?

Example: User enters REASON = 'Vacation', but the model only knows 'DebtCon' and 'HomeImp'.

1. pd.get_dummies creates a new column: REASON_Vacation = 1.
2. The code then adds the *known* columns: REASON_DebtCon = 0 and REASON_HomeImp = 0.
3. The final filtering step *drops* the unknown REASON_Vacation column because it's not in the
   model's expected feature list.

Result: The model receives REASON_DebtCon = 0 and REASON_HomeImp = 0, which correctly
treats the unknown 'Vacation' input as "none of the known categories" (i.e., "Other").
"""
