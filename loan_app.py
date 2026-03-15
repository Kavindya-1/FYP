import sys
import streamlit as st
st.write(sys.path)
st.write(sys.version)

try:
    import joblib
    st.success("joblib imported successfully")
except ImportError as e:
    st.error(f"joblib import failed: {e}")

st.stop()



# app.py
import streamlit as st
import joblib
import pandas as pd

# Load models
xgb_model = joblib.load("credit_model.pkl")
kproto_model = joblib.load("kproto_cluster_model.pkl")
eligible_customers = joblib.load("eligible_customers.pkl")

st.title("Loan Approval")
st.header("Enter your details")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
avg_monthly_credit = st.number_input("Average Salary", min_value=0.0, value=500.0)
gender = st.selectbox("Gender", ["MALE", "FEMALE", "UNKNOWN"])
occupation = st.text_input("Occupation")
marital_status = st.selectbox("Marital Status", ["SINGLE", "MARRIED", "UNKNOWN"])
employment_status = st.selectbox("Employment Status", ["Employed", "UNEMPLOYED", "Self-employed"])
nic = st.text_input("Enter your NIC")  # NIC for validation

# Combine inputs into DataFrame
input_df = pd.DataFrame([{
    "AGE": age,
    "Avg_Monthly_Credit": avg_monthly_credit,
    "GENDER": gender,
    "OCCUPATION": occupation,
    "MARITAL_STATUS": marital_status,
    "EMPLOYMENT_STATUS": employment_status
}])

# Single "Proceed" button with a unique key
if st.button("Proceed", key="proceed_button"):

    if not nic:
        st.error("Please enter your NIC to validate.")
    else:
        # Step 1: Check if NIC exists in eligible customers
        matched_customer = eligible_customers[eligible_customers['MASKED_LEGAL_ID'] == nic]

        if matched_customer.empty:
            st.error("NIC not found in our records. Cannot proceed.")
        else:
            # Step 2: Validate input data matches backend record
            customer_record = matched_customer.iloc[0]
            errors = []

            if age != customer_record['AGE']:
                errors.append("Age does not match our records.")
            if avg_monthly_credit != customer_record['Avg_Monthly_Credit']:
                errors.append("Average salary does not match our records.")
            if gender != customer_record['GENDER']:
                errors.append("Gender does not match our records.")
            if occupation.upper() != customer_record['OCCUPATION']:
                errors.append("Occupation does not match our records.")
            if marital_status != customer_record['MARITAL_STATUS']:
                errors.append("Marital status does not match our records.")
            if employment_status != customer_record['EMPLOYMENT_STATUS']:
                errors.append("Employment status does not match our records.")

            # Step 3: Show errors or proceed with prediction
            if errors:
                for e in errors:
                    st.error(e)
                st.warning("Please correct the inputs to match your record.")
            else:
                st.success("Validation successful! Proceeding with prediction...")

                # Encode categorical columns as in training
                categorical_cols = ["GENDER", "OCCUPATION", "MARITAL_STATUS", "EMPLOYMENT_STATUS"]
                for col in categorical_cols:
                    input_df[col] = input_df[col].astype(str).cat.codes

                # Credit risk prediction
                risk_prob = xgb_model.predict_proba(input_df)[:, 1][0]
                risk_pred = xgb_model.predict(input_df)[0]
                st.write(f"Predicted Default Probability: {risk_prob:.2f}")
                st.write(f"Predicted Default (0=No, 1=Yes): {risk_pred}")

                # Clustering
                cat_idx = [input_df.columns.get_loc(col) for col in categorical_cols]
                cluster_label = kproto_model.predict(input_df.values, categorical=cat_idx)[0]
                st.write(f"Cluster: {cluster_label}")