import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models
xgb_model = joblib.load("credit_model.pkl")
kproto_model = joblib.load("kproto_cluster_model.pkl")
eligible_customers = joblib.load("eligible_customers.pkl")

st.title("Loan Approval System")
st.write("Please enter your NIC number to check your loan eligibility.")

nic = st.text_input("NIC Number")

if st.button("Proceed", key="proceed_button"):

    if not nic:
        st.error("Please enter your NIC number to proceed.")
    else:
        # Check if NIC exists in backend
        matched_customer = eligible_customers[eligible_customers['MASKED_LEGAL_ID'] == nic]

        if matched_customer.empty:
            st.error("NIC number not found in our records. Please contact your nearest branch.")
        else:
            customer_record = matched_customer.iloc[0]

            # Check eligibility flag
            if customer_record['Eligibility_Flag'] == 'REJECT':
                st.error("You are not eligible to apply for a loan at this time.")
            else:
                # Run prediction silently using backend data
                score = customer_record['Internal_Bank_Default_Score']
                band = customer_record['Score_Band']

                st.success("Verification successful!")

                # Show result only
                if band == "Very Low Risk" or band == "Low Risk":
                    st.success(f"Congratulations! Your loan application can proceed.")
                elif band == "Medium Risk":
                    st.warning(f"Your application is under review. A loan officer will contact you.")
                else:
                    st.error(f"Unfortunately, your loan application cannot be approved at this time.")