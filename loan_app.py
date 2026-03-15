import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(page_title="Loan Eligibility Portal", page_icon="🏦", layout="centered")

# Load models
xgb_model = joblib.load("credit_model.pkl")
kproto_model = joblib.load("kproto_cluster_model.pkl")
eligible_customers = joblib.load("eligible_customers.pkl")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, .stApp {
    background: linear-gradient(160deg, #1a5a9a 0%, #0C447C 40%, #042C53 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 4rem !important; max-width: 480px !important;}

.stTextInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    caret-color: white !important;
    font-size: 15px !important;
    padding: 14px !important;
}
.stTextInput input::placeholder {color: rgba(255,255,255,0.35) !important; -webkit-text-fill-color: rgba(255,255,255,0.35) !important;}
.stTextInput label {color: rgba(255,255,255,0.6) !important; letter-spacing: 2px; font-size: 12px !important; text-transform: uppercase;}

.stButton > button {
    background: white !important;
    color: #042C53 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {opacity: 0.9 !important;}
.stAlert {border-radius: 10px !important;}
</style>

<div style="position:fixed;top:-100px;right:-100px;width:400px;height:400px;border-radius:50%;background:rgba(255,255,255,0.05);pointer-events:none;z-index:0"></div>
<div style="position:fixed;bottom:-60px;left:-60px;width:250px;height:250px;border-radius:50%;background:rgba(255,255,255,0.04);pointer-events:none;z-index:0"></div>

<div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);border-radius:20px;padding:2.5rem 2rem;margin-bottom:1.5rem">
    <p style="font-size:12px;letter-spacing:3px;color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:1.2rem">National Bank &bull; Loan Portal</p>
    <h1 style="font-family:'DM Serif Display',serif;font-size:32px;color:white;line-height:1.2;margin-bottom:0.75rem">Check your loan eligibility</h1>
    <p style="font-size:14px;color:rgba(255,255,255,0.55);line-height:1.6;margin:0">Enter your NIC number to instantly verify your eligibility. Your data is secure and confidential.</p>
</div>
""", unsafe_allow_html=True)

nic = st.text_input("NIC NUMBER", placeholder="e.g. 199012345678")

if st.button("Proceed"):
    if not nic:
        st.error("Please enter your NIC number to proceed.")
    else:
        with st.spinner("Verifying your NIC..."):
            time.sleep(2)

        matched = eligible_customers[eligible_customers['MASKED_LEGAL_ID'] == nic]

        if matched.empty:
            st.error("❌ NIC number not found in our records. Please contact your nearest branch.")
        else:
            customer_record = matched.iloc[0]
            if customer_record['Eligibility_Flag'] == 'REJECT':
                st.error("❌ You are not eligible to apply for a loan at this time.")
            else:
                band = customer_record['Score_Band']
                if band in ["Very Low Risk", "Low Risk"]:
                    st.success("✅ Congratulations! Your loan application can proceed.")
                elif band == "Medium Risk":
                    st.warning("⚠️ Your application is under review. A loan officer will contact you.")
                else:
                    st.error("❌ Unfortunately, your loan application cannot be approved at this time.")

st.markdown('<p style="text-align:center;color:rgba(255,255,255,0.3);font-size:12px;margin-top:1rem">Secured · Confidential · Instant results</p>', unsafe_allow_html=True)