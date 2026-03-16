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

.stTextInput input, .stNumberInput input {
    background: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    color: #042C53 !important;
    -webkit-text-fill-color: #042C53 !important;
    caret-color: #042C53 !important;
    font-size: 15px !important;
    padding: 14px !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder {
    color: rgba(4,44,83,0.4) !important;
    -webkit-text-fill-color: rgba(4,44,83,0.4) !important;
}
.stTextInput label, .stNumberInput label {
    color: rgba(255,255,255,0.6) !important;
    letter-spacing: 2px;
    font-size: 12px !important;
    text-transform: uppercase;
}

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
.stAlert,
div[data-testid="stAlert"],
div[data-testid="stAlert"] > div,
.element-container div[data-testid="stAlert"] {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 100% !important;
    border-radius: 10px !important;
    box-sizing: border-box !important;
    display: block !important;
    float: none !important;
}
.element-container {
    width: 100% !important;
}

.step-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.verified-card {
    background: rgba(255,255,255,0.07);
    border: 0.5px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.verified-label {
    font-size: 11px;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    margin-bottom: 4px;
}

.verified-value {
    font-size: 15px;
    color: white;
    font-weight: 500;
}

.stSelectbox label {
    color: rgba(255,255,255,0.6) !important;
    letter-spacing: 2px;
    font-size: 12px !important;
    text-transform: uppercase;
}
.stSelectbox > div > div {
    background: white !important;
    border-radius: 10px !important;
    color: #042C53 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    font-size: 15px !important;
}
</style>

<div style="position:fixed;top:-100px;right:-100px;width:400px;height:400px;border-radius:50%;
background:rgba(255,255,255,0.05);pointer-events:none;z-index:0"></div>
<div style="position:fixed;bottom:-60px;left:-60px;width:250px;height:250px;border-radius:50%;
background:rgba(255,255,255,0.04);pointer-events:none;z-index:0"></div>
""", unsafe_allow_html=True)

# ── Session state initialisation ───────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1
if "customer_record" not in st.session_state:
    st.session_state.customer_record = None
if "nic_value" not in st.session_state:
    st.session_state.nic_value = ""
if "step2_error" not in st.session_state:
    st.session_state.step2_error = ""
if "step4_error" not in st.session_state:
    st.session_state.step4_error = ""
if "loan_amount" not in st.session_state:
    st.session_state.loan_amount = None
if "loan_product" not in st.session_state:
    st.session_state.loan_product = ""

# ══════════════════════════════════════════════════════════════
# STEP 1 — NIC Entry
# ══════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2.5rem 2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 1 of 4</div>
        <h1 style="font-family:'DM Serif Display',serif;font-size:32px;color:white;
        line-height:1.2;margin-bottom:0.75rem">Check your loan eligibility</h1>
        <p style="font-size:14px;color:rgba(255,255,255,0.55);line-height:1.6;margin:0">
        Enter your NIC number to instantly verify your eligibility.
        Your data is secure and confidential.</p>
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
                    st.session_state.customer_record = customer_record
                    st.session_state.nic_value = nic
                    st.session_state.step = 2
                    st.rerun()

# ══════════════════════════════════════════════════════════════
# STEP 2 — Age & Salary Verification
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    record = st.session_state.customer_record

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 2 of 4</div>
        <h1 style="font-family:'DM Serif Display',serif;font-size:28px;color:white;
        line-height:1.2;margin-bottom:0.5rem">Verify your details</h1>
        <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:0">
        Please confirm your personal details to continue.</p>
    </div>

    <div class="verified-card">
        <div class="verified-label">NIC Verified</div>
        <div class="verified-value">✅ &nbsp;{st.session_state.nic_value}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    age_input = st.number_input(
        "YOUR AGE",
        min_value=1,
        max_value=120,
        step=1,
        value=None,
        placeholder="Enter your age"
    )

    salary_input = st.number_input(
        "AVERAGE MONTHLY SALARY (LKR)",
        min_value=0.0,
        step=1000.0,
        format="%.2f",
        value=None,
        placeholder="e.g. 75000.00"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.step2_error = ""
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Verify & Continue"):
            st.session_state.step2_error = ""
            if age_input is None or salary_input is None:
                st.session_state.step2_error = "fill"
            else:
                stored_age = int(record['AGE'])
                if int(age_input) != stored_age:
                    st.session_state.step2_error = "age"
                else:
                    stored_salary = float(record['Avg_Monthly_Credit'])
                    tolerance     = 0.20
                    lower         = stored_salary * (1 - tolerance)
                    upper         = stored_salary * (1 + tolerance)
                    if not (lower <= float(salary_input) <= upper):
                        st.session_state.step2_error = "salary"
                    else:
                        st.session_state.step2_error = ""
                        st.session_state.step = 3
                        st.rerun()

    # Render error OUTSIDE columns — full width
    if st.session_state.step2_error == "fill":
        st.error("Please fill in both your age and monthly salary.")
    elif st.session_state.step2_error == "age":
        st.error("❌ The age you entered does not match our records. Please ensure you enter your correct age.")
    elif st.session_state.step2_error == "salary":
        st.error("❌ The salary you entered could not be verified against our records. Please ensure it reflects your true average monthly income.")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Eligibility Result + Proceed to Loan Details
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    record = st.session_state.customer_record
    band   = record['Score_Band']

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 3 of 4</div>
        <h1 style="font-family:'DM Serif Display',serif;font-size:28px;color:white;
        line-height:1.2;margin-bottom:0.5rem">Eligibility result</h1>
        <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:0">
        Based on your verified profile.</p>
    </div>

    <div class="verified-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <div class="verified-label">NIC</div>
                <div class="verified-value">{st.session_state.nic_value}</div>
            </div>
            <div style="text-align:right">
                <div class="verified-label">Risk Band</div>
                <div class="verified-value">{band}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if band in ["Very Low Risk", "Low Risk"]:
        st.success("✅ You are eligible! Please proceed to select your loan product.")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Start Over"):
                st.session_state.step = 1
                st.session_state.customer_record = None
                st.session_state.nic_value = ""
                st.rerun()
        with col2:
            if st.button("Continue →"):
                st.session_state.step = 4
                st.rerun()
    elif band == "Medium Risk":
        st.warning("⚠️ Your application is under review. A loan officer will contact you shortly.")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("← Start Over"):
            st.session_state.step = 1
            st.session_state.customer_record = None
            st.session_state.nic_value = ""
            st.rerun()
    else:
        st.error("❌ Unfortunately, your loan application cannot be approved at this time.")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("← Start Over"):
            st.session_state.step = 1
            st.session_state.customer_record = None
            st.session_state.nic_value = ""
            st.rerun()

# ══════════════════════════════════════════════════════════════
# STEP 4 — Loan Product & Amount
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 4:

    LOAN_PRODUCTS = {
        "🎓  Personal Education Loan": "Fund your own tuition, professional certifications, or short courses to advance your career.",
        "🏥  Personal Medical Loan": "Cover unexpected medical bills, surgeries, or treatments for yourself or an immediate family member.",
        "✈️  Personal Travel Loan": "Finance a dream holiday, family trip, or religious pilgrimage with easy monthly repayments.",
        "💍  Personal Wedding Loan": "Fund wedding expenses including venue, catering, and arrangements without straining your savings.",
        "🛋️  Personal Home Improvement Loan": "Renovate, furnish, or upgrade your home with a flexible personal loan.",
    }

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 4 of 4</div>
        <h1 style="font-family:'DM Serif Display',serif;font-size:28px;color:white;
        line-height:1.2;margin-bottom:0.5rem">Loan details</h1>
        <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:0">
        Select a loan product and enter the amount you require.</p>
    </div>

    <div class="verified-card">
        <div class="verified-label">NIC Verified</div>
        <div class="verified-value">✅ &nbsp;{st.session_state.nic_value}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    loan_product = st.selectbox(
        "LOAN PRODUCT",
        options=["— Select a loan product —"] + list(LOAN_PRODUCTS.keys()),
    )

    # Show description card for selected product
    if loan_product and loan_product != "— Select a loan product —":
        desc = LOAN_PRODUCTS[loan_product]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-left:3px solid rgba(255,255,255,0.3);
        border-radius:8px;padding:0.9rem 1.1rem;margin:0.5rem 0 1rem 0">
            <p style="font-size:13px;color:rgba(255,255,255,0.65);margin:0;line-height:1.6">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    loan_amount = st.number_input(
        "REQUIRED LOAN AMOUNT (LKR)",
        min_value=0.0,
        step=10000.0,
        format="%.2f",
        value=None,
        placeholder="e.g. 500000.00"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.step4_error = ""
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Submit Application"):
            if loan_product == "— Select a loan product —" or not loan_product:
                st.session_state.step4_error = "product"
            elif loan_amount is None or loan_amount <= 0:
                st.session_state.step4_error = "amount"
            else:
                st.session_state.loan_product = loan_product
                st.session_state.loan_amount  = loan_amount
                st.session_state.step4_error  = ""
                st.session_state.step = 5
                st.rerun()

    if st.session_state.step4_error == "product":
        st.error("❌ Please select a loan product to continue.")
    elif st.session_state.step4_error == "amount":
        st.error("❌ Please enter a valid loan amount to continue.")

# ══════════════════════════════════════════════════════════════
# STEP 5 — Final Confirmation Summary
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    record = st.session_state.customer_record
    band   = record['Score_Band']

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2rem;margin-bottom:1.5rem">
        <h1 style="font-family:'DM Serif Display',serif;font-size:28px;color:white;
        line-height:1.2;margin-bottom:0.5rem">Application submitted</h1>
        <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:0">
        Here is a summary of your loan request.</p>
    </div>

    <div class="verified-card">
        <div style="margin-bottom:1rem">
            <div class="verified-label">NIC</div>
            <div class="verified-value">{st.session_state.nic_value}</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Loan Product</div>
            <div class="verified-value">{st.session_state.loan_product}</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Requested Amount</div>
            <div class="verified-value">LKR {st.session_state.loan_amount:,.2f}</div>
        </div>
        <div>
            <div class="verified-label">Risk Band</div>
            <div class="verified-value">{band}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Your application has been received. A loan officer will be in touch within 2 business days.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("← Start Over"):
        for key in ["step","customer_record","nic_value","step2_error",
                    "step4_error","loan_amount","loan_product"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.markdown(
    '<p style="text-align:center;color:rgba(255,255,255,0.3);font-size:12px;margin-top:1rem">'
    'Secured · Confidential · Instant results</p>',
    unsafe_allow_html=True
)