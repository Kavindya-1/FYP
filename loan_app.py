import math
import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(page_title="Loan Eligibility Portal", page_icon="🏦", layout="centered")

# Load models
xgb_model          = joblib.load("credit_model.pkl")
kproto_model       = joblib.load("kproto_cluster_model.pkl")
eligible_customers = joblib.load("eligible_customers.pkl")

# ══════════════════════════════════════════════════════════════
# DECISION ENGINE
# ══════════════════════════════════════════════════════════════

AGE_CEILINGS = {
    "Young Adult":   500_000,
    "Adult":       2_000_000,
    "Middle-Aged": 1_500_000,
    "Senior":        750_000,
}

BAND_CONFIG = {
    "Very Low Risk": {"base_multiplier": 36, "band_floor": 750, "band_ceiling": 850},
    "Low Risk":      {"base_multiplier": 24, "band_floor": 650, "band_ceiling": 750},
    "Medium Risk":   {"base_multiplier": 12, "band_floor": 550, "band_ceiling": 650},
    "High Risk":     {"base_multiplier":  0, "band_floor": 300, "band_ceiling": 550},
}

EMPLOYMENT_FACTORS = {
    "Core Working Group": 1.0,
    "Special Segment":    0.8,
    "Other":              0.6,
    "Not valid segment":  0.6,
}

def floor_to_nearest_1000(value):
    """Always round DOWN to the nearest 1,000 to ensure affordability."""
    return math.floor(value / 1000) * 1000

def compute_max_eligible(record):
    band          = str(record.get("Score_Band", "High Risk"))
    score         = float(record.get("Internal_Bank_Default_Score", 300))
    salary        = float(record.get("Avg_Monthly_Credit", 0))
    age_bucket    = str(record.get("Age_Bucket", "Adult"))
    emp_segment   = str(record.get("Employment_Segment", "Other"))
    max_ood       = float(record.get("MAX_OOD", 0))
    capital_due   = float(record.get("TOTAL_CAPITAL_DUE", 0))
    net_ratio     = float(record.get("NET_RATIO", 0))

    cfg           = BAND_CONFIG.get(band, BAND_CONFIG["High Risk"])
    base_mult     = cfg["base_multiplier"]
    band_floor    = cfg["band_floor"]
    band_ceiling  = cfg["band_ceiling"]

    # 1 — Score ratio within band (0.0 → 1.0)
    score_ratio   = (score - band_floor) / max(band_ceiling - band_floor, 1)
    score_ratio   = max(0.0, min(1.0, score_ratio))

    # 2 — Adjusted multiplier
    adj_mult      = base_mult * (0.85 + 0.30 * score_ratio)

    # 3 — Max by salary
    max_by_salary = salary * adj_mult

    # 4 — Age ceiling
    age_ceiling   = AGE_CEILINGS.get(age_bucket, 1_000_000)

    # 5 — Raw max before adjustments
    raw_max       = min(max_by_salary, age_ceiling)

    # 6 — Employment factor
    emp_factor    = EMPLOYMENT_FACTORS.get(emp_segment, 0.6)

    # 7 — OOD penalty
    if max_ood >= 60:
        ood_penalty       = -1
        ood_penalty_label = "Rejected (≥60 days overdue)"
    elif max_ood >= 30:
        ood_penalty       = 0.70
        ood_penalty_label = "×0.70 (30–59 days overdue)"
    else:
        ood_penalty       = 1.0
        ood_penalty_label = "No penalty"

    # 8 — Net ratio penalty
    if net_ratio < 0:
        net_penalty       = 0.80
        net_penalty_label = "×0.80 (spending > income)"
    else:
        net_penalty       = 1.0
        net_penalty_label = "No penalty"

    # 9 — Apply all factors
    if ood_penalty == -1:
        max_eligible = 0
    else:
        max_eligible = (raw_max * emp_factor * ood_penalty * net_penalty) - max(capital_due, 0)
        max_eligible = max(0, max_eligible)

    return {
        "band":               band,
        "score":              score,
        "score_ratio":        round(score_ratio, 3),
        "base_multiplier":    base_mult,
        "adj_multiplier":     round(adj_mult, 2),
        "salary":             salary,
        "max_by_salary":      round(max_by_salary, 2),
        "age_ceiling":        age_ceiling,
        "age_bucket":         age_bucket,
        "raw_max":            round(raw_max, 2),
        "emp_segment":        emp_segment,
        "emp_factor":         emp_factor,
        "max_ood":            max_ood,
        "ood_penalty":        ood_penalty,
        "ood_penalty_label":  ood_penalty_label,
        "net_ratio":          round(net_ratio, 3),
        "net_penalty":        net_penalty,
        "net_penalty_label":  net_penalty_label,
        "capital_due":        capital_due,
        "max_eligible":       round(max_eligible, 2),
        "hard_ood_reject":    ood_penalty == -1,
    }

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
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
    letter-spacing: 2px; font-size: 12px !important; text-transform: uppercase;
}
.stButton > button {
    background: white !important; color: #042C53 !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 15px !important;
    padding: 14px !important; width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {opacity: 0.9 !important;}
.stAlert, div[data-testid="stAlert"],
div[data-testid="stAlert"] > div,
.element-container div[data-testid="stAlert"] {
    width: 100% !important; max-width: 100% !important;
    min-width: 100% !important; border-radius: 10px !important;
    box-sizing: border-box !important; display: block !important; float: none !important;
}
.element-container { width: 100% !important; }
.step-badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 20px;
    padding: 4px 14px; font-size: 11px; letter-spacing: 2px;
    color: rgba(255,255,255,0.6); text-transform: uppercase; margin-bottom: 1rem;
}
.verified-card {
    background: rgba(255,255,255,0.07); border: 0.5px solid rgba(255,255,255,0.15);
    border-radius: 20px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.verified-label {
    font-size: 11px; letter-spacing: 2px; color: rgba(255,255,255,0.45);
    text-transform: uppercase; margin-bottom: 4px;
}
.verified-value { font-size: 15px; color: white; font-weight: 500; }
.breakdown-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 0.5px solid rgba(255,255,255,0.08);
    font-size: 13px;
}
.breakdown-row:last-child { border-bottom: none; }
.breakdown-key { color: rgba(255,255,255,0.55); }
.breakdown-val { color: white; font-weight: 500; text-align: right; }
.stSelectbox label {
    color: rgba(255,255,255,0.6) !important; letter-spacing: 2px;
    font-size: 12px !important; text-transform: uppercase;
}
.stSelectbox > div > div {
    background: white !important; border-radius: 10px !important;
    color: #042C53 !important; border: 1px solid rgba(255,255,255,0.2) !important;
    font-size: 15px !important;
}
</style>

<div style="position:fixed;top:-100px;right:-100px;width:400px;height:400px;border-radius:50%;
background:rgba(255,255,255,0.05);pointer-events:none;z-index:0"></div>
<div style="position:fixed;bottom:-60px;left:-60px;width:250px;height:250px;border-radius:50%;
background:rgba(255,255,255,0.04);pointer-events:none;z-index:0"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
defaults = {
    "step": 1, "customer_record": None, "nic_value": "",
    "step2_error": "", "step4_error": "",
    "loan_amount": None, "loan_product": "",
    "decision": None,
    "loan_term":     None,
    "loan_rate":     None,
    "loan_emi":      None,
    "emi_details":   {},
    "loan_total_int":  None,
    "loan_total_pay":  None,
    "suggested_amount": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def start_over():
    for k in defaults:
        st.session_state[k] = defaults[k]
    for k in ["typed_loan_amount", "emi_details"]:
        if k in st.session_state:
            del st.session_state[k]

def fmt(n):
    return f"LKR {n:,.0f}"

# ══════════════════════════════════════════════════════════════
# STEP 1 — NIC Entry
# ══════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2.5rem 2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 1 of 5</div>
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
                rec = matched.iloc[0]
                if rec['Eligibility_Flag'] == 'REJECT':
                    st.error("❌ You are not eligible to apply for a loan at this time.")
                elif int(rec.get('Number_of_Active_Accounts', 0)) == 0:
                    st.error("❌ Sorry, we could not find any active accounts linked to your NIC. Please visit your nearest branch for assistance.")
                else:
                    st.session_state.customer_record = rec
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
        <div class="step-badge">Step 2 of 5</div>
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

    age_input    = st.number_input("YOUR AGE", min_value=1, max_value=120,
                                    step=1, value=None, placeholder="Enter your age")
    salary_input = st.number_input("AVERAGE MONTHLY SALARY (LKR)", min_value=0.0,
                                    step=1000.0, format="%.2f", value=None,
                                    placeholder="e.g. 75000.00")

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
                if int(age_input) != int(record['AGE']):
                    st.session_state.step2_error = "age"
                else:
                    stored_salary = float(record['Avg_Monthly_Credit'])
                    lower = stored_salary * 0.80
                    upper = stored_salary * 1.20
                    if not (lower <= float(salary_input) <= upper):
                        st.session_state.step2_error = "salary"
                    else:
                        st.session_state.step2_error = ""
                        st.session_state.step = 3
                        st.rerun()

    if st.session_state.step2_error == "fill":
        st.error("Please fill in both your age and monthly salary.")
    elif st.session_state.step2_error == "age":
        st.error("❌ The age you entered does not match our records. Please ensure you enter your correct age.")
    elif st.session_state.step2_error == "salary":
        st.error("❌ The salary you entered could not be verified against our records. Please ensure it reflects your true average monthly income.")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Eligibility Check (Score Band + Decision Engine)
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    record  = st.session_state.customer_record
    band    = str(record['Score_Band'])
    dec     = compute_max_eligible(record)
    st.session_state.decision = dec

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 3 of 5</div>
        <h1 style="font-family:'DM Serif Display',serif;font-size:28px;color:white;
        line-height:1.2;margin-bottom:0.5rem">Eligibility result</h1>
        <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:0">
        Based on your verified profile.</p>
    </div>
    <div class="verified-card">
        <div class="verified-label">NIC</div>
        <div class="verified-value">{st.session_state.nic_value}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hard reject: High Risk or OOD ≥ 60
    if band == "High Risk" or dec["hard_ood_reject"]:
        st.error("❌ Unfortunately, your loan application cannot be approved at this time.")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("← Start Over"):
            start_over(); st.rerun()

    # ── Medium Risk: suggest conservative amount
    elif band == "Medium Risk":
        # FIX: floor to nearest 1,000 (never round up) to guarantee amount stays within limits
        suggested = floor_to_nearest_1000(dec["max_eligible"] * 0.70)
        suggested = max(suggested, 0)
        st.session_state.suggested_amount = suggested
        st.warning("⚠️ Based on your profile, you may qualify for a limited loan amount.")
        st.markdown(f"""
        <div class="verified-card" style="margin-top:1rem">
            <div class="verified-label">Suggested loan amount</div>
            <div class="verified-value" style="font-size:22px">{fmt(suggested)}</div>
            <div style="font-size:12px;color:rgba(255,255,255,0.45);margin-top:6px">
            A loan officer will review and confirm your final offer.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("No thanks"):
                start_over(); st.rerun()
        with col2:
            if st.button("Proceed →"):
                st.session_state.loan_amount = suggested
                st.session_state.step = 4
                st.rerun()

    # ── Low / Very Low Risk: full eligibility — show breakdown then proceed
    else:
        st.success("✅ You are eligible! Please proceed to select your loan product.")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── NEW: Eligibility amount breakdown card ──
        st.markdown(f"""
        <div class="verified-card" style="margin-top:0.5rem">
            <div class="verified-label">Maximum loan you are eligible for</div>
            <div class="verified-value" style="font-size:28px;color:rgba(100,220,130,0.95);
            font-weight:700;margin-bottom:4px">{fmt(dec['max_eligible'])}</div>
            <div style="font-size:12px;color:rgba(255,255,255,0.4);margin-bottom:14px">
            Based on your credit profile — you may request up to this amount in the next step.</div>
            <div style="border-top:0.5px solid rgba(255,255,255,0.1);padding-top:12px">
                <div class="breakdown-row">
                    <span class="breakdown-key">Credit score</span>
                    <span class="breakdown-val">{int(dec['score'])} &nbsp;·&nbsp; {dec['band']}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Adjusted multiplier</span>
                    <span class="breakdown-val">{dec['adj_multiplier']}×</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Monthly income used</span>
                    <span class="breakdown-val">{fmt(dec['salary'])}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Max by salary</span>
                    <span class="breakdown-val">{fmt(dec['max_by_salary'])}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Age group cap</span>
                    <span class="breakdown-val">{dec['age_bucket']} &nbsp;·&nbsp; {fmt(dec['age_ceiling'])}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Employment factor</span>
                    <span class="breakdown-val">{dec['emp_segment']} &nbsp;·&nbsp; ×{dec['emp_factor']}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Overdue penalty</span>
                    <span class="breakdown-val">{dec['ood_penalty_label']}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Spending penalty</span>
                    <span class="breakdown-val">{dec['net_penalty_label']}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Existing debt deducted</span>
                    <span class="breakdown-val">{fmt(dec['capital_due'])}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Start Over"):
                start_over(); st.rerun()
        with col2:
            if st.button("Continue →"):
                st.session_state.step = 4
                st.rerun()

# ══════════════════════════════════════════════════════════════
# STEP 4 — Loan Product & Amount
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    record = st.session_state.customer_record
    dec    = st.session_state.decision
    band   = str(record['Score_Band'])
    salary = float(record.get('Avg_Monthly_Credit', 0))

    LOAN_PRODUCTS = {
        "🎓  Personal Education Loan — 12% p.a.":        ("Fund your own tuition, professional certifications, or short courses to advance your career.",  12.0),
        "🏥  Personal Medical Loan — 10% p.a.":          ("Cover unexpected medical bills, surgeries, or treatments for yourself or an immediate family member.", 10.0),
        "✈️  Personal Travel Loan — 15% p.a.":           ("Finance a dream holiday, family trip, or religious pilgrimage with easy monthly repayments.",   15.0),
        "💍  Personal Wedding Loan — 13% p.a.":          ("Fund wedding expenses including venue, catering, and arrangements without straining your savings.", 13.0),
        "🛋️  Personal Home Improvement Loan — 11% p.a.": ("Renovate, furnish, or upgrade your home with a flexible personal loan.",                         11.0),
    }
    TERM_OPTIONS = [12, 24, 36, 48, 60]

    def calc_emi(principal, annual_rate_pct, months):
        r = (annual_rate_pct / 100) / 12
        if r == 0:
            return principal / months
        return principal * r * (1 + r)**months / ((1 + r)**months - 1)

    def max_affordable_principal(annual_rate_pct, months, max_emi):
        r = (annual_rate_pct / 100) / 12
        if r == 0:
            return max_emi * months
        return max_emi * ((1 + r)**months - 1) / (r * (1 + r)**months)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 4 of 5</div>
        <h1 style="font-family:'DM Serif Display',serif;font-size:28px;color:white;
        line-height:1.2;margin-bottom:0.5rem">Loan details</h1>
        <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:0">
        Select a product, enter the amount you need, and choose your repayment term.</p>
    </div>
    <div class="verified-card">
        <div class="verified-label">NIC Verified</div>
        <div class="verified-value">✅ &nbsp;{st.session_state.nic_value}</div>
    </div>
    """, unsafe_allow_html=True)

    loan_product = st.selectbox(
        "LOAN PRODUCT",
        options=["— Select a loan product —"] + list(LOAN_PRODUCTS.keys()),
    )

    selected_rate = None
    if loan_product and loan_product != "— Select a loan product —":
        desc, selected_rate = LOAN_PRODUCTS[loan_product]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-left:3px solid rgba(255,255,255,0.3);
        border-radius:8px;padding:0.9rem 1.1rem;margin:0.5rem 0 1rem 0">
            <p style="font-size:13px;color:rgba(255,255,255,0.65);margin:0;line-height:1.6">
            {desc}</p>
        </div>
        """, unsafe_allow_html=True)

    if band == "Medium Risk":
        loan_amount = st.session_state.suggested_amount
        st.markdown(f"""
        <div class="verified-card">
            <div class="verified-label">Approved loan amount</div>
            <div class="verified-value">{fmt(loan_amount)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if "typed_loan_amount" not in st.session_state:
            st.session_state.typed_loan_amount = 0.0
        st.number_input(
            "REQUIRED LOAN AMOUNT (LKR)",
            min_value=0.0, step=10000.0, format="%.2f",
            key="typed_loan_amount"
        )

    loan_term = st.selectbox(
        "REPAYMENT TERM",
        options=["— Select a term —"] + [f"{t} months" for t in TERM_OPTIONS],
    )

    preview_amount = st.session_state.get("typed_loan_amount", 0.0) if band != "Medium Risk" else st.session_state.get("suggested_amount", 0.0)
    if (selected_rate is not None and loan_term != "— Select a term —"
            and preview_amount and preview_amount > 0):
        term_months = int(loan_term.split()[0])
        emi         = calc_emi(preview_amount, selected_rate, term_months)
        max_emi     = salary * 0.40
        emi_ratio   = (emi / salary * 100) if salary > 0 else 0
        color       = "rgba(100,220,130,0.85)" if emi <= max_emi else "rgba(255,100,100,0.85)"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;
        padding:1rem 1.2rem;margin:0.8rem 0">
            <div style="font-size:11px;letter-spacing:1px;color:rgba(255,255,255,0.45);
            margin-bottom:6px">ESTIMATED MONTHLY REPAYMENT</div>
            <div style="font-size:26px;font-weight:700;color:{color}">{fmt(emi)}</div>
            <div style="font-size:12px;color:rgba(255,255,255,0.45);margin-top:4px">
            {emi_ratio:.1f}% of your monthly salary &nbsp;·&nbsp; max allowed 40%</div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.step4_error = ""
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Submit Application"):
            st.session_state.step4_error = ""
            entered_amount = st.session_state.get("typed_loan_amount", 0.0) if band != "Medium Risk" else st.session_state.suggested_amount
            max_eligible   = compute_max_eligible(st.session_state.customer_record)["max_eligible"]

            if loan_product == "— Select a loan product —" or not loan_product:
                st.session_state.step4_error = "product"
            elif loan_term == "— Select a term —":
                st.session_state.step4_error = "term"
            elif band != "Medium Risk" and (entered_amount is None or entered_amount <= 0):
                st.session_state.step4_error = "amount"
            else:
                term_months  = int(loan_term.split()[0])
                _, rate      = LOAN_PRODUCTS[loan_product]
                emi          = calc_emi(entered_amount, rate, term_months)
                max_emi      = salary * 0.40
                max_afford   = round(max_affordable_principal(rate, term_months, max_emi), 2)

                # FIX: floor to nearest 1,000 — never round up — so suggested amount is always affordable
                true_max     = floor_to_nearest_1000(min(max_eligible, max_afford))
                true_max     = max(true_max, 0)

                if emi > max_emi and entered_amount > max_afford:
                    st.session_state.step4_error      = "emi"
                    st.session_state.suggested_amount = true_max
                    st.session_state.emi_details      = {
                        "emi": emi, "max_emi": max_emi,
                        "max_afford": max_afford, "true_max": true_max,
                        "rate": rate, "term": term_months,
                    }
                elif band != "Medium Risk" and entered_amount > max_eligible:
                    st.session_state.step4_error      = "over"
                    # FIX: floor instead of round for the "over" path too
                    st.session_state.suggested_amount = floor_to_nearest_1000(min(max_eligible, max_afford))
                else:
                    _emi     = calc_emi(entered_amount, rate, term_months)
                    _totpay  = _emi * term_months
                    st.session_state.loan_product      = loan_product
                    st.session_state.loan_amount       = entered_amount
                    st.session_state.loan_term         = term_months
                    st.session_state.loan_rate         = rate
                    st.session_state.loan_emi          = round(_emi, 2)
                    st.session_state.loan_total_int    = round(_totpay - entered_amount, 2)
                    st.session_state.loan_total_pay    = round(_totpay, 2)
                    st.session_state.step = 5
                    st.rerun()

    # ── Error / warning blocks ──
    if st.session_state.step4_error == "product":
        st.error("❌ Please select a loan product to continue.")
    elif st.session_state.step4_error == "term":
        st.error("❌ Please select a repayment term to continue.")
    elif st.session_state.step4_error == "amount":
        st.error("❌ Please enter a valid loan amount to continue.")

    elif st.session_state.step4_error in ("emi", "over"):
        ed       = st.session_state.get("emi_details", {})
        sug      = st.session_state.suggested_amount
        rate     = ed.get("rate", 0)
        term_m   = ed.get("term", 0)
        sug_emi  = calc_emi(sug, rate, term_m) if (rate and term_m) else 0
        tot_pay  = sug_emi * term_m
        tot_int  = tot_pay - sug

        if st.session_state.step4_error == "emi":
            msg = (
                f"Based on your monthly income, we'd like to suggest a loan amount that keeps "
                f"your repayments comfortable. We recommend **{fmt(sug)}** over {term_m} months, "
                f"which brings your monthly repayment to **{fmt(sug_emi)}** — well within a "
                f"manageable range for your salary. Would you like to proceed with this instead?"
            )
        else:
            msg = (
                f"The amount you entered is a little above what we can offer based on your profile. "
                f"We'd be happy to proceed with **{fmt(sug)}** — would that work for you?"
            )
        st.warning(f"⚠️ {msg}")

        if sug > 0 and rate > 0 and term_m > 0:
            monthly_capital = sug / term_m
            monthly_int     = sug_emi - monthly_capital
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border-radius:14px;
            padding:1.2rem 1.4rem;margin:0.8rem 0 0.5rem 0">
                <div style="font-size:11px;letter-spacing:1px;color:rgba(255,255,255,0.45);
                margin-bottom:12px">REPAYMENT BREAKDOWN</div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Loan amount (capital)</span>
                    <span class="breakdown-val">{fmt(sug)}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Total interest payable</span>
                    <span class="breakdown-val">{fmt(tot_int)}</span>
                </div>
                <div class="breakdown-row" style="border-bottom:1px solid rgba(255,255,255,0.15);padding-bottom:10px;margin-bottom:10px">
                    <span class="breakdown-key" style="font-weight:600;color:white">Total repayment</span>
                    <span class="breakdown-val" style="font-weight:600;color:white">{fmt(tot_pay)}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Monthly capital repayment</span>
                    <span class="breakdown-val">{fmt(monthly_capital)}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-key">Monthly interest</span>
                    <span class="breakdown-val">{fmt(monthly_int)}</span>
                </div>
                <div class="breakdown-row" style="border-bottom:none;padding-bottom:0">
                    <span class="breakdown-key" style="font-weight:600;color:rgba(100,220,130,0.95)">Monthly repayment (EMI)</span>
                    <span class="breakdown-val" style="font-weight:600;color:rgba(100,220,130,0.95)">{fmt(sug_emi)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("No thanks", key="warn_no"):
                start_over(); st.rerun()
        with col2:
            if st.button("Confirm & proceed →", key="warn_yes"):
                st.session_state.loan_product      = loan_product
                st.session_state.loan_amount       = sug
                st.session_state.loan_term         = term_m
                st.session_state.loan_rate         = rate
                st.session_state.loan_emi          = round(sug_emi, 2)
                st.session_state.loan_total_int    = round(tot_int, 2)
                st.session_state.loan_total_pay    = round(tot_pay, 2)
                st.session_state.step4_error       = ""
                st.session_state.step = 5
                st.rerun()

# ══════════════════════════════════════════════════════════════
# STEP 5 — Final Confirmation Summary
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    record = st.session_state.customer_record
    dec    = st.session_state.decision

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.15);
    border-radius:20px;padding:2rem;margin-bottom:1.5rem">
        <div class="step-badge">Step 5 of 5</div>
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
            <div class="verified-label">Loan Amount</div>
            <div class="verified-value">{fmt(st.session_state.loan_amount)}</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Repayment Term</div>
            <div class="verified-value">{st.session_state.loan_term} months</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Interest Rate</div>
            <div class="verified-value">{st.session_state.loan_rate}% p.a.</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Loan Amount (Capital)</div>
            <div class="verified-value">{fmt(st.session_state.loan_amount)}</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Total Interest Payable</div>
            <div class="verified-value">{fmt(st.session_state.loan_total_int)}</div>
        </div>
        <div style="margin-bottom:1.2rem;padding-bottom:1.2rem;border-bottom:1px solid rgba(255,255,255,0.1)">
            <div class="verified-label">Total Repayment</div>
            <div class="verified-value" style="font-weight:700">{fmt(st.session_state.loan_total_pay)}</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Monthly Capital Repayment</div>
            <div class="verified-value">{fmt(st.session_state.loan_amount / st.session_state.loan_term)}</div>
        </div>
        <div style="margin-bottom:1rem">
            <div class="verified-label">Monthly Interest</div>
            <div class="verified-value">{fmt(st.session_state.loan_emi - (st.session_state.loan_amount / st.session_state.loan_term))}</div>
        </div>
        <div style="margin-bottom:0">
            <div class="verified-label">Monthly Repayment (EMI)</div>
            <div class="verified-value" style="color:rgba(100,220,130,0.95);font-weight:700">{fmt(st.session_state.loan_emi)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Your application has been received. A loan officer will be in touch within 2 business days.")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("← Start Over"):
        start_over(); st.rerun()

st.markdown(
    '<p style="text-align:center;color:rgba(255,255,255,0.3);font-size:12px;margin-top:1rem">'
    'Secured · Confidential · Instant results</p>',
    unsafe_allow_html=True
)