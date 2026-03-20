import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from db_utils import get_all_applications, update_application_status

st.set_page_config(
    page_title="Loan Officer Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ── Load all data ───────────────────────────────────────────
@st.cache_data
def load_data():
    eligible     = joblib.load("eligible_customers.pkl")
    accounts     = joblib.load("account_df_full.pkl")
    repayments   = joblib.load("repayment_history.pkl")
    transactions = joblib.load("transaction_history.pkl")
    return eligible, accounts, repayments, transactions

eligible_customers, account_df, repayment_df, transaction_df = load_data()

def fmt(n):
    try:    return f"LKR {float(n):,.0f}"
    except: return "LKR 0"

def is_loan_product(product):
    p = str(product).upper()
    return 'LOAN' in p or 'BORROW' in p

def format_date(d):
    try:
        return pd.to_datetime(d).strftime('%Y-%m-%d')
    except:
        return str(d)

# ══════════════════════════════════════════════════════════════
# CSS — Dark Green/Gold Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background: linear-gradient(160deg, #0D1F0F 0%, #0A1A0C 40%, #060F07 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #D4E8D0 !important;
}
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] { display: none; }

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #F0FAF0 !important; }

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(134,239,172,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s, background 0.2s;
}
.metric-card:hover { border-color: rgba(134,239,172,0.5); background: rgba(255,255,255,0.07); }
.metric-label { font-size: 11px; color: rgba(134,239,172,0.6); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; }
.metric-value { font-size: 30px; font-weight: 600; color: #F0FAF0; font-family: 'DM Mono', monospace; }

.info-card {
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(134,239,172,0.15);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    height: 100%;
}
.info-card-title {
    font-size: 10px; color: #86EFAC; text-transform: uppercase;
    letter-spacing: 3px; margin-bottom: 1rem; font-weight: 500;
}
.info-row {
    display: flex; justify-content: space-between;
    padding: 8px 0; border-bottom: 0.5px solid rgba(134,239,172,0.08);
    font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: rgba(212,232,208,0.55); }
.info-val { color: #F0FAF0; font-weight: 500; font-family: 'DM Mono', monospace; font-size: 12px; text-align: right; }

.badge-pending  { background:rgba(251,191,36,0.15); color:#FCD34D; padding:4px 14px; border-radius:20px; font-size:11px; border:0.5px solid rgba(251,191,36,0.4); }
.badge-approved { background:rgba(134,239,172,0.15); color:#86EFAC; padding:4px 14px; border-radius:20px; font-size:11px; border:0.5px solid rgba(134,239,172,0.4); }
.badge-rejected { background:rgba(248,113,113,0.15); color:#FCA5A5; padding:4px 14px; border-radius:20px; font-size:11px; border:0.5px solid rgba(248,113,113,0.4); }

.section-header {
    font-size: 10px; color: #86EFAC; text-transform: uppercase;
    letter-spacing: 3px; font-weight: 500;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 8px;
    border-bottom: 0.5px solid rgba(134,239,172,0.2);
}

.stButton > button {
    background: rgba(255,255,255,0.05) !important;
    color: #D4E8D0 !important;
    border: 0.5px solid rgba(134,239,172,0.25) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(134,239,172,0.1) !important;
    border-color: rgba(134,239,172,0.5) !important;
    color: #86EFAC !important;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 0.5px solid rgba(134,239,172,0.15) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary { color: #D4E8D0 !important; }
div[data-testid="stExpander"] summary:hover { color: #86EFAC !important; }

.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(134,239,172,0.2) !important;
    color: #D4E8D0 !important;
    border-radius: 10px !important;
}
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(134,239,172,0.2) !important;
    color: #D4E8D0 !important;
    border-radius: 10px !important;
}
label { color: rgba(212,232,208,0.6) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }

.stMetric { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 0.8rem; border: 0.5px solid rgba(134,239,172,0.15); }
[data-testid="stMetricValue"] { color: #F0FAF0 !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: rgba(134,239,172,0.6) !important; }

.app-card {
    background: rgba(255,255,255,0.03);
    border: 0.5px solid rgba(134,239,172,0.12);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s, background 0.2s;
}
.app-card:hover { border-color: rgba(134,239,172,0.35); background: rgba(255,255,255,0.05); }

/* Decorative circles */
.deco { position: fixed; border-radius: 50%; pointer-events: none; z-index: 0; }
</style>

<div class="deco" style="top:-120px;right:-120px;width:500px;height:500px;background:radial-gradient(circle,rgba(134,239,172,0.04) 0%,transparent 70%)"></div>
<div class="deco" style="bottom:-80px;left:-80px;width:350px;height:350px;background:radial-gradient(circle,rgba(251,191,36,0.04) 0%,transparent 70%)"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "officer_name"   not in st.session_state: st.session_state.officer_name   = ""
if "selected_app"   not in st.session_state: st.session_state.selected_app   = None
if "loan_detail_acc" not in st.session_state: st.session_state.loan_detail_acc = None

# ══════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════
if not st.session_state.officer_name:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center;padding:5rem 0 2rem'>
            <div style='font-size:52px;margin-bottom:1.2rem'>🏦</div>
            <h1 style='font-size:26px;margin-bottom:0.5rem'>Officer Portal</h1>
            <p style='color:rgba(212,232,208,0.45);font-size:14px'>Internal loan management system</p>
        </div>
        """, unsafe_allow_html=True)
        name = st.text_input("Officer name", placeholder="Enter your full name")
        if st.button("Sign in →", use_container_width=True) and name.strip():
            st.session_state.officer_name = name.strip()
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# LOAN REPAYMENT DETAIL PAGE
# ══════════════════════════════════════════════════════════════
def show_loan_repayment_page(acc_row, cust_repayments):
    if st.button("← Back to customer profile"):
        st.session_state.loan_detail_acc = None
        st.rerun()

    product = str(acc_row.get('ACTIVE_PRODUCT', 'Loan Account'))
    opened  = format_date(acc_row.get('ORIG_CONTRACT_DATE', ''))
    term    = acc_row.get('TERM', 'N/A')

    st.markdown(f"""
    <div style='margin:1rem 0 2rem'>
        <div style='font-size:10px;color:#86EFAC;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px'>
            Loan Repayment Detail
        </div>
        <h1 style='font-size:26px;margin:0'>{product}</h1>
        <p style='color:rgba(212,232,208,0.45);font-size:13px;margin-top:4px'>
            Opened: {opened} &nbsp;·&nbsp; Term: {term}
        </p>
    </div>
    """, unsafe_allow_html=True)

    acc_id = acc_row.get('ACC_MASKED_ID', '')
    if 'ACC_MASKED_ID' in cust_repayments.columns:
        acc_rep = cust_repayments[cust_repayments['ACC_MASKED_ID'] == acc_id].copy()
    else:
        acc_rep = cust_repayments.copy()

    if acc_rep.empty:
        st.info("No repayment records found for this loan account.")
        return

    acc_rep['PAYMENT_DATE'] = pd.to_datetime(acc_rep['PAYMENT_DATE'], errors='coerce')
    acc_rep = acc_rep.dropna(subset=['PAYMENT_DATE']).sort_values('PAYMENT_DATE')

    if acc_rep.empty:
        st.info("No dated repayment records found.")
        return

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Capital Paid</div>
            <div class="metric-value" style="font-size:20px">{fmt(acc_rep['CAPITAL_PAIED'].sum())}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Interest Paid</div>
            <div class="metric-value" style="font-size:20px;color:#FCD34D">{fmt(acc_rep['INTEREST_PAIED'].sum())}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Paid</div>
            <div class="metric-value" style="font-size:20px;color:#86EFAC">{fmt(acc_rep['TOTAL_PAID'].sum())}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">No. of Payments</div>
            <div class="metric-value" style="font-size:20px">{len(acc_rep)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Capital vs Interest Line Graph
    st.markdown('<div class="section-header">Capital & Interest Repayments Over Time</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'],
        y=acc_rep['CAPITAL_PAIED'],
        name='Capital Paid',
        mode='lines+markers',
        line=dict(color='#86EFAC', width=2),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(134,239,172,0.08)'
    ))
    fig.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'],
        y=acc_rep['INTEREST_PAIED'],
        name='Interest Paid',
        mode='lines+markers',
        line=dict(color='#FCD34D', width=2),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(252,211,77,0.08)'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(212,232,208,0.7)', family='DM Sans'),
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor='rgba(134,239,172,0.08)', title='Payment Date'),
        yaxis=dict(gridcolor='rgba(134,239,172,0.08)', title='Amount (LKR)'),
        legend=dict(
            bgcolor='rgba(255,255,255,0.05)',
            bordercolor='rgba(134,239,172,0.2)',
            font=dict(color='#D4E8D0')
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative line graph
    st.markdown('<div class="section-header">Cumulative Repayment Progress</div>', unsafe_allow_html=True)

    acc_rep['Cumulative_Capital']  = acc_rep['CAPITAL_PAIED'].cumsum()
    acc_rep['Cumulative_Interest'] = acc_rep['INTEREST_PAIED'].cumsum()
    acc_rep['Cumulative_Total']    = acc_rep['TOTAL_PAID'].cumsum()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['Cumulative_Total'],
        name='Total Cumulative', mode='lines',
        line=dict(color='#86EFAC', width=3)
    ))
    fig2.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['Cumulative_Capital'],
        name='Capital Cumulative', mode='lines',
        line=dict(color='#6EE7B7', width=2, dash='dot')
    ))
    fig2.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['Cumulative_Interest'],
        name='Interest Cumulative', mode='lines',
        line=dict(color='#FCD34D', width=2, dash='dot')
    ))
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(212,232,208,0.7)', family='DM Sans'),
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor='rgba(134,239,172,0.08)'),
        yaxis=dict(gridcolor='rgba(134,239,172,0.08)', title='Cumulative Amount (LKR)'),
        legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(134,239,172,0.2)', font=dict(color='#D4E8D0'))
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Repayment table
    st.markdown('<div class="section-header">Repayment Records</div>', unsafe_allow_html=True)
    display_df = acc_rep[['PAYMENT_DATE', 'CAPITAL_PAIED', 'INTEREST_PAIED', 'TOTAL_PAID']].copy()
    display_df['PAYMENT_DATE'] = display_df['PAYMENT_DATE'].dt.strftime('%Y-%m-%d')
    display_df.columns = ['Payment Date', 'Capital Paid (LKR)', 'Interest Paid (LKR)', 'Total Paid (LKR)']
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# CUSTOMER DETAIL PAGE
# ══════════════════════════════════════════════════════════════
def show_customer_page(app):
    # If viewing loan repayment detail
    if st.session_state.loan_detail_acc is not None:
        nic = app["nic"]
        cust = eligible_customers[eligible_customers["MASKED_LEGAL_ID"] == nic]
        if not cust.empty:
            masked_id = cust.iloc[0].get("MASKED_ID", "")
            cust_repayments = repayment_df[repayment_df["MASKED_ID"] == masked_id].copy() if masked_id else pd.DataFrame()
        else:
            cust_repayments = pd.DataFrame()
        show_loan_repayment_page(st.session_state.loan_detail_acc, cust_repayments)
        return

    nic = app["nic"]

    if st.button("← Back to applications"):
        st.session_state.selected_app = None
        st.rerun()

    badge = {
        "Pending":  "<span class='badge-pending'>⏳ Pending</span>",
        "Approved": "<span class='badge-approved'>✅ Approved</span>",
        "Rejected": "<span class='badge-rejected'>❌ Rejected</span>",
    }.get(app["status"], app["status"])

    st.markdown(f"""
    <div style='margin:1rem 0 2rem'>
        <div style='font-size:10px;color:#86EFAC;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px'>
            Application #{app['id']} &nbsp;·&nbsp; {badge}
        </div>
        <h1 style='font-size:28px;margin:0'>Customer Profile</h1>
    </div>
    """, unsafe_allow_html=True)

    cust = eligible_customers[eligible_customers["MASKED_LEGAL_ID"] == nic]
    if cust.empty:
        st.error("Customer record not found.")
        return
    c = cust.iloc[0]
    masked_id = c.get("MASKED_ID", "")

    cust_accounts    = account_df[account_df["MASKED_ID"] == masked_id] if masked_id else pd.DataFrame()
    cust_repayments  = repayment_df[repayment_df["MASKED_ID"] == masked_id].copy() if masked_id else pd.DataFrame()
    cust_transactions= transaction_df[transaction_df["MASKED_ID"] == masked_id].copy() if masked_id else pd.DataFrame()

    # ── Top metrics ─────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        score_color = "#86EFAC" if float(c.get('Internal_Bank_Default_Score',0)) >= 650 else "#FCA5A5"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Internal Score</div>
            <div class="metric-value" style="color:{score_color}">{int(float(c.get('Internal_Bank_Default_Score',0)))}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Score Band</div>
            <div class="metric-value" style="font-size:15px;padding-top:8px">{c.get('Score_Band','N/A')}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Monthly Income</div>
            <div class="metric-value" style="font-size:17px;padding-top:6px">{fmt(c.get('Avg_Monthly_Credit',0))}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        ood_val = int(float(c.get('MAX_OOD', 0)))
        ood_color = "#FCA5A5" if ood_val >= 30 else "#86EFAC"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Max Days Overdue</div>
            <div class="metric-value" style="color:{ood_color}">{ood_val}</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Active Accounts</div>
            <div class="metric-value">{int(c.get('Number_of_Active_Accounts',0))}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Three info columns ───────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
            <div class="info-row"><span class="info-key">NIC</span><span class="info-val">{nic}</span></div>
            <div class="info-row"><span class="info-key">Age</span><span class="info-val">{int(c.get('AGE',0))}</span></div>
            <div class="info-row"><span class="info-key">Gender</span><span class="info-val">{str(c.get('GENDER','N/A')).title()}</span></div>
            <div class="info-row"><span class="info-key">Marital status</span><span class="info-val">{str(c.get('MARITAL_STATUS','N/A')).title()}</span></div>
            <div class="info-row"><span class="info-key">District</span><span class="info-val">{str(c.get('DISTRICT','N/A')).title()}</span></div>
            <div class="info-row"><span class="info-key">Occupation</span><span class="info-val">{str(c.get('OCCUPATION','N/A')).title()}</span></div>
            <div class="info-row"><span class="info-key">Employment</span><span class="info-val">{str(c.get('EMPLOYMENT_STATUS','N/A')).title()}</span></div>
            <div class="info-row"><span class="info-key">Segment</span><span class="info-val">{str(c.get('Employment_Segment','N/A'))}</span></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Risk Profile</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
            <div class="info-row"><span class="info-key">Customer risk</span><span class="info-val">{str(c.get('CUSTOMER_RISK_NAME','N/A')).title()}</span></div>
            <div class="info-row"><span class="info-key">Target tier</span><span class="info-val">{c.get('TARGET_DESC','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Financial capacity</span><span class="info-val">{c.get('Financial_Capacity','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Cluster</span><span class="info-val">{c.get('Cluster_Name','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Age bucket</span><span class="info-val">{c.get('Age_Bucket','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Existing debt</span><span class="info-val">{fmt(c.get('TOTAL_CAPITAL_DUE',0))}</span></div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-header">This Application</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
            <div style='margin-bottom:12px'>{badge}</div>
            <div class="info-row"><span class="info-key">Product</span><span class="info-val" style="font-size:11px">{app['loan_product'].split('—')[0].strip()}</span></div>
            <div class="info-row"><span class="info-key">Amount</span><span class="info-val">{fmt(app['loan_amount'])}</span></div>
            <div class="info-row"><span class="info-key">Term</span><span class="info-val">{app['loan_term']} months</span></div>
            <div class="info-row"><span class="info-key">Rate</span><span class="info-val">{app['loan_rate']}% p.a.</span></div>
            <div class="info-row"><span class="info-key">Monthly EMI</span><span class="info-val">{fmt(app['loan_emi'])}</span></div>
            <div class="info-row"><span class="info-key">Total interest</span><span class="info-val">{fmt(app['total_interest'])}</span></div>
            <div class="info-row"><span class="info-key">Total repayment</span><span class="info-val">{fmt(app['total_repayment'])}</span></div>
            <div class="info-row"><span class="info-key">Submitted</span><span class="info-val">{app['submitted_at']}</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Accounts Section ─────────────────────────────────────
    st.markdown('<div class="section-header">Accounts</div>', unsafe_allow_html=True)

    if not cust_accounts.empty:
        for _, acc in cust_accounts.iterrows():
            product = str(acc.get('ACTIVE_PRODUCT', 'N/A'))
            status  = str(acc.get('ACCT_STATUS', 'N/A')).title()
            opened  = format_date(acc.get('ORIG_CONTRACT_DATE', ''))
            balance = float(acc.get('MONTHEND_CONVERTED_BALANCE', 0))
            term    = acc.get('TERM', 'N/A')
            is_loan = is_loan_product(product)

            expander_label = (
                f"{product}  |  {status}  |  Opened: {opened}  |  "
                f"{'Term: ' + str(term) if is_loan else 'Balance: ' + fmt(balance)}"
            )

            with st.expander(expander_label):
                if is_loan:
                    # Loan account — show term, balance, NO ood
                    lc1, lc2, lc3 = st.columns(3)
                    with lc1: st.metric("Balance", fmt(balance))
                    with lc2: st.metric("Term", str(term))
                    with lc3: st.metric("Status", status)

                    # Button to go to repayment detail page
                    if st.button(
                        f"View repayment history →",
                        key=f"repay_{acc.get('ACC_MASKED_ID','')}"
                    ):
                        st.session_state.loan_detail_acc = acc.to_dict()
                        st.rerun()
                else:
                    # Savings / Current — show balance only, NO ood
                    lc1, lc2 = st.columns(2)
                    with lc1: st.metric("Balance", fmt(balance))
                    with lc2: st.metric("Status", status)
    else:
        st.info("No account records found for this customer.")

    # ── Average Monthly Balance Line Graph ───────────────────
    balance_cols = ['JUN_25', 'JUL_25', 'AUG_25', 'SEP_25', 'OCT_25', 'NOV_25']
    existing_cols = [col for col in balance_cols if col in account_df.columns]

    if not cust_accounts.empty and existing_cols:
        st.markdown('<div class="section-header">Average Monthly Balance Trend</div>', unsafe_allow_html=True)
        monthly_avg  = cust_accounts[existing_cols].mean()
        month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'][:len(existing_cols)]

        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(
            x=month_labels, y=monthly_avg.values,
            mode='lines+markers', name='Avg Balance',
            line=dict(color='#86EFAC', width=3),
            marker=dict(size=8, color='#86EFAC'),
            fill='tozeroy', fillcolor='rgba(134,239,172,0.08)'
        ))
        fig_bal.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(212,232,208,0.7)', family='DM Sans'),
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(134,239,172,0.08)'),
            yaxis=dict(gridcolor='rgba(134,239,172,0.08)', title='Average Balance (LKR)')
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    # ── Transaction Activity Line Graph ──────────────────────
    if not cust_transactions.empty and 'BOOKING_DATE' in cust_transactions.columns:
        st.markdown('<div class="section-header">Transaction Activity</div>', unsafe_allow_html=True)
        txn = cust_transactions.copy()
        txn['BOOKING_DATE'] = pd.to_datetime(txn['BOOKING_DATE'], errors='coerce')
        txn = txn.dropna(subset=['BOOKING_DATE'])
        txn['Month']   = txn['BOOKING_DATE'].dt.to_period('M').astype(str)
        txn['INFLOW']  = txn['AMOUNT_LCY'].apply(lambda x: x if x > 0 else 0)
        txn['OUTFLOW'] = txn['AMOUNT_LCY'].apply(lambda x: abs(x) if x < 0 else 0)
        monthly_txn = txn.groupby('Month').agg(Inflow=('INFLOW','sum'), Outflow=('OUTFLOW','sum')).reset_index()

        fig_txn = go.Figure()
        fig_txn.add_trace(go.Scatter(
            x=monthly_txn['Month'], y=monthly_txn['Inflow'],
            name='Inflow', mode='lines+markers',
            line=dict(color='#86EFAC', width=2), marker=dict(size=5),
            fill='tozeroy', fillcolor='rgba(134,239,172,0.08)'
        ))
        fig_txn.add_trace(go.Scatter(
            x=monthly_txn['Month'], y=monthly_txn['Outflow'],
            name='Outflow', mode='lines+markers',
            line=dict(color='#FCD34D', width=2), marker=dict(size=5),
            fill='tozeroy', fillcolor='rgba(252,211,77,0.08)'
        ))
        fig_txn.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(212,232,208,0.7)', family='DM Sans'),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(134,239,172,0.08)'),
            yaxis=dict(gridcolor='rgba(134,239,172,0.08)', title='Amount (LKR)'),
            legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(134,239,172,0.2)', font=dict(color='#D4E8D0'))
        )
        st.plotly_chart(fig_txn, use_container_width=True)

    # ── Review Section ───────────────────────────────────────
    if app["status"] == "Pending":
        st.markdown('<div class="section-header">Review Decision</div>', unsafe_allow_html=True)
        notes = st.text_area(
            "Officer notes",
            key=f"notes_detail_{app['id']}",
            placeholder="Add comments, conditions, or reasons here...",
            height=100
        )
        col_a, col_b, col_c = st.columns([1, 1, 3])
        with col_a:
            if st.button("✅ Approve", key=f"approve_detail_{app['id']}", use_container_width=True):
                update_application_status(app["id"], "Approved", st.session_state.officer_name, notes)
                st.success("Application approved!")
                st.session_state.selected_app = None
                st.rerun()
        with col_b:
            if st.button("❌ Reject", key=f"reject_detail_{app['id']}", use_container_width=True):
                update_application_status(app["id"], "Rejected", st.session_state.officer_name, notes)
                st.error("Application rejected.")
                st.session_state.selected_app = None
                st.rerun()
    else:
        st.markdown('<div class="section-header">Review Details</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
            <div class="info-row"><span class="info-key">Reviewed by</span><span class="info-val">{app.get('reviewed_by','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Reviewed at</span><span class="info-val">{app.get('reviewed_at','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Notes</span><span class="info-val">{app.get('officer_notes','N/A')}</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════
def show_dashboard():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("<h1 style='font-size:26px;margin-bottom:4px'>Loan Officer Dashboard</h1>", unsafe_allow_html=True)
        st.caption(f"Signed in as **{st.session_state.officer_name}**")
    with col2:
        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            st.session_state.officer_name = ""
            st.rerun()

    st.markdown("<hr style='border-color:rgba(134,239,172,0.15);margin:1rem 0'>", unsafe_allow_html=True)

    col_r, _ = st.columns([1, 5])
    with col_r:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()

    applications = get_all_applications()
    if not applications:
        st.info("No applications received yet.")
        return

    total    = len(applications)
    pending  = sum(1 for a in applications if a["status"] == "Pending")
    approved = sum(1 for a in applications if a["status"] == "Approved")
    rejected = sum(1 for a in applications if a["status"] == "Rejected")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total</div>
            <div class="metric-value">{total}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Pending</div>
            <div class="metric-value" style="color:#FCD34D">{pending}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Approved</div>
            <div class="metric-value" style="color:#86EFAC">{approved}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Rejected</div>
            <div class="metric-value" style="color:#FCA5A5">{rejected}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    status_filter = st.selectbox("Filter", ["All", "Pending", "Approved", "Rejected"])
    filtered = applications if status_filter == "All" else [
        a for a in applications if a["status"] == status_filter
    ]

    st.markdown(f"<p style='color:rgba(134,239,172,0.5);font-size:12px;letter-spacing:1px'>{len(filtered)} APPLICATION(S)</p>", unsafe_allow_html=True)

    for app in filtered:
        badge = {
            "Pending":  "<span class='badge-pending'>⏳ Pending</span>",
            "Approved": "<span class='badge-approved'>✅ Approved</span>",
            "Rejected": "<span class='badge-rejected'>❌ Rejected</span>",
        }.get(app["status"], app["status"])

        col_info, col_btn = st.columns([6, 1])
        with col_info:
            st.markdown(f"""
            <div class="app-card">
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div style='display:flex;align-items:center;gap:16px'>
                        <span style='color:rgba(134,239,172,0.4);font-size:11px;font-family:DM Mono'>#{app['id']}</span>
                        <span style='color:#F0FAF0;font-weight:500'>{app['nic']}</span>
                        {badge}
                    </div>
                    <div style='display:flex;gap:2rem;align-items:center'>
                        <span style='color:rgba(212,232,208,0.5);font-size:12px'>{app['loan_product'].split('—')[0].strip()}</span>
                        <span style='color:#F0FAF0;font-family:DM Mono;font-size:13px'>{fmt(app['loan_amount'])}</span>
                        <span style='color:rgba(212,232,208,0.35);font-size:11px'>{app['submitted_at']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            if st.button("View →", key=f"view_{app['id']}", use_container_width=True):
                st.session_state.selected_app = app
                st.session_state.loan_detail_acc = None
                st.rerun()


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
if st.session_state.selected_app is not None:
    show_customer_page(st.session_state.selected_app)
else:
    show_dashboard()