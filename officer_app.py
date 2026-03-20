import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
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
    eligible    = joblib.load("eligible_customers.pkl")
    accounts    = joblib.load("account_df_full.pkl")
    repayments  = joblib.load("repayment_history.pkl")
    transactions= joblib.load("transaction_history.pkl")
    return eligible, accounts, repayments, transactions

eligible_customers, account_df, repayment_df, transaction_df = load_data()

def fmt(n):
    try:    return f"LKR {float(n):,.0f}"
    except: return "LKR 0"

# ══════════════════════════════════════════════════════════════
# CSS — Dark Navy Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, .stApp {
    background-color: #0A0F1E !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #E2E8F0 !important;
}
.stApp { background-color: #0A0F1E !important; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #F8FAFC !important; }
p, span, div { color: #CBD5E1; }

.metric-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3B82F6; }
.metric-label { font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px; }
.metric-value { font-size: 32px; font-weight: 600; color: #F8FAFC; font-family: 'IBM Plex Mono', monospace; }

.info-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.info-card-title {
    font-size: 11px; color: #3B82F6; text-transform: uppercase;
    letter-spacing: 2px; margin-bottom: 1rem; font-weight: 500;
}
.info-row {
    display: flex; justify-content: space-between;
    padding: 8px 0; border-bottom: 1px solid #1E293B;
    font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #64748B; }
.info-val { color: #F8FAFC; font-weight: 500; font-family: 'IBM Plex Mono', monospace; }

.badge-pending  { background:#2D2006; color:#FCD34D; padding:3px 12px; border-radius:20px; font-size:11px; border:1px solid #92400E; }
.badge-approved { background:#052E16; color:#34D399; padding:3px 12px; border-radius:20px; font-size:11px; border:1px solid #065F46; }
.badge-rejected { background:#2D0A0A; color:#F87171; padding:3px 12px; border-radius:20px; font-size:11px; border:1px solid #7F1D1D; }
.badge-loan     { background:#1E1B4B; color:#818CF8; padding:3px 12px; border-radius:20px; font-size:11px; border:1px solid #3730A3; }
.badge-savings  { background:#042F2E; color:#2DD4BF; padding:3px 12px; border-radius:20px; font-size:11px; border:1px solid #134E4A; }

.app-row {
    background: #111827; border: 1px solid #1E293B;
    border-radius: 10px; padding: 1rem 1.5rem;
    margin-bottom: 0.5rem; cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    display: flex; justify-content: space-between; align-items: center;
}
.app-row:hover { border-color: #3B82F6; background: #0F172A; }
.app-row-selected { border-color: #3B82F6 !important; background: #0F172A !important; }

.stButton > button {
    background: #1E293B !important; color: #E2E8F0 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #334155 !important; border-color: #3B82F6 !important; }

.approve-btn > button { background: #052E16 !important; color: #34D399 !important; border-color: #065F46 !important; }
.reject-btn > button  { background: #2D0A0A !important; color: #F87171 !important; border-color: #7F1D1D !important; }

div[data-testid="stExpander"] {
    background: #111827 !important; border: 1px solid #1E293B !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary { color: #E2E8F0 !important; }

.stSelectbox > div > div { background: #111827 !important; border-color: #1E293B !important; color: #E2E8F0 !important; }
.stTextInput input, .stTextArea textarea {
    background: #111827 !important; border-color: #1E293B !important;
    color: #E2E8F0 !important; border-radius: 8px !important;
}
label { color: #94A3B8 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
.stDivider { border-color: #1E293B !important; }

.section-header {
    font-size: 11px; color: #3B82F6; text-transform: uppercase;
    letter-spacing: 3px; font-weight: 500; margin: 1.5rem 0 1rem 0;
    padding-bottom: 8px; border-bottom: 1px solid #1E293B;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════
if "officer_name" not in st.session_state:
    st.session_state.officer_name = ""
if "selected_app" not in st.session_state:
    st.session_state.selected_app = None

if not st.session_state.officer_name:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center;padding:4rem 0 2rem'>
            <div style='font-size:48px;margin-bottom:1rem'>🏦</div>
            <h1 style='font-size:24px;color:#F8FAFC;margin-bottom:0.5rem'>Officer Portal</h1>
            <p style='color:#64748B;font-size:14px'>Internal loan management system</p>
        </div>
        """, unsafe_allow_html=True)
        name = st.text_input("Officer name", placeholder="Enter your full name")
        if st.button("Sign in →", use_container_width=True) and name.strip():
            st.session_state.officer_name = name.strip()
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# CUSTOMER DETAIL PAGE
# ══════════════════════════════════════════════════════════════
def show_customer_page(app):
    nic = app["nic"]

    if st.button("← Back to applications"):
        st.session_state.selected_app = None
        st.rerun()

    st.markdown(f"""
    <div style='margin:1rem 0 2rem'>
        <div style='font-size:11px;color:#3B82F6;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px'>
            Application #{app['id']} · {app['status']}
        </div>
        <h1 style='font-size:28px;color:#F8FAFC;margin:0'>Customer Profile</h1>
    </div>
    """, unsafe_allow_html=True)

    # Get customer record
    cust = eligible_customers[eligible_customers["MASKED_LEGAL_ID"] == nic]
    if cust.empty:
        st.error("Customer record not found.")
        return
    c = cust.iloc[0]

    # Get accounts
    cust_accounts = account_df[account_df["MASKED_ID"] == c.get("MASKED_ID", "")]

    # Get repayments
    cust_repayments = repayment_df[repayment_df["MASKED_ID"] == c.get("MASKED_ID", "")]

    # Get transactions
    cust_transactions = transaction_df[transaction_df["MASKED_ID"] == c.get("MASKED_ID", "")]

    # ── Row 1: Key metrics ──────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Credit Score</div>
            <div class="metric-value" style="color:#3B82F6">{int(c.get('Internal_Bank_Default_Score', 0))}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Score Band</div>
            <div class="metric-value" style="font-size:16px;padding-top:8px">{c.get('Score_Band','N/A')}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Profile Score</div>
            <div class="metric-value" style="color:#10B981">{int(c.get('profile_score', 0) if 'profile_score' in c.index else 0)}/100</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Monthly Income</div>
            <div class="metric-value" style="font-size:18px;padding-top:6px">{fmt(c.get('Avg_Monthly_Credit',0))}</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        ood_color = "#F87171" if float(c.get('MAX_OOD', 0)) >= 30 else "#34D399"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Max Days Overdue</div>
            <div class="metric-value" style="color:{ood_color}">{int(float(c.get('MAX_OOD',0)))}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Row 2: Personal info + Risk profile ────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
            <div class="info-row"><span class="info-key">NIC</span><span class="info-val">{nic}</span></div>
            <div class="info-row"><span class="info-key">Age</span><span class="info-val">{int(c.get('AGE',0))}</span></div>
            <div class="info-row"><span class="info-key">Gender</span><span class="info-val">{c.get('GENDER','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Marital Status</span><span class="info-val">{c.get('MARITAL_STATUS','N/A')}</span></div>
            <div class="info-row"><span class="info-key">District</span><span class="info-val">{c.get('DISTRICT','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Occupation</span><span class="info-val">{c.get('OCCUPATION','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Employment</span><span class="info-val">{c.get('EMPLOYMENT_STATUS','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Segment</span><span class="info-val">{c.get('Employment_Segment','N/A')}</span></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Risk Profile</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
            <div class="info-row"><span class="info-key">Customer Risk</span><span class="info-val">{c.get('CUSTOMER_RISK_NAME','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Target Tier</span><span class="info-val">{c.get('TARGET_DESC','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Financial Capacity</span><span class="info-val">{c.get('Financial_Capacity','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Cluster</span><span class="info-val">{c.get('Cluster_Name','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Age Bucket</span><span class="info-val">{c.get('Age_Bucket','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Existing Debt</span><span class="info-val">{fmt(c.get('TOTAL_CAPITAL_DUE',0))}</span></div>
            <div class="info-row"><span class="info-key">Active Accounts</span><span class="info-val">{int(c.get('Number_of_Active_Accounts',0))}</span></div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-header">This Application</div>', unsafe_allow_html=True)
        badge = {
            "Pending":  "<span class='badge-pending'>⏳ Pending</span>",
            "Approved": "<span class='badge-approved'>✅ Approved</span>",
            "Rejected": "<span class='badge-rejected'>❌ Rejected</span>",
        }.get(app["status"], app["status"])
        st.markdown(f"""<div class="info-card">
            <div style='margin-bottom:12px'>{badge}</div>
            <div class="info-row"><span class="info-key">Product</span><span class="info-val" style="font-size:11px">{app['loan_product']}</span></div>
            <div class="info-row"><span class="info-key">Amount</span><span class="info-val">{fmt(app['loan_amount'])}</span></div>
            <div class="info-row"><span class="info-key">Term</span><span class="info-val">{app['loan_term']} months</span></div>
            <div class="info-row"><span class="info-key">Rate</span><span class="info-val">{app['loan_rate']}% p.a.</span></div>
            <div class="info-row"><span class="info-key">Monthly EMI</span><span class="info-val">{fmt(app['loan_emi'])}</span></div>
            <div class="info-row"><span class="info-key">Total Interest</span><span class="info-val">{fmt(app['total_interest'])}</span></div>
            <div class="info-row"><span class="info-key">Total Repayment</span><span class="info-val">{fmt(app['total_repayment'])}</span></div>
            <div class="info-row"><span class="info-key">Submitted</span><span class="info-val" style="font-size:11px">{app['submitted_at']}</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Accounts Section ────────────────────────────────────
    st.markdown('<div class="section-header">Accounts</div>', unsafe_allow_html=True)

    if not cust_accounts.empty:
        for _, acc in cust_accounts.iterrows():
            product  = str(acc.get('ACTIVE_PRODUCT', 'N/A'))
            status   = str(acc.get('ACCT_STATUS', 'N/A'))
            opened   = str(acc.get('ORIG_CONTRACT_DATE', 'N/A'))
            if hasattr(opened, 'strftime'):
                opened = opened.strftime('%Y-%m-%d')
            balance  = float(acc.get('MONTHEND_CONVERTED_BALANCE', 0))
            ood      = float(acc.get('OOD', 0))
            is_loan  = 'LOAN' in product.upper() or 'BORROW' in product.upper()
            badge_class = 'badge-loan' if is_loan else 'badge-savings'

            with st.expander(f"{product}  |  {status}  |  Opened: {opened}  |  Balance: {fmt(balance)}"):
                ac1, ac2, ac3, ac4 = st.columns(4)
                with ac1:
                    st.metric("Balance", fmt(balance))
                with ac2:
                    st.metric("Days Overdue", int(ood))
                with ac3:
                    st.metric("Status", status)
                with ac4:
                    st.metric("Product", product)

                # ── Repayment graph for loan accounts ──
                if is_loan and not cust_repayments.empty:
                    acc_id = acc.get('ACC_MASKED_ID', '')
                    acc_repayments = cust_repayments[
                        cust_repayments['ACC_MASKED_ID'] == acc_id
                    ] if 'ACC_MASKED_ID' in cust_repayments.columns else cust_repayments

                    if not acc_repayments.empty and 'PAYMENT_DATE' in acc_repayments.columns:
                        acc_rep = acc_repayments.copy()
                        acc_rep['PAYMENT_DATE'] = pd.to_datetime(acc_rep['PAYMENT_DATE'], errors='coerce')
                        acc_rep = acc_rep.dropna(subset=['PAYMENT_DATE']).sort_values('PAYMENT_DATE')

                        if not acc_rep.empty:
                            st.markdown("**Repayment History**")
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=acc_rep['PAYMENT_DATE'],
                                y=acc_rep['CAPITAL_PAIED'],
                                name='Capital Paid',
                                marker_color='#3B82F6'
                            ))
                            fig.add_trace(go.Bar(
                                x=acc_rep['PAYMENT_DATE'],
                                y=acc_rep['INTEREST_PAIED'],
                                name='Interest Paid',
                                marker_color='#8B5CF6'
                            ))
                            fig.add_trace(go.Scatter(
                                x=acc_rep['PAYMENT_DATE'],
                                y=acc_rep['OOD'],
                                name='Days Overdue',
                                yaxis='y2',
                                line=dict(color='#F87171', width=2),
                                mode='lines+markers'
                            ))
                            fig.update_layout(
                                paper_bgcolor='#111827',
                                plot_bgcolor='#111827',
                                font=dict(color='#94A3B8', family='IBM Plex Sans'),
                                barmode='stack',
                                height=300,
                                margin=dict(l=0, r=0, t=20, b=0),
                                legend=dict(bgcolor='#1E293B', bordercolor='#334155'),
                                xaxis=dict(gridcolor='#1E293B'),
                                yaxis=dict(gridcolor='#1E293B', title='Amount (LKR)'),
                                yaxis2=dict(
                                    title='Days Overdue',
                                    overlaying='y',
                                    side='right',
                                    gridcolor='#1E293B'
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No account records found for this customer.")

    # ── Average Monthly Balance Line Graph ─────────────────
    st.markdown('<div class="section-header">Average Monthly Balance Trend</div>', unsafe_allow_html=True)

    balance_cols = ['JUN_25', 'JUL_25', 'AUG_25', 'SEP_25', 'OCT_25', 'NOV_25']
    existing_cols = [c for c in balance_cols if c in account_df.columns]

    if not cust_accounts.empty and existing_cols:
        monthly_avg = cust_accounts[existing_cols].mean()
        month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'][:len(existing_cols)]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=month_labels,
            y=monthly_avg.values,
            mode='lines+markers',
            name='Avg Balance',
            line=dict(color='#10B981', width=3),
            marker=dict(size=8, color='#10B981'),
            fill='tozeroy',
            fillcolor='rgba(16,185,129,0.1)'
        ))
        fig2.update_layout(
            paper_bgcolor='#111827',
            plot_bgcolor='#111827',
            font=dict(color='#94A3B8', family='IBM Plex Sans'),
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#1E293B'),
            yaxis=dict(gridcolor='#1E293B', title='Average Balance (LKR)')
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Transaction Volume Chart ────────────────────────────
    if not cust_transactions.empty and 'BOOKING_DATE' in cust_transactions.columns:
        st.markdown('<div class="section-header">Transaction Activity</div>', unsafe_allow_html=True)

        txn = cust_transactions.copy()
        txn['BOOKING_DATE'] = pd.to_datetime(txn['BOOKING_DATE'], errors='coerce')
        txn = txn.dropna(subset=['BOOKING_DATE'])
        txn['Month'] = txn['BOOKING_DATE'].dt.to_period('M').astype(str)
        txn['INFLOW']  = txn['AMOUNT_LCY'].apply(lambda x: x if x > 0 else 0)
        txn['OUTFLOW'] = txn['AMOUNT_LCY'].apply(lambda x: abs(x) if x < 0 else 0)

        monthly_txn = txn.groupby('Month').agg(
            Inflow=('INFLOW', 'sum'),
            Outflow=('OUTFLOW', 'sum')
        ).reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=monthly_txn['Month'], y=monthly_txn['Inflow'],
            name='Inflow', mode='lines+markers',
            line=dict(color='#10B981', width=2),
            marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.08)'
        ))
        fig3.add_trace(go.Scatter(
            x=monthly_txn['Month'], y=monthly_txn['Outflow'],
            name='Outflow', mode='lines+markers',
            line=dict(color='#F87171', width=2),
            marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(248,113,113,0.08)'
        ))
        fig3.update_layout(
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            font=dict(color='#94A3B8', family='IBM Plex Sans'),
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#1E293B'),
            yaxis=dict(gridcolor='#1E293B', title='Amount (LKR)'),
            legend=dict(bgcolor='#1E293B', bordercolor='#334155')
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Review Section ──────────────────────────────────────
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
            st.markdown('<div class="approve-btn">', unsafe_allow_html=True)
            if st.button("✅ Approve Application", key=f"approve_detail_{app['id']}", use_container_width=True):
                update_application_status(app["id"], "Approved", st.session_state.officer_name, notes)
                st.success("Application approved!")
                st.session_state.selected_app = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="reject-btn">', unsafe_allow_html=True)
            if st.button("❌ Reject Application", key=f"reject_detail_{app['id']}", use_container_width=True):
                update_application_status(app["id"], "Rejected", st.session_state.officer_name, notes)
                st.error("Application rejected.")
                st.session_state.selected_app = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">Review Details</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-card">
            <div class="info-row"><span class="info-key">Reviewed by</span><span class="info-val">{app.get('reviewed_by','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Reviewed at</span><span class="info-val">{app.get('reviewed_at','N/A')}</span></div>
            <div class="info-row"><span class="info-key">Notes</span><span class="info-val">{app.get('officer_notes','N/A')}</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════
def show_dashboard():
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <h1 style='font-size:26px;color:#F8FAFC;margin-bottom:4px'>Loan Officer Dashboard</h1>
        """, unsafe_allow_html=True)
        st.caption(f"Signed in as **{st.session_state.officer_name}**")
    with col2:
        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            st.session_state.officer_name = ""
            st.rerun()

    st.markdown("<hr style='border-color:#1E293B;margin:1rem 0'>", unsafe_allow_html=True)

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()

    applications = get_all_applications()

    if not applications:
        st.info("No applications received yet.")
        return

    # Metrics
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
            <div class="metric-value" style="color:#34D399">{approved}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Rejected</div>
            <div class="metric-value" style="color:#F87171">{rejected}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Filter
    status_filter = st.selectbox("Filter", ["All", "Pending", "Approved", "Rejected"])
    filtered = applications if status_filter == "All" else [
        a for a in applications if a["status"] == status_filter
    ]

    st.markdown(f"<p style='color:#64748B;font-size:13px'>{len(filtered)} application(s)</p>",
                unsafe_allow_html=True)

    # Application list
    for app in filtered:
        badge = {
            "Pending":  "<span class='badge-pending'>⏳ Pending</span>",
            "Approved": "<span class='badge-approved'>✅ Approved</span>",
            "Rejected": "<span class='badge-rejected'>❌ Rejected</span>",
        }.get(app["status"], app["status"])

        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(f"""
            <div class="app-row">
                <div>
                    <span style='color:#64748B;font-size:11px;font-family:IBM Plex Mono'>#{app['id']}</span>
                    &nbsp;&nbsp;
                    <span style='color:#F8FAFC;font-weight:500'>{app['nic']}</span>
                    &nbsp;&nbsp;
                    {badge}
                </div>
                <div style='display:flex;gap:2rem;align-items:center'>
                    <span style='color:#94A3B8;font-size:13px'>{app['loan_product'].split('—')[0].strip()}</span>
                    <span style='color:#F8FAFC;font-family:IBM Plex Mono;font-size:13px'>{fmt(app['loan_amount'])}</span>
                    <span style='color:#64748B;font-size:12px'>{app['submitted_at']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            if st.button("View →", key=f"view_{app['id']}", use_container_width=True):
                st.session_state.selected_app = app
                st.rerun()


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
if st.session_state.selected_app is not None:
    show_customer_page(st.session_state.selected_app)
else:
    show_dashboard()