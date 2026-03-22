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
# SVG Bank Building Illustration
# ══════════════════════════════════════════════════════════════
BANK_SVG = """
<svg width="160" height="140" viewBox="0 0 160 140" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#dbeafe"/>
      <stop offset="100%" stop-color="#bfdbfe"/>
    </linearGradient>
    <linearGradient id="buildGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1e40af"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="glassGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#93c5fd" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#60a5fa" stop-opacity="0.3"/>
    </linearGradient>
    <linearGradient id="signGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fbbf24"/>
      <stop offset="100%" stop-color="#d97706"/>
    </linearGradient>
  </defs>

  <!-- Sky background -->
  <rect width="160" height="140" rx="16" fill="url(#skyGrad)"/>

  <!-- Main building body -->
  <rect x="20" y="45" width="120" height="80" rx="2" fill="url(#buildGrad)"/>

  <!-- Glass windows grid on building -->
  <rect x="28" y="55" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="50" y="55" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="72" y="55" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="94" y="55" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="116" y="55" width="16" height="12" rx="1" fill="url(#glassGrad)"/>

  <rect x="28" y="73" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="50" y="73" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="72" y="73" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="94" y="73" width="16" height="12" rx="1" fill="url(#glassGrad)"/>
  <rect x="116" y="73" width="16" height="12" rx="1" fill="url(#glassGrad)"/>

  <!-- Pediment / roof triangle -->
  <polygon points="10,45 80,12 150,45" fill="#1e3a8a"/>
  <!-- Roof highlight -->
  <polygon points="10,45 80,14 150,45 148,45 80,16 12,45" fill="#2563eb" opacity="0.4"/>

  <!-- Cornice bar -->
  <rect x="15" y="43" width="130" height="5" rx="1" fill="#1e3a8a"/>

  <!-- BANK sign plate -->
  <rect x="42" y="28" width="76" height="18" rx="3" fill="url(#signGrad)"/>
  <text x="80" y="41" text-anchor="middle" font-family="Georgia, serif" font-weight="700"
        font-size="13" fill="#7c2d12" letter-spacing="3">BANK</text>

  <!-- Columns -->
  <rect x="30" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>
  <rect x="53" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>
  <rect x="97" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>
  <rect x="120" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>

  <!-- Column caps -->
  <rect x="28" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>
  <rect x="51" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>
  <rect x="95" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>
  <rect x="118" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>

  <!-- Column bases -->
  <rect x="27" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>
  <rect x="50" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>
  <rect x="94" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>
  <rect x="117" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>

  <!-- Door -->
  <rect x="67" y="99" width="26" height="26" rx="2" fill="#1e3a8a"/>
  <rect x="69" y="101" width="10" height="22" rx="1" fill="url(#glassGrad)"/>
  <rect x="81" y="101" width="10" height="22" rx="1" fill="url(#glassGrad)"/>
  <!-- Door handle -->
  <circle cx="79" cy="113" r="1.5" fill="#fbbf24"/>
  <circle cx="81" cy="113" r="1.5" fill="#fbbf24"/>

  <!-- Base steps -->
  <rect x="10" y="125" width="140" height="5" rx="1" fill="#1e3a8a" opacity="0.7"/>
  <rect x="5"  y="130" width="150" height="5" rx="1" fill="#1e3a8a" opacity="0.5"/>
  <rect x="0"  y="135" width="160" height="5" rx="0" fill="#1e3a8a" opacity="0.3"/>
</svg>
"""

# ══════════════════════════════════════════════════════════════
# CSS — Blue & Orange Analytics Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background: linear-gradient(150deg, #e8f0fe 0%, #dbeafe 30%, #eff6ff 60%, #fef3e8 85%, #fff7ed 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #0c1a4e !important;
}
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] { display: none; }

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #0c1a4e !important; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.72);
    border: 0.5px solid rgba(30,64,175,0.18);
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s, background 0.2s;
    box-shadow: 0 2px 16px rgba(30,64,175,0.07);
    aspect-ratio: 1 / 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.metric-card:hover {
    border-color: rgba(30,64,175,0.4);
    background: rgba(255,255,255,0.9);
}
.metric-label {
    font-size: 10px;
    color: #1e40af;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
    font-weight: 700;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #1d4ed8;
    font-family: 'DM Mono', monospace;
    line-height: 1.15;
    word-break: break-word;
    text-align: center;
}
.metric-value-lg {
    font-size: 40px !important;
}

/* ── Info cards — equal height via flex ── */
.profile-row {
    display: flex;
    gap: 1.2rem;
    align-items: stretch;
    margin-bottom: 1rem;
}
.info-card {
    background: rgba(255,255,255,0.68);
    border: 0.5px solid rgba(30,64,175,0.15);
    border-radius: 14px;
    padding: 1.4rem;
    flex: 1;
    backdrop-filter: blur(10px);
    box-shadow: 0 2px 12px rgba(30,64,175,0.06);
    min-height: 0;
}
.info-card-title {
    font-size: 10px;
    color: #1e40af;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1rem;
    font-weight: 700;
    border-bottom: 1.5px solid #f97316;
    padding-bottom: 6px;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 0.5px solid rgba(30,64,175,0.08);
    font-size: 14px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #2d4a8a; font-weight: 400; }
.info-val {
    color: #0c1a4e;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    text-align: right;
}

/* ── Badges ── */
.badge-pending  { background:rgba(251,191,36,0.18); color:#92400e; padding:4px 14px; border-radius:20px; font-size:11px; border:1px solid rgba(251,191,36,0.5); font-weight:600; }
.badge-approved { background:rgba(34,197,94,0.14);  color:#14532d; padding:4px 14px; border-radius:20px; font-size:11px; border:1px solid rgba(34,197,94,0.4);  font-weight:600; }
.badge-rejected { background:rgba(239,68,68,0.12);  color:#7f1d1d; padding:4px 14px; border-radius:20px; font-size:11px; border:1px solid rgba(239,68,68,0.35); font-weight:600; }

/* ── Section headers ── */
.section-header {
    font-size: 10px;
    color: #1e40af;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 700;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 8px;
    border-bottom: 1.5px solid #f97316;
    display: inline-block;
    width: 100%;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(255,255,255,0.75) !important;
    color: #1e3a8a !important;
    border: 1px solid rgba(30,64,175,0.3) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    backdrop-filter: blur(4px) !important;
}
.stButton > button:hover {
    background: rgba(30,64,175,0.1) !important;
    border-color: #1e40af !important;
    color: #1e3a8a !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.55) !important;
    border: 0.5px solid rgba(30,64,175,0.18) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
}
div[data-testid="stExpander"] summary { color: #0c1a4e !important; font-weight: 500 !important; }
div[data-testid="stExpander"] summary:hover { color: #1e40af !important; }

/* ── Form inputs ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.75) !important;
    border-color: rgba(30,64,175,0.25) !important;
    color: #0c1a4e !important;
    border-radius: 10px !important;
}
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.75) !important;
    border-color: rgba(30,64,175,0.25) !important;
    color: #0c1a4e !important;
    border-radius: 10px !important;
    -webkit-text-fill-color: #0c1a4e !important;
    caret-color: #0c1a4e !important;
}
label {
    color: #1e40af !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600 !important;
}

/* ── Streamlit metrics ── */
.stMetric {
    background: rgba(255,255,255,0.65);
    border-radius: 12px;
    padding: 0.8rem;
    border: 0.5px solid rgba(30,64,175,0.18);
}
[data-testid="stMetricValue"] { color: #0c1a4e !important; font-family: 'DM Mono', monospace !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #1e40af !important; font-weight: 600 !important; font-size: 11px !important; }

/* ── App list cards ── */
.app-card {
    background: rgba(255,255,255,0.62);
    border: 0.5px solid rgba(30,64,175,0.14);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s, background 0.2s;
    backdrop-filter: blur(8px);
    box-shadow: 0 1px 8px rgba(30,64,175,0.05);
}
.app-card:hover {
    border-color: rgba(30,64,175,0.38);
    background: rgba(255,255,255,0.85);
}

/* ── Alert / info boxes ── */
div[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── Decorative blobs ── */
.deco { position: fixed; border-radius: 50%; pointer-events: none; z-index: 0; }

/* ── Login card ── */
.login-card {
    background: rgba(255,255,255,0.72);
    border: 0.5px solid rgba(30,64,175,0.15);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(30,64,175,0.1);
    text-align: center;
}
.login-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: #0c1a4e;
    margin: 1.2rem 0 0.4rem;
}
.login-sub {
    color: #374151;
    font-size: 14px;
    margin-bottom: 1.8rem;
}

/* ── Orange accent bar ── */
.orange-accent { width: 40px; height: 3px; background: #f97316; border-radius: 2px; margin: 0.5rem auto 1.2rem; }
</style>

<!-- Decorative gradient blobs -->
<div class="deco" style="top:-140px;right:-140px;width:520px;height:520px;background:radial-gradient(circle,rgba(147,197,253,0.3) 0%,transparent 70%)"></div>
<div class="deco" style="bottom:-100px;left:-80px;width:420px;height:420px;background:radial-gradient(circle,rgba(249,115,22,0.15) 0%,transparent 70%)"></div>
<div class="deco" style="top:50%;left:55%;width:280px;height:280px;background:radial-gradient(circle,rgba(59,130,246,0.1) 0%,transparent 70%)"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "officer_name"    not in st.session_state: st.session_state.officer_name    = ""
if "selected_app"    not in st.session_state: st.session_state.selected_app    = None
if "loan_detail_acc" not in st.session_state: st.session_state.loan_detail_acc = None

# ══════════════════════════════════════════════════════════════
# PLOTLY THEME — Blue / Orange
# ══════════════════════════════════════════════════════════════
PLOT_BG   = 'rgba(0,0,0,0)'
GRID_COL  = 'rgba(30,64,175,0.1)'
FONT_COL  = '#374151'
LINE_A    = '#1d4ed8'   # blue
LINE_B    = '#f97316'   # orange
LINE_C    = '#60a5fa'   # light blue
FILL_A    = 'rgba(29,78,216,0.08)'
FILL_B    = 'rgba(249,115,22,0.08)'
LEGEND_BG = 'rgba(255,255,255,0.65)'
LEGEND_BD = 'rgba(30,64,175,0.2)'

# ══════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════
if not st.session_state.officer_name:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(f"""
        <div class="login-card">
            <div style="display:flex;justify-content:center">{BANK_SVG}</div>
            <div class="orange-accent"></div>
            <div class="login-title">Officer Portal</div>
            <div class="login-sub" style="color:#2d4a8a">Internal loan management system</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
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
        <div style='font-size:10px;color:#1e40af;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;font-weight:700'>
            Loan Repayment Detail
        </div>
        <h1 style='font-size:26px;margin:0;color:#0c1a4e'>{product}</h1>
        <p style='color:#2d4a8a;font-size:13px;margin-top:4px'>
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

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Capital Paid</div>
            <div class="metric-value" style="font-size:20px">{fmt(acc_rep['CAPITAL_PAIED'].sum())}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Interest Paid</div>
            <div class="metric-value" style="font-size:20px;color:#f97316">{fmt(acc_rep['INTEREST_PAIED'].sum())}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Paid</div>
            <div class="metric-value" style="font-size:20px;color:#1d4ed8">{fmt(acc_rep['TOTAL_PAID'].sum())}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">No. of Payments</div>
            <div class="metric-value" style="font-size:20px">{len(acc_rep)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Capital & Interest Repayments Over Time</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['CAPITAL_PAIED'],
        name='Capital Paid', mode='lines+markers',
        line=dict(color=LINE_A, width=2), marker=dict(size=5),
        fill='tozeroy', fillcolor=FILL_A
    ))
    fig.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['INTEREST_PAIED'],
        name='Interest Paid', mode='lines+markers',
        line=dict(color=LINE_B, width=2), marker=dict(size=5),
        fill='tozeroy', fillcolor=FILL_B
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL, family='DM Sans'),
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor=GRID_COL, title='Payment Date', color=FONT_COL),
        yaxis=dict(gridcolor=GRID_COL, title='Amount (LKR)', color=FONT_COL),
        legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BD, font=dict(color='#0c1a4e'))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Cumulative Repayment Progress</div>', unsafe_allow_html=True)
    acc_rep['Cumulative_Capital']  = acc_rep['CAPITAL_PAIED'].cumsum()
    acc_rep['Cumulative_Interest'] = acc_rep['INTEREST_PAIED'].cumsum()
    acc_rep['Cumulative_Total']    = acc_rep['TOTAL_PAID'].cumsum()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['Cumulative_Total'],
        name='Total Cumulative', mode='lines',
        line=dict(color=LINE_A, width=3)
    ))
    fig2.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['Cumulative_Capital'],
        name='Capital Cumulative', mode='lines',
        line=dict(color=LINE_C, width=2, dash='dot')
    ))
    fig2.add_trace(go.Scatter(
        x=acc_rep['PAYMENT_DATE'], y=acc_rep['Cumulative_Interest'],
        name='Interest Cumulative', mode='lines',
        line=dict(color=LINE_B, width=2, dash='dot')
    ))
    fig2.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL, family='DM Sans'),
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor=GRID_COL, color=FONT_COL),
        yaxis=dict(gridcolor=GRID_COL, title='Cumulative Amount (LKR)', color=FONT_COL),
        legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BD, font=dict(color='#0c1a4e'))
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Repayment Records</div>', unsafe_allow_html=True)
    display_df = acc_rep[['PAYMENT_DATE', 'CAPITAL_PAIED', 'INTEREST_PAIED', 'TOTAL_PAID']].copy()
    display_df['PAYMENT_DATE'] = display_df['PAYMENT_DATE'].dt.strftime('%Y-%m-%d')
    display_df.columns = ['Payment Date', 'Capital Paid (LKR)', 'Interest Paid (LKR)', 'Total Paid (LKR)']
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# CUSTOMER DETAIL PAGE
# ══════════════════════════════════════════════════════════════
def show_customer_page(app):
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
        <div style='font-size:10px;color:#1e40af;letter-spacing:3px;text-transform:uppercase;
                    margin-bottom:8px;font-weight:700'>
            Application #{app['id']} &nbsp;·&nbsp; {badge}
        </div>
        <h1 style='font-size:28px;margin:0;color:#0c1a4e'>Customer Profile</h1>
        <div style='width:48px;height:3px;background:#f97316;border-radius:2px;margin-top:8px'></div>
    </div>
    """, unsafe_allow_html=True)

    cust = eligible_customers[eligible_customers["MASKED_LEGAL_ID"] == nic]
    if cust.empty:
        st.error("Customer record not found.")
        return
    c = cust.iloc[0]
    masked_id = c.get("MASKED_ID", "")

    cust_accounts     = account_df[account_df["MASKED_ID"] == masked_id] if masked_id else pd.DataFrame()
    cust_repayments   = repayment_df[repayment_df["MASKED_ID"] == masked_id].copy() if masked_id else pd.DataFrame()
    cust_transactions = transaction_df[transaction_df["MASKED_ID"] == masked_id].copy() if masked_id else pd.DataFrame()

    # ── Top metric cards ─────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        score_val   = int(float(c.get('Internal_Bank_Default_Score', 0)))
        score_color = "#1d4ed8" if score_val >= 650 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Internal Score</div>
            <div class="metric-value metric-value-lg" style="color:{score_color}">{score_val}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        cluster_raw = str(c.get('Cluster_Name', c.get('Cluster_KProto', 'N/A')))
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Cluster</div>
            <div class="metric-value" style="font-size:18px">{cluster_raw}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Score Band</div>
            <div class="metric-value" style="font-size:18px">{c.get('Score_Band', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Monthly Income</div>
            <div class="metric-value" style="font-size:18px">{fmt(c.get('Avg_Monthly_Credit', 0))}</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        ood_val   = int(float(c.get('MAX_OOD', 0)))
        ood_color = "#dc2626" if ood_val >= 30 else "#1d4ed8"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Max Days Overdue</div>
            <div class="metric-value metric-value-lg" style="color:{ood_color}">{ood_val}</div>
        </div>""", unsafe_allow_html=True)
    with m6:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Active Accounts</div>
            <div class="metric-value metric-value-lg">{int(c.get('Number_of_Active_Accounts', 0))}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ══ THREE INFO CARDS — equal height via flex row ══════════
    # Personal has 8 rows, Risk has 7 rows, Application has 9 rows
    # We force all three to the same row count with a spacer row in Risk card
    st.markdown(f"""
    <div class="profile-row">

      <!-- PERSONAL INFORMATION -->
      <div class="info-card">
        <div class="info-card-title">Personal Information</div>
        <div class="info-row"><span class="info-key">NIC</span><span class="info-val">{nic}</span></div>
        <div class="info-row"><span class="info-key">Age</span><span class="info-val">{int(c.get('AGE', 0))}</span></div>
        <div class="info-row"><span class="info-key">Gender</span><span class="info-val">{str(c.get('GENDER', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Marital status</span><span class="info-val">{str(c.get('MARITAL_STATUS', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">District</span><span class="info-val">{str(c.get('DISTRICT', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Occupation</span><span class="info-val">{str(c.get('OCCUPATION', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Employment</span><span class="info-val">{str(c.get('EMPLOYMENT_STATUS', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Segment</span><span class="info-val">{str(c.get('Employment_Segment', 'N/A'))}</span></div>
      </div>

      <!-- RISK PROFILE — 7 rows, no monthly salary, salary band from cluster -->
      <div class="info-card">
        <div class="info-card-title">Risk Profile</div>
        <div class="info-row"><span class="info-key">Customer risk</span><span class="info-val">{str(c.get('CUSTOMER_RISK_NAME', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Target tier</span><span class="info-val">{c.get('TARGET_DESC', 'N/A')}</span></div>
        <div class="info-row"><span class="info-key">Financial capacity</span><span class="info-val">{c.get('Financial_Capacity', 'N/A')}</span></div>
        <div class="info-row"><span class="info-key">Cluster</span><span class="info-val">{c.get('Cluster_Name', 'N/A')}</span></div>
        <div class="info-row"><span class="info-key">Age bucket</span><span class="info-val">{c.get('Age_Bucket', 'N/A')}</span></div>
        <div class="info-row"><span class="info-key">Existing debt</span><span class="info-val">{fmt(c.get('TOTAL_CAPITAL_DUE', 0))}</span></div>
        <div class="info-row"><span class="info-key">Salary band</span><span class="info-val">{str(c.get('Salary_Band', c.get('salary_band', c.get('SALARY_BAND', c.get('Cluster_Name', 'N/A')))))}</span></div>
        <div class="info-row"><span class="info-key">Score band</span><span class="info-val">{c.get('Score_Band', 'N/A')}</span></div>
      </div>

      <!-- THIS APPLICATION (badge + 8 rows) -->
      <div class="info-card">
        <div class="info-card-title">This Application</div>
        <div style='margin-bottom:12px'>{badge}</div>
        <div class="info-row"><span class="info-key">Product</span><span class="info-val" style="font-size:11px">{app['loan_product'].split('—')[0].strip()}</span></div>
        <div class="info-row"><span class="info-key">Amount</span><span class="info-val">{fmt(app['loan_amount'])}</span></div>
        <div class="info-row"><span class="info-key">Term</span><span class="info-val">{app['loan_term']} months</span></div>
        <div class="info-row"><span class="info-key">Rate</span><span class="info-val">{app['loan_rate']}% p.a.</span></div>
        <div class="info-row"><span class="info-key">Monthly EMI</span><span class="info-val">{fmt(app['loan_emi'])}</span></div>
        <div class="info-row"><span class="info-key">Total interest</span><span class="info-val">{fmt(app['total_interest'])}</span></div>
        <div class="info-row"><span class="info-key">Total repayment</span><span class="info-val">{fmt(app['total_repayment'])}</span></div>
        <div class="info-row"><span class="info-key">Submitted</span><span class="info-val">{app['submitted_at']}</span></div>
      </div>

    </div>
    """, unsafe_allow_html=True)

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
                    lc1, lc2, lc3 = st.columns(3)
                    with lc1: st.metric("Balance", fmt(balance))
                    with lc2: st.metric("Term", str(term))
                    with lc3: st.metric("Status", status)
                    if st.button("View repayment history →", key=f"repay_{acc.get('ACC_MASKED_ID', '')}"):
                        st.session_state.loan_detail_acc = acc.to_dict()
                        st.rerun()
                else:
                    lc1, lc2 = st.columns(2)
                    with lc1: st.metric("Balance", fmt(balance))
                    with lc2: st.metric("Status", status)
    else:
        st.info("No account records found for this customer.")

    # ── Average Monthly Balance ──────────────────────────────
    balance_cols  = ['JUN_25', 'JUL_25', 'AUG_25', 'SEP_25', 'OCT_25', 'NOV_25']
    existing_cols = [col for col in balance_cols if col in account_df.columns]

    if not cust_accounts.empty and existing_cols:
        st.markdown('<div class="section-header">Average Monthly Balance Trend</div>', unsafe_allow_html=True)
        monthly_avg  = cust_accounts[existing_cols].mean()
        month_labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'][:len(existing_cols)]

        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(
            x=month_labels, y=monthly_avg.values,
            mode='lines+markers', name='Avg Balance',
            line=dict(color=LINE_A, width=3),
            marker=dict(size=8, color=LINE_A),
            fill='tozeroy', fillcolor=FILL_A
        ))
        fig_bal.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COL, family='DM Sans'),
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor=GRID_COL, color=FONT_COL),
            yaxis=dict(gridcolor=GRID_COL, title='Average Balance (LKR)', color=FONT_COL)
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    # ── Transaction Activity ─────────────────────────────────
    if not cust_transactions.empty and 'BOOKING_DATE' in cust_transactions.columns:
        st.markdown('<div class="section-header">Transaction Activity</div>', unsafe_allow_html=True)
        txn = cust_transactions.copy()
        txn['BOOKING_DATE'] = pd.to_datetime(txn['BOOKING_DATE'], errors='coerce')
        txn = txn.dropna(subset=['BOOKING_DATE'])
        txn['Month']   = txn['BOOKING_DATE'].dt.to_period('M').astype(str)
        txn['INFLOW']  = txn['AMOUNT_LCY'].apply(lambda x: x if x > 0 else 0)
        txn['OUTFLOW'] = txn['AMOUNT_LCY'].apply(lambda x: abs(x) if x < 0 else 0)
        monthly_txn    = txn.groupby('Month').agg(Inflow=('INFLOW', 'sum'), Outflow=('OUTFLOW', 'sum')).reset_index()

        fig_txn = go.Figure()
        fig_txn.add_trace(go.Scatter(
            x=monthly_txn['Month'], y=monthly_txn['Inflow'],
            name='Inflow', mode='lines+markers',
            line=dict(color=LINE_A, width=2), marker=dict(size=5),
            fill='tozeroy', fillcolor=FILL_A
        ))
        fig_txn.add_trace(go.Scatter(
            x=monthly_txn['Month'], y=monthly_txn['Outflow'],
            name='Outflow', mode='lines+markers',
            line=dict(color=LINE_B, width=2), marker=dict(size=5),
            fill='tozeroy', fillcolor=FILL_B
        ))
        fig_txn.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COL, family='DM Sans'),
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor=GRID_COL, color=FONT_COL),
            yaxis=dict(gridcolor=GRID_COL, title='Amount (LKR)', color=FONT_COL),
            legend=dict(bgcolor=LEGEND_BG, bordercolor=LEGEND_BD, font=dict(color='#0c1a4e'))
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
            <div class="info-row"><span class="info-key">Reviewed by</span><span class="info-val">{app.get('reviewed_by', 'N/A')}</span></div>
            <div class="info-row"><span class="info-key">Reviewed at</span><span class="info-val">{app.get('reviewed_at', 'N/A')}</span></div>
            <div class="info-row"><span class="info-key">Notes</span><span class="info-val">{app.get('officer_notes', 'N/A')}</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════
def show_dashboard():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <h1 style='font-size:26px;margin-bottom:2px;color:#0c1a4e'>Loan Officer Dashboard</h1>
        <div style='width:48px;height:3px;background:#f97316;border-radius:2px;margin-bottom:4px'></div>
        """, unsafe_allow_html=True)
        st.caption(f"Signed in as **{st.session_state.officer_name}**")
    with col2:
        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            st.session_state.officer_name = ""
            st.rerun()

    st.markdown("<hr style='border-color:rgba(30,64,175,0.15);margin:1rem 0'>", unsafe_allow_html=True)

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
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Applications</div>
            <div class="metric-value">{total}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Pending Review</div>
            <div class="metric-value" style="color:#92400e">{pending}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Approved</div>
            <div class="metric-value" style="color:#14532d">{approved}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Rejected</div>
            <div class="metric-value" style="color:#7f1d1d">{rejected}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    status_filter = st.selectbox("Filter by status", ["All", "Pending", "Approved", "Rejected"])
    filtered = applications if status_filter == "All" else [
        a for a in applications if a["status"] == status_filter
    ]

    st.markdown(f"<p style='color:#1e40af;font-size:12px;letter-spacing:1px;font-weight:600'>{len(filtered)} APPLICATION(S)</p>", unsafe_allow_html=True)

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
                        <span style='color:#1e40af;font-size:11px;font-family:DM Mono;font-weight:700'>#{app['id']}</span>
                        <span style='color:#0c1a4e;font-weight:700'>{app['nic']}</span>
                        {badge}
                    </div>
                    <div style='display:flex;gap:2rem;align-items:center'>
                        <span style='color:#2d4a8a;font-size:13px'>{app['loan_product'].split('—')[0].strip()}</span>
                        <span style='color:#0c1a4e;font-family:DM Mono;font-size:14px;font-weight:700'>{fmt(app['loan_amount'])}</span>
                        <span style='color:#3b5fad;font-size:12px'>{app['submitted_at']}</span>
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