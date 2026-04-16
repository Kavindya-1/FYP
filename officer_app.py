import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
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

def safe_float(val, default=0.0):
    """Safely convert any value to float, returning default on failure."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

def safe_int(val, default=0):
    """Safely convert any value to int, returning default on failure."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default

# ══════════════════════════════════════════════════════════════
# EMI RATIO HELPER — always computed from live data
# ══════════════════════════════════════════════════════════════
def compute_emi_ratio(app, customer_row):
    """
    Returns (emi_pct: float|None, exceeds: bool).
    emi_pct is None when salary is zero/missing.
    """
    try:
        emi_val    = safe_float(app.get("loan_emi"))
        salary_val = safe_float(customer_row.get("Avg_Monthly_Credit"))
        if emi_val > 0 and salary_val > 0:
            pct = round((emi_val / salary_val) * 100, 1)
            return pct, pct > 40.0
        elif emi_val > 0 and salary_val == 0:
            return None, True   # can't confirm affordability
        return 0.0, False
    except (TypeError, ValueError):
        return None, False


# ══════════════════════════════════════════════════════════════
# INSUFFICIENT INFORMATION DETECTION
# ══════════════════════════════════════════════════════════════
def get_data_quality_flags(nic, app=None):
    flags = []

    row = eligible_customers[eligible_customers["MASKED_LEGAL_ID"] == nic]
    if row.empty:
        return {"has_issues": True, "flags": [("Customer record not found", "NIC not in eligible_customers")], "severity": "critical"}

    c = row.iloc[0]

    # 1. Thin File
    if safe_int(c.get("Thin_File_Flag")) == 1:
        flags.append(("Thin File", "Customer has limited or no financial history"))

    # 2. Missing / Unknown Financial Capacity
    fin_cap = str(c.get("Financial_Capacity", "")).strip()
    if fin_cap in ("Unknown / Missing Balance Data", "", "nan"):
        flags.append(("Unknown Financial Capacity", "No balance data available to assess financial capacity"))

    # 3. Unknown Customer Risk
    cust_risk = str(c.get("CUSTOMER_RISK_NAME", "")).strip()
    if cust_risk in ("Unknown", "", "nan"):
        flags.append(("Unknown Customer Risk", "Customer risk rating could not be determined"))

    # 4. Unknown Target Tier
    target = str(c.get("TARGET_DESC", "")).strip()
    if target in ("Unknown", "", "nan"):
        flags.append(("Unknown Target Tier", "Customer has not been assigned a target/tier segment"))

    # 5. Invalid / Weak Employment Segment
    emp_seg = str(c.get("Employment_Segment", "")).strip()
    if emp_seg in ("Not valid segment", "", "nan"):
        flags.append(("Invalid Employment Segment", "Employment segment could not be validated"))
    elif emp_seg == "Other":
        flags.append(("Unclassified Employment Segment", "Employment segment is 'Other' — reduced scoring factor (0.6x)"))

    # 6. Missing / zero NET_RATIO (no transaction data)
    net_ratio  = c.get("NET_RATIO", None)
    avg_credit = safe_float(c.get("Avg_Monthly_Credit"))
    try:
        net_ratio_val = float(net_ratio)
    except (TypeError, ValueError):
        net_ratio_val = None

    if net_ratio_val is None or (net_ratio_val == 0.0 and avg_credit == 0.0):
        flags.append(("No Transaction Data", "NET_RATIO and Avg_Monthly_Credit are both zero or missing — income unverifiable"))
    elif net_ratio_val < 0:
        flags.append(("Negative Net Ratio", f"Customer spending exceeds income (NET_RATIO = {net_ratio_val:.3f}) — 0.80x penalty applied"))

    # 7. Missing internal score
    score = safe_float(c.get("Internal_Bank_Default_Score"))
    if score == 0 or pd.isna(score):
        flags.append(("Missing Credit Score", "Internal bank default score is zero or missing"))

    # 8. EMI vs salary — ALWAYS computed from live data, never just from stored flag
    if app:
        emi_pct, emi_exceeds = compute_emi_ratio(app, c)
        if emi_exceeds:
            high_emi_stored = app.get("high_emi_flag", False)
            if emi_pct is None:
                detail = (
                    "Monthly repayment cannot be confirmed — no salary on record. "
                    "Affordability cannot be assessed. Manual verification required."
                )
            elif high_emi_stored:
                detail = (
                    f"Monthly repayment is {emi_pct}% of salary — far exceeds the 40% cap. "
                    f"Customer was warned at submission and chose to proceed anyway. Extreme repayment risk."
                )
            else:
                detail = (
                    f"Monthly repayment is {emi_pct}% of salary — exceeds the 40% threshold. "
                    f"This was not caught at submission. Immediate manual review required."
                )
            flags.append(("EMI Exceeds Salary Threshold", detail))

    # 9. Overdue history present
    max_ood = safe_float(c.get("MAX_OOD"))
    if max_ood >= 30:
        flags.append(("Overdue History", f"Customer has {int(max_ood)} days max overdue — penalty applied in eligibility calculation"))

    # 10. No active accounts
    active_accounts = safe_int(c.get("Number_of_Active_Accounts"))
    if active_accounts == 0:
        flags.append(("No Active Accounts", "Customer has no active accounts linked to NIC"))

    # ── Determine overall severity ──────────────────────────
    critical_labels = {
        "Thin File", "No Transaction Data", "Missing Credit Score",
        "Customer record not found", "No Active Accounts",
        "EMI Exceeds Salary Threshold",
    }
    warning_labels = {
        "Unknown Financial Capacity", "Unknown Customer Risk",
        "Unknown Target Tier", "Invalid Employment Segment",
        "Overdue History", "Negative Net Ratio", "Unclassified Employment Segment",
    }

    has_critical = any(label in critical_labels for label, _ in flags)
    has_warning  = any(label in warning_labels  for label, _ in flags)

    severity = "critical" if has_critical else ("warning" if has_warning else "info")

    return {
        "has_issues": len(flags) > 0,
        "flags":      flags,
        "severity":   severity,
    }


def get_thin_flag(nic):
    row = eligible_customers[eligible_customers["MASKED_LEGAL_ID"] == nic]
    if row.empty:
        return False
    return safe_int(row.iloc[0].get("Thin_File_Flag")) == 1


# ══════════════════════════════════════════════════════════════
# BADGE BUILDER
# ══════════════════════════════════════════════════════════════
def build_alert_badges(is_thin, dq, inline=True):
    badges = []

    if is_thin:
        badges.append(
            "<span style='background:rgba(249,115,22,0.15);color:#92400e;padding:3px 12px;"
            "border-radius:20px;font-size:11px;border:1px solid rgba(249,115,22,0.5);font-weight:600;'>"
            "⚠️ Thin File</span>"
        )

    if dq["has_issues"]:
        critical_labels = {
            "Thin File", "No Transaction Data", "Missing Credit Score",
            "Customer record not found", "No Active Accounts",
            "EMI Exceeds Salary Threshold",
        }
        critical_reasons = [label for label, _ in dq["flags"] if label in critical_labels]

        if dq["severity"] == "critical":
            reason_parts = []
            if "Thin File" in critical_reasons:
                reason_parts.append("Thin File")
            if "EMI Exceeds Salary Threshold" in critical_reasons:
                reason_parts.append("EMI Exceeds 40%")
            if "No Transaction Data" in critical_reasons:
                reason_parts.append("No Transaction Data")
            if "Missing Credit Score" in critical_reasons:
                reason_parts.append("No Credit Score")
            if "No Active Accounts" in critical_reasons:
                reason_parts.append("No Active Accounts")
            if "Customer record not found" in critical_reasons:
                reason_parts.append("Record Not Found")

            if is_thin and "Thin File" in reason_parts:
                reason_parts.remove("Thin File")

            label = "🔴 Critical" + (f": {' · '.join(reason_parts)}" if reason_parts else "")
            badges.append(
                f"<span style='background:rgba(220,38,38,0.15);color:#7f1d1d;padding:3px 12px;"
                f"border-radius:20px;font-size:11px;border:1px solid rgba(220,38,38,0.45);font-weight:600;'>"
                f"{label}</span>"
            )
        elif dq["severity"] == "warning":
            badges.append(
                "<span style='background:rgba(251,191,36,0.18);color:#78350f;padding:3px 12px;"
                "border-radius:20px;font-size:11px;border:1px solid rgba(251,191,36,0.5);font-weight:600;'>"
                "🟠 Insufficient Info</span>"
            )

    return " ".join(badges)


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
  <rect width="160" height="140" rx="16" fill="url(#skyGrad)"/>
  <rect x="20" y="45" width="120" height="80" rx="2" fill="url(#buildGrad)"/>
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
  <polygon points="10,45 80,12 150,45" fill="#1e3a8a"/>
  <polygon points="10,45 80,14 150,45 148,45 80,16 12,45" fill="#2563eb" opacity="0.4"/>
  <rect x="15" y="43" width="130" height="5" rx="1" fill="#1e3a8a"/>
  <rect x="42" y="28" width="76" height="18" rx="3" fill="url(#signGrad)"/>
  <text x="80" y="41" text-anchor="middle" font-family="Georgia, serif" font-weight="700"
        font-size="13" fill="#7c2d12" letter-spacing="3">BANK</text>
  <rect x="30" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>
  <rect x="53" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>
  <rect x="97" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>
  <rect x="120" y="45" width="10" height="45" rx="2" fill="#dbeafe" opacity="0.9"/>
  <rect x="28" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>
  <rect x="51" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>
  <rect x="95" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>
  <rect x="118" y="44" width="14" height="4" rx="1" fill="#93c5fd"/>
  <rect x="27" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>
  <rect x="50" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>
  <rect x="94" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>
  <rect x="117" y="86" width="16" height="4" rx="1" fill="#93c5fd"/>
  <rect x="67" y="99" width="26" height="26" rx="2" fill="#1e3a8a"/>
  <rect x="69" y="101" width="10" height="22" rx="1" fill="url(#glassGrad)"/>
  <rect x="81" y="101" width="10" height="22" rx="1" fill="url(#glassGrad)"/>
  <circle cx="79" cy="113" r="1.5" fill="#fbbf24"/>
  <circle cx="81" cy="113" r="1.5" fill="#fbbf24"/>
  <rect x="10" y="125" width="140" height="5" rx="1" fill="#1e3a8a" opacity="0.7"/>
  <rect x="5"  y="130" width="150" height="5" rx="1" fill="#1e3a8a" opacity="0.5"/>
  <rect x="0"  y="135" width="160" height="5" rx="0" fill="#1e3a8a" opacity="0.3"/>
</svg>
"""

# ══════════════════════════════════════════════════════════════
# CSS
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

.metric-card {
    background: rgba(255,255,255,0.72);
    border: 0.5px solid rgba(30,64,175,0.18);
    border-radius: 14px;
    padding: 1.6rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s, background 0.2s;
    box-shadow: 0 2px 16px rgba(30,64,175,0.07);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 110px;
}
.metric-card:hover { border-color: rgba(30,64,175,0.4); background: rgba(255,255,255,0.9); }
.metric-label { font-size: 10px; color: #1e40af; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; font-weight: 700; }
.metric-value { font-size: 20px; font-weight: 700; color: #1d4ed8; font-family: 'DM Mono', monospace; line-height: 1.2; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }

.profile-row { display: flex; gap: 1.2rem; align-items: stretch; margin-bottom: 1rem; }
.info-card { background: rgba(255,255,255,0.68); border: 0.5px solid rgba(30,64,175,0.15); border-radius: 14px; padding: 1.4rem; flex: 1; backdrop-filter: blur(10px); box-shadow: 0 2px 12px rgba(30,64,175,0.06); min-height: 0; }
.info-card-title { font-size: 10px; color: #1e40af; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 1rem; font-weight: 700; border-bottom: 1.5px solid #f97316; padding-bottom: 6px; }
.info-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 0.5px solid rgba(30,64,175,0.08); font-size: 14px; }
.info-row:last-child { border-bottom: none; }
.info-key { color: #2d4a8a; font-weight: 400; }
.info-val { color: #0c1a4e; font-weight: 700; font-family: 'DM Mono', monospace; font-size: 14px; text-align: right; }

.badge-pending   { background:rgba(251,191,36,0.18); color:#92400e; padding:4px 14px; border-radius:20px; font-size:11px; border:1px solid rgba(251,191,36,0.5); font-weight:600; }
.badge-approved  { background:rgba(34,197,94,0.14);  color:#14532d; padding:4px 14px; border-radius:20px; font-size:11px; border:1px solid rgba(34,197,94,0.4);  font-weight:600; }
.badge-rejected  { background:rgba(239,68,68,0.12);  color:#7f1d1d; padding:4px 14px; border-radius:20px; font-size:11px; border:1px solid rgba(239,68,68,0.35); font-weight:600; }

.section-header { font-size: 10px; color: #1e40af; text-transform: uppercase; letter-spacing: 3px; font-weight: 700; margin: 1.8rem 0 1rem 0; padding-bottom: 8px; border-bottom: 1.5px solid #f97316; display: inline-block; width: 100%; }

.stButton > button { background: rgba(255,255,255,0.75) !important; color: #1e3a8a !important; border: 1px solid rgba(30,64,175,0.3) !important; border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; transition: all 0.2s !important; backdrop-filter: blur(4px) !important; }
.stButton > button:hover { background: rgba(30,64,175,0.1) !important; border-color: #1e40af !important; color: #1e3a8a !important; }

div[data-testid="stExpander"] { background: rgba(255,255,255,0.55) !important; border: 0.5px solid rgba(30,64,175,0.18) !important; border-radius: 12px !important; backdrop-filter: blur(8px) !important; }
div[data-testid="stExpander"] summary { color: #0c1a4e !important; font-weight: 500 !important; }
div[data-testid="stExpander"] summary:hover { color: #1e40af !important; }

.stSelectbox > div > div { background: rgba(255,255,255,0.75) !important; border-color: rgba(30,64,175,0.25) !important; color: #0c1a4e !important; border-radius: 10px !important; }
.stTextInput input, .stTextArea textarea { background: rgba(255,255,255,0.75) !important; border-color: rgba(30,64,175,0.25) !important; color: #0c1a4e !important; border-radius: 10px !important; -webkit-text-fill-color: #0c1a4e !important; caret-color: #0c1a4e !important; }
label { color: #1e40af !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 600 !important; }

.stMetric { background: rgba(255,255,255,0.65); border-radius: 12px; padding: 0.8rem; border: 0.5px solid rgba(30,64,175,0.18); }
[data-testid="stMetricValue"] { color: #0c1a4e !important; font-family: 'DM Mono', monospace !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #1e40af !important; font-weight: 600 !important; font-size: 11px !important; }

.app-card { background: rgba(255,255,255,0.62); border: 0.5px solid rgba(30,64,175,0.14); border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 0.5rem; transition: border-color 0.2s, background 0.2s; backdrop-filter: blur(8px); box-shadow: 0 1px 8px rgba(30,64,175,0.05); }
.app-card:hover { border-color: rgba(30,64,175,0.38); background: rgba(255,255,255,0.85); }
.app-card-warn { background: rgba(255,247,237,0.75); border: 0.5px solid rgba(249,115,22,0.35); border-left: 3px solid #f97316 !important; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 0.5rem; transition: border-color 0.2s; backdrop-filter: blur(8px); box-shadow: 0 1px 8px rgba(249,115,22,0.08); }
.app-card-warn:hover { background: rgba(255,247,237,0.95); }
.app-card-critical { background: rgba(254,242,242,0.8); border: 0.5px solid rgba(220,38,38,0.35); border-left: 3px solid #dc2626 !important; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 0.5rem; transition: border-color 0.2s; backdrop-filter: blur(8px); box-shadow: 0 1px 8px rgba(220,38,38,0.08); }
.app-card-critical:hover { background: rgba(254,242,242,0.95); }

div[data-testid="stAlert"] { border-radius: 12px !important; }
.deco { position: fixed; border-radius: 50%; pointer-events: none; z-index: 0; }

.login-card { background: rgba(255,255,255,0.72); border: 0.5px solid rgba(30,64,175,0.15); border-radius: 20px; padding: 3rem 2.5rem 2.5rem; backdrop-filter: blur(12px); box-shadow: 0 8px 40px rgba(30,64,175,0.1); text-align: center; }
.login-title { font-family: 'DM Serif Display', serif; font-size: 26px; color: #0c1a4e; margin: 1.2rem 0 0.4rem; }
.login-sub { color: #374151; font-size: 14px; margin-bottom: 1.8rem; }
.orange-accent { width: 40px; height: 3px; background: #f97316; border-radius: 2px; margin: 0.5rem auto 1.2rem; }
</style>

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
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════
PLOT_BG   = 'rgba(0,0,0,0)'
GRID_COL  = 'rgba(30,64,175,0.1)'
FONT_COL  = '#374151'
LINE_A    = '#1d4ed8'
LINE_B    = '#f97316'
LINE_C    = '#60a5fa'
FILL_A    = 'rgba(29,78,216,0.08)'
FILL_B    = 'rgba(249,115,22,0.08)'
LEGEND_BG = 'rgba(255,255,255,0.65)'
LEGEND_BD = 'rgba(30,64,175,0.2)'

# ══════════════════════════════════════════════════════════════
# DATA QUALITY BANNER RENDERER
# ══════════════════════════════════════════════════════════════
def render_dq_banner(dq):
    if not dq["has_issues"]:
        return

    sev = dq["severity"]

    sev_cfg = {
        "critical": {
            "border_left": "#dc2626", "bg": "rgba(220,38,38,0.07)", "border": "rgba(220,38,38,0.4)",
            "icon": "🔴", "title": "Critical — Manual Review Required",
            "title_color": "#7f1d1d", "label_color": "#b91c1c", "detail_color": "#991b1b",
            "row_border": "rgba(220,38,38,0.1)",
        },
        "warning": {
            "border_left": "#f97316", "bg": "rgba(249,115,22,0.07)", "border": "rgba(249,115,22,0.4)",
            "icon": "🟠", "title": "Warning — Additional Verification Needed",
            "title_color": "#92400e", "label_color": "#c2410c", "detail_color": "#9a3412",
            "row_border": "rgba(249,115,22,0.1)",
        },
        "info": {
            "border_left": "#3b82f6", "bg": "rgba(59,130,246,0.06)", "border": "rgba(59,130,246,0.3)",
            "icon": "🔵", "title": "Notice — Minor Data Gaps Detected",
            "title_color": "#1e40af", "label_color": "#1d4ed8", "detail_color": "#2563eb",
            "row_border": "rgba(59,130,246,0.1)",
        },
    }
    cfg = sev_cfg.get(sev, sev_cfg["warning"])

    flag_rows = ""
    for label, detail in dq["flags"]:
        flag_rows += f"""
        <div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;
                    border-bottom:0.5px solid {cfg['row_border']};">
            <span style="font-size:13px;flex-shrink:0;">⚠️</span>
            <span style="font-weight:700;min-width:220px;flex-shrink:0;
                         color:{cfg['label_color']};font-size:13px;">{label}</span>
            <span style="color:{cfg['detail_color']};font-size:13px;line-height:1.5;">{detail}</span>
        </div>"""

    n_flags   = len(dq["flags"])
    row_h     = 42
    header_h  = 48
    banner_h  = header_h + n_flags * row_h + 24

    html = f"""<!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>body{{margin:0;padding:0;background:transparent;font-family:'DM Sans',sans-serif;}}</style>
    </head><body>
    <div style="background:{cfg['bg']};border:1px solid {cfg['border']};
                border-left:4px solid {cfg['border_left']};border-radius:10px;
                padding:14px 18px;">
        <div style="font-size:14px;font-weight:700;color:{cfg['title_color']};margin-bottom:10px;">
            {cfg['icon']} &nbsp; Requires Officer Attention &nbsp;·&nbsp; {cfg['title']}
        </div>
        {flag_rows}
    </div>
    </body></html>"""

    components.html(html, height=banner_h, scrolling=False)


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

    st.markdown('<div class="section-header">Repayment Records</div>', unsafe_allow_html=True)

    total_capital_paid   = acc_rep['CAPITAL_PAIED'].sum()
    total_interest_paid  = acc_rep['INTEREST_PAIED'].sum()
    total_paid           = acc_rep['TOTAL_PAID'].sum()
    acc_balance          = safe_float(st.session_state.loan_detail_acc.get('MONTHEND_CONVERTED_BALANCE'))
    remaining_capital    = max(acc_balance, 0)

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(f"""<div class="metric-card" style="background:rgba(29,78,216,0.04);border-color:rgba(29,78,216,0.2)">
            <div class="metric-label">Total Scheduled</div>
            <div class="metric-value" style="color:#3b5fad">{fmt(total_capital_paid + total_interest_paid + remaining_capital)}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="metric-card" style="background:rgba(29,78,216,0.07);border-color:rgba(29,78,216,0.25)">
            <div class="metric-label">Total Capital Paid</div>
            <div class="metric-value">{fmt(total_capital_paid)}</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="metric-card" style="background:rgba(249,115,22,0.07);border-color:rgba(249,115,22,0.25)">
            <div class="metric-label">Total Interest Paid</div>
            <div class="metric-value" style="color:#c2410c">{fmt(total_interest_paid)}</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="metric-card" style="background:rgba(29,78,216,0.05);border-color:rgba(29,78,216,0.2)">
            <div class="metric-label">Total Amount Paid</div>
            <div class="metric-value">{fmt(total_paid)}</div>
        </div>""", unsafe_allow_html=True)
    with s5:
        st.markdown(f"""<div class="metric-card" style="background:rgba(220,38,38,0.06);border-color:rgba(220,38,38,0.22)">
            <div class="metric-label">Amount Still Owed</div>
            <div class="metric-value" style="color:#b91c1c">{fmt(remaining_capital)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    display_df = acc_rep[['PAYMENT_DATE', 'CAPITAL_PAIED', 'INTEREST_PAIED', 'TOTAL_PAID']].copy()
    display_df['PAYMENT_DATE'] = display_df['PAYMENT_DATE'].dt.strftime('%Y-%m-%d')
    display_df['SCHEDULED'] = display_df['CAPITAL_PAIED'] + display_df['INTEREST_PAIED']

    rows_html = ""
    for _, row in display_df.iterrows():
        rows_html += (
            f'<tr>'
            f'<td>{row["PAYMENT_DATE"]}</td>'
            f'<td>{row["SCHEDULED"]:,.2f}</td>'
            f'<td>{row["CAPITAL_PAIED"]:,.2f}</td>'
            f'<td style="color:#c2410c">{row["INTEREST_PAIED"]:,.2f}</td>'
            f'<td>{row["TOTAL_PAID"]:,.2f}</td>'
            f'</tr>'
        )

    table_html = f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      body {{ margin:0; padding:0; background:transparent; }}
      .wrap {{ background:rgba(255,255,255,0.75);border:0.5px solid rgba(29,78,216,0.18);border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(29,78,216,0.07); }}
      table {{ width:100%;border-collapse:collapse; }}
      thead tr {{ background:rgba(29,78,216,0.08); }}
      th {{ padding:13px 16px;font-family:'DM Sans',sans-serif;font-size:10px;letter-spacing:2px;text-transform:uppercase;font-weight:700;color:#1e40af;border-bottom:1.5px solid rgba(29,78,216,0.18); }}
      th:not(:first-child) {{ text-align:right; }}
      tbody td {{ padding:11px 16px;font-size:14px;color:#1e40af; }}
      tbody td:first-child {{ font-family:'DM Sans',sans-serif;color:#2d4a8a;font-size:14px; }}
      tbody td:not(:first-child) {{ font-family:'DM Mono',monospace;text-align:right;color:#1e40af; }}
      tbody td:nth-child(2) {{ color:#3b5fad;font-style:italic; }}
      tbody td:last-child {{ font-weight:700;font-size:15px;color:#1d4ed8; }}
      tbody tr {{ border-bottom:0.5px solid rgba(29,78,216,0.08); }}
      tbody tr:nth-child(even) {{ background:rgba(29,78,216,0.025); }}
      tbody tr:nth-child(odd)  {{ background:rgba(255,255,255,0.6); }}
      tfoot tr.total {{ background:rgba(29,78,216,0.08);border-top:2px solid #f97316; }}
      tfoot tr.owed  {{ background:rgba(220,38,38,0.04);border-top:0.5px solid rgba(220,38,38,0.2); }}
      tfoot td {{ padding:12px 16px;font-family:'DM Mono',monospace;font-weight:700;font-size:15px; }}
      .lbl {{ font-family:'DM Sans',sans-serif !important;font-size:11px;letter-spacing:2px;text-transform:uppercase; }}
    </style></head><body>
    <div class="wrap">
      <table>
        <thead>
          <tr>
            <th style="text-align:left">Payment Date</th>
            <th>Scheduled (LKR)</th>
            <th>Capital Paid (LKR)</th>
            <th>Interest Paid (LKR)</th>
            <th>Total Paid (LKR)</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
        <tfoot>
          <tr class="total">
            <td class="lbl" style="color:#1e40af">Paid to Date</td>
            <td style="text-align:right;color:#3b5fad;font-style:italic">{display_df['SCHEDULED'].sum():,.2f}</td>
            <td style="text-align:right;color:#1d4ed8">{total_capital_paid:,.2f}</td>
            <td style="text-align:right;color:#c2410c">{total_interest_paid:,.2f}</td>
            <td style="text-align:right;color:#1d4ed8">{total_paid:,.2f}</td>
          </tr>
          <tr class="owed">
            <td class="lbl" colspan="4" style="color:#b91c1c">Remaining Balance (Still Owed)</td>
            <td style="text-align:right;color:#b91c1c">{remaining_capital:,.2f}</td>
          </tr>
        </tfoot>
      </table>
    </div>
    </body></html>
    """

    table_height = min(80 + len(display_df) * 42 + 90, 800)
    components.html(table_html, height=table_height, scrolling=True)


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

    # ── Compute DQ flags ─────────────────────────────────────
    dq       = get_data_quality_flags(nic, app)
    is_thin  = get_thin_flag(nic)

    status_badge = {
        "Pending":  "<span class='badge-pending'>⏳ Pending</span>",
        "Approved": "<span class='badge-approved'>✅ Approved</span>",
        "Rejected": "<span class='badge-rejected'>❌ Rejected</span>",
    }.get(app["status"], app["status"])

    alert_badges = build_alert_badges(is_thin, dq)

    st.markdown(f"""
    <div style='margin:1rem 0 1.5rem'>
        <div style='font-size:10px;color:#1e40af;letter-spacing:3px;text-transform:uppercase;
                    margin-bottom:8px;font-weight:700'>
            Application #{app['id']} &nbsp;·&nbsp; {status_badge} &nbsp; {alert_badges}
        </div>
        <h1 style='font-size:28px;margin:0;color:#0c1a4e'>Customer Profile</h1>
        <div style='width:48px;height:3px;background:#f97316;border-radius:2px;margin-top:8px'></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data quality banner ───────────────────────────────────
    render_dq_banner(dq)

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
        # ── FIX: safe cast for internal score ──
        score_val   = safe_int(c.get('Internal_Bank_Default_Score'))
        score_color = "#1d4ed8" if score_val >= 650 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Internal Score</div>
            <div class="metric-value" style="color:{score_color}">{score_val if score_val > 0 else '—'}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        cluster_raw = str(c.get('Cluster_Name', c.get('Cluster_KProto', 'N/A')))
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Cluster</div>
            <div class="metric-value">{cluster_raw}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        score_band = str(c.get('Score_Band', 'N/A'))
        band_color = "#f97316" if score_band in ("Unknown Risk", "N/A") else "#1d4ed8"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Score Band</div>
            <div class="metric-value" style="color:{band_color}">{score_band}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        # ── FIX: safe_float so None/NaN → 0.0, never crashes ──
        avg_credit   = safe_float(c.get('Avg_Monthly_Credit'))
        income_color = "#dc2626" if avg_credit == 0 else "#1d4ed8"
        income_label = fmt(avg_credit) if avg_credit > 0 else "— No data"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Monthly Income</div>
            <div class="metric-value" style="color:{income_color}">{income_label}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        # ── FIX: safe_float for OOD ──
        ood_val   = safe_float(c.get('MAX_OOD'))
        ood_color = "#dc2626" if ood_val >= 30 else "#1d4ed8"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Max Days Overdue</div>
            <div class="metric-value" style="color:{ood_color}">{int(ood_val)}</div>
        </div>""", unsafe_allow_html=True)

    with m6:
        dq_count = len(dq["flags"])
        dq_col   = "#dc2626" if dq["severity"] == "critical" else ("#f97316" if dq["severity"] == "warning" else "#1d4ed8")
        dq_bg    = "background:rgba(220,38,38,0.07);border-color:rgba(220,38,38,0.3)" if dq["severity"] == "critical" else ("background:rgba(249,115,22,0.07);border-color:rgba(249,115,22,0.3)" if dq["severity"] == "warning" else "")
        st.markdown(f"""<div class="metric-card" style="{dq_bg}">
            <div class="metric-label" style="color:{dq_col}">Data Quality Flags</div>
            <div class="metric-value" style="color:{dq_col}">{dq_count if dq_count > 0 else '✓ Clean'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Derive display values ─────────────────────────────────
    fin_cap     = str(c.get('Financial_Capacity', 'N/A'))
    cust_risk   = str(c.get('CUSTOMER_RISK_NAME', 'N/A'))
    target_desc = str(c.get('TARGET_DESC', 'N/A'))
    emp_seg     = str(c.get('Employment_Segment', 'N/A'))
    net_ratio   = c.get('NET_RATIO', None)

    # ── FIX: safe age display ──
    age_display = safe_int(c.get('AGE'))

    def flag_val(val, bad_values=("Unknown", "", "nan", "N/A", "Not valid segment")):
        v      = str(val).strip()
        is_bad = v in bad_values
        color  = "#dc2626" if is_bad else "#0c1a4e"
        label  = f"⚠️ {v}" if is_bad else v
        return f'<span style="color:{color};font-weight:700;font-family:DM Mono,monospace;font-size:14px">{label}</span>'

    try:
        net_ratio_val     = safe_float(net_ratio)
        net_ratio_display = f"{net_ratio_val:.3f}"
        net_ratio_color   = "#dc2626" if net_ratio_val < 0 else "#0c1a4e"
        net_ratio_html    = f'<span style="color:{net_ratio_color};font-weight:700;font-family:DM Mono,monospace;font-size:14px">{net_ratio_display}</span>'
    except:
        net_ratio_html = '<span style="color:#dc2626;font-weight:700;font-family:DM Mono,monospace;font-size:14px">⚠️ Missing</span>'

    # ── FIX: EMI vs salary — always compute live with safe helpers ──
    emi_pct, emi_exceeds = compute_emi_ratio(app, c)
    if emi_exceeds:
        if emi_pct is None:
            emi_flag_html = (
                '<span style="color:#dc2626;font-weight:700;font-family:DM Mono,monospace;font-size:13px">'
                '⚠️ Cannot verify — no salary data</span>'
            )
        else:
            emi_flag_html = (
                f'<span style="color:#dc2626;font-weight:700;font-family:DM Mono,monospace;font-size:13px">'
                f'⚠️ {emi_pct}% of salary — exceeds 40% cap</span>'
            )
    else:
        pct_label     = f"{emi_pct}% of salary" if emi_pct is not None else "Within limit"
        emi_flag_html = (
            f'<span style="color:#14532d;font-weight:700;font-family:DM Mono,monospace;font-size:14px">'
            f'✓ {pct_label}</span>'
        )

    # ── FIX: safe existing debt display ──
    total_capital_due = safe_float(c.get('TOTAL_CAPITAL_DUE'))

    st.markdown(f"""
    <div class="profile-row">
      <div class="info-card">
        <div class="info-card-title">Personal Information</div>
        <div class="info-row"><span class="info-key">NIC</span><span class="info-val">{nic}</span></div>
        <div class="info-row"><span class="info-key">Age</span><span class="info-val">{age_display}</span></div>
        <div class="info-row"><span class="info-key">Gender</span><span class="info-val">{str(c.get('GENDER', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Marital status</span><span class="info-val">{str(c.get('MARITAL_STATUS', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">District</span><span class="info-val">{str(c.get('DISTRICT', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Occupation</span><span class="info-val">{str(c.get('OCCUPATION', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Employment</span><span class="info-val">{str(c.get('EMPLOYMENT_STATUS', 'N/A')).title()}</span></div>
        <div class="info-row"><span class="info-key">Segment</span>{flag_val(emp_seg, ("Not valid segment", "", "nan", "N/A"))}</div>
      </div>

      <div class="info-card">
        <div class="info-card-title">Risk Profile</div>
        <div class="info-row"><span class="info-key">Customer risk</span>{flag_val(cust_risk)}</div>
        <div class="info-row"><span class="info-key">Target tier</span>{flag_val(target_desc)}</div>
        <div class="info-row"><span class="info-key">Financial capacity</span>{flag_val(fin_cap, ("Unknown / Missing Balance Data", "", "nan", "N/A"))}</div>
        <div class="info-row"><span class="info-key">Cluster</span><span class="info-val">{c.get('Cluster_Name', 'N/A')}</span></div>
        <div class="info-row"><span class="info-key">Age bucket</span><span class="info-val">{c.get('Age_Bucket', 'N/A')}</span></div>
        <div class="info-row"><span class="info-key">Existing debt</span><span class="info-val">{fmt(total_capital_due)}</span></div>
        <div class="info-row"><span class="info-key">Net ratio</span>{net_ratio_html}</div>
        <div class="info-row"><span class="info-key">Thin file</span><span class="info-val" style="color:{'#f97316' if is_thin else '#1d4ed8'}">{'⚠️ Yes' if is_thin else 'No'}</span></div>
      </div>

      <div class="info-card">
        <div class="info-card-title">This Application</div>
        <div style='margin-bottom:12px'>{status_badge} &nbsp; {alert_badges}</div>
        <div class="info-row"><span class="info-key">Product</span><span class="info-val" style="font-size:11px">{app['loan_product'].split('—')[0].strip()}</span></div>
        <div class="info-row"><span class="info-key">Amount</span><span class="info-val">{fmt(app['loan_amount'])}</span></div>
        <div class="info-row"><span class="info-key">Term</span><span class="info-val">{app['loan_term']} months</span></div>
        <div class="info-row"><span class="info-key">Rate</span><span class="info-val">{app['loan_rate']}% p.a.</span></div>
        <div class="info-row"><span class="info-key">Monthly EMI</span><span class="info-val">{fmt(app['loan_emi'])}</span></div>
        <div class="info-row"><span class="info-key">Total interest</span><span class="info-val">{fmt(app['total_interest'])}</span></div>
        <div class="info-row"><span class="info-key">Total repayment</span><span class="info-val">{fmt(app['total_repayment'])}</span></div>
        <div class="info-row"><span class="info-key">EMI vs salary</span>{emi_flag_html}</div>
        <div class="info-row"><span class="info-key">Submitted</span><span class="info-val">{app['submitted_at']}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Accounts ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Accounts</div>', unsafe_allow_html=True)

    if not cust_accounts.empty:
        for _, acc in cust_accounts.iterrows():
            product = str(acc.get('ACTIVE_PRODUCT', 'N/A'))
            status  = str(acc.get('ACCT_STATUS', 'N/A')).title()
            opened  = format_date(acc.get('ORIG_CONTRACT_DATE', ''))
            balance = safe_float(acc.get('MONTHEND_CONVERTED_BALANCE'))
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

    # ── Monthly Balance Trend ─────────────────────────────────
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

    # ── Transaction Activity ──────────────────────────────────
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
        if dq["has_issues"]:
            sev_msg = {
                "critical": "🔴 This application has **critical data gaps**. Manual verification is mandatory before any decision.",
                "warning":  "🟠 This application has **insufficient information** in one or more areas. Please verify before proceeding.",
                "info":     "🔵 Minor data gaps detected. Review flags above before deciding.",
            }.get(dq["severity"], "")
            st.warning(sev_msg)
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

    col_r, col_clear, _ = st.columns([1, 1, 4])
    with col_r:
        if st.button("↻ Refresh", use_container_width=True):
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            if "confirm_clear" not in st.session_state:
                st.session_state.confirm_clear = False
            st.session_state.confirm_clear = True

    if st.session_state.get("confirm_clear", False):
        st.warning("⚠️ Are you sure you want to delete ALL applications? This cannot be undone.")
        col_yes, col_no, _ = st.columns([1, 1, 4])
        with col_yes:
            if st.button("✅ Yes, delete all", use_container_width=True):
                from db_utils import clear_all_applications
                clear_all_applications()
                st.session_state.confirm_clear = False
                st.success("All applications cleared!")
                st.rerun()
        with col_no:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    applications = get_all_applications()
    if not applications:
        st.info("No applications received yet.")
        return

    # Pre-compute DQ for all apps
    app_dq = {app["id"]: get_data_quality_flags(app["nic"], app) for app in applications}

    total    = len(applications)
    pending  = sum(1 for a in applications if a["status"] == "Pending")
    approved = sum(1 for a in applications if a["status"] == "Approved")
    rejected = sum(1 for a in applications if a["status"] == "Rejected")
    thin     = sum(1 for a in applications if get_thin_flag(a["nic"]))
    insuff   = sum(1 for a in applications if app_dq[a["id"]]["has_issues"])
    critical = sum(1 for a in applications if app_dq[a["id"]]["severity"] == "critical")

    # ── Summary metric cards ────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total</div>
            <div class="metric-value">{total}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Pending</div>
            <div class="metric-value" style="color:#92400e">{pending}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Approved</div>
            <div class="metric-value" style="color:#14532d">{approved}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Rejected</div>
            <div class="metric-value" style="color:#7f1d1d">{rejected}</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card" style="background:rgba(249,115,22,0.07);border-color:rgba(249,115,22,0.3)">
            <div class="metric-label" style="color:#92400e">Thin File</div>
            <div class="metric-value" style="color:#f97316">{thin}</div></div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="metric-card" style="background:rgba(251,191,36,0.07);border-color:rgba(251,191,36,0.4)">
            <div class="metric-label" style="color:#78350f">Insuff. Info</div>
            <div class="metric-value" style="color:#d97706">{insuff}</div></div>""", unsafe_allow_html=True)
    with c7:
        st.markdown(f"""<div class="metric-card" style="background:rgba(220,38,38,0.07);border-color:rgba(220,38,38,0.3)">
            <div class="metric-label" style="color:#7f1d1d">Critical Gaps</div>
            <div class="metric-value" style="color:#dc2626">{critical}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        status_filter = st.selectbox("Filter by status", ["All", "Pending", "Approved", "Rejected"])
    with col_f2:
        thin_filter = st.selectbox("Filter by file type", ["All", "Thin File Only", "Normal Only"])
    with col_f3:
        dq_filter = st.selectbox("Filter by data quality", ["All", "Insufficient Info", "Critical Gaps Only", "Clean Only"])

    filtered = applications if status_filter == "All" else [
        a for a in applications if a["status"] == status_filter
    ]
    if thin_filter == "Thin File Only":
        filtered = [a for a in filtered if get_thin_flag(a["nic"])]
    elif thin_filter == "Normal Only":
        filtered = [a for a in filtered if not get_thin_flag(a["nic"])]

    if dq_filter == "Insufficient Info":
        filtered = [a for a in filtered if app_dq[a["id"]]["has_issues"]]
    elif dq_filter == "Critical Gaps Only":
        filtered = [a for a in filtered if app_dq[a["id"]]["severity"] == "critical"]
    elif dq_filter == "Clean Only":
        filtered = [a for a in filtered if not app_dq[a["id"]]["has_issues"]]

    st.markdown(f"<p style='color:#1e40af;font-size:12px;letter-spacing:1px;font-weight:600'>{len(filtered)} APPLICATION(S)</p>", unsafe_allow_html=True)

    for app in filtered:
        dq      = app_dq[app["id"]]
        is_thin = get_thin_flag(app["nic"])

        if dq["severity"] == "critical":
            card_inline = "background:rgba(254,242,242,0.88);border:0.5px solid rgba(220,38,38,0.35);border-left:3px solid #dc2626;border-radius:12px;padding:12px 18px;"
        elif dq["has_issues"]:
            card_inline = "background:rgba(255,247,237,0.82);border:0.5px solid rgba(249,115,22,0.35);border-left:3px solid #f97316;border-radius:12px;padding:12px 18px;"
        else:
            card_inline = "background:rgba(255,255,255,0.62);border:0.5px solid rgba(30,64,175,0.14);border-radius:12px;padding:12px 18px;backdrop-filter:blur(8px);"

        badge_styles = {
            "pending":  "background:rgba(251,191,36,0.18);color:#92400e;padding:3px 12px;border-radius:20px;font-size:11px;border:1px solid rgba(251,191,36,0.5);font-weight:600;",
            "approved": "background:rgba(34,197,94,0.14);color:#14532d;padding:3px 12px;border-radius:20px;font-size:11px;border:1px solid rgba(34,197,94,0.4);font-weight:600;",
            "rejected": "background:rgba(239,68,68,0.12);color:#7f1d1d;padding:3px 12px;border-radius:20px;font-size:11px;border:1px solid rgba(239,68,68,0.35);font-weight:600;",
        }
        badge_s           = badge_styles.get(app["status"].lower(), "")
        badge_html_inline = f"<span style='{badge_s}'>{app['status']}</span>"

        alert_html = build_alert_badges(is_thin, dq)

        fc_inline = ""
        if dq["has_issues"]:
            fc       = len(dq["flags"])
            fc_color = "#dc2626" if dq["severity"] == "critical" else "#d97706"
            fc_inline = f'<span style="font-size:12px;color:{fc_color};font-weight:600">{fc} flag{"s" if fc != 1 else ""}</span>'

        product_label = app['loan_product'].split('—')[0].strip()

        card_html = f"""
        <!DOCTYPE html><html><head>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>body{{margin:0;padding:0;background:transparent;}}</style>
        </head><body>
        <div style="{card_inline}">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span style="color:#1e40af;font-size:11px;font-family:'DM Mono',monospace;font-weight:700;">#{app['id']}</span>
                    <span style="color:#0c1a4e;font-weight:700;font-family:'DM Sans',sans-serif;">{app['nic']}</span>
                    {badge_html_inline}
                    {alert_html}
                    {fc_inline}
                </div>
                <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;">
                    <span style="color:#2d4a8a;font-size:13px;font-family:'DM Sans',sans-serif;">{product_label}</span>
                    <span style="color:#0c1a4e;font-family:'DM Mono',monospace;font-size:14px;font-weight:700;">{fmt(app['loan_amount'])}</span>
                    <span style="color:#3b5fad;font-size:12px;font-family:'DM Sans',sans-serif;">{app['submitted_at']}</span>
                </div>
            </div>
        </div>
        </body></html>
        """

        col_info, col_btn = st.columns([6, 1])
        with col_info:
            components.html(card_html, height=68, scrolling=False)
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