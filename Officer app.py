import streamlit as st
import joblib
from db_utils import (
    get_all_applications,
    update_application_status,
)

st.set_page_config(
    page_title="Loan Officer Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ── Load customer data ──────────────────────────────────────
eligible_customers = joblib.load("eligible_customers.pkl")

def fmt(n):
    try:
        return f"LKR {float(n):,.0f}"
    except Exception:
        return "LKR 0"

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label {
    font-size: 12px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #042C53;
}
.badge-pending  { background:#fff3cd; color:#856404;
                  padding:3px 10px; border-radius:20px; font-size:12px; }
.badge-approved { background:#d1e7dd; color:#0a3622;
                  padding:3px 10px; border-radius:20px; font-size:12px; }
.badge-rejected { background:#f8d7da; color:#58151c;
                  padding:3px 10px; border-radius:20px; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIMPLE OFFICER LOGIN
# ══════════════════════════════════════════════════════════════
if "officer_name" not in st.session_state:
    st.session_state.officer_name = ""

if not st.session_state.officer_name:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0 1rem">
            <h1 style="font-size:28px">🏦 Officer Portal</h1>
            <p style="color:#666">Enter your name to access the dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        name = st.text_input("Your name", placeholder="e.g. Kasun Perera")
        if st.button("Login", use_container_width=True) and name.strip():
            st.session_state.officer_name = name.strip()
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🏦 Loan Officer Dashboard")
    st.caption(f"Logged in as **{st.session_state.officer_name}**")
with col2:
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        st.session_state.officer_name = ""
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# LOAD APPLICATIONS
# ══════════════════════════════════════════════════════════════
if st.button("Refresh", use_container_width=False):
    st.rerun()

applications = get_all_applications()

if not applications:
    st.info("No applications received yet.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# SUMMARY METRICS
# ══════════════════════════════════════════════════════════════
total    = len(applications)
pending  = sum(1 for a in applications if a["status"] == "Pending")
approved = sum(1 for a in applications if a["status"] == "Approved")
rejected = sum(1 for a in applications if a["status"] == "Rejected")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total</div>
        <div class="metric-value">{total}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pending</div>
        <div class="metric-value" style="color:#856404">{pending}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Approved</div>
        <div class="metric-value" style="color:#0a3622">{approved}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rejected</div>
        <div class="metric-value" style="color:#58151c">{rejected}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════
# FILTER
# ══════════════════════════════════════════════════════════════
status_filter = st.selectbox(
    "Filter by status",
    ["All", "Pending", "Approved", "Rejected"],
    index=0
)

filtered = applications if status_filter == "All" else [
    a for a in applications if a["status"] == status_filter
]

st.markdown(f"**{len(filtered)} application(s)**")
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# APPLICATION CARDS
# ══════════════════════════════════════════════════════════════
for row in filtered:

    badge = {
        "Pending":  "<span class='badge-pending'>⏳ Pending</span>",
        "Approved": "<span class='badge-approved'>✅ Approved</span>",
        "Rejected": "<span class='badge-rejected'>❌ Rejected</span>",
    }.get(row["status"], row["status"])

    with st.expander(
        f"#{row['id']}  |  {row['nic']}  |  "
        f"{fmt(row['loan_amount'])}  |  {row['submitted_at']}"
    ):

        # ── Status badge ──
        st.markdown(
            f"<div style='margin-bottom:1rem'>{badge}</div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 1])

        # ── Loan details ──
        with col1:
            st.markdown("**Loan Details**")
            st.write(f"**NIC:** {row['nic']}")
            st.write(f"**Product:** {row['loan_product']}")
            st.write(f"**Amount:** {fmt(row['loan_amount'])}")
            st.write(f"**Term:** {row['loan_term']} months")
            st.write(f"**Rate:** {row['loan_rate']}% p.a.")
            st.write(f"**Monthly EMI:** {fmt(row['loan_emi'])}")
            st.write(f"**Total Interest:** {fmt(row['total_interest'])}")
            st.write(f"**Total Repayment:** {fmt(row['total_repayment'])}")
            st.write(f"**Score Band:** {row['score_band']}")
            st.write(f"**Profile Score:** {row['profile_score']} / 100")

        # ── Customer profile ──
        with col2:
            st.markdown("**Customer Profile**")
            match = eligible_customers[
                eligible_customers["MASKED_LEGAL_ID"] == row["nic"]
            ]
            if not match.empty:
                c = match.iloc[0]
                st.write(f"**Age:** {int(c.get('AGE', 0))}")
                st.write(f"**Employment:** {c.get('Employment_Segment', 'N/A')}")
                st.write(f"**Customer Risk:** {c.get('CUSTOMER_RISK_NAME', 'N/A')}")
                st.write(f"**Target Tier:** {c.get('TARGET_DESC', 'N/A')}")
                st.write(f"**Financial Capacity:** {c.get('Financial_Capacity', 'N/A')}")
                st.write(f"**Avg Monthly Income:** {fmt(float(c.get('Avg_Monthly_Credit', 0)))}")
                st.write(f"**Max Days Overdue:** {int(c.get('MAX_OOD', 0))} days")
                st.write(f"**Existing Debt:** {fmt(float(c.get('TOTAL_CAPITAL_DUE', 0)))}")
            else:
                st.warning("Customer profile not found.")

        # ── Review section (only for pending) ──
        if row["status"] == "Pending":
            st.divider()
            st.markdown("**Review this application**")
            notes = st.text_area(
                "Officer notes",
                key=f"notes_{row['id']}",
                placeholder="Add any comments, conditions, or reasons here..."
            )
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                if st.button(
                    "✅ Approve",
                    key=f"approve_{row['id']}",
                    use_container_width=True
                ):
                    update_application_status(
                        row["id"], "Approved",
                        st.session_state.officer_name,
                        notes
                    )
                    st.success("Application approved successfully.")
                    st.rerun()
            with col_b:
                if st.button(
                    "❌ Reject",
                    key=f"reject_{row['id']}",
                    use_container_width=True
                ):
                    update_application_status(
                        row["id"], "Rejected",
                        st.session_state.officer_name,
                        notes
                    )
                    st.error("Application rejected.")
                    st.rerun()

        # ── Already reviewed ──
        else:
            st.divider()
            st.markdown("**Review Details**")
            st.write(f"**Reviewed by:** {row.get('reviewed_by', 'N/A')}")
            st.write(f"**Reviewed at:** {row.get('reviewed_at', 'N/A')}")
            st.write(f"**Notes:** {row.get('officer_notes', 'N/A')}")