import streamlit as st
from supabase import create_client
from datetime import datetime


def get_client():
    """Create and return a Supabase client using Streamlit secrets."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def save_application(data: dict):
    """Save a new loan application to Supabase."""
    supabase = get_client()
    supabase.table("applications").insert({
        "nic":             data["nic"],
        "loan_product":    data["loan_product"],
        "loan_amount":     data["loan_amount"],
        "loan_term":       data["loan_term"],
        "loan_rate":       data["loan_rate"],
        "loan_emi":        data["loan_emi"],
        "total_interest":  data["total_interest"],
        "total_repayment": data["total_repayment"],
        "score_band":      data["score_band"],
        "profile_score":   data["profile_score"],
        "status":          "Pending",
        "submitted_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }).execute()


def get_all_applications():
    """Fetch all applications ordered by most recent first."""
    supabase = get_client()
    result = (
        supabase.table("applications")
        .select("*")
        .order("submitted_at", desc=True)
        .execute()
    )
    return result.data


def get_customer_applications(nic: str):
    """Fetch all applications for a specific NIC."""
    supabase = get_client()
    result = (
        supabase.table("applications")
        .select("*")
        .eq("nic", nic)
        .order("submitted_at", desc=True)
        .execute()
    )
    return result.data


def update_application_status(app_id: int, status: str,
                               officer_name: str, notes: str):
    """Update an application status with officer review details."""
    supabase = get_client()
    supabase.table("applications").update({
        "status":        status,
        "reviewed_by":   officer_name,
        "officer_notes": notes,
        "reviewed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }).eq("id", app_id).execute()
