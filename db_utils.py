import streamlit as st
from supabase import create_client, Client
from datetime import datetime


def get_client() -> Client:
    """Create and return a Supabase client using Streamlit secrets."""
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def save_application(data: dict):
    """Save a new loan application to Supabase."""
    try:
        supabase = get_client()
        response = supabase.table("applications").insert({
            "nic":             str(data["nic"]),
            "loan_product":    str(data["loan_product"]),
            "loan_amount":     float(data["loan_amount"]),
            "loan_term":       int(data["loan_term"]),
            "loan_rate":       float(data["loan_rate"]),
            "loan_emi":        float(data["loan_emi"]),
            "total_interest":  float(data["total_interest"]),
            "total_repayment": float(data["total_repayment"]),
            "score_band":      str(data["score_band"]),
            "profile_score":   int(data["profile_score"]),
            "status":          "Pending",
            "submitted_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }).execute()
        return response
    except Exception as e:
        st.error(f"Failed to save application: {e}")
        return None


def get_all_applications():
    """Fetch all applications ordered by most recent first."""
    try:
        supabase = get_client()
        response = supabase.table("applications") \
            .select("*") \
            .order("submitted_at", desc=True) \
            .execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Failed to fetch applications: {e}")
        return []


def get_customer_applications(nic: str):
    """Fetch all applications for a specific NIC."""
    try:
        supabase = get_client()
        response = supabase.table("applications") \
            .select("*") \
            .eq("nic", nic) \
            .order("submitted_at", desc=True) \
            .execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Failed to fetch customer applications: {e}")
        return []


def update_application_status(app_id: int, status: str,
                               officer_name: str, notes: str):
    """Update an application status with officer review details."""
    try:
        supabase = get_client()
        response = supabase.table("applications").update({
            "status":        status,
            "reviewed_by":   officer_name,
            "officer_notes": notes,
            "reviewed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }).eq("id", app_id).execute()
        return response
    except Exception as e:
        st.error(f"Failed to update application: {e}")
        return None


def clear_all_applications():
    """Delete all applications from Supabase."""
    try:
        supabase = get_client()
        response = supabase.table("applications") \
            .delete() \
            .neq("id", 0) \
            .execute()
        return response
    except Exception as e:
        st.error(f"Failed to clear applications: {e}")
        return None