import re
import pyodbc
import pandas as pd
import gradio as gr

# -----------------------
# 1️⃣ Database connection
# -----------------------
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=LOAN_PORTFOLIO_DB;"
    "Trusted_Connection=yes"
)

# -----------------------
# 2️⃣ NIC validation
# -----------------------
def check_nic(nic):
   
    cursor = conn.cursor()
    cursor.execute("SELECT MASKED_LEGAL_ID FROM customer_details WHERE MASKED_LEGAL_ID = ?", nic)
    result = cursor.fetchone()
    
    if result:
        return "✅ NIC valid", gr.update(visible=True)  # Show Next button
    else:
        return "❌ NIC not found", gr.update(visible=False)

# -----------------------
# 3️⃣ Customer validation
# -----------------------
def process_customer(age, income, occupation, employment_status, gender, marital_status):
    # Validation rules
    if age < 18 or age > 80:
        return {"Eligibility": "REJECT", "Reason": "Regulatory Age Restriction", "Loan Decision": "Rejected"}
    
    if employment_status not in ["EMPLOYED","SELF-EMPLOYED","BUSINESS","STUDENT","RETIRED","UNEMPLOYED","FREELANCE"]:
        return {"Eligibility": "REJECT", "Reason": "Invalid Employment Status", "Loan Decision": "Rejected"}
    
    # Clustering placeholder
    cluster_label = "Cluster 1"  # Replace with actual clustering model
    
    cluster_risk_mapping = {
        "Cluster 1": "Low Risk",
        "Cluster 2": "Medium Risk",
        "Cluster 3": "High Risk"
    }
    
    risk_score = cluster_risk_mapping.get(cluster_label, "Unknown")
    loan_decision = "Rejected" if risk_score == "High Risk" else "Approved"
    
    return {
        "Eligibility": "ELIGIBLE",
        "Reason": None,
        "Cluster": cluster_label,
        "Risk Score": risk_score,
        "Loan Decision": loan_decision
    }

# -----------------------
# 4️⃣ Gradio Interface
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("## Step 1: Enter NIC")
    
    nic_input = gr.Textbox(label="NIC")
    nic_status = gr.Label()
    next_button = gr.Button("Next", visible=False)
    
    # Customer details (hidden initially)
    with gr.Row(visible=False) as customer_form:
        age_input = gr.Number(label="Age")
        income_input = gr.Number(label="Monthly Income")
        occupation_input = gr.Textbox(label="Occupation")
        employment_input = gr.Dropdown(["EMPLOYED","SELF-EMPLOYED","BUSINESS","STUDENT","RETIRED","UNEMPLOYED","FREELANCE"], label="Employment Status")
        gender_input = gr.Dropdown(["Male","Female","Other"], label="Gender")
        marital_input = gr.Dropdown(["Single","Married","Divorced","Widowed"], label="Marital Status")
        submit_button = gr.Button("Submit")
        output = gr.JSON(label="Loan Decision")
    
    # -----------------------
    # Events
    # -----------------------
    nic_input.change(check_nic, inputs=[nic_input], outputs=[nic_status, next_button])
    
    def show_customer_form():
        return gr.update(visible=True)
    
    next_button.click(show_customer_form, outputs=[customer_form])
    
    submit_button.click(
        process_customer,
        inputs=[age_input, income_input, occupation_input, employment_input, gender_input, marital_input],
        outputs=output
    )

# Launch
demo.launch

