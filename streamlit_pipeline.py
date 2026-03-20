#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyodbc
import pandas as pd
from pycaret.classification import setup, compare_models
import numpy as np


# In[2]:


import pyodbc
import pandas as pd

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"        # default instance
    "DATABASE=LOAN_PORTFOLIO_DB;"
    "Trusted_Connection=yes"   # use Windows Authentication
)


# In[3]:


customer_df = pd.read_sql("SELECT * FROM CUSTOMER_DETAILS", conn)
account_df = pd.read_sql("SELECT * FROM ACCOUNT_DETAILS", conn)
transaction_df = pd.read_sql("SELECT * FROM TRANSACTION_DETAILS", conn)
loan_cashflow_df = pd.read_sql("SELECT * FROM LOAN_CASHFLOW", conn)
repayment_df = pd.read_sql("SELECT * FROM REPAYMENT", conn)


# In[4]:


customer_df['MARITAL_STATUS'].fillna('Unknown', inplace=True)
customer_df['EMPLOYMENT_STATUS'].fillna('Unknown', inplace=True)
customer_df['DISTRICT'].fillna('Unknown', inplace=True)
customer_df['OCCUPATION'].fillna('Unknown', inplace=True)

# Step 1 — Calculate median risk number
median_risk = customer_df['CUSTOMER_RISK'].median()

# Step 2 — Fill missing CUSTOMER_RISK number with median
customer_df['CUSTOMER_RISK'].fillna(median_risk, inplace=True)

# Step 3 — Build the number → name mapping from existing data
risk_map = (
    customer_df[['CUSTOMER_RISK', 'CUSTOMER_RISK_NAME']]
    .dropna()                        # only rows that have both values
    .drop_duplicates()               # remove duplicates
    .set_index('CUSTOMER_RISK')      # use number as the key
    ['CUSTOMER_RISK_NAME']           # get the name column
    .to_dict()                       # convert to dictionary
)

# Step 4 — Fill missing CUSTOMER_RISK_NAME using the mapping
customer_df['CUSTOMER_RISK_NAME'] = customer_df.apply(
    lambda row: risk_map.get(row['CUSTOMER_RISK'], 'Unknown')
    if pd.isna(row['CUSTOMER_RISK_NAME'])
    else row['CUSTOMER_RISK_NAME'],
    axis=1
)



# In[5]:


accts_numeric_cols = ['TERM', 'TERM_AMOUNT', 'OOD', 'MONTHEND_CONVERTED_BALANCE', 'CONVERTED_BALANCE',
                'JUN_25','JUL_25','AUG_25','SEP_25','OCT_25','NOV_25']

account_df[accts_numeric_cols] = account_df[accts_numeric_cols].fillna(0)

account_df['ACCT_CLOSE_DATE'] = pd.to_datetime(account_df['ACCT_CLOSE_DATE'], errors='coerce')


# In[6]:


loan_numeric_cols = ['OOD', 'CAPITAL_TO_BE_PAIED', 'INTEREST_TO_BE_PAIED']
loan_cashflow_df[loan_numeric_cols] = loan_cashflow_df[loan_numeric_cols].fillna(0)

loan_cashflow_df['TO_BE_PAYMENT_DATE'] = pd.to_datetime(loan_cashflow_df['TO_BE_PAYMENT_DATE'], errors='coerce')


# In[7]:


repayment_numeric_cols = ['OOD', 'CAPITAL_PAIED']
repayment_df[repayment_numeric_cols] = repayment_df[repayment_numeric_cols].fillna(0)


# In[8]:


# Rule 1 - Hard Eligibility & Regulatory Rules


# In[9]:


## Regulatory Age Check


# In[10]:


customer_df['Eligibility_Flag'] = 'ELIGIBLE'
customer_df['Rejection_Reason'] = None


# In[11]:


customer_df.loc[
    (customer_df['AGE'] < 18) | (customer_df['AGE'] > 80),
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Regulatory Age Restriction']


# In[12]:


eligible_cus_df = customer_df.copy()


# In[13]:


## Account Eligibility Check


# In[14]:


active_accounts = account_df[
    (account_df['ACCT_STATUS'] == 'ACTIVE') |
    (account_df['ACCT_CLOSE_DATE'].isna())
]

active_account_count = (
    active_accounts
    .groupby('MASKED_ID')
    .size()
    .reset_index(name='Number_of_Active_Accounts')
)


# In[15]:


eligible_cus_df = eligible_cus_df.merge(
    active_account_count,
    on='MASKED_ID',
    how='left'
)


# In[16]:



# if no accounts isnted of Nan use 0 
eligible_cus_df['Number_of_Active_Accounts'] = (
    eligible_cus_df['Number_of_Active_Accounts']
    .fillna(0)
    .astype(int)
)


# In[17]:


# Step 1: Set ELSE condition (default)
eligible_cus_df['Eligibility_Flag'] = 'ELIGIBLE'
eligible_cus_df['Rejection_Reason'] = 'Existing Customer'

# Step 2: Apply IF condition
eligible_cus_df.loc[
    eligible_cus_df['Number_of_Active_Accounts'] == 0,
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Non-Existing Customer']


# In[18]:


## Employment Status Validation


# In[19]:


valid_employment_status = pd.read_sql(
    "SELECT DISTINCT EMPLOYMENT_STATUS FROM CUSTOMER_DETAILS",
    conn
)['EMPLOYMENT_STATUS'].dropna().tolist()


# In[20]:


eligible_cus_df['Employment_Status_Flag'] = 'Valid Employment Status'

eligible_cus_df.loc[
    ~eligible_cus_df['EMPLOYMENT_STATUS'].isin(valid_employment_status),
    'Employment_Status_Flag'
] = 'Invalid Employment Status'


# In[21]:


# Rule 2 - Employment-Based Routing & Participation Rules


# In[22]:


## Routing & Participation 


# In[23]:


eligible_cus_df['Employment_Segment'] = 'Other'


# In[24]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin(['EMPLOYED', 'SELF-EMPLOYED', 'BUSINESS'])) &
    (eligible_cus_df['AGE'].between(18, 60)),
    'Employment_Segment'
] = 'Core Working Group'


# In[25]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin(['UNEMPLOYED	', 'RETIRED', 'STUDENT', 'FREELANCE'])) &
    (eligible_cus_df['AGE'].between(18, 65)),
    'Employment_Segment'
] = 'Special Segment'


# In[26]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin([
        'UNEMPLOYED', 'RETIRED', 'STUDENT', 'FREELANCE', 
        'EMPLOYED', 'SELF-EMPLOYED', 'BUSINESS'
    ])) &
    (~eligible_cus_df['AGE'].between(18, 65)),  # not between 18 and 60
    'Employment_Segment'
] = 'Not valid segment'


# In[27]:


# Rule 3 - Age-Based Segments


# In[28]:


eligible_cus_df['Age_Bucket'] = pd.cut(
    eligible_cus_df['AGE'],
    bins=[17, 25, 40, 60, 80],
    labels=['Young Adult', 'Adult', 'Middle-Aged', 'Senior']
)


# In[29]:


# Rule 4 - Financial capacity


# In[30]:


balance_cols = ['JUN_25', 'JUL_25', 'AUG_25', 'SEP_25', 'OCT_25', 'NOV_25']


# In[31]:


account_df['Monthly_Avg_Balance'] = account_df[balance_cols].mean(axis=1)


# In[32]:


customer_balance_df = (
    account_df
    .groupby('MASKED_ID', as_index=False)['Monthly_Avg_Balance']
    .mean()
)


# In[33]:


eligible_cus_df = eligible_cus_df.merge(
    customer_balance_df,
    on='MASKED_ID',
    how='left'
)


# In[34]:


eligible_cus_df['Financial_Capacity'] = 'Unknown / Missing Balance Data'

eligible_cus_df.loc[
    eligible_cus_df['Monthly_Avg_Balance'] >= 100000,
    'Financial_Capacity'
] = 'High Financial Capacity'

eligible_cus_df.loc[
    eligible_cus_df['Monthly_Avg_Balance'].between(50000, 99999),
    'Financial_Capacity'
] = 'Medium Financial Capacity'

eligible_cus_df.loc[
    eligible_cus_df['Monthly_Avg_Balance'] < 50000,
    'Financial_Capacity'
] = 'Low Financial Capacity'


# In[35]:


# Rule 6 - Salary Verification Check


# In[36]:


transaction_df['AMOUNT_LCY'] = pd.to_numeric(
    transaction_df['AMOUNT_LCY'],
    errors='coerce'
)


# In[37]:


credit_df = transaction_df[
    transaction_df['AMOUNT_LCY'] > 0
].copy()


# In[38]:


credit_df['Month'] = pd.to_datetime(
    credit_df['BOOKING_DATE']
).dt.to_period('M')


# In[39]:


monthly_credit = (
    credit_df
    .groupby(['MASKED_ID', 'Month'], as_index=False)['AMOUNT_LCY']
    .sum()
)


# In[40]:


avg_monthly_income = (
    monthly_credit
    .groupby('MASKED_ID', as_index=False)['AMOUNT_LCY']
    .mean()
    .rename(columns={'AMOUNT_LCY': 'Avg_Monthly_Credit'})
)


# In[41]:


eligible_cus_df['MASKED_ID'] = eligible_cus_df['MASKED_ID'].astype(str)


# In[42]:


avg_monthly_income['MASKED_ID'] = avg_monthly_income['MASKED_ID'].astype(str)


# In[43]:


eligible_cus_df = eligible_cus_df.merge(
    avg_monthly_income,
    on='MASKED_ID',
    how='left'
)


# In[44]:


mask = (
    eligible_cus_df['Avg_Monthly_Credit'].isna() &
    (eligible_cus_df['Monthly_Avg_Balance'] > 0)
)

eligible_cus_df.loc[mask, 'Avg_Monthly_Credit'] = (
    eligible_cus_df.loc[mask, 'Monthly_Avg_Balance']
)


# In[45]:



eligible_cus_df['Cluster_Name'] = 'Unknown / Missing Salary'

eligible_cus_df.loc[
    eligible_cus_df['Avg_Monthly_Credit'] >= 100000,
    'Cluster_Name'
] = 'High Salary'

eligible_cus_df.loc[
    eligible_cus_df['Avg_Monthly_Credit'].between(50000, 99999),
    'Cluster_Name'
] = 'Medium Salary'

eligible_cus_df.loc[
    eligible_cus_df['Avg_Monthly_Credit'] < 50000,
    'Cluster_Name'  
] = 'Low Salary'


# In[46]:


#Creating Credit scores


# In[47]:



# # Ensure AMOUNT_LCY is numeric

# transaction_df["AMOUNT_LCY"] = pd.to_numeric(transaction_df["AMOUNT_LCY"], errors='coerce').fillna(0)


# Create inflow, outflow, net, and absolute activity

transaction_df["INFLOW"] = transaction_df["AMOUNT_LCY"].apply(lambda x: x if x > 0 else 0)
transaction_df["OUTFLOW"] = transaction_df["AMOUNT_LCY"].apply(lambda x: abs(x) if x < 0 else 0)
transaction_df["NET_AMOUNT"] = transaction_df["AMOUNT_LCY"]  # keeps signs
transaction_df["ABS_ACTIVITY"] = transaction_df["AMOUNT_LCY"].abs()


#  Aggregate per customer

txn_features = transaction_df.groupby("MASKED_ID").agg({
    "INFLOW": "sum",
    "OUTFLOW": "sum",
    "NET_AMOUNT": "sum",
    "ABS_ACTIVITY": "sum",
    "AMOUNT_LCY": ["mean", "std", "count"]  # original avg/std/count
}).reset_index()


# Flatten MultiIndex columns from aggregation

txn_features.columns = [
    "MASKED_ID",
    "TOTAL_INFLOW",
    "TOTAL_OUTFLOW",
    "NET_TRANSACTION_SUM",
    "ABS_ACTIVITY",
    "AVG_TRANSACTION",
    "STD_TRANSACTION",
    "TXN_COUNT"
]

txn_features["INFLOW_RATIO"] = txn_features["TOTAL_INFLOW"] / (txn_features["ABS_ACTIVITY"] + 1)
txn_features["OUTFLOW_RATIO"] = txn_features["TOTAL_OUTFLOW"] / (txn_features["ABS_ACTIVITY"] + 1)
txn_features["NET_RATIO"] = txn_features["NET_TRANSACTION_SUM"] / (txn_features["ABS_ACTIVITY"] + 1)


# Average transaction per day (assuming 30 days)
txn_features["AVG_TXN_PER_DAY"] = txn_features["TXN_COUNT"] / 30

# Transaction volatility: std relative to average
txn_features["TRANSACTION_VOLATILITY"] = txn_features["STD_TRANSACTION"] / (txn_features["AVG_TRANSACTION"] + 1)


# Average monthly inflow as income proxy
txn_features["AVG_MONTHLY_INFLOW"] = txn_features["TOTAL_INFLOW"] / 12

# Log-transformed average inflow for modeling stability
txn_features["LOG_AVG_INFLOW"] = np.log1p(txn_features["AVG_MONTHLY_INFLOW"])


# In[48]:



# Fill missing values with 0

num_cols = txn_features.select_dtypes(include=["float64", "int64"]).columns
txn_features[num_cols] = txn_features[num_cols].fillna(0)


# In[49]:


# Ensure numeric columns in loan_cashflow
loan_cashflow_df["CAPITAL_TO_BE_PAIED"] = pd.to_numeric(
    loan_cashflow_df["CAPITAL_TO_BE_PAIED"].astype(str).str.replace(",", "").str.replace(" ", ""),
    errors='coerce'
).fillna(0)

loan_cashflow_df["INTEREST_TO_BE_PAIED"] = pd.to_numeric(
    loan_cashflow_df["INTEREST_TO_BE_PAIED"].astype(str).str.replace(",", "").str.replace(" ", ""),
    errors='coerce'
).fillna(0)

loan_cashflow_df["TERM_AMOUNT"] = pd.to_numeric(
    loan_cashflow_df["TERM_AMOUNT"].astype(str).str.replace(",", "").str.replace(" ", ""),
    errors='coerce'
).fillna(0)


# Total scheduled per customer
scheduled = loan_cashflow_df.groupby("MASKED_ID").agg(
    TOTAL_CAPITAL_SCHEDULED=("CAPITAL_TO_BE_PAIED", "sum"),
    TOTAL_INTEREST_SCHEDULED=("INTEREST_TO_BE_PAIED", "sum"),
    TOTAL_AMOUNT_SCHEDULED=("TERM_AMOUNT", "sum")
).reset_index()



# Total actually paid from repayment table
actually_paid = repayment_df.groupby("MASKED_ID").agg(
    TOTAL_CAPITAL_PAID=("CAPITAL_PAIED", "sum"),
    TOTAL_INTEREST_PAID=("INTEREST_PAIED", "sum")
).reset_index()


# Merge
cash_features = scheduled.merge(actually_paid, on="MASKED_ID", how="left")

# ── Fix: force all columns to numeric BEFORE doing subtraction ──
cash_features["TOTAL_CAPITAL_PAID"]      = pd.to_numeric(cash_features["TOTAL_CAPITAL_PAID"],      errors='coerce').fillna(0)
cash_features["TOTAL_INTEREST_PAID"]     = pd.to_numeric(cash_features["TOTAL_INTEREST_PAID"],     errors='coerce').fillna(0)
cash_features["TOTAL_CAPITAL_SCHEDULED"] = pd.to_numeric(cash_features["TOTAL_CAPITAL_SCHEDULED"], errors='coerce').fillna(0)
cash_features["TOTAL_INTEREST_SCHEDULED"]= pd.to_numeric(cash_features["TOTAL_INTEREST_SCHEDULED"],errors='coerce').fillna(0)
cash_features["TOTAL_AMOUNT_SCHEDULED"]  = pd.to_numeric(cash_features["TOTAL_AMOUNT_SCHEDULED"],  errors='coerce').fillna(0)

# Due = Scheduled - Already Paid
cash_features["TOTAL_CAPITAL_DUE"]  = cash_features["TOTAL_CAPITAL_SCHEDULED"]  - cash_features["TOTAL_CAPITAL_PAID"]
cash_features["TOTAL_INTEREST_DUE"] = cash_features["TOTAL_INTEREST_SCHEDULED"] - cash_features["TOTAL_INTEREST_PAID"]
cash_features["AVG_PAYMENT_RATIO"]  = (cash_features["TOTAL_CAPITAL_PAID"]+cash_features["TOTAL_INTEREST_PAID"]) / cash_features["TOTAL_AMOUNT_SCHEDULED"].replace(0, 1)

# Keep only your original columns
cash_features = cash_features[["MASKED_ID", "TOTAL_CAPITAL_DUE", "TOTAL_INTEREST_DUE", "AVG_PAYMENT_RATIO"]]
cash_features = cash_features.fillna(0)

# # Aggregate per customer
# cash_features = loan_cashflow_df.groupby("MASKED_ID").agg({
#     "CAPITAL_TO_BE_PAIED": "sum",
#     "INTEREST_TO_BE_PAIED": "sum",
#     "PAYMENT_RATIO": "mean"
# }).reset_index()

# Rename columns
cash_features.columns = [
    "MASKED_ID",
    "TOTAL_CAPITAL_DUE",
    "TOTAL_INTEREST_DUE",
    "AVG_PAYMENT_RATIO"
]

# Fill missing values
cash_features = cash_features.fillna(0)


# In[50]:


import numpy as np

# Ensure numeric columns
account_df["MONTHEND_CONVERTED_BALANCE"] = pd.to_numeric(account_df["MONTHEND_CONVERTED_BALANCE"], errors='coerce').fillna(0)
account_df["CONVERTED_BALANCE"] = pd.to_numeric(account_df["CONVERTED_BALANCE"], errors='coerce').fillna(0)
account_df["OOD"] = pd.to_numeric(account_df["OOD"], errors='coerce').fillna(0)

# Calculate Utilization: ratio of loan balance / term amount
# (if TERM_AMOUNT available, else skip or adjust)
account_df["UTILIZATION"] = account_df["CONVERTED_BALANCE"] / account_df["TERM_AMOUNT"].replace(0, 1)

# Calculate ACCOUNT_AGE_MONTHS
# ----------------------------
# Ensure date columns are datetime
account_df["ORIG_CONTRACT_DATE"] = pd.to_datetime(account_df["ORIG_CONTRACT_DATE"], errors='coerce')
account_df["ACCT_CLOSE_DATE"] = pd.to_datetime(account_df["ACCT_CLOSE_DATE"], errors='coerce')

# Fix reference date for reproducibility (e.g., model training cutoff)
reference_date = pd.to_datetime("2026-01-31")

# Use ACCT_CLOSE_DATE if exists, else reference_date
account_df["END_DATE"] = account_df["ACCT_CLOSE_DATE"].fillna(reference_date)

# Calculate account age in months
# account_df["ACCOUNT_AGE_MONTHS"] = ((account_df["END_DATE"] - account_df["ORIG_CONTRACT_DATE"]) / np.timedelta64(1, "M")).round(0)
account_df["ACCOUNT_AGE_MONTHS"] = ((account_df["END_DATE"] - account_df["ORIG_CONTRACT_DATE"]) / np.timedelta64(1, "D") / 30).round(0)

# Aggregate per customer if multiple accounts
acc_features = account_df.groupby("MASKED_ID").agg({
    "MONTHEND_CONVERTED_BALANCE": "mean",
    "UTILIZATION": "mean",
    # "OOD": "max",
    "ACCOUNT_AGE_MONTHS": "mean"
}).reset_index()

# Fill missing numeric values
acc_features = acc_features.fillna(0)


# In[51]:


repayment_df["OOD"] = pd.to_numeric(
    repayment_df["OOD"],
    errors="coerce"
)


# In[52]:


re_numeric_cols = ["CAPITAL_PAIED", "INTEREST_PAIED"]

for col in re_numeric_cols:
    repayment_df[col] = pd.to_numeric(repayment_df[col], errors="coerce")  # convert strings to numbers
    repayment_df[col].fillna(0, inplace=True)  # replace NaN with 0


# In[53]:


ood_features = repayment_df.groupby("MASKED_ID").agg({
    "OOD": "max"
}).reset_index()

ood_features.rename(columns={"OOD": "MAX_OOD"}, inplace=True)


# In[54]:


payment_features = repayment_df.groupby("MASKED_ID").agg({
    "CAPITAL_PAIED": "sum",
    "INTEREST_PAIED": "sum"
}).reset_index()

payment_features["TOTAL_PAID"] = (
    payment_features["CAPITAL_PAIED"] +
    payment_features["INTEREST_PAIED"]
)

# import joblib

# # eligible_cus_df is your cleaned DataFrame
# import joblib
# joblib.dump(eligible_cus_df, "eligible_customers.pkl", protocol=4)

# In[55]:


model_data = eligible_cus_df.copy()


# In[56]:


model_data = model_data.merge(txn_features, on="MASKED_ID", how="left")


# In[57]:


model_data = model_data.merge(cash_features, on="MASKED_ID", how="left")


# In[58]:


model_data = model_data.merge(acc_features, on="MASKED_ID", how="left")


# In[59]:


# Merge customer-level cash features with MAX_OOD
model_data = model_data.merge(ood_features, on="MASKED_ID", how="left")

# Fill missing MAX_OOD for customers with no loans
model_data["MAX_OOD"] = model_data["MAX_OOD"].fillna(0)
#


# In[60]:


model_data = model_data.merge(payment_features, on="MASKED_ID", how="left")

# Fill missing MAX_OOD for customers with no loans
model_data["TOTAL_PAID"] = model_data["TOTAL_PAID"].fillna(0)


# In[61]:


num_cols = model_data.select_dtypes(include=["float64", "int64"]).columns
model_data[num_cols] = model_data[num_cols].fillna(0)


# In[62]:


# Identify categorical columns
cat_cols = model_data.select_dtypes(include=["category", "object"]).columns


# In[63]:



for col in cat_cols:
    if pd.api.types.is_categorical_dtype(model_data[col]):
        # Add 'Unknown' to categories if not already present
        if "Unknown" not in model_data[col].cat.categories:
            model_data[col] = model_data[col].cat.add_categories("Unknown")
    # Fill missing values
    model_data[col] = model_data[col].fillna("Unknown")


# In[64]:


# Creating the default flag


# In[65]:


# Filter only credit accounts (LOAN ACCOUNT or BORROWINGS)
loan_accounts = account_df[account_df["ACTIVE_PRODUCT"].isin(["LOAN ACCOUNT", "BORROWINGS"])]


# In[66]:



model_data["DEFAULT"] = model_data.apply(
    lambda x: 1 if (x["MAX_OOD"] >= 60 or (x["TOTAL_CAPITAL_DUE"] > 0 and x["AVG_PAYMENT_RATIO"] < 0.6)) else 0,
    axis=1
)


# In[67]:


#Select features for credit scoring
features = [
    # Transaction behavior
    "TOTAL_INFLOW", "TOTAL_OUTFLOW", "NET_TRANSACTION_SUM", "ABS_ACTIVITY",
    "AVG_TRANSACTION", "STD_TRANSACTION", "TXN_COUNT",
    "INFLOW_RATIO", "OUTFLOW_RATIO", "NET_RATIO",
    "AVG_TXN_PER_DAY", "TRANSACTION_VOLATILITY",
    "AVG_MONTHLY_INFLOW", "LOG_AVG_INFLOW",

    # # Loan / credit exposure (exclude TOTAL_CAPITAL_DUE, AVG_PAYMENT_RATIO)
    "TOTAL_INTEREST_DUE",

    # Account-level / customer-level
    "MONTHEND_CONVERTED_BALANCE", "UTILIZATION", "ACCOUNT_AGE_MONTHS"
]

# Demographic / profile features
demographic_features = [
    "AGE", "EMPLOYMENT_STATUS", "OCCUPATION", "GENDER", "MARITAL_STATUS", "DISTRICT"
]

# Combine features
features = features + demographic_features


# Convert categorical features
# -------------------------------
categorical_cols = ["EMPLOYMENT_STATUS", "OCCUPATION", "GENDER", "MARITAL_STATUS", "DISTRICT"]
for col in categorical_cols:
    model_data[col] = model_data[col].astype("category")


# In[68]:


X = model_data[features]
y = model_data["DEFAULT"]


# In[69]:


#xgboost


# In[70]:





# In[71]:


from xgboost import XGBClassifier

# Encode categorical columns for full model
X_full = X.copy()
for col in categorical_cols:
    X_full[col] = X_full[col].cat.codes

# Train final model on all customers
xgb_reg_full = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    scale_pos_weight=(len(y)-sum(y))/sum(y),
    reg_alpha=5,
    reg_lambda=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_reg_full.fit(X_full, y)


# In[72]:


# import joblib

# # ----- SAVE THE MODEL -----
# joblib.dump(xgb_reg_full, "credit_model.pkl", protocol=4)
# print("Model saved as credit_model.pkl")


# In[73]:


# Predict credit scores (probability of default)
model_data['default_probability'] = xgb_reg_full.predict_proba(X_full)[:, 1]


# In[74]:


y_proba = xgb_reg_full.predict_proba(X_full)[:,1]
y_pred = xgb_reg_full.predict(X_full)


# In[75]:


results = pd.DataFrame({
    'MASKED_ID': model_data['MASKED_ID'],
    'Default_Probability': y_proba,
    'Predicted_Default': y_pred
})


# In[76]:



min_score = 300
max_score = 850
results['Internal_Bank_Score'] = max_score - (y_proba * (max_score - min_score))
results['Internal_Bank_Default_Score'] = results['Internal_Bank_Score'].round().astype(int)


# In[77]:


bins = [300, 550, 650, 750, 850]
labels = ["High Risk","Medium Risk","Low Risk","Very Low Risk"]

results["Score_Band"] = pd.cut(
    results["Internal_Bank_Default_Score"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# In[78]:


final_table = model_data.merge(results, on="MASKED_ID", how="left")


# In[79]:


## Model 6 K-Prototypes


# In[80]:





# In[81]:


import pandas as pd
from kmodes.kprototypes import KPrototypes


# In[82]:


from kmodes.kprototypes import KPrototypes
import pandas as pd

# Use a separate copy for final clustering
eligible_final = final_table.copy()


# In[83]:


# Define features
numeric_feats_final = ['Monthly_Avg_Balance', 'Avg_Monthly_Credit','AGE']
categorical_feats_final = ['OCCUPATION','CUSTOMER_RISK_NAME','GENDER', 
                           'EMPLOYMENT_STATUS', 'MARITAL_STATUS','TARGET_DESC','Score_Band']


# In[84]:


# Fill missing values
eligible_final[numeric_feats_final] = eligible_final[numeric_feats_final].fillna(0)
for col in categorical_feats_final:
    eligible_final[col] = eligible_final[col].astype(str).fillna('Unknown')


# In[85]:


# Prepare cluster data
cluster_data_final = eligible_final[numeric_feats_final + categorical_feats_final].copy()
# Weight Internal_Bank_Default_Score higher
weight_factor = 5  # you can tune this
cluster_data_final['Internal_Bank_Default_Score'] = eligible_final['Internal_Bank_Default_Score'] * weight_factor


# In[86]:


eligible_final[categorical_feats_final] = eligible_final[categorical_feats_final].fillna('Unknown').astype(str)


# In[87]:



cat_idx_final = [cluster_data_final.columns.get_loc(col) for col in categorical_feats_final]


# In[88]:


# Fit K-Prototypes with k = 4
kproto_final = KPrototypes(n_clusters=4, init='Cao', random_state=42)
cluster_labels_final = kproto_final.fit_predict(cluster_data_final.values, categorical=cat_idx_final)


# # In[89]:


# import joblib

# # ----- SAVE THE CLUSTER MODEL -----
# joblib.dump(kproto_final, "kproto_cluster_model.pkl", protocol=4)
# print("K-Prototypes cluster model saved as kproto_cluster_model.pkl")


# In[90]:


# Assign cluster labels to dataframe
eligible_final['Cluster_KProto'] = cluster_labels_final


# In[91]:


# Merging to the original df


# In[92]:


# Keep only MASKED_ID and NAME_MASKED_ID from customer_df
customer_basic = customer_df[['MASKED_ID']].copy()

# Drop duplicate MASKED_IDs in eligible_final first
eligible_final = eligible_final.drop_duplicates(subset='MASKED_ID')

# Perform left join on MASKED_ID
customer_full_df = customer_basic.merge(
    eligible_final,
    on='MASKED_ID',
    how='left'  # keeps all customers from customer_basic
)


import joblib

joblib.dump(customer_full_df, "eligible_customers.pkl", protocol=4)
print("eligible_customers.pkl saved successfully!")

joblib.dump(xgb_reg_full, "credit_model.pkl", protocol=4)
print("credit_model.pkl saved successfully!")

joblib.dump(kproto_final, "kproto_cluster_model.pkl", protocol=4)
print("kproto_cluster_model.pkl saved successfully!")

joblib.dump(account_df, "account_df_full.pkl", protocol=4)
print("account_df_full.pkl saved successfully!")

joblib.dump(repayment_df, "repayment_df.pkl", protocol=4)
print("repayment_df.pkl saved successfully!")

joblib.dump(transaction_df, "transaction_df.pkl", protocol=4)
print("transaction_df.pkl saved successfully!")