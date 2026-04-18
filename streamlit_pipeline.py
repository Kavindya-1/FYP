#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the nessary libraries
import pyodbc
import pandas as pd
from pycaret.classification import setup, compare_models
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler





#connecting to the database
import pyodbc
import pandas as pd

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"        # default instance
    "DATABASE=LOAN_PORTFOLIO_DB;"
    "Trusted_Connection=yes"   # use Windows Authentication
)


# In[3]:


#creating the dataframes
customer_df = pd.read_sql("SELECT * FROM CUSTOMER_DETAILS", conn)
account_df = pd.read_sql("SELECT * FROM ACCOUNT_DETAILS", conn)
transaction_df = pd.read_sql("SELECT * FROM TRANSACTION_DETAILS", conn)
loan_cashflow_df = pd.read_sql("SELECT * FROM LOAN_CASHFLOW", conn)
repayment_df = pd.read_sql("SELECT * FROM REPAYMENT", conn)



# # Exploratory Data Analysis



# filling the missing values in customer df
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




# filling the missing values in account df
accts_numeric_cols = ['TERM', 'TERM_AMOUNT', 'OOD', 'MONTHEND_CONVERTED_BALANCE', 'CONVERTED_BALANCE',
                'JUN_25','JUL_25','AUG_25','SEP_25','OCT_25','NOV_25']


account_df['ACCT_CLOSE_DATE'] = pd.to_datetime(account_df['ACCT_CLOSE_DATE'], errors='coerce')



#filling Missing values in loan_cashflow df

loan_numeric_cols = ['OOD', 'CAPITAL_TO_BE_PAIED', 'INTEREST_TO_BE_PAIED']

loan_cashflow_df['TO_BE_PAYMENT_DATE'] = pd.to_datetime(loan_cashflow_df['TO_BE_PAYMENT_DATE'], errors='coerce')





# In[18]:


# filling Missing values in account df

repayment_numeric_cols = ['OOD', 'CAPITAL_PAIED']



# ## categorical features summary in each dataframe

import matplotlib.pyplot as plt
import seaborn as sns






customer_df['Eligibility_Flag'] = 'ELIGIBLE'
customer_df['Rejection_Reason'] = None




customer_df.loc[
    (customer_df['AGE'] < 18) | (customer_df['AGE'] > 80),
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Regulatory Age Restriction']




customer_df[customer_df['Eligibility_Flag'] == 'REJECT'][['AGE', 'Eligibility_Flag', 'Rejection_Reason']]




eligible_cus_df = customer_df.copy()



# ## Account Eligibility Check

# In[37]:


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


# In[38]:


eligible_cus_df = eligible_cus_df.merge(
    active_account_count,
    on='MASKED_ID',
    how='left'
)


# In[39]:



# if no accounts isnted of Nan use 0 
eligible_cus_df['Number_of_Active_Accounts'] = (
    eligible_cus_df['Number_of_Active_Accounts']
    .fillna(0)
    .astype(int)
)


# In[40]:


# Step 1: Set ELSE condition (default)
eligible_cus_df['Eligibility_Flag'] = 'ELIGIBLE'
eligible_cus_df['Rejection_Reason'] = 'Existing Customer'

# Step 2: Apply IF condition
eligible_cus_df.loc[
    eligible_cus_df['Number_of_Active_Accounts'] == 0,
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Non-Existing Customer']




eligible_cus_df['Number_of_Active_Accounts'].value_counts().sort_index()



# Count of eligible customers
eligible_count = eligible_cus_df[eligible_cus_df['Eligibility_Flag'] == 'ELIGIBLE'].shape[0]



eligible_cus_df['Eligibility_Flag'].unique()




valid_employment_status = pd.read_sql(
    "SELECT DISTINCT EMPLOYMENT_STATUS FROM CUSTOMER_DETAILS",
    conn
)['EMPLOYMENT_STATUS'].dropna().tolist()



eligible_cus_df['Employment_Status_Flag'] = 'Valid Employment Status'

eligible_cus_df.loc[
    ~eligible_cus_df['EMPLOYMENT_STATUS'].isin(valid_employment_status),
    'Employment_Status_Flag'
] = 'Invalid Employment Status'






# # Rule 2 - Employment-Based Routing & Participation Rules

# ## Routing & Participation 

# In[50]:


eligible_cus_df['Employment_Segment'] = 'Other'


# In[51]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin(['EMPLOYED', 'SELF-EMPLOYED', 'BUSINESS'])) &
    (eligible_cus_df['AGE'].between(18, 60)),
    'Employment_Segment'
] = 'Core Working Group'


# In[52]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin(['UNEMPLOYED	', 'RETIRED', 'STUDENT', 'FREELANCE'])) &
    (eligible_cus_df['AGE'].between(18, 65)),
    'Employment_Segment'
] = 'Special Segment'


# In[53]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin([
        'UNEMPLOYED', 'RETIRED', 'STUDENT', 'FREELANCE', 
        'EMPLOYED', 'SELF-EMPLOYED', 'BUSINESS'
    ])) &
    (~eligible_cus_df['AGE'].between(18, 65)),  # not between 18 and 60
    'Employment_Segment'
] = 'Not valid segment'


# # Rule 3 - Age-Based Segments

# In[54]:


eligible_cus_df['Age_Bucket'] = pd.cut(
    eligible_cus_df['AGE'],
    bins=[17, 25, 40, 60, 80],
    labels=['Young Adult', 'Adult', 'Middle-Aged', 'Senior']
)


# In[55]:


eligible_cus_df['Age_Bucket'].value_counts(dropna=False)



# In[57]:


balance_cols = ['JUN_25', 'JUL_25', 'AUG_25', 'SEP_25', 'OCT_25', 'NOV_25']



account_df['Monthly_Avg_Balance'] = account_df[balance_cols].mean(axis=1)





customer_balance_df = (
    account_df
    .groupby('MASKED_ID', as_index=False)['Monthly_Avg_Balance']
    .mean()
)



eligible_cus_df = eligible_cus_df.merge(
    customer_balance_df,
    on='MASKED_ID',
    how='left'
)




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


# In[63]:


eligible_cus_df[['Monthly_Avg_Balance', 'Financial_Capacity']].sample(10)








transaction_df['AMOUNT_LCY'] = pd.to_numeric(
    transaction_df['AMOUNT_LCY'],
    errors='coerce'
)




credit_df = transaction_df[
    transaction_df['AMOUNT_LCY'] > 0
].copy()



credit_df[['AMOUNT_LCY']].head()


# In[70]:


credit_df['Month'] = pd.to_datetime(
    credit_df['BOOKING_DATE']
).dt.to_period('M')






monthly_credit = (
    credit_df
    .groupby(['MASKED_ID', 'Month'], as_index=False)['AMOUNT_LCY']
    .sum()
)





avg_monthly_income = (
    monthly_credit
    .groupby('MASKED_ID', as_index=False)['AMOUNT_LCY']
    .mean()
    .rename(columns={'AMOUNT_LCY': 'Avg_Monthly_Credit'})
)





eligible_cus_df['MASKED_ID'] = eligible_cus_df['MASKED_ID'].astype(str)




avg_monthly_income['MASKED_ID'] = avg_monthly_income['MASKED_ID'].astype(str)





eligible_cus_df = eligible_cus_df.merge(
    avg_monthly_income,
    on='MASKED_ID',
    how='left'
)




mask = (
    eligible_cus_df['Avg_Monthly_Credit'].isna() &
    (eligible_cus_df['Monthly_Avg_Balance'] > 0)
)

eligible_cus_df.loc[mask, 'Avg_Monthly_Credit'] = (
    eligible_cus_df.loc[mask, 'Monthly_Avg_Balance']
)







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







import pandas as pd

# Export to Excel
# pip install openpyxl
eligible_cus_df.to_excel("eligible_customers.xlsx", index=False, sheet_name="Eligible_Customers")





# Creating Credit scores



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




# Fill missing values with 0

num_cols = txn_features.select_dtypes(include=["float64", "int64"]).columns
txn_features[num_cols] = txn_features[num_cols].fillna(0)



# In[91]:


# ── Step 1: Clean loan_cashflow_df numeric columns ──
for col in ["CAPITAL_TO_BE_PAIED", "INTEREST_TO_BE_PAIED", "TERM_AMOUNT"]:
    loan_cashflow_df[col] = pd.to_numeric(
        loan_cashflow_df[col].astype(str).str.replace(",", "").str.replace(" ", ""),
        errors='coerce'
    ).fillna(0)

# ── Step 2: Clean repayment_df numeric columns ──
for col in ["CAPITAL_PAIED", "INTEREST_PAIED"]:
    repayment_df[col] = pd.to_numeric(
        repayment_df[col].astype(str).str.replace(",", "").str.replace(" ", ""),
        errors='coerce'
    ).fillna(0)

# ── Step 3: Total scheduled amounts per customer ──
scheduled = loan_cashflow_df.groupby("MASKED_ID").agg(
    TOTAL_CAPITAL_SCHEDULED  = ("CAPITAL_TO_BE_PAIED", "sum"),
    TOTAL_INTEREST_SCHEDULED = ("INTEREST_TO_BE_PAIED", "sum"),
    TOTAL_AMOUNT_SCHEDULED   = ("TERM_AMOUNT",          "sum")
).reset_index()

# ── Step 4: Total actually paid per customer ──
actually_paid = repayment_df.groupby("MASKED_ID").agg(
    TOTAL_CAPITAL_PAID  = ("CAPITAL_PAIED",  "sum"),
    TOTAL_INTEREST_PAID = ("INTEREST_PAIED", "sum")
).reset_index()

# ── Step 5: Payment behaviour — how consistently do they pay ──
payment_behaviour = repayment_df.groupby("MASKED_ID").agg(
    MONTHS_WITH_PAYMENT = ("CAPITAL_PAIED", lambda x: (x > 0).sum()),
    TOTAL_MONTHS        = ("CAPITAL_PAIED", "count")
).reset_index()

payment_behaviour["PAYMENT_FREQUENCY"] = (
    payment_behaviour["MONTHS_WITH_PAYMENT"] /
    payment_behaviour["TOTAL_MONTHS"].replace(0, 1)
)

# ── Step 6: Merge everything together ──
cash_features = scheduled     .merge(actually_paid,      on="MASKED_ID", how="left")     .merge(payment_behaviour,  on="MASKED_ID", how="left")

# ── Step 7: Force numeric on all columns ──
numeric_cols = [
    "TOTAL_CAPITAL_PAID",
    "TOTAL_INTEREST_PAID",
    "TOTAL_CAPITAL_SCHEDULED",
    "TOTAL_INTEREST_SCHEDULED",
    "TOTAL_AMOUNT_SCHEDULED",
    "MONTHS_WITH_PAYMENT",
    "TOTAL_MONTHS",
    "PAYMENT_FREQUENCY"
]
for col in numeric_cols:
    cash_features[col] = pd.to_numeric(
        cash_features[col], errors='coerce'
    ).fillna(0)

# ── Step 8: Calculate DUE columns (for DEFAULT definition only) ──
cash_features["TOTAL_CAPITAL_DUE"] = (
    cash_features["TOTAL_CAPITAL_SCHEDULED"] - cash_features["TOTAL_CAPITAL_PAID"]
)
cash_features["AVG_PAYMENT_RATIO"] = (
    (cash_features["TOTAL_CAPITAL_PAID"] + 
     cash_features["TOTAL_INTEREST_PAID"]) /
    cash_features["TOTAL_AMOUNT_SCHEDULED"].replace(0, 1)

)

# ── Step 9: Keep final columns ──
cash_features = cash_features[[
    "MASKED_ID",

    # ── For DEFAULT definition only — do NOT put in features ──
    "TOTAL_CAPITAL_DUE",       # used in DEFAULT condition 3
    "AVG_PAYMENT_RATIO",       # used in DEFAULT condition 3

    # ── Safe to use as model features ──
    "TOTAL_CAPITAL_SCHEDULED", # how much they borrowed
    "TOTAL_INTEREST_SCHEDULED",# cost of the loan
    "TOTAL_AMOUNT_SCHEDULED",  # total loan obligation
    "PAYMENT_FREQUENCY",       # how consistently they pay — key behavioural signal
    "MONTHS_WITH_PAYMENT",     # total months they made a payment
    "TOTAL_MONTHS",            # total months on record
]].fillna(0)






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






# In[98]:


repayment_df["OOD"] = pd.to_numeric(
    repayment_df["OOD"],
    errors="coerce"
)


# In[99]:


re_numeric_cols = ["CAPITAL_PAIED", "INTEREST_PAIED"]

for col in re_numeric_cols:
    repayment_df[col] = pd.to_numeric(repayment_df[col], errors="coerce")  # convert strings to numbers
    repayment_df[col].fillna(0, inplace=True)  # replace NaN with 0


# In[100]:


ood_features = repayment_df.groupby("MASKED_ID").agg({
    "OOD": "max"
}).reset_index()

ood_features.rename(columns={"OOD": "MAX_OOD"}, inplace=True)


# In[101]:


payment_features = repayment_df.groupby("MASKED_ID").agg({
    "CAPITAL_PAIED": "sum",
    "INTEREST_PAIED": "sum"
}).reset_index()

payment_features["TOTAL_PAID"] = (
    payment_features["CAPITAL_PAIED"] +
    payment_features["INTEREST_PAIED"]
)



# In[103]:


model_data = eligible_cus_df.copy()


# In[104]:


# eligible_cus_df


# In[105]:


model_data = model_data.merge(txn_features, on="MASKED_ID", how="left")



# In[107]:


model_data = model_data.merge(cash_features, on="MASKED_ID", how="left")




# In[109]:


model_data = model_data.merge(acc_features, on="MASKED_ID", how="left")



# In[111]:


# Merge customer-level cash features with MAX_OOD
model_data = model_data.merge(ood_features, on="MASKED_ID", how="left")

# Fill missing MAX_OOD for customers with no loans
model_data["MAX_OOD"] = model_data["MAX_OOD"].fillna(0)
#



# In[113]:


model_data = model_data.merge(payment_features, on="MASKED_ID", how="left")



# Fill missing MAX_OOD for customers with no loans
model_data["TOTAL_PAID"] = model_data["TOTAL_PAID"].fillna(0)


# In[114]:


num_cols = model_data.select_dtypes(include=["float64", "int64"]).columns
model_data[num_cols] = model_data[num_cols].fillna(0)




# In[117]:


# Identify categorical columns
cat_cols = model_data.select_dtypes(include=["category", "object"]).columns



# In[118]:



for col in cat_cols:
    if pd.api.types.is_categorical_dtype(model_data[col]):
        # Add 'Unknown' to categories if not already present
        if "Unknown" not in model_data[col].cat.categories:
            model_data[col] = model_data[col].cat.add_categories("Unknown")
    # Fill missing values
    model_data[col] = model_data[col].fillna("Unknown")






#defining the defualt 
loan_status = account_df[
    account_df["ACTIVE_PRODUCT"].isin(["LOAN ACCOUNT", "BORROWINGS"])
].groupby("MASKED_ID").agg(
    MAX_OOD_RAW  = ("OOD", "max"),
    WORST_STATUS = ("ACCT_STATUS", lambda x: 
                   "BAD" if any(s in ["ABANDONED", "UNCLAIMED", "EXPIRED"] 
                   for s in x) else "OK")
).reset_index()

# Merge into model_data
if "MAX_OOD_RAW" not in model_data.columns:
    model_data = model_data.merge(loan_status, on="MASKED_ID", how="left")
else:
    model_data = model_data.drop(columns=["MAX_OOD_RAW"], errors="ignore")
    model_data = model_data.merge(loan_status, on="MASKED_ID", how="left")

model_data["MAX_OOD_RAW"]  = model_data["MAX_OOD_RAW"].fillna(0)
model_data["WORST_STATUS"] = model_data["WORST_STATUS"].fillna("OK")

# ── Combined DEFAULT definition ──
model_data["DEFAULT"] = (
    # Condition 1: 30+ days overdue
    (model_data["MAX_OOD_RAW"] >= 90) |

    # Condition 2: Bad account status
    (model_data["WORST_STATUS"] == "BAD") |

    # Condition 3: Has capital due AND very poor payment ratio
    # (use TOTAL_CAPITAL_DUE and AVG_PAYMENT_RATIO from account_df
    #  NOT from model features — separate calculation!)
    (
        (model_data["TOTAL_CAPITAL_DUE"] > 0) &
        (model_data["AVG_PAYMENT_RATIO"] < 0.3)
    )
).astype(int)








# Ignore loans closed BEFORE 2025-05-01
# Because we have no repayment records for those within our window

cutoff_date = pd.to_datetime('2025-05-01')

trackable_loans = account_df[
    (account_df['ACTIVE_PRODUCT'].isin(['LOAN ACCOUNT', 'BORROWINGS'])) &
    (
        (account_df['ACCT_CLOSE_DATE'].isna()) |          # still active
        (account_df['ACCT_CLOSE_DATE'] >= cutoff_date)    # closed after May 2025
                                                           # so repayments exist
    )
]

customers_with_trackable_loans = trackable_loans['MASKED_ID'].unique()

model_data['Has_Trackable_Loan'] = model_data['MASKED_ID'].isin(customers_with_trackable_loans).astype(int)

# ── Step 2: Thin File Flag ──
# Thin file = never had a loan OR all loans closed before 2025-05-01
model_data['Thin_File_Flag'] = (model_data['Has_Trackable_Loan'] == 0).astype(int)



features = [
    # Transaction behaviour — ✅ all clean, no link to DEFAULT
    "NET_TRANSACTION_SUM",
    "AVG_TRANSACTION",
    "STD_TRANSACTION",
    "TXN_COUNT",
    "NET_RATIO",
    "AVG_TXN_PER_DAY",
    "TRANSACTION_VOLATILITY",
    "LOG_AVG_INFLOW",

    # Loan size — ✅ scheduled amounts, exist before default happens
    "TOTAL_CAPITAL_SCHEDULED",    # how much they borrowed
    "TOTAL_INTEREST_SCHEDULED",   # cost of their loan
 

    # Payment behaviour — ✅ pattern based, not consequence of default
    "PAYMENT_FREQUENCY",          # how consistently they pay (0.0 to 1.0)
    


    # Account level — ✅ all clean
    "MONTHEND_CONVERTED_BALANCE",
    "UTILIZATION",
    "ACCOUNT_AGE_MONTHS",
    "Number_of_Active_Accounts",

    # Risk
    "CUSTOMER_RISK_NAME",
]

# Demographic features
demographic_features = [
    "AGE",
    "EMPLOYMENT_STATUS",
    "OCCUPATION",
    "DISTRICT",
]

# Combine
features = features + demographic_features

# Categorical columns
categorical_cols = [
    "EMPLOYMENT_STATUS",
    "OCCUPATION",
    "DISTRICT",
    "CUSTOMER_RISK_NAME",
  
]

for col in categorical_cols:
    model_data[col] = model_data[col].astype("category")




# In[127]:


X = model_data[features]
y = model_data["DEFAULT"]


# In[128]:


from sklearn.model_selection import train_test_split

#Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 30% for testing, 70% for training
    random_state=42,     # for reproducibility
    stratify=y           # preserves the default/non-default ratio
)






# # can i train on the full data set 

# In[146]:


# ── Step 1: Prepare full dataset with category dtype ──
X_full = X.copy()
for col in categorical_cols:
    X_full[col] = X_full[col].astype("category")

# ── Step 2: Rebuild monotone constraints from X_full columns ──
# (X_full has same columns as X_train but we rebuild to be safe)
constraint_map = {
      # Transaction behaviour
    "NET_TRANSACTION_SUM"        : -1,  # more net flow       → lower risk
    "AVG_TRANSACTION"            : -1,  # higher avg txn      → lower risk
    "STD_TRANSACTION"            :  1,  # high volatility     → higher risk
    "TXN_COUNT"                  : -1,  # more transactions   → lower risk
    "NET_RATIO"                  : -1,  # better ratio        → lower risk
    "AVG_TXN_PER_DAY"            : -1,  # more activity       → lower risk
    "TRANSACTION_VOLATILITY"     :  1,  # more volatile       → higher risk
    "LOG_AVG_INFLOW"             : -1,  # more inflow         → lower risk

    # Loan size
    "TOTAL_CAPITAL_SCHEDULED"    :  1,  # bigger loan         → higher risk
    "TOTAL_INTEREST_SCHEDULED"   :  1,  # higher interest     → higher risk

    # Payment behaviour
    "PAYMENT_FREQUENCY"          : -1,  # pays more often     → lower risk


    # Account level
    "MONTHEND_CONVERTED_BALANCE" : -1,  # higher balance      → lower risk
    "UTILIZATION"                :  1,  # higher utilization  → higher risk
    "ACCOUNT_AGE_MONTHS"         : -1,  # older account       → lower risk
    "Number_of_Active_Accounts"  :  0,  # unclear direction

    # Categoricals — cannot be constrained
    "CUSTOMER_RISK_NAME"         :  0,
    "AGE"                        :  0,
    "EMPLOYMENT_STATUS"          :  0,
    "OCCUPATION"                 :  0,
    "DISTRICT"                   :  0,

}

# Build from X_full columns — guarantees correct order
monotone_constraints_full = tuple(
    constraint_map.get(f, 0) for f in X_full.columns
)

# ── Step 3: Retrain on full dataset with same rules ──
non_default  = (y == 0).sum()
default      = (y == 1).sum()
weight_full  = non_default / default

xgb_monotone_full = xgb.XGBClassifier(
    # ── Tuned hyperparameters ──
    n_estimators         = 400,   # ← changed
    max_depth            = 4,     # ← same
    learning_rate        = 0.01,  # ← changed
    subsample            = 0.8,   # ← same
    colsample_bytree     = 0.8,   # ← same
    reg_alpha            = 0.5,   # ← changed
    reg_lambda           = 1,     # ← changed
    min_child_weight     = 3,     # ← changed
    scale_pos_weight     = weight_full,
    random_state         = 42,
    eval_metric          = "logloss",
    enable_categorical   = True,
    # ── Same business rules enforced ──
    monotone_constraints = monotone_constraints_full
)
xgb_monotone_full.fit(X_full, y)



# In[147]:




# ── Step 4: Predict ──
mon_y_proba_all = xgb_monotone_full.predict_proba(X_full)[:, 1]
mon_y_pred_all  = xgb_monotone_full.predict(X_full)

# ── Step 5: Build results ──
results = pd.DataFrame({
    "MASKED_ID"           : model_data["MASKED_ID"],
    "Default_Probability" : mon_y_proba_all,
    "Predicted_Default"   : mon_y_pred_all
})

# ── Step 6: Generate credit score (250 to 900) ──
results["Internal_Bank_Default_Score"] = (
    900 - (mon_y_proba_all * (900 - 250))
).round().astype(int)


# In[148]:


# ── Step 7: Score bands ──
bins   = [250, 500, 620, 750, 900]
labels = ["High Risk", "Medium Risk", "Low Risk", "Very Low Risk"]

results["Score_Band"] = pd.cut(
    results["Internal_Bank_Default_Score"],
    bins           = bins,
    labels         = labels,
    include_lowest = True
)

# ── Convert to string so Unknown Risk can be added later ──
results["Score_Band"] = results["Score_Band"].astype(str)





final_table = model_data.merge(results, on="MASKED_ID", how="left")



score_summary = final_table.groupby("Score_Band").size().reset_index(name="Customer_Count")




# # Start - Defining clusters
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from kmodes.kprototypes import KPrototypes
import pandas as pd

# features = [
#     'AGE',
#     'Monthly_Avg_Balance',
#     'Avg_Monthly_Credit',
#     'Number_of_Active_Accounts'
# ]

# X = eligible_cus_df[features]

# # Handle missing values
# imputer = SimpleImputer(strategy='median')
# X_imputed = imputer.fit_transform(X)

# # Scale
# scaler = RobustScaler()
# X_scaled = scaler.fit_transform(X_imputed)


# Use a separate copy for final clustering
# eligible_final = final_table.copy()

# # Define features
# numeric_feats_final = ['Internal_Bank_Default_Score', 'Monthly_Avg_Balance', 'Avg_Monthly_Credit', 'Number_of_Active_Accounts', 'AVG_MONTHLY_INFLOW', 'MAX_OOD']
# categorical_feats_final = ['CUSTOMER_RISK_NAME']

# # Fill missing values
# eligible_final[numeric_feats_final] = eligible_final[numeric_feats_final].fillna(0)
# for col in categorical_feats_final:
#     eligible_final[col] = eligible_final[col].astype(str).fillna('Unknown')

# eligible_final[categorical_feats_final] = eligible_final[categorical_feats_final].fillna('Unknown').astype(str)

# # Prepare cluster data (copy — eligible_final is never touched)
# cluster_data_final = eligible_final[numeric_feats_final + categorical_feats_final].copy()

# # ✅ Scale only the copy
# cluster_scaler = RobustScaler()
# cluster_data_final[numeric_feats_final] = cluster_scaler.fit_transform(cluster_data_final[numeric_feats_final])

# # Weight Internal_Bank_Default_Score higher (applied AFTER scaling)
# weight_factor = 5
# cluster_data_final['Internal_Bank_Default_Score'] = cluster_data_final['Internal_Bank_Default_Score'] * weight_factor

# # Categorical indices
# cat_idx_final = [cluster_data_final.columns.get_loc(col) for col in categorical_feats_final]

# # Fit K-Prototypes with k = 4
# kproto_final = KPrototypes(n_clusters=4, init='Cao', random_state=42)
# cluster_labels_final = kproto_final.fit_predict(cluster_data_final.values, categorical=cat_idx_final)

# # ✅ Only cluster label is added to eligible_final — nothing else changed
# eligible_final['Cluster_KProto'] = cluster_labels_final

# eligible_final.to_excel("eligible_final.xlsx", index=False)

from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd

# ── Keep ALL selected features ────────────────────────────────────────────────
numeric_features     = [
    'Internal_Bank_Default_Score',
    'Monthly_Avg_Balance',
    'Avg_Monthly_Credit',
    'Number_of_Active_Accounts',
    'AVG_MONTHLY_INFLOW',
    'MAX_OOD'
]
categorical_features = ['CUSTOMER_RISK_NAME']

# ── 1. Filter eligible records ────────────────────────────────────────────────
eligible_final = final_table[
    final_table['Eligibility_Flag'].str.upper() == 'ELIGIBLE'
].copy().reset_index(drop=True)


# ── 2. Fill missing numeric with median ───────────────────────────────────────
eligible_final[numeric_features] = eligible_final[numeric_features].fillna(
    eligible_final[numeric_features].median()
)

# ── 3. Apply weight to Internal_Bank_Default_Score BEFORE scaling ─────────────
weight_factor = 5
eligible_final['Internal_Bank_Default_Score'] = (
    eligible_final['Internal_Bank_Default_Score'] * weight_factor
)

# ── 4. Fill missing categorical ───────────────────────────────────────────────
for col in categorical_features:
    eligible_final[col] = eligible_final[col].astype('category')
    if 'Unknown' not in eligible_final[col].cat.categories:
        eligible_final[col] = eligible_final[col].cat.add_categories('Unknown')
    eligible_final[col] = eligible_final[col].fillna('Unknown').astype(str)

# # ── 5. Collapse rare categories ───────────────────────────────────────────────
# eligible_final['CUSTOMER_RISK_NAME'] = eligible_final['CUSTOMER_RISK_NAME'].replace({
#     'HIGH'    : 'MEDIUM',
#     'CRITICAL': 'MEDIUM',
#     'UNKNOWN' : 'MEDIUM'
# })


# ── 6. Correct transformations per feature ────────────────────────────────────

# Monthly_Avg_Balance — has NEGATIVE values, use signed log
# signed log preserves direction: log1p(|x|) × sign(x)
eligible_final['Monthly_Avg_Balance'] = (
    np.sign(eligible_final['Monthly_Avg_Balance']) *
    np.log1p(np.abs(eligible_final['Monthly_Avg_Balance']))
)

# Avg_Monthly_Credit — all positive, standard log1p
eligible_final['Avg_Monthly_Credit'] = np.log1p(
    eligible_final['Avg_Monthly_Credit']
)

# AVG_MONTHLY_INFLOW — all positive, standard log1p
eligible_final['AVG_MONTHLY_INFLOW'] = np.log1p(
    eligible_final['AVG_MONTHLY_INFLOW']
)

# MAX_OOD — 75% zeros, use binary + magnitude split
# Instead of transforming, create a meaningful representation:
# log1p is fine here since all values >= 0
eligible_final['MAX_OOD'] = np.log1p(eligible_final['MAX_OOD'])

# Internal_Bank_Default_Score — already well distributed, no transform needed
# Number_of_Active_Accounts  — discrete, no transform needed

# ── 7. Check distributions after transformation ───────────────────────────────


# ── 8. Scale with RobustScaler ────────────────────────────────────────────────
scaler = RobustScaler()
eligible_final[numeric_features] = scaler.fit_transform(
    eligible_final[numeric_features]
)

# ── 9. Build input array ──────────────────────────────────────────────────────
cluster_data_final = eligible_final[
    numeric_features + categorical_features
].copy()

for col in categorical_features:
    cluster_data_final[col] = cluster_data_final[col].astype(str)

numeric_array_final = cluster_data_final[numeric_features].values.astype(float)
cat_array_final     = cluster_data_final[categorical_features].values.astype(str)
X_final             = np.concatenate([numeric_array_final, cat_array_final], axis=1)

cat_idx_final = [cluster_data_final.columns.get_loc(col)
                 for col in categorical_features]


# ── 10. Fit K-Prototypes with k=3 ─────────────────────────────────────────────

kproto_final = KPrototypes(
    n_clusters=3,
    init='Huang',
    n_init=10,
    random_state=42,
    verbose=0
)

eligible_final['Cluster_KProto'] = kproto_final.fit_predict(
    X_final, categorical=cat_idx_final
)


# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# features = [
#     'AGE',
#     'Monthly_Avg_Balance',
#     'Avg_Monthly_Credit',
#     'Number_of_Active_Accounts'
# ]

# X = eligible_cus_df[features]

# # Handle missing values
# imputer = SimpleImputer(strategy='median')
# X_imputed = imputer.fit_transform(X)

# # Scale
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_imputed)


# # # Chosing K prototype

# # In[200]:


# from kmodes.kprototypes import KPrototypes
# import pandas as pd

# # Use a separate copy for final clustering
# eligible_final = final_table.copy()




# # In[202]:


# # Define features
# numeric_feats_final = ['Internal_Bank_Default_Score', 'Monthly_Avg_Balance', 'Avg_Monthly_Credit', 'Number_of_Active_Accounts', 'AVG_MONTHLY_INFLOW', 'MAX_OOD']
# categorical_feats_final = ['OCCUPATION','CUSTOMER_RISK_NAME','GENDER', 
#                            'EMPLOYMENT_STATUS', 'MARITAL_STATUS','TARGET_DESC']


# # In[203]:


# # Fill missing values
# eligible_final[numeric_feats_final] = eligible_final[numeric_feats_final].fillna(0)
# for col in categorical_feats_final:
#     eligible_final[col] = eligible_final[col].astype(str).fillna('Unknown')

# # #scaler = RobustScaler()
# # scaler =StandardScaler()
# # eligible_final[numeric_feats_final] = scaler.fit_transform(eligible_final[numeric_feats_final])



# # Prepare cluster data
# cluster_data_final = eligible_final[numeric_feats_final + categorical_feats_final].copy()
# # Weight Internal_Bank_Default_Score higher
# weight_factor = 5  # you can tune this
# cluster_data_final['Internal_Bank_Default_Score'] = eligible_final['Internal_Bank_Default_Score'] * weight_factor



# eligible_final[categorical_feats_final] = eligible_final[categorical_feats_final].fillna('Unknown').astype(str)


# # In[206]:



# cat_idx_final = [cluster_data_final.columns.get_loc(col) for col in categorical_feats_final]


# # In[207]:


# # Fit K-Prototypes with k = 4
# kproto_final = KPrototypes(n_clusters=4, init='Cao', random_state=42)
# cluster_labels_final = kproto_final.fit_predict(cluster_data_final.values, categorical=cat_idx_final)


# # In[208]:


# # Assign cluster labels to dataframe
# eligible_final['Cluster_KProto'] = cluster_labels_final





# eligible_final.to_excel("eligible_final.xlsx", index=False)


import joblib

joblib.dump(eligible_final, "eligible_customers.pkl", protocol=4)
print(f"✅ eligible_customers.pkl saved — {len(eligible_final)} rows")


joblib.dump(xgb_monotone_full, "credit_model.pkl", protocol=4)
print("✅ credit_model.pkl saved successfully!")



joblib.dump(kproto_final, "kproto_cluster_model.pkl", protocol=4)
print("✅ kproto_cluster_model.pkl saved successfully!")


joblib.dump(account_df, "account_df_full.pkl", protocol=4)
print(f"✅ account_df_full.pkl saved — {len(account_df)} rows")

# print("account_df_full.pkl saved successfully!")

# joblib.dump(repayment_df, "repayment_df.pkl", protocol=4)
# print("repayment_df.pkl saved successfully!")

# joblib.dump(transaction_df, "transaction_df.pkl", protocol=4)
# print("transaction_df.pkl saved successfully!")



import joblib
import pandas as pd

# Clean repayment columns
repayment_df['CAPITAL_PAIED']  = pd.to_numeric(repayment_df['CAPITAL_PAIED'],  errors='coerce').fillna(0)
repayment_df['INTEREST_PAIED'] = pd.to_numeric(repayment_df['INTEREST_PAIED'], errors='coerce').fillna(0)
repayment_df['OOD']            = pd.to_numeric(repayment_df['OOD'],            errors='coerce').fillna(0)
repayment_df['TERM_AMOUNT']    = pd.to_numeric(repayment_df['TERM_AMOUNT'],    errors='coerce').fillna(0)
repayment_df['PAYMENT_DATE']   = pd.to_datetime(repayment_df['PAYMENT_DATE'],  errors='coerce')
repayment_df['TOTAL_PAID']     = repayment_df['CAPITAL_PAIED'] + repayment_df['INTEREST_PAIED']

joblib.dump(repayment_df, "repayment_history.pkl", protocol=4)
print(f"✅ repayment_history.pkl saved — {len(repayment_df)} rows")

import os
size = os.path.getsize("repayment_history.pkl") / (1024*1024)



import joblib
import pandas as pd

# Clean transaction columns
transaction_df['AMOUNT_LCY']   = pd.to_numeric(transaction_df['AMOUNT_LCY'],    errors='coerce').fillna(0)
transaction_df['BOOKING_DATE'] = pd.to_datetime(transaction_df['BOOKING_DATE'],  errors='coerce')

joblib.dump(transaction_df, "transaction_history.pkl", protocol=4)
print(f"✅ transaction_history.pkl saved — {len(transaction_df)} rows")

import os
size = os.path.getsize("transaction_history.pkl") / (1024*1024)
# %%
