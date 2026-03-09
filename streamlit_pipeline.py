#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyodbc
import pandas as pd
from pycaret.classification import setup, compare_models
import numpy as np

print("All libraries imported successfully!")


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


# Exploratory Data Analysis

# In[4]:


# Number of rows & columns
print(customer_df.shape, account_df.shape, transaction_df.shape)


# In[5]:


# # Columns and types
# print(customer_df.dtypes)
# print(account_df.dtypes)

# # Quick info summary
# customer_df.info()
# account_df.info()


# In[6]:


# Missing values
customer_df.isnull().sum()


# In[7]:


customer_df['MARITAL_STATUS'].fillna('Unknown', inplace=True)
customer_df['EMPLOYMENT_STATUS'].fillna('Unknown', inplace=True)
customer_df['DISTRICT'].fillna('Unknown', inplace=True)
customer_df['OCCUPATION'].fillna('Unknown', inplace=True)
median_risk = customer_df['CUSTOMER_RISK'].median()
customer_df['CUSTOMER_RISK'].fillna(median_risk, inplace=True)
# chcking after replacing the Missing values
customer_df.isnull().sum()


# In[8]:


account_df.isnull().sum()


# In[9]:


accts_numeric_cols = ['TERM', 'TERM_AMOUNT', 'OOD', 'MONTHEND_CONVERTED_BALANCE', 'CONVERTED_BALANCE',
                'JUN_25','JUL_25','AUG_25','SEP_25','OCT_25','NOV_25']

account_df[accts_numeric_cols] = account_df[accts_numeric_cols].fillna(0)

account_df['ACCT_CLOSE_DATE'] = pd.to_datetime(account_df['ACCT_CLOSE_DATE'], errors='coerce')

#checking after filling with 0
account_df.isnull().sum()


# In[10]:


# Missing values
transaction_df.isnull().sum()


# In[11]:


# Missing values
loan_cashflow_df.isnull().sum()


# In[12]:


loan_numeric_cols = ['OOD', 'CAPITAL_TO_BE_PAIED', 'INTEREST_TO_BE_PAIED']
loan_cashflow_df[loan_numeric_cols] = loan_cashflow_df[loan_numeric_cols].fillna(0)

loan_cashflow_df['TO_BE_PAYMENT_DATE'] = pd.to_datetime(loan_cashflow_df['TO_BE_PAYMENT_DATE'], errors='coerce')

#checking after filling with 0
loan_cashflow_df.isnull().sum()


# In[13]:


repayment_df.isnull().sum()


# In[14]:


repayment_numeric_cols = ['OOD', 'CAPITAL_PAIED']
repayment_df[repayment_numeric_cols] = repayment_df[repayment_numeric_cols].fillna(0)

#checking after filling with 0
repayment_df.isnull().sum()


# In[15]:


# Numeric features summary
customer_df.describe()


# In[16]:


account_df.describe()


# In[17]:


transaction_df.describe()


# In[18]:


repayment_df.describe()


# In[19]:


loan_cashflow_df.describe()


# In[20]:



# Categorical features summary
customer_df.select_dtypes(include='object').nunique()


# In[21]:


account_df.select_dtypes(include='object').nunique()


# In[22]:


loan_cashflow_df.select_dtypes(include='object').nunique()


# In[23]:


repayment_df.select_dtypes(include='object').nunique()


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# Example: AGE distribution
sns.histplot(customer_df['AGE'], bins=20, kde=True)
plt.show()


# In[25]:


# Account balances
sns.boxplot(x=account_df['CONVERTED_BALANCE'])
plt.show()


# In[26]:


# Count of customers by risk
sns.countplot(x='CUSTOMER_RISK_NAME', data=customer_df)
plt.show()

# Account status
sns.countplot(x='ACCT_STATUS', data=account_df)
plt.show()


# In[27]:


account_df


# In[28]:


customer_df


# # Rule 1 - Hard Eligibility & Regulatory Rules

# ## Regulatory Age Check

# In[29]:


customer_df['Eligibility_Flag'] = 'ELIGIBLE'
customer_df['Rejection_Reason'] = None


# In[30]:


customer_df.loc[
    (customer_df['AGE'] < 18) | (customer_df['AGE'] > 80),
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Regulatory Age Restriction']


# In[31]:


customer_df[customer_df['Eligibility_Flag'] == 'REJECT'][['AGE', 'Eligibility_Flag', 'Rejection_Reason']]


# In[32]:


customer_df['Eligibility_Flag'].value_counts()


# In[33]:


eligible_cus_df = customer_df[customer_df['Eligibility_Flag'] == 'ELIGIBLE'].copy()


# ## Account Eligibility Check

# In[34]:


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


# In[35]:


eligible_cus_df = eligible_cus_df.merge(
    active_account_count,
    on='MASKED_ID',
    how='left'
)


# In[36]:



# if no accounts isnted of Nan use 0 
eligible_cus_df['Number_of_Active_Accounts'] = (
    eligible_cus_df['Number_of_Active_Accounts']
    .fillna(0)
    .astype(int)
)


# In[37]:


# Step 1: Set ELSE condition (default)
eligible_cus_df['Eligibility_Flag'] = 'ELIGIBLE'
eligible_cus_df['Rejection_Reason'] = 'Existing Customer'

# Step 2: Apply IF condition
eligible_cus_df.loc[
    eligible_cus_df['Number_of_Active_Accounts'] == 0,
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Non-Existing Customer']


# In[38]:


eligible_cus_df


# In[39]:


eligible_cus_df['Number_of_Active_Accounts'].describe()


# In[40]:


eligible_cus_df['Number_of_Active_Accounts'].value_counts().sort_index()


# In[41]:


# Count of eligible customers
eligible_count = eligible_cus_df[eligible_cus_df['Eligibility_Flag'] == 'ELIGIBLE'].shape[0]
print("Number of eligible customers:", eligible_count)


# In[42]:


eligible_cus_df['Eligibility_Flag'].unique()


# ## Employment Status Validation

# In[43]:


valid_employment_status = pd.read_sql(
    "SELECT DISTINCT EMPLOYMENT_STATUS FROM CUSTOMER_DETAILS",
    conn
)['EMPLOYMENT_STATUS'].dropna().tolist()


# In[44]:


eligible_cus_df['Employment_Status_Flag'] = 'Valid Employment Status'

eligible_cus_df.loc[
    ~eligible_cus_df['EMPLOYMENT_STATUS'].isin(valid_employment_status),
    'Employment_Status_Flag'
] = 'Invalid Employment Status'


# In[45]:


eligible_cus_df


# In[46]:


# Check age range
print(eligible_cus_df['AGE'].min(), eligible_cus_df['AGE'].max())

# Check active accounts
print(eligible_cus_df['Number_of_Active_Accounts'].min())

# Check employment status
print(eligible_cus_df['EMPLOYMENT_STATUS'].unique())

# Check eligibility flag
print(eligible_cus_df['Eligibility_Flag'].unique())


# # Rule 2 - Employment-Based Routing & Participation Rules

# ## Routing & Participation 

# In[47]:


eligible_cus_df['Employment_Segment'] = 'Other'


# In[48]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin(['EMPLOYED', 'SELF-EMPLOYED', 'BUSINESS'])) &
    (eligible_cus_df['AGE'].between(18, 60)),
    'Employment_Segment'
] = 'Core Working Group'


# In[49]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin(['UNEMPLOYED	', 'RETIRED', 'STUDENT', 'FREELANCE'])) &
    (eligible_cus_df['AGE'].between(18, 65)),
    'Employment_Segment'
] = 'Special Segment'


# In[50]:


eligible_cus_df.loc[
    (eligible_cus_df['EMPLOYMENT_STATUS'].isin([
        'UNEMPLOYED', 'RETIRED', 'STUDENT', 'FREELANCE', 
        'EMPLOYED', 'SELF-EMPLOYED', 'BUSINESS'
    ])) &
    (~eligible_cus_df['AGE'].between(18, 60)),  # not between 18 and 60
    'Employment_Segment'
] = 'Not valid segment'


# # Rule 3 - Age-Based Segments

# In[51]:


eligible_cus_df['Age_Bucket'] = pd.cut(
    eligible_cus_df['AGE'],
    bins=[17, 25, 40, 60, 80],
    labels=['Young Adult', 'Adult', 'Middle-Aged', 'Senior']
)


# In[52]:


eligible_cus_df['Age_Bucket'].value_counts(dropna=False)


# # Rule 4 - Financial capacity

# In[53]:


balance_cols = ['JUN_25', 'JUL_25', 'AUG_25', 'SEP_25', 'OCT_25', 'NOV_25']


# In[54]:


account_df['Monthly_Avg_Balance'] = account_df[balance_cols].mean(axis=1)


# In[55]:


account_df


# In[56]:


customer_balance_df = (
    account_df
    .groupby('MASKED_ID', as_index=False)['Monthly_Avg_Balance']
    .mean()
)


# In[57]:


eligible_cus_df = eligible_cus_df.merge(
    customer_balance_df,
    on='MASKED_ID',
    how='left'
)


# In[58]:


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


# In[59]:


eligible_cus_df[['Monthly_Avg_Balance', 'Financial_Capacity']].sample(10)


# # Rule 6 - Salary Verification Check

# In[60]:


transaction_df


# In[61]:


transaction_df['AMOUNT_LCY'].dtype


# In[62]:


transaction_df['AMOUNT_LCY'] = pd.to_numeric(
    transaction_df['AMOUNT_LCY'],
    errors='coerce'
)


# In[63]:


transaction_df['AMOUNT_LCY'].dtype


# In[64]:


credit_df = transaction_df[
    transaction_df['AMOUNT_LCY'] > 0
].copy()


# In[65]:


credit_df[['AMOUNT_LCY']].head()


# In[66]:


credit_df['Month'] = pd.to_datetime(
    credit_df['BOOKING_DATE']
).dt.to_period('M')


# In[67]:


credit_df


# In[68]:


monthly_credit = (
    credit_df
    .groupby(['MASKED_ID', 'Month'], as_index=False)['AMOUNT_LCY']
    .sum()
)


# In[69]:


monthly_credit


# In[70]:


avg_monthly_income = (
    monthly_credit
    .groupby('MASKED_ID', as_index=False)['AMOUNT_LCY']
    .mean()
    .rename(columns={'AMOUNT_LCY': 'Avg_Monthly_Credit'})
)


# In[71]:


eligible_cus_df['MASKED_ID'] = eligible_cus_df['MASKED_ID'].astype(str)


# In[72]:


avg_monthly_income['MASKED_ID'] = avg_monthly_income['MASKED_ID'].astype(str)


# In[73]:


eligible_cus_df['MASKED_ID'].dtype


# In[74]:


avg_monthly_income['MASKED_ID'].dtype


# In[75]:


eligible_cus_df = eligible_cus_df.merge(
    avg_monthly_income,
    on='MASKED_ID',
    how='left'
)


# In[76]:


mask = (
    eligible_cus_df['Avg_Monthly_Credit'].isna() &
    (eligible_cus_df['Monthly_Avg_Balance'] > 0)
)

eligible_cus_df.loc[mask, 'Avg_Monthly_Credit'] = (
    eligible_cus_df.loc[mask, 'Monthly_Avg_Balance']
)


# In[77]:


eligible_cus_df


# In[78]:



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


# In[79]:


# Total number of rows
total_rows = len(eligible_cus_df)
print("Total rows in table:", total_rows)


# In[80]:


eligible_cus_df


# In[81]:


import pandas as pd

# Export to Excel
# pip install openpyxl
eligible_cus_df.to_excel("eligible_customers.xlsx", index=False, sheet_name="Eligible_Customers")

print("Data successfully exported to Excel!")


# Creating Credit scores

# In[82]:



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


# In[83]:


txn_features


# In[84]:



# Fill missing values with 0

num_cols = txn_features.select_dtypes(include=["float64", "int64"]).columns
txn_features[num_cols] = txn_features[num_cols].fillna(0)

# Check result

print(txn_features.head())


# In[85]:


# Ensure numeric columns
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

# Payment ratio per installment
loan_cashflow_df["PAYMENT_RATIO"] = loan_cashflow_df["CAPITAL_TO_BE_PAIED"] / loan_cashflow_df["TERM_AMOUNT"]

# Aggregate per customer
cash_features = loan_cashflow_df.groupby("MASKED_ID").agg({
    "CAPITAL_TO_BE_PAIED": "sum",
    "INTEREST_TO_BE_PAIED": "sum",
    "PAYMENT_RATIO": "mean"
}).reset_index()

# Rename columns
cash_features.columns = [
    "MASKED_ID",
    "TOTAL_CAPITAL_DUE",
    "TOTAL_INTEREST_DUE",
    "AVG_PAYMENT_RATIO"
]

# Fill missing values
cash_features = cash_features.fillna(0)


# In[86]:


cash_features


# In[87]:


# # Aggregate per customer
# cash_features = loan_cashflow_df.groupby("MASKED_ID").agg({
#     "CAPITAL_TO_BE_PAIED": "sum",
#     "INTEREST_TO_BE_PAIED": "sum",
#     "PAYMENT_RATIO": "mean"
# }).reset_index()

# # Rename columns
# cash_features.columns = [
#     "MASKED_ID",
#     "TOTAL_CAPITAL_DUE",
#     "TOTAL_INTEREST_DUE",
#     "AVG_PAYMENT_RATIO"
# ]

# # Fill NaN for customers with no loans
# cash_features.fillna(0, inplace=True)


# In[88]:


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
account_df["ACCOUNT_AGE_MONTHS"] = ((account_df["END_DATE"] - account_df["ORIG_CONTRACT_DATE"]) / np.timedelta64(1, "M")).round(0)

# Aggregate per customer if multiple accounts
acc_features = account_df.groupby("MASKED_ID").agg({
    "MONTHEND_CONVERTED_BALANCE": "mean",
    "UTILIZATION": "mean",
    # "OOD": "max",
    "ACCOUNT_AGE_MONTHS": "mean"
}).reset_index()

# Fill missing numeric values
acc_features = acc_features.fillna(0)


# In[89]:


acc_features


# In[90]:


repayment_df["OOD"].dtype


# In[91]:


repayment_df["CAPITAL_PAIED"].dtype


# In[92]:


repayment_df["INTEREST_PAIED"].dtype


# In[93]:


repayment_df["OOD"] = pd.to_numeric(
    repayment_df["OOD"],
    errors="coerce"
)


# In[94]:


re_numeric_cols = ["CAPITAL_PAIED", "INTEREST_PAIED"]

for col in re_numeric_cols:
    repayment_df[col] = pd.to_numeric(repayment_df[col], errors="coerce")  # convert strings to numbers
    repayment_df[col].fillna(0, inplace=True)  # replace NaN with 0


# In[95]:


ood_features = repayment_df.groupby("MASKED_ID").agg({
    "OOD": "max"
}).reset_index()

ood_features.rename(columns={"OOD": "MAX_OOD"}, inplace=True)


print(ood_features.head())


# In[96]:


payment_features = repayment_df.groupby("MASKED_ID").agg({
    "CAPITAL_PAIED": "sum",
    "INTEREST_PAIED": "sum"
}).reset_index()

payment_features["TOTAL_PAID"] = (
    payment_features["CAPITAL_PAIED"] +
    payment_features["INTEREST_PAIED"]
)


# In[97]:


payment_features


# In[98]:


model_data = eligible_cus_df.copy()


# In[99]:


# eligible_cus_df


# In[100]:


model_data = model_data.merge(txn_features, on="MASKED_ID", how="left")


# In[101]:


print(model_data.columns)


# In[102]:


model_data = model_data.merge(cash_features, on="MASKED_ID", how="left")


# In[103]:


print(model_data.columns)


# In[104]:


model_data = model_data.merge(acc_features, on="MASKED_ID", how="left")


# In[105]:


print(model_data.columns)


# In[106]:


# Merge customer-level cash features with MAX_OOD
model_data = model_data.merge(ood_features, on="MASKED_ID", how="left")

# Fill missing MAX_OOD for customers with no loans
model_data["MAX_OOD"] = model_data["MAX_OOD"].fillna(0)
#


# In[107]:


print(model_data.columns)


# In[108]:


model_data = model_data.merge(payment_features, on="MASKED_ID", how="left")




# model_data = model_data.merge(payment_features[["MASKED_ID", "TOTAL_PAID"]], 
#                           on="MASKED_ID", how="left")

# Fill missing MAX_OOD for customers with no loans
model_data["TOTAL_PAID"] = model_data["TOTAL_PAID"].fillna(0)


# In[109]:


num_cols = model_data.select_dtypes(include=["float64", "int64"]).columns
model_data[num_cols] = model_data[num_cols].fillna(0)


# In[110]:


print(model_data.columns)


# In[111]:


# Display column names and their data types
print(model_data.dtypes)


# In[112]:


# Identify categorical columns
cat_cols = model_data.select_dtypes(include=["category", "object"]).columns

# Display the list
print(cat_cols)


# In[113]:



for col in cat_cols:
    if pd.api.types.is_categorical_dtype(model_data[col]):
        # Add 'Unknown' to categories if not already present
        if "Unknown" not in model_data[col].cat.categories:
            model_data[col] = model_data[col].cat.add_categories("Unknown")
    # Fill missing values
    model_data[col] = model_data[col].fillna("Unknown")


# In[114]:


print(model_data.head())
# print(model_data.info())
# print(model_data.isna().sum())


# # Creating the default flag

# In[115]:


# Filter only credit accounts (LOAN ACCOUNT or BORROWINGS)
loan_accounts = account_df[account_df["ACTIVE_PRODUCT"].isin(["LOAN ACCOUNT", "BORROWINGS"])]


# In[116]:




# # Create customer-level default
# customer_default = loan_accounts.groupby("MASKED_ID").apply(
#     lambda x: 1 if (x["OOD"] > 30).any() or x["ACCT_STATUS"].isin(["UNCLAIMED","ABANDONED","EXPIRED"]).any() else 0
# ).reset_index(name="DEFAULT")


# # Fill customers with no loan accounts with 0 (no default)
# model_data["DEFAULT"] = model_data["DEFAULT"].fillna(0)

# # Check distribution
# print(model_data["DEFAULT"].value_counts())

model_data["DEFAULT"] = model_data.apply(
    lambda x: 1 if (x["MAX_OOD"] >= 60 or (x["TOTAL_CAPITAL_DUE"] > 0 and x["AVG_PAYMENT_RATIO"] < 0.6)) else 0,
    axis=1
)

print(model_data["DEFAULT"].value_counts())


# In[117]:


# Customers flagged as default
print(model_data[model_data["DEFAULT"] == 1][["MASKED_ID", "MAX_OOD", "AVG_PAYMENT_RATIO", "TOTAL_CAPITAL_DUE"]].head())


# In[118]:



# Customers flagged as non-default
print(model_data[model_data["DEFAULT"] == 0][["MASKED_ID", "MAX_OOD", "AVG_PAYMENT_RATIO", "TOTAL_CAPITAL_DUE"]].head())


# In[119]:


model_data


# In[120]:


print(model_data.columns.tolist())


# In[121]:


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


# 2️⃣ Convert categorical features
# -------------------------------
categorical_cols = ["EMPLOYMENT_STATUS", "OCCUPATION", "GENDER", "MARITAL_STATUS", "DISTRICT"]
for col in categorical_cols:
    model_data[col] = model_data[col].astype("category")


# In[122]:


X = model_data[features]
y = model_data["DEFAULT"]


# In[123]:


from sklearn.model_selection import train_test_split

#Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 30% for testing, 70% for training
    random_state=42,     # for reproducibility
    stratify=y           # preserves the default/non-default ratio
)

# Optional: check the distribution
print("Train set class distribution:\n", y_train.value_counts())
print("Test set class distribution:\n", y_test.value_counts())


# # CatBoost

# In[124]:


pip install catboost


# In[125]:


# 1️⃣ Import libraries
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd


# In[126]:


# 3️⃣ Create CatBoost Pools
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_cols)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_cols)


# In[127]:


# 4️⃣ Initialize CatBoost model
cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)


# In[128]:


# 5️⃣ Train the model
cat_model.fit(train_pool)


# In[129]:


# 6️⃣ Predict on test set
y_pred = cat_model.predict(X_test)
y_proba = cat_model.predict_proba(X_test)[:, 1]  # probability of default


# In[130]:


# 7️⃣ Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[131]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Non-Default', 'Default'], columns=['Predicted Non-Default', 'Predicted Default'])
print("\nConfusion Matrix:")
print(cm_df)


# In[132]:



# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print("\nROC-AUC:", roc_auc)


# # XG boost

# In[133]:


get_ipython().system('pip install xgboost')


# In[134]:


# from xgboost import XGBClassifier

# xgb_model = XGBClassifier(
#     n_estimators=200,
#     max_depth=6,
#     scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
#     random_state=42,
#     enable_categorical=True,
#     # use_label_encoder=False,
#     eval_metric='logloss'
#     #  reg_alpha=0.1,  # L1 regularization
#     # reg_lambda=1    # L2 regularization
# )


# # RFE (Recursive Feature Elimination) selects the top 10 features that contribute most to prediction

# In[135]:


# from xgboost import XGBClassifier
# from sklearn.feature_selection import RFE

# xgb_RFE= XGBClassifier(
#     n_estimators=200,
#     max_depth=6,
#     scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
#     random_state=42,
#     enable_categorical=True,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )


# In[136]:


# rfe = RFE(estimator=xgb_model, n_features_to_select=10)


# In[137]:


# rfe.fit(X_train, y_train)


# In[138]:


# selected_features = X_train.columns[rfe.support_].tolist()
# print("Top selected features:", selected_features)


# In[139]:


# X_train_sel = X_train[selected_features]
# X_test_sel = X_test[selected_features]

# xgb_RFE = XGBClassifier(
#     n_estimators=200,
#     max_depth=6,
#     scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
#     random_state=42,
#     enable_categorical=True,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )

# xgb_RFE.fit(X_train_sel, y_train)
# y_pred = xgb_RFE.predict(X_test_sel)


# In[140]:


# from sklearn.feature_selection import RFE

# # Use RFE to select top 10–15 features
# rfe = RFE(estimator=xgb_model, n_features_to_select=10)


# In[141]:


# # Get selected features
# selected_features = X_train.columns[rfe.support_].tolist()
# print("Top selected features:", selected_features)


# In[142]:


# xgb_RFE= XGBClassifier(
#     n_estimators=200,
#     max_depth=6,
#     scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
#     random_state=42,
#     enable_categorical=True,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )

# X_train_sel = X_train[selected_features]
# X_test_sel = X_test[selected_features]

# xgb_RFE.fit(X_train_sel, y_train)
# y_pred = xgb_RFE.predict(X_test_sel)


# In[143]:


# # Step 2: Get feature importance by gain
# importance = xgb_RFE.get_booster().get_score(importance_type='gain')

# importance_df = pd.DataFrame({
#     'Feature': importance.keys(),
#     'Importance': importance.values()
# })

# # Step 3: Normalize to percentage
# importance_df['Importance_%'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()

# # Step 4: Sort descending
# importance_df = importance_df.sort_values(by='Importance_%', ascending=False)

# print(importance_df)


# In[144]:


# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6))
# plt.barh(importance_df['Feature'], importance_df['Importance_%'])
# plt.xlabel("Importance (%)")
# plt.title("XGBoost Feature Importance by Gain")
# plt.gca().invert_yaxis()
# plt.show()


# In[145]:


# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# # Predictions
# y_pred = xgb_RFE.predict(X_test_sel)
# y_proba = xgb_RFE.predict_proba(X_test_sel)[:, 1]


# In[146]:


# # Accuracy
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # ROC-AUC
# print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# In[147]:


# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Compute confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix:")
# print(cm)

# # Optional: visualize
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()


# # trying with regularization parameters in XGBoost (reg_alpha, reg_lambda) to reduce over-reliance on a single feature.

# In[148]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Use regularization
xgb_reg = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),  # balance classes
    reg_alpha=5,     # L1 regularization (try 0.5, 1, 5)
    reg_lambda=5,    # L2 regularization (try 0.5, 1, 5)
    random_state=42,
    enable_categorical=True,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Fit the model
xgb_reg.fit(X_train, y_train)

# Predictions
y_pred = xgb_reg.predict(X_test)
y_proba = xgb_reg.predict_proba(X_test)[:,1]


# In[149]:


import pandas as pd

# Get feature importance by gain
importance = xgb_reg.get_booster().get_score(importance_type='gain')

# Convert to DataFrame
importance_df = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
})

# Normalize to percentage
importance_df['Importance_%'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()

# Sort descending
importance_df = importance_df.sort_values(by='Importance_%', ascending=False)

print(importance_df)


# In[150]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sort by Importance %
importance_df = importance_df.sort_values(by='Importance_%', ascending=True)  # ascending for horizontal bar

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='Importance_%', y='Feature', data=importance_df, palette='viridis')
plt.title('XGBoost Feature Importance (%) by Gain')
plt.xlabel('Importance (%)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# In[151]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[152]:


# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# In[153]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # Log-transform or cap extreme values

# In[154]:


import numpy as np
import pandas as pd
from xgboost import XGBClassifier




# Cap at 95th percentile
cap_value = model_data["TOTAL_INTEREST_DUE"].quantile(0.95)
model_data["TOTAL_INTEREST_DUE_CAPPED"] = np.minimum(model_data["TOTAL_INTEREST_DUE"], cap_value)

# Optional: log transform
model_data["TOTAL_INTEREST_DUE_LOG"] = np.log1p(model_data["TOTAL_INTEREST_DUE_CAPPED"])

# Use the transformed column in features
features_transformed = features.copy()
features_transformed = [f if f != "TOTAL_INTEREST_DUE" else "TOTAL_INTEREST_DUE_LOG" for f in features_transformed]


# In[155]:


xgb_default_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    reg_alpha=1,       # L1 regularization
    reg_lambda=2,      # L2 regularization
    scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
    random_state=42,
    use_label_encoder=False,
    enable_categorical=True,
    eval_metric='logloss'
)

xgb_default_model.fit(X_train, y_train)


# In[156]:


y_pred = xgb_default_model.predict(X_test)
y_prob = xgb_default_model.predict_proba(X_test)[:,1]


# In[157]:


importance = xgb_default_model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
})
importance_df['Importance_%'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()
importance_df = importance_df.sort_values(by='Importance_%', ascending=False)
print(importance_df)


# In[158]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sort by Importance for better plotting
importance_df_sorted = importance_df.sort_values(by='Importance_%', ascending=True)  # horizontal bar

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance_%', y='Feature', data=importance_df_sorted, palette='viridis')
plt.title('XGBoost Feature Importance (%) by Gain')
plt.xlabel('Importance (%)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# In[159]:


# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# In[160]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Optional: visualize as heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # trying monotone_constraints

# In[161]:


import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

# Create monotone constraints: +1 = increasing, 0 = unconstrained
# Ensure the length matches X_train columns
# Example: only TOTAL_INTEREST_DUE is increasing, rest are unconstrained
monotone_constraints = []
for f in X_train.columns:
    if f == "TOTAL_INTEREST_DUE":
        monotone_constraints.append(1)
    else:
        monotone_constraints.append(0)

# Convert to tuple (XGBoost requires tuple)
monotone_constraints = tuple(monotone_constraints)


# In[162]:


# Example: constrain only TOTAL_INTEREST_DUE to increase
monotone_constraints = tuple(1 if f == "TOTAL_INTEREST_DUE" else 0 for f in X_train.columns)

# Create model
xgb_monotone_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    enable_categorical=True,
    monotone_constraints=monotone_constraints  # must be a tuple
)

# Train
xgb_monotone_model.fit(X_train, y_train)


# In[163]:




# Predict and evaluate
y_pred = xgb_monotone_model.predict(X_test)
y_proba = xgb_monotone_model.predict_proba(X_test)[:, 1]


# In[164]:


# Feature importance by gain
importance = xgb_monotone_model.get_booster().get_score(importance_type='gain')
importance_df_m = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
})
importance_df_m['Importance_%'] = 100 * importance_df_m['Importance'] / importance_df_m['Importance'].sum()
importance_df_m = importance_df_m.sort_values(by='Importance_%', ascending=False)
print(importance_df_m)


# In[165]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.barh(importance_df_m['Feature'], importance_df_m['Importance_%'])
plt.xlabel("Importance (%)")
plt.title("XGBoost Feature Importance by Gain")
plt.gca().invert_yaxis()
plt.show()


# In[166]:


# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# In[167]:


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Non-Default (0)", "Default (1)"],
            yticklabels=["Non-Default (0)", "Default (1)"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - XGBoost Monotone Model")
plt.show()


# # Chosing the monotone_constraints
# With Monotone Constraint
# 
# If we add a monotone constraint on TOTAL_INTEREST_DUE:
# 
# CatBoost/XGBoost is told: “predicted risk should increase or stay the same as TOTAL_INTEREST_DUE increases.”
# 
# After training, the predictions might look like:
# 
# Customer	TOTAL_INTEREST_DUE	Predicted Default Probability (with constraint)
# A	500	0.10
# B	2000	0.15
# C	5000	0.25
# 
# ✅ Now the predictions align with business intuition: higher debt → higher default risk.
# Why it matters in credit scoring
# 
# Regulatory compliance – Credit models are often audited. Monotone constraints make the model more interpretable.
# 
# Trust for business users – Relationship managers or credit officers can trust the model outputs.
# 
# Reduces nonsensical predictions – Without constraints, extreme or sparse values could create anomalies where a customer with very high debt gets a lower predicted risk.

# ## Chking the model data table count and the eligible data count

# In[168]:


# Count of rows (customers) in model_data
num_model_data = model_data.shape[0]

# Count of rows (customers) in eligible_cus_df
num_eligible = eligible_cus_df.shape[0]

print(f"Number of customers in model_data: {num_model_data}")
print(f"Number of eligible customers: {num_eligible}")


# # can i train on the full data set 

# In[169]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Copy features
X_cv = X.copy()

# Encode categorical columns as numeric codes
for col in categorical_cols:
    X_cv[col] = X_cv[col].cat.codes

# Define model with regularization
xgb_reg_cv = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    scale_pos_weight=(len(y)-sum(y))/sum(y),
    reg_alpha=5,
    reg_lambda=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 5-fold cross-validation with ROC-AUC
cv_scores = cross_val_score(xgb_reg_cv, X_cv, y, cv=5, scoring='roc_auc')
print("Cross-validation AUC scores:", cv_scores)
print("Mean CV AUC:", np.mean(cv_scores))


# In[170]:


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

# Predict credit scores (probability of default)
model_data['default_probability'] = xgb_reg_full.predict_proba(X_full)[:, 1]


# In[171]:


import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# Feature importance
importance = xgb_reg_full.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
})
importance_df['Importance_%'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()
importance_df = importance_df.sort_values(by='Importance_%', ascending=False)
print(importance_df)

# Evaluate model on full data (for reference)
y_pred_full = xgb_reg_full.predict(X_full)
print(classification_report(y, y_pred_full))
print("ROC-AUC:", roc_auc_score(y, model_data['default_probability']))


# In[172]:


y_proba = xgb_reg_full.predict_proba(X_full)[:,1]
y_pred = xgb_reg_full.predict(X_full)


# In[173]:


results = pd.DataFrame({
    'MASKED_ID': model_data['MASKED_ID'],
    'Default_Probability': y_proba,
    'Predicted_Default': y_pred
})


# In[174]:



min_score = 300
max_score = 850
results['Internal_Bank_Score'] = max_score - (y_proba * (max_score - min_score))
results['Internal_Bank_Default_Score'] = results['Internal_Bank_Score'].round().astype(int)
results


# In[175]:


bins = [300, 550, 650, 750, 850]
labels = ["High Risk","Medium Risk","Low Risk","Very Low Risk"]

results["Score_Band"] = pd.cut(
    results["Internal_Bank_Default_Score"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# In[176]:


final_table = model_data.merge(results, on="MASKED_ID", how="left")


# In[177]:


# Total number of rows
total_rows = len(final_table)
print("Total rows in table:", total_rows)


# In[178]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Make sure actual labels exist
y_true = final_table['DEFAULT']          # Actual defaults
y_pred = final_table['Predicted_Default']  # Predicted defaults

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Default", "Default"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Full Dataset")
plt.show()

# Print counts
print(cm)


# In[179]:


len(final_table)


# In[180]:


score_summary = final_table.groupby("Score_Band").size().reset_index(name="Customer_Count")
print(score_summary)


# In[181]:


final_table


# # Testing SMOTE since the data is imbalanced

# In[182]:


# pip install imbalanced-learn


# In[183]:


# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, roc_auc_score
# from sklearn.model_selection import train_test_split


# In[184]:


# # Split data (separate variables)
# X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# # Scale numeric columns (separate scaler)
# numeric_cols = X.select_dtypes(include=["int64","float64"]).columns
# scaler_smote = StandardScaler()
# X_train_scaled = X_orig_train.copy()
# X_test_scaled = X_orig_test.copy()


# In[185]:


# X_train_scaled[numeric_cols] = scaler_smote.fit_transform(X_train_scaled[numeric_cols])
# X_test_scaled[numeric_cols] = scaler_smote.transform(X_test_scaled[numeric_cols])

# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# In[186]:


# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_res, y_train_res)


# In[187]:


# from sklearn.metrics import classification_report, roc_auc_score
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:,1]

# print(classification_report(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# In[188]:


# min_score = 250
# max_score = 900

# model_data["CREDIT_SCORE"] = (
#     (1 - model.predict_proba(scaler.transform(X))[:,1]) * (max_score - min_score) + min_score
# ).round(0)


# # Start - Defining clusters

# In[189]:


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

features = [
    'AGE',
    'Monthly_Avg_Balance',
    'Avg_Monthly_Credit',
    'Number_of_Active_Accounts'
]

X = eligible_cus_df[features]

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# # Model 1 -  K-Means

# In[190]:


# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=4, random_state=42)
# labels_km = kmeans.fit_predict(X)


# In[191]:


eligible_cus_df[features].isna().sum()


# In[192]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[193]:


features = [
    'AGE',
    'Monthly_Avg_Balance',
    'Avg_Monthly_Credit',
    'Number_of_Active_Accounts'
]

X = eligible_cus_df[features]

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(random_state=42))
])


# Imputation step
# → fills missing values (median)

# Scaling step
# → standardizes features (required for distance-based clustering)

# KMeans clustering
# → the actual clustering algorithm


# In[194]:


from sklearn.metrics import silhouette_score

best_score = -1  # initialize BEFORE the loop
best_k = None

for k in range(2, 8):
    pipe.set_params(kmeans__n_clusters=k)
    labels = pipe.fit_predict(X)

    X_transformed = pipe[:-1].transform(X)
    score = silhouette_score(X_transformed, labels)

    print(f'k={k}, silhouette score={score:.3f}')

    # Keep track of the best score
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nOptimal number of clusters: {best_k} with silhouette score: {best_score:.3f}")


# In[195]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K = range(2, 8)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# # Model 2 - GMM

# In[196]:


from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

gmm_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('gmm', GaussianMixture(random_state=42))
])


# In[197]:


bic_scores = []
aic_scores = []

for k in range(2, 8):
    gmm_pipe.set_params(gmm__n_components=k)
    gmm_pipe.fit(X)

    gmm = gmm_pipe.named_steps['gmm']
    bic_scores.append(gmm.bic(gmm_pipe[:-1].transform(X)))
    aic_scores.append(gmm.aic(gmm_pipe[:-1].transform(X)))

    print(f'k={k}, BIC={bic_scores[-1]:.0f}, AIC={aic_scores[-1]:.0f}')
    
    
# Lower BIC = better

# BIC penalizes too many clusters

# Usually smoother and safer than AIC


# In[198]:


from sklearn.metrics import silhouette_score

for k in range(2, 8):
    gmm_pipe.set_params(gmm__n_components=k)
    labels = gmm_pipe.fit_predict(X)

    X_transformed = gmm_pipe[:-1].transform(X)
    score = silhouette_score(X_transformed, labels)

    print(f'k={k}, silhouette={score:.3f}')


# # Model 3 - Hierarchical Clustering

# In[199]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Preprocess (same logic, no leakage)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)


# In[200]:


from sklearn.metrics import silhouette_score

for k in range(2, 8):
    agg = AgglomerativeClustering(
        n_clusters=k,
        linkage='ward'  # best for numeric risk data
    )
    labels = agg.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    print(f'k={k}, silhouette={score:.3f}')


# In[201]:


from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

sample_idx = np.random.choice(len(X_scaled), size=300, replace=False)
Z = linkage(X_scaled[sample_idx], method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram (Sample)')
plt.show()


# # Model 6 K-Prototypes

# In[202]:


get_ipython().system('pip install kmodes')


# In[203]:


import pandas as pd
from kmodes.kprototypes import KPrototypes


# In[204]:


# Filter eligible customers
eligible_df_k = final_table[final_table['Eligibility_Flag'].str.upper() == 'ELIGIBLE'].copy()


# In[205]:


# Total number of rows
total_rows = len(eligible_df_k)
print("Total rows in table:", total_rows)


# In[206]:


eligible_df_k


# In[207]:


# Define features
numeric_features = ['Monthly_Avg_Balance', 'Avg_Monthly_Credit','AGE','Internal_Bank_Default_Score']
categorical_features = ['OCCUPATION','CUSTOMER_RISK_NAME','GENDER', 'EMPLOYMENT_STATUS', 'MARITAL_STATUS','TARGET_DESC']


# In[208]:


# Fill missing values
eligible_df_k[numeric_features] = eligible_df_k[numeric_features].fillna(0)
for col in categorical_features:
    eligible_df_k[col] = eligible_df_k[col].astype(str).fillna('Unknown')


# In[209]:


cluster_data = eligible_df_k[numeric_features + categorical_features].copy()
cat_idx = [cluster_data.columns.get_loc(col) for col in categorical_features]


# In[210]:


cluster_data


# In[211]:


# import matplotlib.pyplot as plt

# cost = []
# K = range(2, 10) 
#  # test 2 to 9 clusters

# for k in K:
#     kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
#     kproto.fit_predict(cluster_data, categorical=cat_idx)
#     cost.append(kproto.cost_)



# In[212]:


from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Prepare numeric + encoded categorical data
cluster_numeric = cluster_data.copy()
for col in categorical_features:
    cluster_numeric[col] = cluster_numeric[col].astype('category').cat.codes


# In[213]:


# Define range of clusters to test
K = range(2, 9)

# Initialize lists to store metrics
cost = []
sil_scores = []
ch_scores = []

# Loop once per k and compute all metrics
for k in K:
    kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
    labels = kproto.fit_predict(cluster_data.values, categorical=cat_idx)

    cost.append(kproto.cost_)
    sil_scores.append(silhouette_score(cluster_numeric, labels, metric='euclidean'))
    ch_scores.append(calinski_harabasz_score(cluster_numeric, labels))
    
  


# In[214]:


print("K:", K)
print("Length of K:", len(K))
print("Cost:", cost)
print("Length of cost:", len(cost))


# In[215]:


# Plot Elbow (Cost)
plt.figure(figsize=(8,5))
plt.plot(K, cost, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost')
plt.title('Elbow Method for K-Prototypes')
plt.show()


# In[216]:


get_ipython().system('pip install kneed')


from kneed import KneeLocator
import matplotlib.pyplot as plt

knee = KneeLocator(K, cost, curve='convex', direction='decreasing')
optimal_k = knee.knee

print("Optimal number of clusters (Elbow Method):", optimal_k)


# In[217]:


# Plot Silhouette Score
plt.figure(figsize=(8,5))
plt.plot(K, sil_scores, 'rx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-Prototypes')
plt.show()


# In[218]:


# Plot Calinski-Harabasz Index
plt.figure(figsize=(8,5))
plt.plot(K, ch_scores, 'gx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Calinski-Harabasz Index')
plt.title('Calinski-Harabasz Index for K-Prototypes')
plt.show()


# In[219]:


# Print all metrics per k
for i, k_val in enumerate(K):
    print(f"k={k_val} -> Cost: {cost[i]:.2f}, Silhouette: {sil_scores[i]:.4f}, CH: {ch_scores[i]:.2f}")


# In[220]:


# Copy cluster_data so we don't modify the original
cluster_numeric = cluster_data.copy()

for col in categorical_features:
    cluster_numeric[col] = cluster_numeric[col].astype('category').cat.codes


# # Model 7 - Gower

# In[221]:


get_ipython().system('pip install gower')


# In[222]:


import pandas as pd
import numpy as np

# Select features
numeric_features = [
    'AGE',
    'Monthly_Avg_Balance',
    'Avg_Monthly_Credit','Internal_Bank_Default_Score'
]


categorical_features = [
    'GENDER',
    'EMPLOYMENT_STATUS',
    'MARITAL_STATUS',
    'Employment_Segment','TARGET_DESC','CUSTOMER_RISK_NAME'
]

gower_df = final_table[
    final_table['Eligibility_Flag'].str.upper() == 'ELIGIBLE'
][numeric_features + categorical_features].copy()

for col in categorical_features:
    gower_df[col] = gower_df[col].astype('category')  # ensure categorical
    if 'Unknown' not in gower_df[col].cat.categories:
        gower_df[col] = gower_df[col].cat.add_categories('Unknown')
    gower_df[col] = gower_df[col].fillna('Unknown')


# In[223]:


# Handle missing values for numeric columns
gower_df[numeric_features] = gower_df[numeric_features].fillna(gower_df[numeric_features].median())

# Convert categorical columns to string and fill missing values
gower_df[categorical_features] = gower_df[categorical_features].astype(str).fillna('Unknown')

# Now compute Gower distance
import gower
gower_dist = gower.gower_matrix(gower_df)


# In[224]:


import gower

gower_dist = gower.gower_matrix(gower_df)


# In[225]:


from sklearn.cluster import AgglomerativeClustering

n_clusters = 4  

hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='precomputed',
    linkage='average'
)

gower_df['Cluster'] = hc.fit_predict(gower_dist)


# In[226]:


from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

sample = gower_df.sample(n=500, random_state=42)
gower_sample = gower.gower_matrix(sample)

Z = linkage(gower_sample, method='average')

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (Gower Distance)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()


# In[227]:


gower_df.groupby('Cluster')[numeric_features].mean()


# In[228]:


for col in categorical_features:
    print(f"\n{col}")
    print(pd.crosstab(gower_df['Cluster'], gower_df[col], normalize='index'))


# # comparing each model

# In[229]:


from sklearn.preprocessing import StandardScaler

# One-hot encode categorical features
encoded_df = pd.get_dummies(cluster_data, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
encoded_df[numeric_features] = scaler.fit_transform(encoded_df[numeric_features])

# Convert to numpy array for metrics
X_encoded = encoded_df.values

print("X_encoded rows:", len(X_encoded))


# In[230]:


k = 4

# K-Prototypes
kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
kproto_labels = kproto.fit_predict(cluster_data.values, categorical=cat_idx)
sil_kproto = silhouette_score(X_encoded, kproto_labels, metric='euclidean')

# Gower
hc = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
gower_labels = hc.fit_predict(gower_dist)
sil_gower = silhouette_score(gower_dist, gower_labels, metric='precomputed')

print("Silhouette K-Prototypes:", sil_kproto)
print("Silhouette Gower:", sil_gower)


# In[231]:


# CH Scores
from sklearn.metrics import calinski_harabasz_score
ch_kproto = calinski_harabasz_score(X_encoded, kproto_labels)
ch_gower = calinski_harabasz_score(X_encoded, gower_labels)

print("Calinski-Harabasz K-Prototypes:", ch_kproto)
print("Calinski-Harabasz Gower:", ch_gower)


# In[232]:


from sklearn.metrics import davies_bouldin_score

# Davies–Bouldin for K-Prototypes
db_kproto = davies_bouldin_score(X_encoded, kproto_labels)
print("Davies–Bouldin Index (K-Prototypes):", db_kproto)

# Davies–Bouldin for Gower clusters
db_gower = davies_bouldin_score(X_encoded, gower_labels)
print("Davies–Bouldin Index (Gower + Agglomerative):", db_gower)


# In[233]:


from sklearn.metrics import adjusted_rand_score

# Assuming you already have:
# kproto_labels -> cluster labels from K-Prototypes
# gower_labels  -> cluster labels from Gower + Agglomerative

# Compute ARI
ari_score = adjusted_rand_score(kproto_labels, gower_labels)

print("Adjusted Rand Index (K-Prototypes vs Gower):", ari_score)


# # Chosing K prototype

# In[234]:


from kmodes.kprototypes import KPrototypes
import pandas as pd

# Use a separate copy for final clustering
eligible_final = final_table.copy()


# In[235]:


eligible_final


# In[236]:


# Define features
numeric_feats_final = ['Monthly_Avg_Balance', 'Avg_Monthly_Credit','AGE']
categorical_feats_final = ['OCCUPATION','CUSTOMER_RISK_NAME','GENDER', 
                           'EMPLOYMENT_STATUS', 'MARITAL_STATUS','TARGET_DESC','Score_Band']


# In[237]:


# Fill missing values
eligible_final[numeric_feats_final] = eligible_final[numeric_feats_final].fillna(0)
for col in categorical_feats_final:
    eligible_final[col] = eligible_final[col].astype(str).fillna('Unknown')


# In[238]:


# Prepare cluster data
cluster_data_final = eligible_final[numeric_feats_final + categorical_feats_final].copy()
# Weight Internal_Bank_Default_Score higher
weight_factor = 5  # you can tune this
cluster_data_final['Internal_Bank_Default_Score'] = eligible_final['Internal_Bank_Default_Score'] * weight_factor


# In[239]:


eligible_final[categorical_feats_final] = eligible_final[categorical_feats_final].fillna('Unknown').astype(str)


# In[240]:



cat_idx_final = [cluster_data_final.columns.get_loc(col) for col in categorical_feats_final]


# In[241]:


# Fit K-Prototypes with k = 4
kproto_final = KPrototypes(n_clusters=4, init='Cao', random_state=42)
cluster_labels_final = kproto_final.fit_predict(cluster_data_final.values, categorical=cat_idx_final)


# In[242]:


# Assign cluster labels to dataframe
eligible_final['Cluster_KProto'] = cluster_labels_final


# In[243]:


eligible_final


# In[244]:


# # Encode categorical columns as numeric (optional, for metrics)
# cluster_numeric_final = cluster_data_final.copy()
# for col in categorical_feats_final:
#     cluster_numeric_final[col] = cluster_numeric_final[col].astype('category').cat.codes


# In[245]:


# Show cluster sizes
cluster_sizes_final = eligible_final['Cluster_KProto'].value_counts().sort_index()
print("Cluster sizes (k=4):")
print(cluster_sizes_final)


# In[246]:


cluster_summary_list = []

for cluster in eligible_final['Cluster_KProto'].unique():
    cluster_data = eligible_final[eligible_final['Cluster_KProto'] == cluster]
    
    summary = {}
    summary['Cluster'] = cluster
    
    # Numeric features: mean
    for col in numeric_feats_final:
        summary[col] = round(cluster_data[col].mean(), 2)
    
    # Categorical features: top category + percentage
    for col in categorical_feats_final:
        top_cat = cluster_data[col].value_counts(normalize=True).idxmax()
        top_pct = round(cluster_data[col].value_counts(normalize=True).max() * 100, 1)
        summary[col] = f"{top_cat} ({top_pct}%)"
    
    cluster_summary_list.append(summary)

# Convert to DataFrame
cluster_summary_df = pd.DataFrame(cluster_summary_list)



# In[247]:


# Show table
cluster_summary_df


# In[248]:


import matplotlib.pyplot as plt
import seaborn as sns

# Choose two numeric features for the scatter plot
x_feature = 'Monthly_Avg_Balance'
y_feature = 'Avg_Monthly_Credit'

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=eligible_final,
    x=x_feature,
    y=y_feature,
    hue='Cluster_KProto',        # color by cluster
    palette='Set2',              # different colors
    s=100,                       # marker size
    alpha=0.7                     # transparency
)

plt.title('K-Prototypes Clusters')
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


# In[249]:


# Use Internal_Bank_Default_Score (or Internal_Bank_Score if scaled 300-850)
scores = eligible_final['Internal_Bank_Default_Score']

plt.figure(figsize=(12,6))

# KDE for smooth bell curve
sns.kdeplot(scores, fill=True, color='skyblue', bw_adjust=0.5, label='Credit Score Density')


# In[250]:


# # 6️⃣ PCA for 2D Visualization (numeric only)
# # -------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(eligible_final[numeric_feats_final])

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# eligible_final['PCA1'] = X_pca[:,0]
# eligible_final['PCA2'] = X_pca[:,1]


# In[251]:


# #Scatter Plot: Clusters Highlighted
# # -------------------------
# plt.figure(figsize=(12,8))
# clusters = eligible_final['Cluster_KProto'].unique()
# colors = sns.color_palette('tab10', n_colors=len(clusters))

# for i, c in enumerate(clusters):
#     subset = eligible_final[eligible_final['Cluster_KProto'] == c]
#     plt.scatter(subset['PCA1'], subset['PCA2'], s=60, alpha=0.7, label=f'Cluster {c}', color=colors[i])

# plt.xlabel('PCA Component 1', fontsize=12)
# plt.ylabel('PCA Component 2', fontsize=12)
# plt.title('Customer Clusters (K-Prototypes) in PCA Space', fontsize=14)
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.show()


# In[252]:


# # 8️⃣ Optional: Scatter colored by CREDIT_SCORE
# # -------------------------
# plt.figure(figsize=(12,8))
# scatter = plt.scatter(
#     eligible_final['PCA1'],
#     eligible_final['PCA2'],
#     c=eligible_final['CREDIT_SCORE'],
#     cmap='viridis',
#     s=60,
#     alpha=0.7
# )
# plt.colorbar(scatter, label='CREDIT_SCORE')
# plt.xlabel('PCA Component 1', fontsize=12)
# plt.ylabel('PCA Component 2', fontsize=12)
# plt.title('Customer PCA Colored by CREDIT_SCORE', fontsize=14)
# plt.grid(True)
# plt.show()


# In[253]:


# Assume kproto is your trained KPrototypes model
centroids = kproto.cluster_centroids_


# In[254]:


centroids


# In[255]:


import numpy as np

numeric_columns = ['Monthly_Avg_Balance', 'Avg_Monthly_Credit', 'AGE', 'CREDIT_SCORE']
categorical_columns = ['OCCUPATION','CUSTOMER_RISK_NAME','GENDER', 'EMPLOYMENT_STATUS', 'MARITAL_STATUS','TARGET_DESC']


# In[256]:


# import numpy as np

# numeric_columns = ['Monthly_Avg_Balance', 'Avg_Monthly_Credit', 'AGE', 'CREDIT_SCORE']
# categorical_columns = ['OCCUPATION','CUSTOMER_RISK_NAME','GENDER', 
#                        'EMPLOYMENT_STATUS', 'MARITAL_STATUS','TARGET_DESC']

# # Ensure numeric columns in eligible_final are float
# eligible_final[numeric_columns] = eligible_final[numeric_columns].astype(float)

# # cluster centroids split properly
# centroid_numeric = np.array(kproto.cluster_centroids_[0][cluster_num], dtype=float)
# centroid_categorical = np.array(kproto.cluster_centroids_[1][cluster_num])

# # get cluster 2 indices
# cluster_indices = np.where(clusters == cluster_num)[0]
# cluster_points = eligible_final.iloc[cluster_indices]

# distances = []
# for i, row in cluster_points.iterrows():
#     # numeric Euclidean distance
#     num_dist = np.linalg.norm(row[numeric_columns].values.astype(float) - centroid_numeric)
    
#     # categorical Hamming distance
#     cat_dist = sum(row[categorical_columns].values != centroid_categorical)
    
#     total_dist = num_dist + cat_dist
#     distances.append(total_dist)

# # identify outlier
# outlier_index = cluster_indices[np.argmax(distances)]
# outlier_point = eligible_final.iloc[outlier_index]

# # print("Outlier in Cluster 2:\n", outlier_point)


# # Meging to the original df

# In[258]:


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

print(customer_full_df.head())


# In[259]:


# Count of rows where Eligibility_Flag is null
null_count = customer_full_df['Eligibility_Flag'].isna().sum()
print("Number of customers with null Eligibility_Flag:", null_count)


# In[260]:


# Fill null Eligibility_Flag with "NOT ELIGIBLE"
customer_full_df['Eligibility_Flag'] = customer_full_df['Eligibility_Flag'].fillna("Regulatory Age Restriction")

# Check that nulls are gone
null_count_after = customer_full_df['Eligibility_Flag'].isna().sum()
print("Number of customers with null Eligibility_Flag after filling:", null_count_after)


# In[261]:


# Count of rows where Eligibility_Flag is null
null_count = customer_full_df['Eligibility_Flag'].isna().sum()
print("Number of customers with null Eligibility_Flag:", null_count)

