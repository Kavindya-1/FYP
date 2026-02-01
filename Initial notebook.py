#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyodbc
import pandas as pd
from pycaret.classification import setup, compare_models

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


# In[4]:


account_df


# In[5]:


customer_df
customer_df


# # Rule 1

# In[6]:


customer_df['Eligibility_Flag'] = 'ELIGIBLE'
customer_df['Rejection_Reason'] = None


# In[7]:


customer_df.loc[
    (customer_df['AGE'] < 18) | (customer_df['AGE'] > 80),
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Regulatory Age Restriction']


# In[8]:


customer_df[customer_df['Eligibility_Flag'] == 'REJECT'][['AGE', 'Eligibility_Flag', 'Rejection_Reason']]


# In[9]:


customer_df['Eligibility_Flag'].value_counts()


# In[10]:


eligible_cus_df = customer_df[customer_df['Eligibility_Flag'] == 'ELIGIBLE'].copy()


# # Rule 2

# In[11]:


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


# In[12]:


eligible_cus_df = eligible_cus_df.merge(
    active_account_count,
    on='MASKED_ID',
    how='left'
)


# In[13]:



# if no accounts isnted of Nan use 0 
eligible_cus_df['Number_of_Active_Accounts'] = (
    eligible_cus_df['Number_of_Active_Accounts']
    .fillna(0)
    .astype(int)
)


# In[14]:


# Step 1: Set ELSE condition (default)
eligible_cus_df['Eligibility_Flag'] = 'ELIGIBLE'
eligible_cus_df['Rejection_Reason'] = 'Existing Customer'

# Step 2: Apply IF condition
eligible_cus_df.loc[
    eligible_cus_df['Number_of_Active_Accounts'] == 0,
    ['Eligibility_Flag', 'Rejection_Reason']
] = ['REJECT', 'Non-Existing Customer']


# In[15]:


eligible_cus_df


# In[16]:


eligible_cus_df['Number_of_Active_Accounts'].describe()


# In[17]:


eligible_cus_df['Number_of_Active_Accounts'].value_counts().sort_index()


# In[18]:


eligible_cus_df['Eligibility_Flag'].unique()


# # Rule 3 

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


eligible_cus_df


# In[22]:


# Check age range
print(eligible_cus_df['AGE'].min(), eligible_cus_df['AGE'].max())

# Check active accounts
print(eligible_cus_df['Number_of_Active_Accounts'].min())

# Check employment status
print(eligible_cus_df['EMPLOYMENT_STATUS'].unique())

# Check eligibility flag
print(eligible_cus_df['Eligibility_Flag'].unique())


# # Rule 4

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
    (~eligible_cus_df['AGE'].between(18, 60)),  # not between 18 and 60
    'Employment_Segment'
] = 'Not valid segment'


# # Rule 5

# In[27]:


eligible_cus_df['Age_Bucket'] = pd.cut(
    eligible_cus_df['AGE'],
    bins=[17, 25, 40, 60, 80],
    labels=['Young Adult', 'Adult', 'Middle-Aged', 'Senior']
)


# In[28]:


eligible_cus_df['Age_Bucket'].value_counts(dropna=False)


# In[29]:


balance_cols = ['JUN_25', 'JUL_25', 'AUG_25', 'SEP_25', 'OCT_25', 'NOV_25']


# In[30]:


account_df['Monthly_Avg_Balance'] = account_df[balance_cols].mean(axis=1)


# In[31]:


account_df


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


eligible_cus_df[['Monthly_Avg_Balance', 'Financial_Capacity']].sample(10)


# # Rule 6

# In[36]:


transaction_df


# In[37]:


transaction_df['AMOUNT_LCY'].dtype


# In[38]:


transaction_df['AMOUNT_LCY'] = pd.to_numeric(
    transaction_df['AMOUNT_LCY'],
    errors='coerce'
)


# In[39]:


transaction_df['AMOUNT_LCY'].dtype


# In[40]:


credit_df = transaction_df[
    transaction_df['AMOUNT_LCY'] > 0
].copy()


# In[41]:


credit_df[['AMOUNT_LCY']].head()


# In[42]:


credit_df['Month'] = pd.to_datetime(
    credit_df['BOOKING_DATE']
).dt.to_period('M')


# In[43]:


credit_df


# In[44]:


monthly_credit = (
    credit_df
    .groupby(['MASKED_ID', 'Month'], as_index=False)['AMOUNT_LCY']
    .sum()
)


# In[45]:


monthly_credit


# In[46]:


avg_monthly_income = (
    monthly_credit
    .groupby('MASKED_ID', as_index=False)['AMOUNT_LCY']
    .mean()
    .rename(columns={'AMOUNT_LCY': 'Avg_Monthly_Credit'})
)


# In[47]:


eligible_cus_df['MASKED_ID'] = eligible_cus_df['MASKED_ID'].astype(str)


# In[48]:


avg_monthly_income['MASKED_ID'] = avg_monthly_income['MASKED_ID'].astype(str)


# In[49]:


eligible_cus_df['MASKED_ID'].dtype


# In[50]:


avg_monthly_income['MASKED_ID'].dtype


# In[51]:


eligible_cus_df = eligible_cus_df.merge(
    avg_monthly_income,
    on='MASKED_ID',
    how='left'
)


# In[52]:


mask = (
    eligible_cus_df['Avg_Monthly_Credit'].isna() &
    (eligible_cus_df['Monthly_Avg_Balance'] > 0)
)

eligible_cus_df.loc[mask, 'Avg_Monthly_Credit'] = (
    eligible_cus_df.loc[mask, 'Monthly_Avg_Balance']
)


# In[53]:


eligible_cus_df


# In[54]:



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


# In[55]:


eligible_cus_df


# In[56]:


import pandas as pd

# Export to Excel
# You need openpyxl installed for Excel export
# pip install openpyxl
eligible_cus_df.to_excel("eligible_customers.xlsx", index=False, sheet_name="Eligible_Customers")

print("Data successfully exported to Excel!")


# # Start - Defining clusters

# In[57]:


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


# In[58]:


#testing model  K-Means


# In[59]:


# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=4, random_state=42)
# labels_km = kmeans.fit_predict(X)


# In[60]:


eligible_cus_df[features].isna().sum()


# In[61]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[62]:


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


# In[63]:


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


# In[64]:


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


# # Model 2 GMM

# In[65]:


from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

gmm_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('gmm', GaussianMixture(random_state=42))
])


# In[66]:


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


# In[67]:


from sklearn.metrics import silhouette_score

for k in range(2, 8):
    gmm_pipe.set_params(gmm__n_components=k)
    labels = gmm_pipe.fit_predict(X)

    X_transformed = gmm_pipe[:-1].transform(X)
    score = silhouette_score(X_transformed, labels)

    print(f'k={k}, silhouette={score:.3f}')


# # MODEL 3 Hierarchical Clustering

# In[68]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Preprocess (same logic, no leakage)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)


# In[69]:


from sklearn.metrics import silhouette_score

for k in range(2, 8):
    agg = AgglomerativeClustering(
        n_clusters=k,
        linkage='ward'  # best for numeric risk data
    )
    labels = agg.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    print(f'k={k}, silhouette={score:.3f}')


# In[70]:


from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

sample_idx = np.random.choice(len(X_scaled), size=300, replace=False)
Z = linkage(X_scaled[sample_idx], method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram (Sample)')
plt.show()


# # model 6 K-Prototypes

# In[71]:


get_ipython().system('pip install kmodes')


# In[72]:


import pandas as pd
from kmodes.kprototypes import KPrototypes


# In[73]:


# Step 2: Filter eligible customers
eligible_df = eligible_cus_df[eligible_cus_df['Eligibility_Flag'].str.upper() == 'ELIGIBLE'].copy()


# In[74]:


# Step 3: Define features
numeric_features = ['Monthly_Avg_Balance', 'Avg_Monthly_Credit','AGE']
categorical_features = ['OCCUPATION','CUSTOMER_RISK_NAME','GENDER', 'EMPLOYMENT_STATUS', 'MARITAL_STATUS','TARGET_DESC']


# In[75]:


# Fill missing values
eligible_df[numeric_features] = eligible_df[numeric_features].fillna(0)
for col in categorical_features:
    eligible_df[col] = eligible_df[col].astype(str).fillna('Unknown')


# In[76]:


cluster_data = eligible_df[numeric_features + categorical_features].copy()
cat_idx = [cluster_data.columns.get_loc(col) for col in categorical_features]


# In[77]:


eligible_df


# In[78]:


# import matplotlib.pyplot as plt

# cost = []
# K = range(2, 10)  # test 2 to 9 clusters

# for k in K:
#     kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
#     kproto.fit_predict(cluster_data, categorical=cat_idx)
#     cost.append(kproto.cost_)

# # Plot the elbow graph
# plt.figure(figsize=(8,5))
# plt.plot(K, cost, 'bx-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Cost')
# plt.title('Elbow Method for Optimal k (K-Prototypes)')
# plt.show()


# In[79]:


get_ipython().system('pip install gower')


# In[82]:


import pandas as pd
import numpy as np

# Select features
numeric_features = [
    'AGE',
    'Monthly_Avg_Balance',
    'Avg_Monthly_Credit'
]


categorical_features = [
    'GENDER',
    'EMPLOYMENT_STATUS',
    'MARITAL_STATUS',
    'Employment_Segment','TARGET_DESC','CUSTOMER_RISK_NAME'
]

gower_df = eligible_cus_df[numeric_features + categorical_features].copy()

# Handle missing values
gower_df[numeric_features] = gower_df[numeric_features].fillna(gower_df[numeric_features].median())
gower_df[categorical_features] = gower_df[categorical_features].fillna('Unknown')


# In[83]:


import gower

gower_dist = gower.gower_matrix(gower_df)


# In[84]:


from sklearn.cluster import AgglomerativeClustering

n_clusters = 4  # change after dendrogram analysis

hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='precomputed',
    linkage='average'
)

gower_df['Cluster'] = hc.fit_predict(gower_dist)


# In[85]:


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


# In[86]:


gower_df.groupby('Cluster')[numeric_features].mean()


# In[87]:


for col in categorical_features:
    print(f"\n{col}")
    print(pd.crosstab(gower_df['Cluster'], gower_df[col], normalize='index'))


# In[ ]:




