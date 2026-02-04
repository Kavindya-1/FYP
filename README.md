# Automated Personal Loan Underwriting Using Integrated Rule-Based and Behavioral Analytics

## Introduction
This project addresses inefficiencies in the personal loan underwriting process in Sri Lanka, where manual procedures cause delays and inconsistent risk assessments. The solution leverages rule-based validation, transactional analytics, and persona generation to deliver instant, consistent, and fair loan decisions for existing customers.

## Problem Statement
Banks currently take 3–7 working days to process personal loans due to manual verification and fragmented data systems. Despite having internal customer data, institutions fail to systematically use it for automation. This project proposes an automated underwriting system to reduce processing time to minutes.

## Project Aim
To develop an automated personal loan underwriting system that evaluates existing customers using rule-based validation, transactional analytics, and risk persona generation.

## Objectives
- Automatically validate customer-entered data against internal records.  
- Analyze salary patterns, transactions, and financial behavior to assess creditworthiness.  
- Determine loan approval outcomes (grant, partial grant, decline) instantly.  
- Reduce staff workload and standardize assessments.  

## Dataset and Sampling Framework
- Source: DFCC Bank anonymized records (~15,000).  
- Sampling: Stratified by age, occupation, sector, etc.  
- Compliance: Excluded minors (<18 years), anonymized all data.  
- Final sample: ~750 customers for each demographic with transaction, account, etc. data.  

## Tools and Software
- **Data Preprocessing & Mining:** Pandas, NumPy, Scikit-learn, Mlxtend  
- **Database:** SQL Server Management Studio, SQL queries  
- **Visualization:** Matplotlib, Seaborn  
- **Interface:** Testing different tools such as Power Apps, Power Automate, Streamlit, and other web-based solutions  
- **Development Environment:** Jupyter Notebook, Python scripts  

## Methodology
1. **Preprocessing:** Cleaning, feature engineering, outlier detection.  
2. **Rule-Based Validation:** Age restrictions, account activity, employment status, financial capacity.  
3. **Behavioral Analytics:** Transaction stability, monthly credits, salary clustering.  
4. **Clustering & Persona Generation:** K-Means, GMM, Hierarchical, K-Prototypes, Gower distance.  
5. **Risk Scoring:** Salary consistency, transaction stability, persona attributes.  
6. **Deployment:** Testing different tools for real-time loan decisions.  

## Implementation Evidence
- **Database Connection:** pyodbc to SQL Server.  
- **Data Extraction:** Customer, account, transaction, repayment tables.  
- **Rule-Based Checks:** Age, accounts, employment, financial capacity.  
- **Clustering Models:** K-Means, GMM, Hierarchical, K-Prototypes.  
- **Validation:** Silhouette score, AIC/BIC, dendrogram analysis.  
- **Export:** Eligible customers exported to Excel.  

## Preliminary Results
- Personas identified: High Salary, Medium Salary, Low Salary.  
- Risk segmentation based on balances and transactions.  
- Evidence: clustering scores, sample outputs, visualizations.  

## Project Timeline
- Literature Review → Data Preprocessing → Clustering → Rule Mining → Risk Scoring → Interface Development → Testing → Documentation.  

## Risk and Mitigation
- **Data Security:** Masking, encryption, audit controls.  
- **Low-Quality Data:** Cleaning, synthetic augmentation.  
- **Biased Outputs:** Expert validation, recalibration.  
- **Integration Issues:** Modular testing.  

## Conclusion
The proposed system reduces loan processing time from days to minutes, improves fairness, and standardizes risk assessment. Future work includes expanding datasets, integrating explainable AI, and aligning with regulatory frameworks.
