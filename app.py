import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob
import numpy as np
 
# --- 1. LOAD THE BRAIN ---
model = joblib.load('attrition_xgb_model.pkl')
scaler = joblib.load('robust_scaler.pkl')
core_features = joblib.load('features_columns.pkl')
 
st.title("🛡️ HR Employee Retention Tool")
st.write("Upload a CSV file to identify which employees are likely to leave.")
 
# --- 2. UPLOAD BUTTON ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
 
if uploaded_file is not None:
    # Read the data the HR member uploaded
    df = pd.read_csv(uploaded_file)
    # --- 3. APPLY YOUR SCIENCE (Feature Engineering) ---
    # We must do the exact same math we did in Kaggle
    df['Comp_Ratio'] = df['Base_Salary'] / (df['Benchmark_Salary'] + 1)
    df['Stagnation_Index'] = df['Tenure_Years'] / (df['Career_Development'] + 0.1)
    df['Is_Contractor'] = np.where(df['Employment_Type'] == 'Contract', 1, 0)
    df['Survey_Sentiment'] = df['Feedback_Comments'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0)
 
    # --- 4. PREDICT ---
    # Scale the data using the scaler we brought from Kaggle
    X_scaled = scaler.transform(df[core_features])
    risk_probs = model.predict_proba(X_scaled)[:, 1]
    # Add the results to the table
    df['Risk_Score_%'] = (risk_probs * 100).round(1)
    # --- 5. SHOW RESULTS ---
    st.subheader("Analysis Results")
    # Show only the most important columns to the HR user
    display_cols = ['Employee_ID', 'Department', 'Risk_Score_%']
    st.dataframe(df[display_cols].sort_values(by='Risk_Score_%', ascending=False))
    st.success("Analysis Complete! Download the full report below.")
    st.download_button("Download Full Report", df.to_csv(index=False), "HR_Report.csv")
