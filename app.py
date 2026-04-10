import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
 
# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HR AI Attrition Predictor", layout="wide")
 
# --- 1. LOAD ASSETS (The 'Brain' from Kaggle) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('attrition_xgb_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        core_features = joblib.load('feature_columns.pkl')
        return model, scaler, core_features
    except FileNotFoundError:
        st.error("⚠️ Error: Missing .pkl files! Ensure 'attrition_model.pkl', 'robust_scaler.pkl', and 'core_features.pkl' are in the same folder.")
        return None, None, None
 
model, scaler, core_features = load_assets()
 
# --- 2. THE UI HEADER ---
st.title("🛡️ AI-Driven Employee Retention Dashboard")
st.markdown("""
Upload your employee survey and compensation data to identify flight risks and receive automated intervention strategies.
""")
 
# --- 3. SIDEBAR TEMPLATE DOWNLOAD ---
with st.sidebar:
    st.header("App Instructions")
    st.write("1. Upload a CSV with employee data.")
    st.write("2. The AI will calculate risk scores.")
    st.write("3. Download the Actionable Report.")
    # Mock data for template download
    template_data = pd.DataFrame({
        'Employee_ID': ['EMP001'], 'Department': ['Sales'], 'Role': ['Manager'], 
        'Work_Location': ['Remote'], 'Base_Salary': [70000], 'Benchmark_Salary': [75000],
        'Job_Satisfaction': [3], 'Engagement_Level': [3], 'Work_Life_Balance': [3], 
        'Management_Support': [3], 'Career_Development': [3], 'Tenure_Years': [2.5], 
        'Employment_Type': ['Full-time'], 'Feedback_Comments': ['Sample feedback here.']
    })
    st.download_button("📥 Download CSV Template", template_data.to_csv(index=False), "hr_template.csv")
 
# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Employee Data (CSV Format)", type="csv")
 
if uploaded_file is not None and model is not None:
    # Read uploaded data
    df = pd.read_csv(uploaded_file)
    with st.spinner('AI is analyzing flight risk factors...'):
        # --- 5. FEATURE ENGINEERING (Science Step) ---
        df_proc = df.copy()
        # Financial & Stagnation Math
        df_proc['Comp_Ratio'] = df_proc['Base_Salary'] / (df_proc['Benchmark_Salary'] + 1)
        df_proc['Stagnation_Index'] = df_proc['Tenure_Years'] / (df_proc['Career_Development'] + 0.1)
        df_proc['Is_Contractor'] = np.where(df_proc['Employment_Type'] == 'Contract', 1, 0)
        # NLP Sentiment Analysis
        df_proc['Survey_Sentiment'] = df_proc['Feedback_Comments'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
        )
 
        # --- 6. MACHINE LEARNING PREDICTION ---
        # Select and Scale features
        X_active = df_proc[core_features]
        X_scaled = scaler.transform(X_active)
        # Get Probabilities
        risk_probs = model.predict_proba(X_scaled)[:, 1]
        df_proc['Risk_Score_%'] = (risk_probs * 100).round(1)
 
        # --- 7. STRATEGY ENGINE (Risk Tiers & Actions) ---
        def assign_strategy(row):
            score = row['Risk_Score_%']
            if score >= 75:
                tier = 'High Risk (Critical)'
                if row['Comp_Ratio'] < 0.9: 
                    action = 'Urgent Salary Correction'
                elif row['Management_Support'] < 3: 
                    action = 'Skip-Level Meeting / Manager Review'
                else: 
                    action = 'Immediate Stay Interview'
            elif score >= 40:
                tier = 'Medium Risk (Monitor)'
                action = 'Engagement Project / Flex Work'
            else:
                tier = 'Low Risk (Stable)'
                action = 'Standard Engagement'
            return pd.Series([tier, action])
 
        df_proc[['Risk_Tier', 'Recommended_Action']] = df_proc.apply(assign_strategy, axis=1)
 
    # --- 8. RESULTS DASHBOARD ---
    st.divider()
    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees Analyzed", len(df_proc))
    col2.metric("Critical Risks Detected", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']))
    col3.metric("Average Risk Score", f"{df_proc['Risk_Score_%'].mean().round(1)}%")
 
    # Results Table
    st.subheader("📋 Employee Triage List")
    display_cols = ['Employee_ID', 'Department', 'Risk_Score_%', 'Risk_Tier', 'Recommended_Action']
    st.dataframe(
        df_proc[display_cols].sort_values(by='Risk_Score_%', ascending=False),
        use_container_width=True
    )
 
    # Export Button
    st.divider()
    st.subheader("💾 Export Actionable Data")
    csv_data = df_proc.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Actionable Report (CSV)",
        data=csv_data,
        file_name='Actionable_HR_Attrition_Report.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to begin the analysis.")
