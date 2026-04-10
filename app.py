import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from textblob import TextBlob
 
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="HR AI Attrition Predictor", layout="wide")
 
# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Using your specific filenames
        model = joblib.load('attrition_xgb_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        core_features = joblib.load('feature_columns.pkl')
        return model, scaler, core_features
    except FileNotFoundError:
        st.error("⚠️ Model files missing! Ensure the .pkl files are in the app directory.")
        return None, None, None
 
model, scaler, core_features = load_assets()
 
# --- 3. HEADER & UI ---
st.title("🛡️ AI-Driven Employee Retention Strategy")
st.markdown("Upload employee data to identify high-risk segments and receive automated intervention plans.")
 
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset Dashboard"):
        st.rerun()
    st.divider()
    st.info("The AI analyzes sentiment, compensation market-parity, and engagement metrics.")
 
# --- 4. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload New HR Data (CSV)", type="csv")
 
if uploaded_file and model:
    # Read data
    df = pd.read_csv(uploaded_file)
    with st.spinner('AI is processing 50,000+ potential data points...'):
        # --- FEATURE ENGINEERING ---
        df_proc = df.copy()
        df_proc['Comp_Ratio'] = df_proc['Base_Salary'] / (df_proc['Benchmark_Salary'] + 1)
        df_proc['Stagnation_Index'] = df_proc['Tenure_Years'] / (df_proc['Career_Development'] + 0.1)
        df_proc['Is_Contractor'] = np.where(df_proc['Employment_Type'] == 'Contract', 1, 0)
        df_proc['Survey_Sentiment'] = df_proc['Feedback_Comments'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
        )
 
        # --- PREDICTION ---
        X_scaled = scaler.transform(df_proc[core_features])
        risk_probs = model.predict_proba(X_scaled)[:, 1]
        df_proc['Risk_Score_%'] = (risk_probs * 100).round(1)
 
        # --- STRATEGY ENGINE ---
        def assign_risk_details(row):
            score = row['Risk_Score_%']
            if score >= 75:
                tier = 'High Risk (Critical)'
                if row['Comp_Ratio'] < 0.9: action = 'Urgent Salary Correction'
                elif row['Management_Support'] < 3: action = 'Skip-Level Manager Meeting'
                else: action = 'Immediate Stay Interview'
            elif score >= 40:
                tier = 'Medium Risk (Monitor)'
                action = 'Engagement Check-in / Project Swap'
            else:
                tier = 'Low Risk (Stable)'
                action = 'Standard Engagement Cycle'
            return pd.Series([tier, action])
 
        df_proc[['Risk_Tier', 'Recommended_Action']] = df_proc.apply(assign_risk_details, axis=1)
 
    # --- 5. INTERACTIVE VISUALIZATION (LEVEL 1) ---
    st.divider()
    st.subheader("📊 Attrition Risk Overview")
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Analyzed", len(df_proc))
    m2.metric("Critical Risks", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']))
    m3.metric("Avg Risk Score", f"{df_proc['Risk_Score_%'].mean().round(1)}%")
 
    # Global Donut Chart
    chart_data = df_proc['Risk_Tier'].value_counts().reset_index()
    chart_data.columns = ['Risk_Tier', 'Count']
 
    fig_global = px.pie(
        chart_data, values='Count', names='Risk_Tier', hole=0.4,
        color='Risk_Tier',
        color_discrete_map={
            'High Risk (Critical)': '#d32f2f', 
            'Medium Risk (Monitor)': '#f57c00', 
            'Low Risk (Stable)': '#2e7d32'
        },
        title="Employee Distribution (Click 'High Risk' slice to drill down)"
    )
    # Interactive Selection
    global_sel = st.plotly_chart(fig_global, use_container_width=True, on_select="rerun")
 
    # --- 6. DRILL-DOWN LOGIC (LEVEL 2 & 3) ---
    if global_sel and len(global_sel["selection"]["points"]) > 0:
        clicked_tier = global_sel["selection"]["points"][0]["label"]
        if clicked_tier == 'High Risk (Critical)':
            st.subheader(f"🔥 Critical Risk Analysis: {clicked_tier}")
            crit_df = df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']
            dept_counts = crit_df['Department'].value_counts().reset_index()
            dept_counts.columns = ['Department', 'Count']
            col_chart, col_table = st.columns([1, 1])
            with col_chart:
                fig_dept = px.pie(
                    dept_counts, values='Count', names='Department',
                    title="Critical Risks by Department (Click to see names)",
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                dept_sel = st.plotly_chart(fig_dept, use_container_width=True, on_select="rerun")
 
            with col_table:
                if dept_sel and len(dept_sel["selection"]["points"]) > 0:
                    clicked_dept = dept_sel["selection"]["points"][0]["label"]
                    st.markdown(f"**Top 10 High-Risk Employees in {clicked_dept}:**")
                    # Filtering and slicing to exactly 10 records
                    dept_top_10 = crit_df[crit_df['Department'] == clicked_dept].sort_values(
                        by='Risk_Score_%', ascending=False
                    ).head(10)
                    st.table(dept_top_10[['Employee_ID', 'Risk_Score_%', 'Recommended_Action']])
                else:
                    st.info("👈 Select a department in the chart to view the top 10 employee names.")
        else:
            st.success(f"Selected: {clicked_tier}. These segments are stable. Focused intervention is not required.")
 
    # --- 7. FULL DATA & DOWNLOAD ---
    st.divider()
    st.subheader("📋 Complete Actionable Database")
    display_cols = ['Employee_ID', 'Department', 'Risk_Score_%', 'Risk_Tier', 'Recommended_Action']
    st.dataframe(df_proc[display_cols].sort_values(by='Risk_Score_%', ascending=False), use_container_width=True)
 
    st.download_button(
        label="📥 Download Full Actionable Report (CSV)",
        data=df_proc.to_csv(index=False).encode('utf-8'),
        file_name='Actionable_HR_Report.csv',
        mime='text/csv'
    )
 
else:
    st.info("Please upload a CSV file to generate the Intelligence Dashboard.")
