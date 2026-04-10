import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from textblob import TextBlob
 
# --- 1. CONFIGURATION ---
st.set_page_config(page_title="HR AI Attrition Dashboard", layout="wide")
 
# --- 2. ASSET LOADING (Updated to your specific filenames) ---
@st.cache_resource
def load_ml_assets():
    try:
        # Loading your specific pkl files
        model = joblib.load('attrition_xgb_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        features = joblib.load('feature_columns.pkl')
        return model, scaler, features
    except FileNotFoundError:
        st.error("⚠️ Files missing! Please ensure 'attrition_xgb_model.pkl', 'robust_scaler.pkl', and 'feature_columns.pkl' are in the directory.")
        return None, None, None
 
model, scaler, core_features = load_ml_assets()
 
# --- 3. UI SIDEBAR ---
with st.sidebar:
    st.title("Admin Controls")
    if st.button("🔄 Reset Dashboard"):
        st.rerun()
    st.divider()
    st.info("Upload the CSV file containing employee metrics to identify risk segments.")
 
# --- 4. MAIN APP LOGIC ---
st.title("🛡️ Employee Retention Strategy AI")
uploaded_file = st.file_uploader("Upload New HR Data (CSV)", type="csv")
 
if uploaded_file and model:
    # Read Data
    df = pd.read_csv(uploaded_file)
    with st.spinner('Calculating Risk Scores...'):
        df_proc = df.copy()
        # FEATURE ENGINEERING (Matching your model's training logic)
        df_proc['Comp_Ratio'] = df_proc['Base_Salary'] / (df_proc['Benchmark_Salary'] + 1)
        df_proc['Stagnation_Index'] = df_proc['Tenure_Years'] / (df_proc['Career_Development'] + 0.1)
        df_proc['Is_Contractor'] = np.where(df_proc['Employment_Type'] == 'Contract', 1, 0)
        df_proc['Survey_Sentiment'] = df_proc['Feedback_Comments'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
        )
 
        # PREDICTION using your core_features list
        X_scaled = scaler.transform(df_proc[core_features])
        df_proc['Risk_Score_%'] = (model.predict_proba(X_scaled)[:, 1] * 100).round(1)
 
        # STRATEGY ENGINE
        def get_strategy(row):
            score = row['Risk_Score_%']
            if score >= 75:
                tier = 'High Risk (Critical)'
                action = 'Urgent Salary Correction' if row['Comp_Ratio'] < 0.9 else 'Immediate Stay Interview'
            elif score >= 40:
                tier = 'Medium Risk (Monitor)'
                action = 'Engagement Check-in'
            else:
                tier = 'Low Risk (Stable)'
                action = 'Standard Engagement'
            return pd.Series([tier, action])
 
        df_proc[['Risk_Tier', 'Recommended_Action']] = df_proc.apply(get_strategy, axis=1)
 
    # --- 5. INTERACTIVE DASHBOARD ---
    st.divider()
    # Summary Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Workforce", len(df_proc))
    m2.metric("Critical Risks", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']))
    m3.metric("Avg Risk Score", f"{df_proc['Risk_Score_%'].mean().round(1)}%")
 
    st.header("📊 Interactive Drill-Down Analysis")
    st.write("Click the **Red Slice (High Risk)** to view specific departmental data.")
 
    # LEVEL 1: GLOBAL PIE
    tier_counts = df_proc['Risk_Tier'].value_counts().reset_index()
    tier_counts.columns = ['Risk_Tier', 'Count']
    fig_global = px.pie(
        tier_counts, values='Count', names='Risk_Tier', hole=0.4,
        title="Overall Risk Distribution",
        color='Risk_Tier',
        color_discrete_map={
            'High Risk (Critical)': '#d32f2f', 
            'Medium Risk (Monitor)': '#f57c00', 
            'Low Risk (Stable)': '#2e7d32'
        }
    )
    # Global Pie Selection (Clickable)
    global_sel = st.plotly_chart(fig_global, use_container_width=True, on_select="rerun")
 
    # LEVEL 2 & 3: DRILL DOWN (Nested Logic)
    if global_sel and len(global_sel["selection"]["points"]) > 0:
        clicked_tier = global_sel["selection"]["points"][0]["label"]
        if clicked_tier == 'High Risk (Critical)':
            st.subheader(f"🔥 Critical Breakdown: {clicked_tier}")
            crit_df = df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']
            dept_counts = crit_df['Department'].value_counts().reset_index()
            dept_counts.columns = ['Department', 'Count']
            # Show Department chart and Top 10 Table in columns
            c_chart, c_table = st.columns([1, 1])
            with c_chart:
                fig_dept = px.pie(
                    dept_counts, values='Count', names='Department',
                    title="Critical Risks by Department (Click slice for names)",
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                dept_sel = st.plotly_chart(fig_dept, use_container_width=True, on_select="rerun")
 
            with c_table:
                if dept_sel and len(dept_sel["selection"]["points"]) > 0:
                    clicked_dept = dept_sel["selection"]["points"][0]["label"]
                    st.markdown(f"**Top 10 High-Risk Employees in {clicked_dept}:**")
                    # Filtering and slicing to 10 records
                    dept_top_10 = crit_df[crit_df['Department'] == clicked_dept].sort_values(
                        by='Risk_Score_%', ascending=False
                    ).head(10)
                    st.table(dept_top_10[['Employee_ID', 'Risk_Score_%', 'Recommended_Action']])
                else:
                    st.info("👈 Select a department in the pie chart to view employee names.")
        else:
            st.success(f"Selected: **{clicked_tier}**. This segment is stable. Click the Red slice to investigate risks.")
 
    # --- 6. EXPORT SECTION ---
    st.divider()
    if st.button("Prepare Final Report for Download"):
        csv = df_proc.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Actionable HR Strategy (CSV)", csv, "HR_Attrition_Action_Plan.csv", "text/csv")
 
else:
    st.info("Waiting for data upload...")
