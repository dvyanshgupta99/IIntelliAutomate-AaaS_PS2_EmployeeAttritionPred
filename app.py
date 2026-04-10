import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from textblob import TextBlob

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="HR AI Risk Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('attrition_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        features = joblib.load('core_features.pkl')
        return model, scaler, features
    except:
        st.error("Model files not found. Please upload .pkl files to the app directory.")
        return None, None, None

model, scaler, core_features = load_assets()

# --- 2. HEADER & FILE UPLOAD ---
st.title("🛡️ Employee Retention Strategy AI")
uploaded_file = st.file_uploader("Upload New HR Data (CSV)", type="csv")

if uploaded_file and model:
    # Load and Process Data
    df = pd.read_csv(uploaded_file)
    
    with st.spinner('Calculating Risk Scores...'):
        df_proc = df.copy()
        # Feature Engineering
        df_proc['Comp_Ratio'] = df_proc['Base_Salary'] / (df_proc['Benchmark_Salary'] + 1)
        df_proc['Stagnation_Index'] = df_proc['Tenure_Years'] / (df_proc['Career_Development'] + 0.1)
        df_proc['Is_Contractor'] = np.where(df_proc['Employment_Type'] == 'Contract', 1, 0)
        df_proc['Survey_Sentiment'] = df_proc['Feedback_Comments'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0.0
        )
        
        # Prediction
        X_scaled = scaler.transform(df_proc[core_features])
        df_proc['Risk_Score_%'] = (model.predict_proba(X_scaled)[:, 1] * 100).round(1)

        # Strategy Engine
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

    # --- 3. TOP LEVEL METRICS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Workforce", len(df_proc))
    m2.metric("Critical Risks", len(df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']))
    m3.metric("Avg Risk Score", f"{df_proc['Risk_Score_%'].mean().round(1)}%")

    # --- 4. INTERACTIVE DASHBOARD ---
    st.divider()
    st.header("📊 Interactive Drill-Down Analysis")
    st.info("Step 1: Click the **Red Slice (High Risk)** to view departments.")

    # LEVEL 1: GLOBAL PIE
    tier_counts = df_proc['Risk_Tier'].value_counts().reset_index()
    tier_counts.columns = ['Risk_Tier', 'Count']
    
    fig_global = px.pie(
        tier_counts, values='Count', names='Risk_Tier', hole=0.4,
        title="Global Risk Distribution",
        color='Risk_Tier',
        color_discrete_map={'High Risk (Critical)': '#d32f2f', 'Medium Risk (Monitor)': '#f57c00', 'Low Risk (Stable)': '#2e7d32'}
    )
    
    global_sel = st.plotly_chart(fig_global, use_container_width=True, on_select="rerun")

    # LEVEL 2: DEPARTMENT PIE
    if global_sel and len(global_sel["selection"]["points"]) > 0:
        clicked_tier = global_sel["selection"]["points"][0]["label"]
        
        if clicked_tier == 'High Risk (Critical)':
            st.subheader(f"🔥 Critical Risk: Departmental Breakdown")
            st.info("Step 2: Click a **Department slice** to see top 10 individual records.")
            
            crit_df = df_proc[df_proc['Risk_Tier'] == 'High Risk (Critical)']
            dept_counts = crit_df['Department'].value_counts().reset_index()
            dept_counts.columns = ['Department', 'Count']
            
            fig_dept = px.pie(
                dept_counts, values='Count', names='Department',
                title="Critical Employees by Department",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            
            dept_sel = st.plotly_chart(fig_dept, use_container_width=True, on_select="rerun")

            # LEVEL 3: TABULAR DATA (10 RECORDS)
            if dept_sel and len(dept_sel["selection"]["points"]) > 0:
                clicked_dept = dept_sel["selection"]["points"][0]["label"]
                
                st.markdown(f"#### 📋 Top 10 Critical Employees in {clicked_dept}")
                
                # Filter, Sort, and Slice
                dept_table = crit_df[crit_df['Department'] == clicked_dept].sort_values(
                    by='Risk_Score_%', ascending=False
                ).head(10)
                
                st.table(dept_table[['Employee_ID', 'Role', 'Risk_Score_%', 'Recommended_Action']])
                st.caption(f"Displaying top 10 highest-risk individuals in the {clicked_dept} department.")
        else:
            st.write(f"Selection: **{clicked_tier}**. No further drill-down required for stable segments.")

    # --- 5. EXPORT ---
    st.divider()
    if st.button("Generate Final CSV Report"):
        csv = df_proc.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Actionable Report", csv, "HR_Strategy_Report.csv", "text/csv")
