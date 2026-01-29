import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# Page setup
st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥", layout="wide")

# Title
st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("### Predict 30-day readmission risk for diabetic patients")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(app_dir, '..', 'models')
        
        model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, feature_names = load_model()

if model is None:
    st.error("❌ Model files not found. Please ensure models folder contains required files.")
    st.stop()

st.sidebar.success("✅ Model loaded!")

# Sidebar
st.sidebar.title("📋 Menu")
page = st.sidebar.radio("Select:", ["🏠 Home", "🔮 Predict", "📊 Batch Predict", "📈 Model Info"])

# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Purpose")
        st.write("""
        Identify patients at risk of hospital readmission within 30 days.
        
        **Benefits:**
        - 🎯 Early risk identification
        - 💰 Reduce costs ($15K per readmission)
        - 🏥 Better patient outcomes
        """)
        
        st.subheader("🔍 How to Use")
        st.write("""
        1. Click "Predict" for single patient
        2. Click "Batch Predict" for multiple patients
        3. Get risk assessments
        4. See recommendations
        """)
    
    with col2:
        st.subheader("📊 Model Stats")
        st.metric("Accuracy", "88.8%")
        st.metric("Model Type", "XGBoost")
        
        st.info("""
        **Top Risk Factors:**
        - Previous hospitalizations
        - Number of medications
        - Multiple diagnoses
        - Hospital stay length
        """)
    
    st.markdown("---")
    st.subheader("💡 Impact")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("US Annual Cost", "$17.4B")
    with col2:
        st.metric("Readmission Rate", "20%")
    with col3:
        st.metric("Preventable", "30%")

# ============================================
# PREDICT PAGE
# ============================================
elif page == "🔮 Predict":
    st.header("Predict Patient Readmission Risk")
    
    st.info("👉 Enter patient information below")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Patient Info")
        age = st.slider("Age", 0, 100, 65)
        time_in_hospital = st.number_input("Days in Hospital", 1, 14, 3)
        num_lab_procedures = st.number_input("Lab Procedures", 0, 150, 40)
        num_procedures = st.number_input("Procedures", 0, 10, 1)
        num_medications = st.number_input("Medications", 1, 50, 15)
        
    with col2:
        st.subheader("🏥 History")
        number_diagnoses = st.number_input("Diagnoses", 1, 16, 7)
        number_inpatient = st.number_input("Previous Admissions", 0, 20, 0)
        number_emergency = st.number_input("Previous ER Visits", 0, 20, 0)
        number_outpatient = st.number_input("Outpatient Visits", 0, 40, 0)
    
    if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            
            # Create input
            input_dict = {
                'admission_type_id': 1,
                'discharge_disposition_id': 1,
                'admission_source_id': 7,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'age_numeric': age,
                'total_visits': number_outpatient + number_emergency + number_inpatient,
                'a1c_tested': 0
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Add missing features
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            input_df = input_df[feature_names]
            
            try:
                # Predict
                prediction_proba = model.predict_proba(input_df)[0][1]
                prediction = 1 if prediction_proba > 0.5 else 0
                
                st.markdown("---")
                st.subheader("📊 Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🔴 HIGH RISK")
                    else:
                        st.success("🟢 LOW RISK")
                
                with col2:
                    st.metric("Probability", f"{prediction_proba*100:.1f}%")
                
                with col3:
                    if prediction_proba > 0.7:
                        st.metric("Level", "🔴 Critical")
                    elif prediction_proba > 0.4:
                        st.metric("Level", "🟡 Moderate")
                    else:
                        st.metric("Level", "🟢 Low")
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba * 100,
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if prediction_proba > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                if prediction == 1:
                    st.warning("⚠️ **High Risk - Recommended Actions:**")
                    st.markdown("""
                    - ✅ Schedule follow-up within 7 days
                    - ✅ Assign care coordinator
                    - ✅ Medication review
                    - ✅ Patient education
                    - ✅ Post-discharge call within 48 hours
                    """)
                else:
                    st.success("✅ **Low Risk - Standard Care:**")
                    st.markdown("""
                    - Standard discharge
                    - Follow-up in 2-4 weeks
                    - Routine medication review
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================
# BATCH PREDICTION PAGE
# ============================================
elif page == "📊 Batch Predict":
    st.header("Batch Patient Risk Assessment")
    st.write("Upload a CSV file with multiple patients for bulk predictions.")
    
    # Show required format
    with st.expander("📋 Required CSV Format (click to expand)"):
        st.write("Your CSV should have these columns:")
        
        sample_df = pd.DataFrame({
            'time_in_hospital': [3, 5, 2],
            'num_lab_procedures': [40, 55, 30],
            'num_procedures': [1, 2, 0],
            'num_medications': [15, 20, 10],
            'number_outpatient': [0, 2, 1],
            'number_emergency': [0, 1, 0],
            'number_inpatient': [0, 1, 0],
            'number_diagnoses': [7, 9, 5],
            'age_numeric': [65, 72, 58]
        })
        
        st.dataframe(sample_df)
        
        # Download sample
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            "📥 Download Sample CSV",
            csv_sample,
            "sample_patients.csv",
            "text/csv",
            use_container_width=True
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read file
            df_batch = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded! Found {len(df_batch)} patients")
            
            # Preview
            st.subheader("Data Preview")
            st.dataframe(df_batch.head(10))
            
            if st.button("🔮 Run Predictions", type="primary", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    
                    # Add default values for required fields
                    if 'admission_type_id' not in df_batch.columns:
                        df_batch['admission_type_id'] = 1
                    if 'discharge_disposition_id' not in df_batch.columns:
                        df_batch['discharge_disposition_id'] = 1
                    if 'admission_source_id' not in df_batch.columns:
                        df_batch['admission_source_id'] = 7
                    
                    # Calculate total_visits if not present
                    if 'total_visits' not in df_batch.columns:
                        df_batch['total_visits'] = (
                            df_batch.get('number_outpatient', 0) + 
                            df_batch.get('number_emergency', 0) + 
                            df_batch.get('number_inpatient', 0)
                        )
                    
                    if 'a1c_tested' not in df_batch.columns:
                        df_batch['a1c_tested'] = 0
                    
                    # Add missing features
                    for feature in feature_names:
                        if feature not in df_batch.columns:
                            df_batch[feature] = 0
                    
                    # Select features in correct order
                    X_batch = df_batch[feature_names]
                    
                    # Predict
                    probabilities = model.predict_proba(X_batch)[:, 1]
                    predictions = (probabilities > 0.5).astype(int)
                    
                    # Add results
                    df_batch['Risk_Probability'] = probabilities
                    df_batch['Risk_Prediction'] = predictions
                    df_batch['Risk_Level'] = ['High' if p > 0.5 else 'Low' for p in probabilities]
                    
                    st.success("✅ Predictions completed!")
                    
                    # Summary stats
                    st.subheader("📊 Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Patients", len(df_batch))
                    with col2:
                        high_risk_count = (df_batch['Risk_Level'] == 'High').sum()
                        st.metric("High Risk", high_risk_count)
                    with col3:
                        low_risk_count = (df_batch['Risk_Level'] == 'Low').sum()
                        st.metric("Low Risk", low_risk_count)
                    with col4:
                        avg_risk = df_batch['Risk_Probability'].mean()
                        st.metric("Avg Risk", f"{avg_risk:.1%}")
                    
                    # Visualization
                    st.subheader("📈 Risk Distribution")
                    
                    fig = px.histogram(
                        df_batch, 
                        x='Risk_Probability',
                        nbins=20,
                        title='Distribution of Readmission Risk',
                        labels={'Risk_Probability': 'Risk Probability'},
                        color_discrete_sequence=['steelblue']
                    )
                    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                                annotation_text="High Risk Threshold (50%)")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    
                    # Show relevant columns
                    display_cols = ['Risk_Probability', 'Risk_Level']
                    
                    # Add input columns if they exist
                    for col in ['time_in_hospital', 'num_medications', 'number_inpatient', 
                               'number_diagnoses', 'age_numeric']:
                        if col in df_batch.columns:
                            display_cols.insert(-1, col)
                    
                    st.dataframe(
                        df_batch[display_cols].sort_values('Risk_Probability', ascending=False),
                        use_container_width=True
                    )
                    
                    # Download results
                    csv_results = df_batch.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv_results,
                        f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # High risk patients
                    high_risk_patients = df_batch[df_batch['Risk_Level'] == 'High']
                    if len(high_risk_patients) > 0:
                        st.warning(f"⚠️ {len(high_risk_patients)} high-risk patients require immediate attention")
                        st.dataframe(high_risk_patients[display_cols].head(10))
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.write("Please ensure your CSV has the required columns. Download the sample to see the format.")

# ============================================
# MODEL INFO PAGE
# ============================================
elif page == "📈 Model Info":
    st.header("Model Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "XGBoost")
    with col2:
        st.metric("Accuracy", "88.8%")
    with col3:
        st.metric("AUC-ROC", "0.681")
    
    st.markdown("---")
    
    st.subheader("🎯 Top Risk Factors")
    
    risk_factors = pd.DataFrame({
        'Factor': [
            'Previous inpatient visits',
            'Discharge disposition',
            'Total hospital visits',
            'Number of diagnoses',
            'Previous ER visits'
        ],
        'Impact': ['Very High', 'High', 'High', 'High', 'Medium']
    })
    
    st.dataframe(risk_factors, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("💡 Clinical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔴 High Risk:**
        - Previous hospitalizations
        - 15+ medications
        - 7+ diagnoses
        - Stay > 7 days
        """)
    
    with col2:
        st.markdown("""
        **🟢 Lower Risk:**
        - First hospitalization
        - Fewer medications
        - Shorter stay
        - Younger age
        """)
    
    st.markdown("---")
    st.subheader("📋 Interventions")
    
    st.markdown("""
    **High Risk Patients:**
    - Intensive discharge planning
    - Follow-up within 7 days
    - Care coordinator
    - Medication reconciliation
    
    **Lower Risk Patients:**
    - Standard discharge
    - Follow-up in 2-4 weeks
    """)

# Footer
st.markdown("---")
st.caption("Hospital Readmission Predictor | XGBoost ML Model | Educational Project")