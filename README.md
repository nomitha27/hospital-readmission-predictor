# 🏥 Hospital Readmission Prediction System

An end-to-end machine learning system that predicts 30-day hospital readmission risk for diabetic patients using XGBoost.

[![Live Demo]https://hospital-readmission-predictor-pljzbnozx45xhttkfv6v6f.streamlit.app/
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io/)

## 🚀 Live Demo

**Try it here:** [Hospital Readmission Predictor]https://hospital-readmission-predictor-pljzbnozx45xhttkfv6v6f.streamlit.app/

## 📊 Project Overview

Hospital readmissions cost the US healthcare system **$17.4 billion annually**. This project uses machine learning to identify high-risk patients before discharge, enabling targeted interventions that can prevent readmissions.

### Key Results
- ✅ **88.8% Accuracy** in predicting readmissions
- ✅ **AUC-ROC: 0.681**
- ✅ Interactive web application for real-time predictions
- ✅ Batch prediction capability for multiple patients

## 🎯 Features

- **Single Patient Prediction**: Enter patient data and get immediate risk assessment
- **Batch Predictions**: Upload CSV files for bulk risk analysis
- **Interactive Visualizations**: Risk gauges and distribution charts
- **Clinical Recommendations**: Actionable interventions based on risk level
- **Model Insights**: Feature importance and performance metrics

## 🛠️ Technologies Used

- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Web App**: Streamlit
- **Deployment**: Streamlit Cloud

## 📁 Project Structure
```
hospital-readmission-predictor/
├── app/
│   └── app.py                 # Streamlit web application
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Feature engineering
│   └── 03_modeling.ipynb      # Model training
├── models/
│   ├── best_model.pkl         # Trained XGBoost model
│   └── feature_names.pkl      # Feature list
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Processed data
└── visualizations/            # Generated plots
```

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 88.8% |
| AUC-ROC | 0.681 |
| Model Type | XGBoost |

### Top Risk Factors
1. Previous inpatient visits
2. Discharge disposition
3. Total hospital visits
4. Number of diagnoses
5. Previous emergency visits

## 💡 Business Impact

**Estimated Annual Savings**: $277,500  
**ROI**: 336%

By identifying high-risk patients, hospitals can:
- Provide intensive discharge planning
- Schedule early follow-up appointments
- Reduce costly readmissions
- Improve patient outcomes

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/nomitha27/hospital-readmission-predictor.git
cd hospital-readmission-predictor
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
streamlit run app/app.py
```

## 📊 Dataset

- **Source**: UCI Machine Learning Repository - Diabetes 130-US Hospitals Dataset
- **Size**: 101,766 patient records
- **Features**: 14 clinical and demographic features
- **Target**: 30-day readmission (binary)

## 🔍 Key Insights

**High Risk Indicators:**
- Multiple previous hospitalizations
- 15+ medications (polypharmacy)
- 7+ diagnoses (comorbidities)
- Hospital stay > 7 days

**Protective Factors:**
- First-time hospitalization
- Fewer medications
- Younger age
- Shorter hospital stays

## 📝 Usage

### Single Prediction
1. Navigate to "Predict" page
2. Enter patient information
3. Click "Predict Risk"
4. View risk assessment and recommendations

### Batch Predictions
1. Navigate to "Batch Predict" page
2. Download sample CSV template
3. Fill in patient data
4. Upload and run predictions
5. Download results

## 🎓 Skills Demonstrated

- Machine Learning model development and evaluation
- Feature engineering and data preprocessing
- Handling imbalanced datasets
- Building interactive web applications
- Model deployment and cloud hosting
- Healthcare analytics and domain knowledge

## 📫 Contact

**Nomitha Sugavasi**
- Email: nsugavas@umd.edu
- LinkedIn: [linkedin.com/in/nomitha-sugavasi-b219b2242](https://linkedin.com/in/nomitha-sugavasi-b219b2242)
- GitHub: [@nomitha27](https://github.com/nomitha27)

