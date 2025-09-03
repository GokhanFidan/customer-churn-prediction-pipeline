# Customer Churn Prediction: Multi-Domain ML Pipeline

### 🌐 Try Live Demo First!
**[➡️ INTERACTIVE DASHBOARD](https://customer-churn-prediction-pipeline-jo9eew5gfurijaejmgof3s.streamlit.app/)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning pipeline for customer churn prediction across **3 different industries**: Telecommunications, Banking, and E-commerce. This project demonstrates industry-standard data science methodology with rigorous data leakage detection, model optimization, and production-ready deployment.

## 🎯 Business Problem

Customer acquisition costs are 5-25x higher than retention costs. This project develops predictive models to identify at-risk customers across multiple industries, enabling proactive retention strategies and reducing revenue loss.

## 📊 Project Overview

- **3 Datasets**: Internet Service, Banking, E-commerce Website
- **8 ML Algorithms**: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Logistic Regression, SVM, KNN
- **Advanced Pipeline**: Data leakage detection, hyperparameter optimization, overfitting analysis
- **Interactive Dashboard**: Streamlit application with comprehensive analytics
- **Production Ready**: Optimized models saved for immediate deployment

## 🔧 Key Features

### ✅ Industry-Standard Methodology
- **9-step ML pipeline** following best practices
- **Data leakage detection** using single-feature AUC testing
- **Cross-validation** with stratified k-fold
- **Hyperparameter optimization** using Optuna TPE sampler

### ✅ Comprehensive Analysis
- **Multi-domain comparison** across industries
- **Model performance analysis** with 5 evaluation metrics
- **Feature engineering** with correlation analysis
- **Overfitting detection** and mitigation

### ✅ Production Deployment
- **Interactive dashboard** for stakeholder presentation
- **Optimized models** ready for real-time prediction
- **Complete documentation** for reproducibility

## 📈 Key Results

| Dataset | Best Model | AUC Score | Leaky Features Removed |
|---------|------------|-----------|------------------------|
| **Internet Service** | LightGBM | 0.910 | 1 (`remaining_contract`) |
| **Banking** | CatBoost | 0.870 | 0 (Clean dataset) |
| **E-commerce** | Random Forest | 0.960 | 2 (`points_in_wallet`, `membership_category`) |

### 🚨 Data Leakage Detection Results

Critical finding: **3 leaky features** detected and removed across datasets using single-feature AUC > 0.90 threshold:

- **Internet**: `remaining_contract` (AUC: 0.936) - Future information leak
- **Website**: `points_in_wallet` (AUC: 0.954) - Post-churn calculation
- **Website**: `membership_category` (AUC: 0.945) - Temporal logic issue

**Impact**: Prevented catastrophic production failures while maintaining realistic performance expectations.

## 🏗️ Project Architecture

### Data Pipeline Evolution
```
data/raw/          → Original datasets (3 industries)
data/processed/    → Basic cleaning (outliers, missing values)
data/conservative/ → Alternative preprocessing approach
data/ml_ready/     → Final clean datasets (leakage-free)
```

### Model Development Pipeline
```python
# 9-Step Industry-Standard ML Pipeline

1. Data Cleaning & EDA           # Outlier detection, missing value analysis
2. Baseline Model Testing        # 8 algorithms on cleaned data
3. Performance Analysis          # AUC evaluation, suspicious score detection
4. Data Leakage Investigation    # Single-feature AUC testing (threshold: 0.90)
5. Feature Engineering           # Remove leaky features, create ML-ready datasets
6. Model Comparison              # Full evaluation on leak-free data
7. Overfitting Detection         # Train-validation gap analysis
8. Model Selection               # Choose top 4 models per dataset
9. Hyperparameter Optimization   # Optuna-based tuning (20 trials per model)
```

## 🛠️ Technical Implementation

### Machine Learning Stack
- **Algorithms**: XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting
- **Optimization**: Optuna with TPE sampler
- **Evaluation**: 5-fold stratified cross-validation
- **Metrics**: AUC, Precision, Recall, F1-Score, Accuracy

### Visualization & Deployment
- **Dashboard**: Streamlit with interactive analytics
- **Charts**: Plotly for professional visualizations
- **Analysis**: ROC curves, confusion matrices, radar charts

### Development Tools
- **Python 3.8+** with scientific computing stack
- **Pandas, NumPy** for data manipulation
- **Scikit-learn** for preprocessing and evaluation
- **Joblib** for model serialization

## 🚀 Quick Start

### 🌐 Try Live Demo First!
**[➡️ INTERACTIVE DASHBOARD](https://customer-churn-prediction-pipeline-jo9eew5gfurijaejmgof3s.streamlit.app/)**

Experience the full pipeline without installation:
- Navigate through 4 comprehensive analysis pages
- Explore real model performance results
- Interact with ROC curves, confusion matrices, and radar charts
- See data leakage detection methodology in action

### 1. Installation
```bash
git clone https://github.com/your-username/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
```

### 2. Run Interactive Dashboard
```bash
streamlit run src/dashboard/streamlit_app.py
```

### 3. Access Dashboard
Open your browser to `http://localhost:8501`

Navigate through:
- **Overview**: Project methodology and dataset summary
- **Data Analysis**: Exploratory analysis and feature distributions
- **Leakage Detection**: Data quality investigation with visualizations
- **Model Performance**: Comprehensive model comparison and insights

## 📁 Repository Structure

```
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Basic cleaned data
│   ├── conservative/           # Alternative preprocessing
│   └── ml_ready/              # Final leakage-free datasets
├── src/
│   ├── dashboard/
│   │   └── streamlit_app.py   # Interactive dashboard
│   ├── data_processing/
│   │   ├── data_cleaner.py    # Data cleaning pipeline
│   │   └── ml_ready_data_processor.py  # Leakage removal
│   ├── evaluation/
│   │   ├── data_leakage_detector.py    # Leakage detection
│   │   └── overfitting_detector.py     # Model validation
│   ├── models/
│   │   └── model_trainer.py   # 8-algorithm training
│   └── optimization/
│       └── hyperparameter_optimizer.py  # Optuna optimization
├── results/
│   ├── optimized_models/      # Production-ready models (.joblib)
│   ├── plots/                 # Analysis visualizations
│   └── leakage_analysis/      # Data quality reports
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## 🔍 Methodology Deep Dive

### Data Leakage Detection Protocol

**Problem**: Features that contain future information or are calculated post-churn lead to unrealistic model performance that fails in production.

**Solution**: Single-feature AUC testing with 0.90 threshold
```python
# For each feature independently:
model = RandomForestClassifier()
model.fit(X_feature.reshape(-1, 1), y)
auc_score = roc_auc_score(y, model.predict_proba(X_feature.reshape(-1, 1))[:, 1])

if auc_score > 0.90:
    flag_as_suspicious(feature)
```

**Validation**: Confirmed by business logic review and temporal analysis.

### Hyperparameter Optimization Strategy

- **Framework**: Optuna with Tree-structured Parzen Estimator (TPE)
- **Trials**: 20 per model (80 total per dataset)
- **Timeout**: 10 minutes per model
- **Cross-validation**: 5-fold stratified
- **Objective**: Maximize AUC score

**Total optimization time**: 16.4 minutes across all datasets and models.

## 📊 Performance Analysis

### Model Performance by Industry

**Telecommunications (Internet Service)**
- **Challenge**: Moderate complexity, 1 leaky feature detected
- **Best Model**: LightGBM (AUC: 0.910)
- **Insight**: Tree-based models significantly outperform linear models

**Banking & Finance**
- **Challenge**: Most difficult dataset, no leakage detected
- **Best Model**: CatBoost (AUC: 0.870)
- **Insight**: Clean dataset validates our leakage detection methodology

**E-commerce Website**
- **Challenge**: Complex feature space, multiple leaky features
- **Best Model**: Random Forest (AUC: 0.960)
- **Insight**: Exceptional performance after proper feature engineering

### Cross-Algorithm Performance
```
Algorithm          | Avg AUC | Best Dataset    | Worst Dataset
-------------------|---------|-----------------|---------------
Random Forest      | 0.910   | Website (0.960) | Banking (0.861)
LightGBM          | 0.909   | Website (0.960) | Banking (0.866)
XGBoost           | 0.906   | Website (0.960) | Banking (0.867)
CatBoost          | 0.908   | Website (0.960) | Banking (0.870)
Gradient Boosting | 0.886   | Website (0.949) | Banking (0.853)
```

## 💡 Business Insights & Recommendations

### Internet Service Provider
- **Risk Factors**: Service failures, short subscription age, high billing amounts
- **Recommendation**: Deploy LightGBM model with 0.7 confidence threshold
- **Expected Impact**: 85% precision in identifying at-risk customers

### Banking Institution
- **Risk Factors**: Low product engagement, younger demographics, account inactivity
- **Recommendation**: Ensemble CatBoost + XGBoost for robust predictions
- **Expected Impact**: Balanced precision-recall for retention campaigns

### E-commerce Platform
- **Risk Factors**: Low engagement time, infrequent logins, low transaction values
- **Recommendation**: Random Forest with feature importance analysis
- **Expected Impact**: 89% precision with actionable business insights

## 🎯 Production Deployment

### Model Usage
```python
import joblib
import pandas as pd

# Load optimized model
model = joblib.load('results/optimized_models/website_random_forest_optimized.joblib')

# Predict churn probability
new_customer_data = pd.DataFrame({...})  # Customer features
churn_probability = model.predict_proba(new_customer_data)[:, 1]

# Business action threshold
if churn_probability > 0.7:
    trigger_retention_campaign(customer_id)
```

### Dashboard Features
- **Real-time Analytics**: Interactive charts and filters
- **Model Comparison**: Performance metrics across all algorithms
- **Business Intelligence**: Actionable insights and recommendations
- **Quality Assurance**: Data leakage detection results and methodology

## 🔬 Research & Development

### Innovative Approaches
1. **Multi-domain Analysis**: Comparative study across 3 industries
2. **Systematic Leakage Detection**: Automated pipeline for data quality
3. **Production-first Methodology**: Focus on real-world deployment readiness

### Future Enhancements
- Real-time prediction API development
- Advanced feature engineering (interaction terms, temporal features)
- A/B testing framework for retention strategies
- MLOps pipeline with automated retraining

## 📚 Technical Documentation

### Dependencies
- **Core**: pandas, numpy, scikit-learn
- **ML Models**: xgboost, lightgbm, catboost
- **Optimization**: optuna
- **Visualization**: plotly, streamlit
- **Utilities**: joblib, pathlib

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for models and data

### Performance Benchmarks
- **Training Time**: ~45 minutes for complete pipeline
- **Optimization Time**: 16.4 minutes for all models
- **Prediction Latency**: <50ms per customer
- **Model Size**: 1-5MB per optimized model

## 🤝 Contributing

This project demonstrates production-ready machine learning methodology suitable for enterprise deployment. The codebase follows industry standards and is designed for scalability and maintainability.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions about methodology, implementation, or potential collaboration opportunities, please feel free to reach out.

---

**Built with industry-standard practices | Production-ready architecture | Comprehensive documentation**
