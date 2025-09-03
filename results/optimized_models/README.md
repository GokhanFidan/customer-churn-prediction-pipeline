  # Optimized Models

  This directory contains production-ready machine learning models optimized using Optuna hyperparameter tuning.

  ## Models by Dataset:

  ### Banking (4 models)
  - ✅ `banking_catboost_optimized.joblib`
  - ✅ `banking_lightgbm_optimized.joblib`
  - ✅ `banking_random_forest_optimized.joblib`
  - ✅ `banking_xgboost_optimized.joblib`

  ### Internet Service (5 models)
  - ✅ `internet_catboost_optimized.joblib`
  - ✅ `internet_gradient_boosting_optimized.joblib`
  - ✅ `internet_lightgbm_optimized.joblib`
  - ❌ `internet_random_forest_optimized.joblib` (180MB - exceeds GitHub limit)
  - ✅ `internet_xgboost_optimized.joblib`

  ### E-commerce Website (4 models)
  - ✅ `website_catboost_optimized.joblib`
  - ✅ `website_lightgbm_optimized.joblib`
  - ✅ `website_random_forest_optimized.joblib`
  - ✅ `website_xgboost_optimized.joblib`

  ## File Size Note
  **Total Models:** 13 production-ready models
  **Available on GitHub:** 12 models (1 exceeds 100MB limit)
  **Missing Model:** `internet_random_forest_optimized.joblib` due to GitHub's file size restriction

  ## Usage
  ```python
  import joblib
  model = joblib.load('results/optimized_models/banking_catboost_optimized.joblib')
  prediction = model.predict(new_data)

  Performance Summary

  All models achieved cross-validated performance with rigorous leakage detection:
  - Internet Service: AUC 0.85-0.91 (LightGBM best)
  - Banking: AUC 0.72-0.87 (CatBoost best)
  - E-commerce: AUC 0.82-0.96 (Random Forest best)
