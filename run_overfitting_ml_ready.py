#!/usr/bin/env python3
"""
Step 7: Overfitting Detection on ML-Ready Results
"""

import sys
sys.path.append('src')

import pandas as pd
from evaluation.overfitting_detector import OverfittingDetector
from models.model_trainer import ModelTrainer
import joblib
import os

def main():
    print('🔍 OVERFITTING DETECTION - ML-READY DATA')
    print('='*60)

    # Load ML-ready datasets
    datasets = {
        'internet': pd.read_csv('data/ml_ready/internet_ml_ready.csv'),
        'banking': pd.read_csv('data/ml_ready/banking_ml_ready.csv'),
        'website': pd.read_csv('data/ml_ready/website_ml_ready.csv')
    }

    print(f'📂 Loaded {len(datasets)} ML-ready datasets:')
    for name, df in datasets.items():
        print(f'   {name}: {df.shape}')
        if 'churn' in df.columns:
            target_dist = df['churn'].value_counts().to_dict()
            print(f'   Target distribution: {target_dist}')

    # Train models to get results for overfitting analysis
    print('\n🚀 Training models for overfitting analysis...')
    trainer = ModelTrainer()
    results = trainer.train_all_datasets(datasets, optimize_hyperparams=False)
    summary = trainer.get_results_summary()

    print(f'\n📊 MODEL RESULTS SUMMARY:')
    print('='*50)
    
    for dataset_name in datasets.keys():
        print(f'\n{dataset_name.upper()} Dataset:')
        dataset_results = summary[summary['Dataset'] == dataset_name]
        if not dataset_results.empty:
            # Sort by performance column (try different column names)
            perf_cols = ['ROC_AUC', 'AUC', 'Test_AUC', 'CV_Mean']
            sort_col = None
            for col in perf_cols:
                if col in dataset_results.columns:
                    sort_col = col
                    break
            
            if sort_col:
                dataset_results = dataset_results.sort_values(sort_col, ascending=False)
                for _, row in dataset_results.iterrows():
                    auc = row[sort_col]
                    model_name = row['Model']
                    print(f'  {model_name:<20}: {auc:.4f}')
                    
                    # Check for overfitting indicators
                    if sort_col == 'ROC_AUC' and auc > 0.95:
                        print(f'    🚨 STILL SUSPICIOUS! Very high AUC ({auc:.4f})')
                    
                    # Check train-test gap if available
                    if 'Train_AUC' in row:
                        gap = abs(row['Train_AUC'] - auc)
                        if gap > 0.05:
                            print(f'    ⚠️  Large train-test gap: {gap:.4f}')

    # Run comprehensive overfitting detection
    print(f'\n🔍 COMPREHENSIVE OVERFITTING ANALYSIS...')
    detector = OverfittingDetector()
    
    # Process each dataset
    for dataset_name, df in datasets.items():
        print(f'\n📊 Analyzing {dataset_name.upper()} for overfitting...')
        
        # Prepare data
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        print(f'   Features: {X.shape[1]}, Samples: {X.shape[0]}')
        
        # Check best models for overfitting
        best_models = ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
        
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbosity=-1),
            'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=False, iterations=100)
        }
        
        for model_name, model in models.items():
            print(f'\n   🔍 {model_name} overfitting analysis...')
            try:
                # Generate learning curves for this model
                detector.plot_learning_curves(model, X, y, dataset_name, model_name, cv=3)
                print(f'     ✅ Learning curves generated')
            except Exception as e:
                print(f'     ❌ Error: {str(e)[:100]}...')

    print(f'\n✅ Overfitting analysis completed!')
    print('📊 Check results/plots/ for learning curves')

if __name__ == "__main__":
    main()