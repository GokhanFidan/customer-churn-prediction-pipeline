#!/usr/bin/env python3
"""
Step 2: Baseline 8-Model Testing with Clean Data
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from models.model_trainer import ModelTrainer
import joblib

def main():
    print('🚀 BASELINE 8-MODEL TESTING WITH CLEAN DATA')
    print('='*60)

    # Load processed datasets
    datasets = {
        'internet': pd.read_csv('data/processed/internet_processed.csv'),
        'banking': pd.read_csv('data/processed/banking_processed.csv'),
        'website': pd.read_csv('data/processed/website_processed.csv')
    }

    print(f'📂 Loaded {len(datasets)} processed datasets:')
    for name, df in datasets.items():
        print(f'   {name}: {df.shape}')
        # Assume target column is 'churn' for all datasets
        if 'churn' in df.columns:
            target_dist = df['churn'].value_counts().to_dict()
            print(f'   Target distribution: {target_dist}')

    # Initialize trainer
    trainer = ModelTrainer()

    # Train all datasets (without hyperparameter optimization)
    print('\n🚀 Training 8 models on each dataset...')
    results = trainer.train_all_datasets(datasets, optimize_hyperparams=False)
    
    print('\n📊 Getting results summary...')
    summary = trainer.get_results_summary()

    print(f'\n🎯 BASELINE RESULTS SUMMARY:')
    print('='*50)
    
    for dataset_name in datasets.keys():
        print(f'\n{dataset_name.upper()} Dataset:')
        dataset_results = summary[summary['Dataset'] == dataset_name].sort_values('ROC_AUC', ascending=False)
        for _, row in dataset_results.iterrows():
            auc = row['ROC_AUC']
            model_name = row['Model']
            print(f'  {model_name:<20}: {auc:.4f}')
            
            # Flag suspicious high performance
            if auc > 0.95:
                print(f'    🚨 SUSPICIOUS! Very high AUC ({auc:.4f})')

    # Save results
    os.makedirs('results', exist_ok=True)
    joblib.dump(results, 'results/baseline_8model_results.joblib')
    summary.to_csv('results/baseline_8model_summary.csv', index=False)
    
    print(f'\n✅ Baseline results saved to results/ directory')
    print('📄 Next: Check for suspicious high AUCs (>0.95) for data leakage investigation')

if __name__ == "__main__":
    main()