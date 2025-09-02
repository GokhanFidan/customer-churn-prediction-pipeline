"""
ML-Ready Data Processing Pipeline
Removes suspicious features identified through data leakage detection analysis.

This module prepares production-ready datasets by:
- Removes 'reamining_contract' from Internet dataset (future information leak)
- Removes 'points_in_wallet' from Website dataset (high AUC overfitting risk)
- Keeps Banking dataset as-is (no leakage detected)

Output: Clean, ML-ready datasets in data/ml_ready/ directory
Decision rationale and analysis process documented in README.md
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MLReadyDataProcessor:
    """
    ML-ready data processing pipeline that removes potentially leaky features
    based on systematic data leakage analysis.
    """

    def __init__(self):
        self.removal_log = {
            'internet': [],
            'banking': [],
            'website': []
        }
        self.analysis_results = {}

    def log_feature_removal(self, dataset: str, feature: str, reason: str, impact_metric: float = None):
        """Log feature removal with rationale"""
        removal_entry = {
            'feature': feature,
            'reason': reason,
            'impact_metric': impact_metric,
            'decision': 'REMOVED - ML-Ready Processing'
        }
        self.removal_log[dataset].append(removal_entry)
        print(f"🚫 {dataset.upper()}: Removed '{feature}' - {reason}")
        if impact_metric:
            print(f"   📊 Impact metric: {impact_metric:.4f}")

    def process_internet_ml_ready(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Internet dataset for ML-ready output.
        
        ANALYSIS FINDINGS:
        - 'reamining_contract' showed AUC: 0.9360 (suspicious)
        - Likely future information leak: contract duration calculated AFTER churn decision
        - ML-Ready decision: REMOVE this feature for production stability
        """
        print("🧹 Processing Internet Dataset (ML-Ready Mode)...")
        df_clean = df.copy()
        original_features = df_clean.shape[1]

        # DECISION: Remove reamining_contract (Future Information Leak)
        if 'reamining_contract' in df_clean.columns:
            self.log_feature_removal(
                'internet',
                'reamining_contract',
                'Future information leak - contract status likely calculated post-churn',
                0.9360  # AUC score that was too high
            )
            df_clean = df_clean.drop('reamining_contract', axis=1)

        # Store analysis results for documentation
        self.analysis_results['internet'] = {
            'original_features': original_features,
            'final_features': df_clean.shape[1],
            'removed_features': original_features - df_clean.shape[1],
            'main_concern': 'Future information leak in contract duration',
            'decision': 'ML-ready removal of suspicious feature'
        }

        print(f"   ✅ ML-ready processing completed: {original_features} → {df_clean.shape[1]} features")
        return df_clean

    def process_website_ml_ready(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Website dataset for ML-ready output.
        
        ANALYSIS FINDINGS:
        - 'points_in_wallet' showed AUC: 0.9536 (extremely high)
        - 'membership_category' showed AUC: 0.9448 (suspicious)  
        - Both features exceed 0.90 AUC threshold for potential overfitting
        - ML-Ready decision: REMOVE both features for production stability
        """
        print("🧹 Processing Website Dataset (ML-Ready Mode)...")
        df_clean = df.copy()
        original_features = df_clean.shape[1]

        # DECISION: Remove points_in_wallet (High AUC overfitting risk)
        if 'points_in_wallet' in df_clean.columns:
            self.log_feature_removal(
                'website',
                'points_in_wallet',
                'Extremely high AUC (0.9536) - removed to prevent overfitting in production',
                0.9536
            )
            df_clean = df_clean.drop('points_in_wallet', axis=1)
            
        # DECISION: Remove membership_category (High AUC overfitting risk)  
        if 'membership_category' in df_clean.columns:
            self.log_feature_removal(
                'website',
                'membership_category',
                'High AUC (0.9448) - removed to prevent overfitting in production',
                0.9448
            )
            df_clean = df_clean.drop('membership_category', axis=1)

        # Store analysis results
        self.analysis_results['website'] = {
            'original_features': original_features,
            'final_features': df_clean.shape[1],
            'removed_features': original_features - df_clean.shape[1],
            'main_concern': 'Two features with AUC > 0.90 suggesting overfitting risk',
            'decision': 'ML-ready removal of both suspicious features for production stability',
            'leaky_features': ['points_in_wallet (0.9536)', 'membership_category (0.9448)']
        }

        print(f"   ✅ ML-ready processing completed: {original_features} → {df_clean.shape[1]} features")
        return df_clean

    def process_banking_ml_ready(self, df: pd.DataFrame) -> pd.DataFrame:
        """Banking dataset - already ML-ready (no suspicious features detected)"""
        print("🧹 Processing Banking Dataset (ML-Ready Mode)...")
        df_clean = df.copy()

        print("   ✅ No suspicious features detected - dataset is already ML-ready")

        self.analysis_results['banking'] = {
            'original_features': df_clean.shape[1],
            'final_features': df_clean.shape[1],
            'removed_features': 0,
            'main_concern': 'None detected',
            'decision': 'No changes needed - already ML-ready'
        }

        return df_clean

    def process_all_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply ML-ready processing to all datasets"""
        print("\n🚀 ML-READY DATA PROCESSING PIPELINE")
        print("=" * 60)
        print("Preparing production-ready datasets after data leakage analysis")

        ml_ready_datasets = {}

        if 'internet' in datasets:
            ml_ready_datasets['internet'] = self.process_internet_ml_ready(datasets['internet'])

        if 'banking' in datasets:
            ml_ready_datasets['banking'] = self.process_banking_ml_ready(datasets['banking'])

        if 'website' in datasets:
            ml_ready_datasets['website'] = self.process_website_ml_ready(datasets['website'])

        self._print_processing_summary()
        return ml_ready_datasets

    def _print_processing_summary(self):
        """Print comprehensive summary of ML-ready processing decisions"""
        print(f"\n📋 ML-READY PROCESSING SUMMARY")
        print("=" * 50)

        total_removed = 0
        for dataset_name, results in self.analysis_results.items():
            removed = results['removed_features']
            total_removed += removed

            print(f"\n📊 {dataset_name.upper()}:")
            print(f"   Features: {results['original_features']} → {results['final_features']} (-{removed})")
            print(f"   Concern: {results['main_concern']}")
            print(f"   Decision: {results['decision']}")

        print(f"\n🎯 OVERALL IMPACT:")
        print(f"   Total features removed: {total_removed}")
        print(f"   Approach: Production-ready (stability over max accuracy)")


# Usage
if __name__ == "__main__":
    print("🧪 Testing ML-Ready Data Processing Pipeline...")

    # Load processed datasets
    datasets = {
        'internet': pd.read_csv("data/processed/internet_processed.csv"),
        'banking': pd.read_csv("data/processed/banking_processed.csv"),
        'website': pd.read_csv("data/processed/website_processed.csv")
    }

    print(f"📂 Loaded {len(datasets)} datasets")
    for name, df in datasets.items():
        print(f"   {name}: {df.shape}")

    # Apply ML-ready processing
    processor = MLReadyDataProcessor()
    ml_ready_datasets = processor.process_all_datasets(datasets)

    # Save ML-ready datasets
    import os
    os.makedirs("data/ml_ready", exist_ok=True)

    for name, df in ml_ready_datasets.items():
        output_path = f"data/ml_ready/{name}_ml_ready.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")

    print("\n✅ ML-ready processing pipeline completed!")