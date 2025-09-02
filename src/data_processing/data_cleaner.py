"""
Simple and Effective Data Cleaning Based on EDA Insights
Clean approach: Remove obvious errors, handle missing values logically.
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import LabelEncoder


class DataCleaner:
    """Simple, EDA-driven data cleaning pipeline."""

    def __init__(self):
        self.cleaning_stats = {}

    def clean_internet_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean internet dataset based on EDA insights."""
        print("🧹 Cleaning Internet Dataset (Conservative Approach)...")

        original_shape = df.shape
        df_clean = df.copy()

        # 1. Remove obvious data errors
        print("   🔍 Removing data errors...")
        df_clean = df_clean[df_clean['subscription_age'] >= 0]  # Remove negative ages

        # 2. Handle missing values logically
        print("   📝 Handling missing values...")
        df_clean['reamining_contract'] = df_clean['reamining_contract'].fillna(0)  # No contract left
        df_clean['download_avg'] = df_clean['download_avg'].fillna(0)  # Didn't use service
        df_clean['upload_avg'] = df_clean['upload_avg'].fillna(0)    # Didn't use service

        # 3. Remove extreme outliers (top 1% to be safe)
        print("   📊 Handling extreme outliers...")
        df_clean = df_clean[df_clean['download_avg'] <= df_clean['download_avg'].quantile(0.99)]
        df_clean = df_clean[df_clean['upload_avg'] <= df_clean['upload_avg'].quantile(0.99)]

        # 4. Drop ID column (not useful for ML)
        df_clean = df_clean.drop('id', axis=1)

        # Store stats
        removed = original_shape[0] - df_clean.shape[0]
        self.cleaning_stats['internet'] = {
            'original_rows': original_shape[0],
            'cleaned_rows': df_clean.shape[0],
            'removed_rows': removed,
            'removal_percentage': f"{(removed/original_shape[0])*100:.1f}%"
        }

        print(f"   📈 Result: {original_shape} → {df_clean.shape} (removed {removed} rows)")
        return df_clean

    def clean_banking_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean banking dataset (minimal cleaning - data is already good)."""
        print("🧹 Cleaning Banking Dataset (Minimal Approach)...")

        original_shape = df.shape
        df_clean = df.copy()

        # 1. Drop ID column
        df_clean = df_clean.drop('customer_id', axis=1)

        # 2. Simple categorical encoding
        print("   🔤 Encoding categorical variables...")
        le_country = LabelEncoder()
        le_gender = LabelEncoder()

        df_clean['country'] = le_country.fit_transform(df_clean['country'])
        df_clean['gender'] = le_gender.fit_transform(df_clean['gender'])

        # 3. No aggressive cleaning - data is clean!
        print("   ✅ No major cleaning needed - data quality is excellent!")

        # Store stats
        self.cleaning_stats['banking'] = {
            'original_rows': original_shape[0],
            'cleaned_rows': df_clean.shape[0],
            'removed_rows': 0,
            'removal_percentage': "0.0%"
        }

        print(f"   📈 Result: {original_shape} → {df_clean.shape} (perfect data!)")
        return df_clean

    def clean_website_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean website dataset (moderate cleaning - has quality issues)."""
        print("🧹 Cleaning Website Dataset (Moderate Approach)...")

        original_shape = df.shape
        df_clean = df.copy()
        step_stats = []

        # Track each cleaning step
        def log_step(description, df_current):
            current_rows = df_current.shape[0]
            removed = original_shape[0] - current_rows
            pct_removed = (removed / original_shape[0]) * 100
            removed_this_step = removed - (step_stats[-1]['removed_cumulative'] if step_stats else 0)
            step_stats.append({
                'step': description,
                'remaining_rows': current_rows,
                'removed_this_step': removed_this_step,
                'removed_cumulative': removed,
                'pct_removed': pct_removed
            })
            if removed_this_step > 0:
                print(f"   📊 {description}: -{removed_this_step:,} rows ({current_rows:,} remaining)")

        # 1. Remove obvious error codes and impossible values
        print("   🔍 Step by step cleaning analysis:")
        log_step("Initial", df_clean)

        df_clean = df_clean[df_clean['days_since_last_login'] != -999]
        log_step("Remove -999 login days", df_clean)

        df_clean = df_clean[df_clean['avg_time_spent'] >= 0]
        log_step("Remove negative time spent", df_clean)

        df_clean = df_clean[df_clean['points_in_wallet'] >= 0]
        log_step("Remove negative points", df_clean)

        df_clean = df_clean[df_clean['churn_risk_score'] >= 1]
        log_step("Remove invalid churn scores", df_clean)

        # 2. Remove unnecessary columns (PII and redundant)
        drop_cols = ['customer_id', 'Name', 'security_no', 'joining_date', 'last_visit_time']
        df_clean = df_clean.drop([col for col in drop_cols if col in df_clean.columns], axis=1)
        log_step("Drop PII columns", df_clean)

        # 3. Clean categorical variables
        df_clean = df_clean[df_clean['gender'] != 'Unknown']
        log_step("Remove Unknown gender", df_clean)

        df_clean = df_clean[df_clean['joined_through_referral'] != '?']
        log_step("Remove ? referral", df_clean)

        df_clean = df_clean[df_clean['medium_of_operation'] != '?']
        log_step("Remove ? medium", df_clean)

        # 4. Handle remaining missing values
        df_clean = df_clean.dropna(subset=['region_category', 'preferred_offer_types'])
        log_step("Remove missing categories", df_clean)

        df_clean['points_in_wallet'] = df_clean['points_in_wallet'].fillna(df_clean['points_in_wallet'].median())

        # 5. Encode categorical variables
        print("   🔤 Encoding categorical variables...")
        categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
        if 'churn_risk_score' in categorical_columns:
            categorical_columns.remove('churn_risk_score')
        
        for col in categorical_columns:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        
        # 6. Convert target to binary (4+ = high risk churn)
        df_clean['churn'] = (df_clean['churn_risk_score'] >= 4).astype(int)
        df_clean = df_clean.drop('churn_risk_score', axis=1)

        # Final summary
        removed = original_shape[0] - df_clean.shape[0]
        self.cleaning_stats['website'] = {
            'original_rows': original_shape[0],
            'cleaned_rows': df_clean.shape[0],
            'removed_rows': removed,
            'removal_percentage': f"{(removed/original_shape[0])*100:.1f}%",
            'step_by_step': step_stats
        }

        print(f"\n   🔥 BIGGEST IMPACT STEPS:")
        # Skip first step (initial) and show top 3 impact steps
        impact_steps = [s for s in step_stats[1:] if s['removed_this_step'] > 0]
        sorted_steps = sorted(impact_steps, key=lambda x: x['removed_this_step'], reverse=True)
        for step in sorted_steps[:3]:
            print(f"      • {step['step']}: -{step['removed_this_step']:,} rows")

        print(f"   📈 Final Result: {original_shape} → {df_clean.shape}")
        return df_clean

    def clean_all_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean all datasets with appropriate strategies."""
        print("\n🚿 STARTING DATA CLEANING - EDA-DRIVEN APPROACH")
        print("=" * 60)

        cleaned_datasets = {}

        if 'internet' in datasets:
            cleaned_datasets['internet'] = self.clean_internet_dataset(datasets['internet'])

        if 'banking' in datasets:
            cleaned_datasets['banking'] = self.clean_banking_dataset(datasets['banking'])

        if 'website' in datasets:
            cleaned_datasets['website'] = self.clean_website_dataset(datasets['website'])

        # Print summary
        print(f"\n📊 CLEANING SUMMARY:")
        print("-" * 30)
        for name, stats in self.cleaning_stats.items():
            print(f"{name.upper()}: {stats['original_rows']} → {stats['cleaned_rows']} "
                f"({stats['removal_percentage']} removed)")

        print(f"\n✨ CLEANING COMPLETED! Ready for modeling.")
        return cleaned_datasets

if __name__ == "__main__":
    print("🧪 Testing EDA-Driven Data Cleaner...")

    # Configuration
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration loaded from config/config.yaml")

    cleaner = DataCleaner()

    # Load datasets manually
    datasets = {}

    # Internet dataset
    internet_df = pd.read_csv("data/raw/internet_service_churn.csv")
    datasets['internet'] = internet_df
    print(f"📂 Loading internet dataset from data/raw/internet_service_churn.csv")
    print(f"✅ internet: {internet_df.shape[0]} rows, {internet_df.shape[1]} columns")

    # Banking dataset  
    banking_df = pd.read_csv("data/raw/Bank Customer Churn Prediction.csv")
    datasets['banking'] = banking_df
    print(f"📂 Loading banking dataset from data/raw/Bank Customer Churn Prediction.csv")
    print(f"✅ banking: {banking_df.shape[0]} rows, {banking_df.shape[1]} columns")

    # Website dataset
    website_df = pd.read_csv("data/raw/website.csv")
    datasets['website'] = website_df
    print(f"📂 Loading website dataset from data/raw/website.csv")
    print(f"✅ website: {website_df.shape[0]} rows, {website_df.shape[1]} columns")

    # Clean all datasets
    cleaned_datasets = cleaner.clean_all_datasets(datasets)

    # Save cleaned data
    print("\n💾 Saving cleaned datasets...")

    import os
    os.makedirs("data/processed", exist_ok=True)

    for name, df in cleaned_datasets.items():
        file_path = f"data/processed/{name}_processed.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ {name}: {file_path} kaydedildi")

    print("\n🎯 Successfully cleaned and saved 3 datasets!")

    # Target distributions
    for name, df in cleaned_datasets.items():
        target_col = 'churn'
        if target_col in df.columns:
            target_dist = df[target_col].value_counts().to_dict()
            print(f"{name}: {df.shape} - Target distribution: {target_dist}")