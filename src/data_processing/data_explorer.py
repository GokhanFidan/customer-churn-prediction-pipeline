"""
Comprehensive Exploratory Data Analysis (EDA)
Deep dive into datasets before any cleaning decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class DataExplorer:
    """Professional EDA pipeline for informed data cleaning decisions."""

    def __init__(self):
        self.analysis_results = {}

    def analyze_missing_patterns(self, df: pd.DataFrame, dataset_name: str):
        """Deep analysis of missing value patterns."""
        print(f"\n🔍 MISSING VALUE ANALYSIS - {dataset_name.upper()}")
        print("-" * 50)

        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        # Show only columns with missing values
        missing_data = pd.DataFrame({
            'Missing Count': missing_counts,
            'Missing %': missing_percentages
        }).round(2)

        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)

        if not missing_data.empty:
            print(missing_data)

            # Patterns analysis
            print(f"\n📊 Missing Value Insights:")
            for col in missing_data.index:
                if missing_data.loc[col, 'Missing %'] > 50:
                    print(f"   ⚠️  {col}: {missing_data.loc[col, 'Missing %']:.1f}% missing (CRITICAL)")
                elif missing_data.loc[col, 'Missing %'] > 20:
                    print(f"   🟡 {col}: {missing_data.loc[col, 'Missing %']:.1f}% missing (HIGH)")
                else:
                    print(f"   🟢 {col}: {missing_data.loc[col, 'Missing %']:.1f}% missing (LOW)")
        else:
            print("✅ No missing values found!")

        return missing_data

    def analyze_target_variable(self, df: pd.DataFrame, target_col: str, dataset_name: str):
        """Analyze target variable distribution."""
        print(f"\n🎯 TARGET VARIABLE ANALYSIS - {dataset_name.upper()}")
        print("-" * 50)

        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found!")
            return

        print(f"Target Column: {target_col}")
        print(f"Data Type: {df[target_col].dtype}")
        print(f"Unique Values: {df[target_col].nunique()}")

        # Value counts
        value_counts = df[target_col].value_counts().head(10)
        print(f"\nValue Distribution:")
        for val, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {val}: {count} ({percentage:.1f}%)")

        # Statistics
        if df[target_col].dtype in ['int64', 'float64']:
            print(f"\nStatistics:")
            print(f"   Mean: {df[target_col].mean():.3f}")
            print(f"   Std: {df[target_col].std():.3f}")
            print(f"   Min: {df[target_col].min()}")
            print(f"   Max: {df[target_col].max()}")

    def analyze_numeric_columns(self, df: pd.DataFrame, dataset_name: str):
        """Analyze numeric columns for outliers and distributions."""
        print(f"\n📊 NUMERIC COLUMNS ANALYSIS - {dataset_name.upper()}")
        print("-" * 50)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols[:5]:  # Show first 5 to avoid clutter
            print(f"\n🔢 {col}:")
            print(f"   Count: {df[col].count()}")
            print(f"   Missing: {df[col].isnull().sum()}")
            print(f"   Mean: {df[col].mean():.2f}")
            print(f"   Std: {df[col].std():.2f}")
            print(f"   Min: {df[col].min()}")
            print(f"   Max: {df[col].max()}")

            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold_low = Q1 - 1.5 * IQR
            outlier_threshold_high = Q3 + 1.5 * IQR

            outliers = df[(df[col] < outlier_threshold_low) | (df[col] > outlier_threshold_high)][col]
            print(f"   Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

    def analyze_categorical_columns(self, df: pd.DataFrame, dataset_name: str):
        """Analyze categorical columns."""
        print(f"\n🏷️  CATEGORICAL COLUMNS ANALYSIS - {dataset_name.upper()}")
        print("-" * 50)

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_cols[:5]:  # Show first 5
            print(f"\n📝 {col}:")
            print(f"   Count: {df[col].count()}")
            print(f"   Missing: {df[col].isnull().sum()}")
            print(f"   Unique: {df[col].nunique()}")

            # Show top values
            top_values = df[col].value_counts().head(3)
            print(f"   Top values:")
            for val, count in top_values.items():
                percentage = (count / df[col].count()) * 100
                print(f"      {val}: {count} ({percentage:.1f}%)")

    def full_dataset_analysis(self, datasets: Dict[str, pd.DataFrame], configs: dict):
        """Complete EDA for all datasets."""
        print("🔬 COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 60)

        for name, df in datasets.items():
            print(f"\n{'='*20} {name.upper()} DATASET {'='*20}")
            print(f"Shape: {df.shape}")
            print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # Target column from config
            target_col = configs['data']['datasets'][name]['target']

            # Analyses
            self.analyze_missing_patterns(df, name)
            self.analyze_target_variable(df, target_col, name)
            self.analyze_numeric_columns(df, name)
            self.analyze_categorical_columns(df, name)

            print(f"\n{'='*50}")


if __name__ == "__main__":
    # Test EDA
    from data_loader import DataLoader

    print("🔬 Starting Comprehensive EDA...")

    loader = DataLoader()
    datasets = loader.load_all_datasets()

    explorer = DataExplorer()
    explorer.full_dataset_analysis(datasets, loader.config)

    print(f"\n✅ EDA completed for {len(datasets)} datasets!")