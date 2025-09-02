"""
Modern Data Loading Module
Handles loading and initial processing of multiple churn datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import yaml


class DataLoader:
    """Advanced data loader for multiple churn prediction datasets."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.datasets: Dict[str, pd.DataFrame] = {}

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"✅ Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            print(f"❌ Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config file: {e}")
            raise

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets."""
        raw_data_path = Path(self.config['data']['raw_data_path'])

        for dataset_name, dataset_config in self.config['data']['datasets'].items():
            file_path = raw_data_path / dataset_config['file']

            print(f"📂 Loading {dataset_name} dataset from {file_path}")

            try:
                df = pd.read_csv(file_path)
                self.datasets[dataset_name] = df

                print(f"✅ {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")

            except FileNotFoundError:
                print(f"❌ Dataset file not found: {file_path}")
                continue
            except Exception as e:
                print(f"❌ Error loading {dataset_name}: {str(e)}")
                continue

        return self.datasets

    def get_dataset_summary(self):
        """Get basic summary of all datasets."""
        print("\n🔍 DATASET SUMMARY")
        print("=" * 50)

        for name, df in self.datasets.items():
            print(f"\n📊 {name.upper()} Dataset:")
            print(f"   Shape: {df.shape}")
            print(f"   Missing values: {df.isnull().sum().sum()}")
            print(f"   Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")


if __name__ == "__main__":
    # Test the data loader
    print("🚀 Testing Data Loader...")

    loader = DataLoader()
    datasets = loader.load_all_datasets()
    loader.get_dataset_summary()

    print(f"\n🎯 Successfully loaded {len(datasets)} datasets!")