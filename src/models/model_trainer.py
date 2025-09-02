"""
Advanced Model Training Pipeline
Handles multiple algorithms, hyperparameter optimization, and cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Professional model training with multiple algorithms and hyperparameter optimization."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.results = {}

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all 8 models with default parameters."""
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss', verbosity=0),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1),
            'CatBoost': cb.CatBoostClassifier(random_state=self.random_state, verbose=False, iterations=100),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        return models

    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for optimization."""
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.01],
                'subsample': [0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'Naive Bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }
        return param_grids

    def train_single_dataset(self, df: pd.DataFrame, dataset_name: str, 
                            target_col: str = 'churn', optimize_hyperparams: bool = True) -> Dict:
        """Train models on a single dataset."""
        print(f"\n🤖 Training models for {dataset_name.upper()} dataset...")

        # Prepare data
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        print(f"   📊 Dataset shape: {X.shape}")
        print(f"   🎯 Target distribution: {y.value_counts().to_dict()}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Initialize models
        models = self.initialize_models()
        dataset_results = {}

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for model_name, model in models.items():
            print(f"   🔧 Training {model_name}...")

            try:
                if optimize_hyperparams and model_name in self.get_hyperparameter_grids():
                    # Hyperparameter optimization
                    param_grid = self.get_hyperparameter_grids()[model_name]

                    # Use smaller grid for faster training
                    if model_name == 'SVM' and X_train.shape[0] > 10000:
                        param_grid = {'C': [1.0], 'kernel': ['rbf'], 'gamma': ['scale']}

                    grid_search = GridSearchCV(
                        model, param_grid, cv=3, scoring='roc_auc',
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    # Use default parameters
                    best_model = model
                    best_model.fit(X_train, y_train)
                    best_params = "Default parameters"

                # Predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred

                # Cross-validation scores
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')

                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'best_params': best_params
                }

                dataset_results[model_name] = metrics

                print(f"      ✅ {model_name}: ROC-AUC = {metrics['roc_auc']:.3f}")

            except Exception as e:
                print(f"      ❌ {model_name}: Error - {str(e)}")
                dataset_results[model_name] = {'error': str(e)}

        return dataset_results

    def train_all_datasets(self, datasets: Dict[str, pd.DataFrame], 
                        optimize_hyperparams: bool = True) -> Dict[str, Dict]:
        """Train models on all datasets."""
        print("🚀 STARTING MODEL TRAINING PIPELINE")
        print("=" * 60)

        all_results = {}

        for dataset_name, df in datasets.items():
            # Determine target column
            target_col = 'churn' if 'churn' in df.columns else df.columns[-1]

            try:
                results = self.train_single_dataset(
                    df, dataset_name, target_col, optimize_hyperparams
                )
                all_results[dataset_name] = results

                # Find best model for this dataset
                valid_results = {k: v for k, v in results.items() if 'error' not in v}
                if valid_results:
                    best_model_name = max(valid_results.keys(),
                                        key=lambda x: valid_results[x]['roc_auc'])
                    print(f"   🏆 Best model for {dataset_name}: {best_model_name} "
                        f"(ROC-AUC: {valid_results[best_model_name]['roc_auc']:.3f})")

            except Exception as e:
                print(f"   ❌ Error training {dataset_name}: {str(e)}")
                all_results[dataset_name] = {'error': str(e)}

        self.results = all_results
        print(f"\n✨ MODEL TRAINING COMPLETED!")
        return all_results

    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all results."""
        summary_data = []

        for dataset_name, dataset_results in self.results.items():
            for model_name, metrics in dataset_results.items():
                if 'error' not in metrics:
                    row = {
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1'],
                        'ROC-AUC': metrics['roc_auc'],
                        'CV Mean': metrics['cv_mean'],
                        'CV Std': metrics['cv_std']
                    }
                    summary_data.append(row)

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Test model training
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data_processing.data_loader import DataLoader
    from data_processing.data_cleaner import DataCleaner
    from feature_engineering.feature_processor import FeatureProcessor

    print("🧪 Testing Model Training Pipeline...")

    # Load, clean, and process data
    loader = DataLoader()
    raw_datasets = loader.load_all_datasets()

    cleaner = DataCleaner()
    clean_datasets = cleaner.clean_all_datasets(raw_datasets)

    processor = FeatureProcessor()
    processed_datasets = processor.process_all_datasets(clean_datasets, is_training=True)

    # Train models (without hyperparameter optimization for speed)
    trainer = ModelTrainer()
    results = trainer.train_all_datasets(processed_datasets, optimize_hyperparams=False)

    # Show summary
    summary_df = trainer.get_results_summary()
    print(f"\n📊 RESULTS SUMMARY:")
    print(summary_df.round(3))

    print("\n🎯 Model Training completed successfully!")