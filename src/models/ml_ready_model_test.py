"""
ML-Ready Model Testing Pipeline
Tests model performance with production-ready datasets.

This module validates our ML-ready data processing approach by:
1. Loading datasets processed with ml_ready_data_processor.py
2. Training models without suspicious features (reamining_contract, points_in_wallet)  
3. Comparing performance to baseline expectations
4. Ensuring models maintain reasonable performance without data leakage risk

Expected Performance (post-ML-ready processing):
- Banking: ~0.76 AUC (unchanged - no suspicious features)
- Internet: ~0.80-0.85 AUC (down from 0.97 due to reamining_contract removal)
- Website: ~0.75-0.80 AUC (down from 0.99+ due to points_in_wallet removal)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')


class MLReadyModelTester:
    """
    Tests model performance with ML-ready processed datasets.
    Validates that removing suspicious features still yields reasonable performance.
    """

    def __init__(self):
        self.results = {}
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1),
            'CatBoost': cb.CatBoostClassifier(iterations=100, random_state=42, verbose=False),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        self.performance_expectations = {
            'banking': {'min_auc': 0.70, 'max_auc': 0.80, 'note': 'Should remain stable (no features removed)'},
            'internet': {'min_auc': 0.75, 'max_auc': 0.90, 'note': 'Expected drop due to reamining_contract removal'},
            'website': {'min_auc': 0.70, 'max_auc': 0.85, 'note': 'Expected significant drop due to points_in_wallet removal'}
        }

    def prepare_features(self, df, target_col, dataset_name):
        """Prepare features for modeling - simple and clean"""
        print(f"🔧 Preparing {dataset_name} features (ML-ready mode)...")

        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        print(f"   📊 Features: {len(feature_cols)}")
        print(f"   📊 Samples: {len(X)}")
        print(f"   📊 Target distribution: {y.value_counts().to_dict()}")

        # Handle categorical features (the real problem)
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

        return X, y

    def train_and_evaluate_models(self, X, y, dataset_name):
        """Train and evaluate all models with cross-validation"""
        print(f"\n🚀 Training models on {dataset_name} (ML-ready dataset)...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features for algorithms that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        dataset_results = {}

        for model_name, model in self.models.items():
            print(f"   🔄 Training {model_name}...")

            # Use scaled data for SVM, Logistic Regression, KNN
            if model_name in ['SVM', 'Logistic Regression', 'KNN']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test

            # Train model
            model.fit(X_train_use, y_train)

            # Make predictions
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            # Calculate metrics
            test_auc = roc_auc_score(y_test, y_pred_proba)

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if model_name in ['SVM', 'Logistic Regression', 'KNN']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

            dataset_results[model_name] = {
                'test_auc': test_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_gap': abs(cv_scores.mean() - test_auc)
            }

            print(f"      📈 {model_name}: Test AUC = {test_auc:.4f}, CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return dataset_results

    def evaluate_ml_ready_performance(self, dataset_name, results):
        """Evaluate if ML-ready performance meets expectations"""
        print(f"\n🎯 Evaluating {dataset_name} ML-ready performance...")

        expectations = self.performance_expectations[dataset_name]
        best_model = max(results.items(), key=lambda x: x[1]['test_auc'])
        best_name, best_metrics = best_model
        best_auc = best_metrics['test_auc']

        print(f"   🏆 Best model: {best_name} (AUC: {best_auc:.4f})")
        print(f"   📊 Expected range: {expectations['min_auc']:.2f} - {expectations['max_auc']:.2f}")
        print(f"   💭 Note: {expectations['note']}")

        # Performance assessment
        if best_auc >= expectations['min_auc'] and best_auc <= expectations['max_auc']:
            status = "✅ MEETS EXPECTATIONS"
            color = "🟢"
        elif best_auc > expectations['max_auc']:
            status = "⚠️  SURPRISINGLY HIGH (check for remaining leakage)"
            color = "🟡"
        else:
            status = "❌ BELOW EXPECTATIONS"
            color = "🔴"

        print(f"   {color} Status: {status}")

        # Overfitting check
        overfitting_gap = best_metrics['overfitting_gap']
        if overfitting_gap < 0.05:
            print(f"   ✅ Stable model (train-test gap: {overfitting_gap:.4f})")
        else:
            print(f"   ⚠️  Potential overfitting (train-test gap: {overfitting_gap:.4f})")

        return {
            'best_model': best_name,
            'best_auc': best_auc,
            'status': status,
            'meets_expectations': best_auc >= expectations['min_auc'],
            'overfitting_gap': overfitting_gap
        }

    def test_all_datasets(self, datasets, targets):
        """Test all datasets with ML-ready processing"""
        print("=" * 70)
        print("🚀 ML-READY MODEL TESTING PIPELINE")
        print("=" * 70)
        print("Testing model performance with production-ready datasets")
        print("Validating ML-ready data processing decisions\n")

        overall_results = {}

        for dataset_name, df in datasets.items():
            target_col = targets[dataset_name]

            print(f"\n{'='*50}")
            print(f"📊 TESTING {dataset_name.upper()} DATASET")
            print(f"{'='*50}")

            # Prepare features
            X, y = self.prepare_features(df, target_col, dataset_name)

            # Train and evaluate models
            model_results = self.train_and_evaluate_models(X, y, dataset_name)

            # Evaluate ML-ready performance
            evaluation = self.evaluate_ml_ready_performance(dataset_name, model_results)

            overall_results[dataset_name] = {
                'model_results': model_results,
                'evaluation': evaluation,
                'feature_count': X.shape[1],
                'sample_count': len(X)
            }

        self._print_final_summary(overall_results)
        return overall_results

    def _print_final_summary(self, results):
        """Print comprehensive summary of ML-ready testing"""
        print(f"\n{'='*70}")
        print("📋 ML-READY MODEL TESTING SUMMARY")
        print(f"{'='*70}")

        print(f"🎯 Performance Validation:")

        all_meet_expectations = True
        for dataset_name, result in results.items():
            evaluation = result['evaluation']
            feature_count = result['feature_count']

            status_emoji = "✅" if evaluation['meets_expectations'] else "❌"
            print(f"   {status_emoji} {dataset_name.upper()}: {evaluation['best_model']} "
                f"(AUC: {evaluation['best_auc']:.4f}, {feature_count} features)")

            if not evaluation['meets_expectations']:
                all_meet_expectations = False

        print(f"\n🏁 OVERALL ASSESSMENT:")
        if all_meet_expectations:
            print("   ✅ ML-ready approach SUCCESSFUL")
            print("   ✅ Models maintain reasonable performance without data leakage risk")
            print("   ✅ Ready for hyperparameter optimization")
        else:
            print("   ⚠️  Some models below expectations - review feature engineering")

        print(f"\n💡 NEXT STEPS:")
        print("   1. Proceed with hyperparameter optimization using Optuna")
        print("   2. Focus on datasets that meet performance expectations")
        print("   3. Consider additional feature engineering for underperforming datasets")


# Testing and usage
if __name__ == "__main__":
    print("🧪 Testing ML-Ready Model Performance...")

    # Load ML-ready datasets
    datasets = {
        'internet': pd.read_csv("data/ml_ready/internet_ml_ready.csv"),
        'banking': pd.read_csv("data/ml_ready/banking_ml_ready.csv"),
        'website': pd.read_csv("data/ml_ready/website_ml_ready.csv")
    }

    # Target columns
    targets = {
        'internet': 'churn',
        'banking': 'churn',
        'website': 'churn'
    }

    print(f"📂 Loaded {len(datasets)} ML-ready datasets:")
    for name, df in datasets.items():
        print(f"   {name}: {df.shape}")

    # Initialize tester and run tests
    tester = MLReadyModelTester()
    results = tester.test_all_datasets(datasets, targets)

    print(f"\n✅ ML-ready model testing completed!")
    print(f"📄 Results stored in results object for further analysis")