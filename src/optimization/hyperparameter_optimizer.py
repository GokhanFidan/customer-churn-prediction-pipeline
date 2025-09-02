"""
Hyperparameter Optimization Pipeline with Optuna
Optimizes model parameters for better performance across all datasets.

This module uses Optuna's intelligent search algorithms to find optimal
hyperparameters for multiple ML algorithms on our ML-ready datasets.
"""

import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Suppress model-specific warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization using Optuna for multiple models.
    """

    def __init__(self, n_trials=20, timeout_per_model=600, cv_folds=5):
        self.n_trials = n_trials
        self.timeout_per_model = timeout_per_model  # 10 minutes per model
        self.cv_folds = cv_folds
        self.results = {}
        self.best_models = {}

        # Create output directories
        self.models_dir = Path("results/optimized_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = Path("results/optimization_plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model_class': RandomForestClassifier,
                'param_space': self._get_rf_param_space,
                'fixed_params': {'random_state': 42, 'n_jobs': -1}
            },
            'xgboost': {
                'model_class': xgb.XGBClassifier,
                'param_space': self._get_xgb_param_space,
                'fixed_params': {'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}
            },
            'lightgbm': {
                'model_class': lgb.LGBMClassifier,
                'param_space': self._get_lgb_param_space,
                'fixed_params': {'random_state': 42, 'verbosity': -1}
            },
            'catboost': {
                'model_class': cb.CatBoostClassifier,
                'param_space': self._get_cb_param_space,
                'fixed_params': {'random_state': 42, 'verbose': False}
            }
        }

    def _get_rf_param_space(self, trial):
        """Random Forest parameter space"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

    def _get_gb_param_space(self, trial):
        """Gradient Boosting parameter space"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

    def _get_xgb_param_space(self, trial):
        """XGBoost parameter space"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }

    def _get_lgb_param_space(self, trial):
        """LightGBM parameter space"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }

    def _get_cb_param_space(self, trial):
        """CatBoost parameter space"""
        return {
            'iterations': trial.suggest_int('iterations', 100, 500, step=50),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10, log=True)
        }

    def _prepare_data(self, df, target_col):
        """Prepare data for optimization"""
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

        return X, y

    def _objective_function(self, trial, model_name, model_config, X, y):
        """Objective function for Optuna optimization"""
        try:
            # Get hyperparameters from trial
            params = model_config['param_space'](trial)
            params.update(model_config['fixed_params'])

            # Create model
            model = model_config['model_class'](**params)

            # Cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

            return scores.mean()

        except Exception as e:
            # Return low score for failed trials
            return 0.5

    def optimize_model(self, model_name, X, y, dataset_name):
        """Optimize single model for given dataset"""
        print(f"\n🔄 Optimizing {model_name.upper()} for {dataset_name}")

        model_config = self.model_configs[model_name]

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Create objective function
        objective = lambda trial: self._objective_function(trial, model_name, model_config, X, y)

        # Optimize
        start_time = time.time()
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_per_model)
        optimization_time = time.time() - start_time

        # Get best results
        best_params = study.best_params
        best_score = study.best_value

        # Create and save best model
        final_params = {**best_params, **model_config['fixed_params']}
        best_model = model_config['model_class'](**final_params)
        best_model.fit(X, y)

        # Save model
        model_filename = f"{dataset_name}_{model_name}_optimized.joblib"
        joblib.dump(best_model, self.models_dir / model_filename)

        print(f"   ✅ Best AUC: {best_score:.4f}")
        print(f"   ⏰ Time: {optimization_time/60:.1f} minutes")
        print(f"   💾 Saved: {model_filename}")

        return {
            'model_name': model_name,
            'best_params': best_params,
            'best_score': best_score,
            'optimization_time': optimization_time,
            'n_completed_trials': len(study.trials),
            'study': study,
            'best_model': best_model
        }

    def optimize_dataset(self, df, target_col, dataset_name):
        """Optimize all models for a single dataset"""
        print(f"\n{'='*60}")
        print(f"🚀 OPTIMIZING {dataset_name.upper()} DATASET")
        print(f"{'='*60}")

        # Prepare data
        X, y = self._prepare_data(df, target_col)
        print(f"📊 Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"📊 Target distribution: {dict(y.value_counts())}")

        dataset_results = {}
        dataset_start_time = time.time()

        # Optimize each model
        for model_name in self.model_configs.keys():
            try:
                result = self.optimize_model(model_name, X, y, dataset_name)
                dataset_results[model_name] = result

            except Exception as e:
                print(f"   ❌ {model_name.upper()} optimization failed: {str(e)}")
                continue

        total_time = time.time() - dataset_start_time

        # Find best model for this dataset
        if dataset_results:
            best_model_name = max(dataset_results.keys(), key=lambda k: dataset_results[k]['best_score'])
            best_result = dataset_results[best_model_name]

            print(f"\n🏆 BEST MODEL FOR {dataset_name.upper()}:")
            print(f"   Model: {best_model_name.upper()}")
            print(f"   AUC: {best_result['best_score']:.4f}")
            print(f"   Parameters: {best_result['best_params']}")

        print(f"\n⏰ Total optimization time: {total_time/60:.1f} minutes")

        return dataset_results

    def optimize_all_datasets(self, datasets, targets):
        """Optimize hyperparameters for all datasets"""
        print("="*80)
        print("🎯 HYPERPARAMETER OPTIMIZATION PIPELINE")
        print("="*80)
        print(f"Using Optuna with {self.n_trials} trials per model")
        print(f"Timeout: {self.timeout_per_model/60} minutes per model")
        print(f"Cross-validation: {self.cv_folds}-fold StratifiedKFold")

        all_results = {}
        pipeline_start_time = time.time()

        # Process each dataset
        for dataset_name, df in datasets.items():
            target_col = targets[dataset_name]

            try:
                results = self.optimize_dataset(df, target_col, dataset_name)
                all_results[dataset_name] = results

            except Exception as e:
                print(f"❌ {dataset_name} optimization failed completely: {str(e)}")
                continue

        total_pipeline_time = time.time() - pipeline_start_time

        # Generate comprehensive summary
        self._generate_optimization_summary(all_results, total_pipeline_time)

        return all_results

    def _generate_optimization_summary(self, all_results, total_time):
        """Generate comprehensive optimization summary"""
        print(f"\n{'='*80}")
        print("📋 HYPERPARAMETER OPTIMIZATION SUMMARY")
        print(f"{'='*80}")

        # Overall statistics
        total_models_optimized = sum(len(results) for results in all_results.values())
        successful_datasets = len(all_results)

        print(f"🎯 OPTIMIZATION STATISTICS:")
        print(f"   📊 Datasets processed: {successful_datasets}")
        print(f"   🤖 Models optimized: {total_models_optimized}")
        print(f"   ⏰ Total time: {total_time/3600:.1f} hours")

        # Best model per dataset
        print(f"\n🏆 BEST MODELS BY DATASET:")
        dataset_champions = {}

        for dataset_name, results in all_results.items():
            if results:
                best_model_name = max(results.keys(), key=lambda k: results[k]['best_score'])
                best_result = results[best_model_name]
                dataset_champions[dataset_name] = (best_model_name, best_result['best_score'])

                print(f"   🥇 {dataset_name.upper()}: {best_model_name.upper()} "
                    f"(AUC: {best_result['best_score']:.4f})")

        # Cross-dataset model performance
        print(f"\n📊 MODEL PERFORMANCE ACROSS DATASETS:")
        model_performances = {}

        for dataset_name, results in all_results.items():
            for model_name, result in results.items():
                if model_name not in model_performances:
                    model_performances[model_name] = []
                model_performances[model_name].append(result['best_score'])

        for model_name, scores in model_performances.items():
            avg_score = np.mean(scores)
            print(f"   🤖 {model_name.upper():<12} Avg AUC: {avg_score:.4f} "
                f"(across {len(scores)} datasets)")

        # Time analysis
        print(f"\n⏰ OPTIMIZATION TIME ANALYSIS:")
        for dataset_name, results in all_results.items():
            total_dataset_time = sum(r['optimization_time'] for r in results.values())
            print(f"   📊 {dataset_name.upper()}: {total_dataset_time/60:.1f} minutes "
                f"({len(results)} models)")

        print(f"\n💾 ARTIFACTS SAVED:")
        print(f"   🤖 Optimized models: {self.models_dir}")
        print(f"   📈 Optimization plots: {self.plots_dir}")

        print(f"\n✅ Hyperparameter optimization completed successfully!")


# Main execution
if __name__ == "__main__":
    print("🧪 Starting Hyperparameter Optimization...")

    # Load ML-ready datasets
    try:
        datasets = {
            'internet': pd.read_csv("data/ml_ready/internet_ml_ready.csv"),
            'banking': pd.read_csv("data/ml_ready/banking_ml_ready.csv"),
            'website': pd.read_csv("data/ml_ready/website_ml_ready.csv")
        }

        targets = {
            'internet': 'churn',
            'banking': 'churn',
            'website': 'churn'
        }

        print(f"📂 Loaded {len(datasets)} ML-ready datasets:")
        for name, df in datasets.items():
            print(f"   {name}: {df.shape}")

        # Initialize optimizer
        optimizer = HyperparameterOptimizer(
            n_trials=20,           # 20 trials per model
            timeout_per_model=600, # 10 minutes per model
            cv_folds=5             # 5-fold cross-validation
        )

        # Run optimization
        results = optimizer.optimize_all_datasets(datasets, targets)

        print(f"\n🎯 Optimization pipeline completed!")
        print(f"📄 Results stored for further analysis")

    except Exception as e:
        print(f"❌ Critical error in optimization pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
