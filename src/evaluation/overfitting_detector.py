"""
Overfitting Detection and Model Validation
Comprehensive analysis to detect model overfitting and ensure reliability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class OverfittingDetector:
    """Professional overfitting detection and model validation."""

    def __init__(self):
        self.analysis_results = {}

    def plot_learning_curves(self, model, X, y, dataset_name, model_name, cv=5):
        """Plot learning curves to detect overfitting."""
        print(f"   📈 Generating learning curves for {model_name}...")

        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc', random_state=42
        )

        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training ROC-AUC')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation ROC-AUC')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('ROC-AUC Score')
        plt.title(f'Learning Curves - {dataset_name} - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # Overfitting analysis
        final_gap = train_mean[-1] - val_mean[-1]

        # Add text analysis
        if final_gap > 0.05:
            plt.text(0.02, 0.98, f'⚠️ Possible Overfitting\nGap: {final_gap:.3f}',
                    transform=plt.gca().transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        elif final_gap < 0.02:
            plt.text(0.02, 0.98, f'✅ Good Generalization\nGap: {final_gap:.3f}',
                    transform=plt.gca().transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            plt.text(0.02, 0.98, f'⚠️ Monitor Closely\nGap: {final_gap:.3f}',
                    transform=plt.gca().transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

        plt.tight_layout()
        plt.savefig(f'results/plots/learning_curve_{dataset_name}_{model_name.replace(" ", "_")}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'final_train_score': train_mean[-1],
            'final_val_score': val_mean[-1],
            'overfitting_gap': final_gap,
            'is_overfitting': final_gap > 0.05
        }

    def analyze_cv_stability(self, results_df):
        """Analyze cross-validation stability to detect overfitting."""
        print("🔍 Analyzing Cross-Validation Stability...")

        stability_analysis = []

        for _, row in results_df.iterrows():
            dataset = row['Dataset']
            model = row['Model']
            cv_mean = row['CV Mean']
            cv_std = row['CV Std']
            test_score = row['ROC-AUC']

            # Calculate stability metrics
            cv_coefficient_variation = cv_std / cv_mean if cv_mean > 0 else 0
            train_test_gap = abs(cv_mean - test_score)

            # Stability flags
            is_unstable = cv_coefficient_variation > 0.1  # High variation
            is_suspicious = cv_std < 0.001  # Too stable (suspicious)
            large_gap = train_test_gap > 0.05  # Large train-test gap

            analysis = {
                'Dataset': dataset,
                'Model': model,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std,
                'Test_ROC_AUC': test_score,
                'CV_Coefficient_Variation': cv_coefficient_variation,
                'Train_Test_Gap': train_test_gap,
                'Is_Unstable': is_unstable,
                'Is_Suspicious': is_suspicious,
                'Large_Gap': large_gap,
                'Overall_Flag': is_unstable or is_suspicious or large_gap
            }

            stability_analysis.append(analysis)

        stability_df = pd.DataFrame(stability_analysis)

        # Summary
        print(f"\n📊 CV STABILITY ANALYSIS:")
        print("=" * 50)

        suspicious_models = stability_df[stability_df['Is_Suspicious']]['Model'].unique()
        unstable_models = stability_df[stability_df['Is_Unstable']]['Model'].unique()
        gap_models = stability_df[stability_df['Large_Gap']]['Model'].unique()

        if len(suspicious_models) > 0:
            print(f"🚩 SUSPICIOUS (Too Stable): {list(suspicious_models)}")
            print("   → CV Std < 0.001 indicates possible data leakage or overfitting")

        if len(unstable_models) > 0:
            print(f"⚠️ UNSTABLE (High Variation): {list(unstable_models)}")
            print("   → High CV variation indicates model instability")

        if len(gap_models) > 0:
            print(f"📈 LARGE GAPS: {list(gap_models)}")
            print("   → Large train-test gap indicates overfitting")

        # Website specific analysis
        website_results = stability_df[stability_df['Dataset'] == 'website']
        if len(website_results) > 0:
            print(f"\n🌐 WEBSITE DATASET ANALYSIS:")
            for _, row in website_results.iterrows():
                if row['Test_ROC_AUC'] > 0.99:
                    print(f"   🚨 {row['Model']}: {row['Test_ROC_AUC']:.3f} ROC-AUC - EXTREMELY HIGH!")
                    print(f"      CV Std: {row['CV_Std']:.4f} - {'SUSPICIOUS' if row['CV_Std'] < 0.001 else 'Normal'}")

        return stability_df

    def detect_data_leakage_patterns(self, processed_datasets, results_df):
        """Look for patterns that might indicate data leakage."""
        print("\n🔍 ANALYZING DATA LEAKAGE PATTERNS...")

        leakage_flags = []

        for dataset_name, df in processed_datasets.items():
            dataset_results = results_df[results_df['Dataset'] == dataset_name]

            # Check for suspiciously high performance
            max_roc_auc = dataset_results['ROC-AUC'].max()

            flags = {
                'dataset': dataset_name,
                'max_roc_auc': max_roc_auc,
                'extremely_high_performance': max_roc_auc > 0.99,
                'perfect_performance': max_roc_auc > 0.999,
                'feature_count': df.shape[1],
                'sample_count': df.shape[0],
                'feature_to_sample_ratio': df.shape[1] / df.shape[0]
            }

            # Additional checks
            if dataset_name == 'website' and max_roc_auc > 0.99:
                flags['website_specific_warning'] = True
                flags['possible_causes'] = [
                    "Target leakage in features",
                    "Time-based data leakage",
                    "Perfect separability due to data artifacts",
                    "Overfitting to specific patterns"
                ]

            leakage_flags.append(flags)

        # Print analysis
        print("📊 DATA LEAKAGE ANALYSIS:")
        print("-" * 30)

        for flag in leakage_flags:
            dataset = flag['dataset'].upper()
            print(f"\n{dataset} Dataset:")
            print(f"   Max ROC-AUC: {flag['max_roc_auc']:.3f}")

            if flag['extremely_high_performance']:
                print(f"   🚨 EXTREMELY HIGH PERFORMANCE - INVESTIGATE!")
                if 'possible_causes' in flag:
                    print(f"   Possible causes:")
                    for cause in flag['possible_causes']:
                        print(f"      • {cause}")

            if flag['feature_to_sample_ratio'] > 0.1:
                print(f"   ⚠️ High feature-to-sample ratio: {flag['feature_to_sample_ratio']:.3f}")

        return leakage_flags

    def plot_performance_comparison(self, results_df):
        """Create comprehensive visual analysis of all models."""
        print("📊 Creating comprehensive performance comparison plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔍 COMPREHENSIVE OVERFITTING ANALYSIS', fontsize=16, fontweight='bold')

        # 1. ROC-AUC by Dataset
        ax1 = axes[0, 0]
        datasets = results_df['Dataset'].unique()
        colors = ['blue', 'orange', 'green']

        for i, dataset in enumerate(datasets):
            data = results_df[results_df['Dataset'] == dataset]
            ax1.scatter(range(len(data)), data['ROC-AUC'],
                        label=dataset.capitalize(), s=100, alpha=0.7, color=colors[i])

        ax1.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='🚨 Suspicious Threshold')
        ax1.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='⚠️ High Performance')
        ax1.set_ylabel('ROC-AUC')
        ax1.set_title('ROC-AUC Performance by Dataset')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.7, 1.0)

        # 2. CV Stability Analysis  
        ax2 = axes[0, 1]
        scatter = ax2.scatter(results_df['CV Mean'], results_df['CV Std'],
                            c=results_df['ROC-AUC'], s=100, alpha=0.7, cmap='viridis')
        ax2.axhline(y=0.001, color='red', linestyle='--', alpha=0.7, label='🚨 Too Stable')
        ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='⚠️ Low Variation')
        ax2.set_xlabel('CV Mean ROC-AUC')
        ax2.set_ylabel('CV Standard Deviation')
        ax2.set_title('Cross-Validation Stability Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2)

        # 3. Train-Test Gap Analysis
        ax3 = axes[0, 2]
        train_test_gap = abs(results_df['CV Mean'] - results_df['ROC-AUC'])
        colors = ['red' if gap < 0.001 else 'orange' if gap < 0.01 else 'green'
                    for gap in train_test_gap]
        bars = ax3.bar(range(len(train_test_gap)), train_test_gap, color=colors, alpha=0.7)
        ax3.set_ylabel('|CV Mean - Test ROC-AUC|')
        ax3.set_title('Train-Test Performance Gap')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Add labels for high gaps
        for i, (gap, bar) in enumerate(zip(train_test_gap, bars)):
            if gap < 0.001:
                ax3.text(i, gap + 0.0001, '🚨', ha='center', va='bottom')

        # 4. Dataset-wise Maximum Performance
        ax4 = axes[1, 0]
        dataset_max_scores = results_df.groupby('Dataset')['ROC-AUC'].max()
        bars = ax4.bar(dataset_max_scores.index, dataset_max_scores.values)

        for bar, score in zip(bars, dataset_max_scores.values):
            if score > 0.99:
                bar.set_color('red')
                bar.set_label('🚨 Suspicious')
            elif score > 0.95:
                bar.set_color('orange')
                bar.set_label('⚠️ High')
            else:
                bar.set_color('green')
                bar.set_label('✅ Normal')

        # Add value labels on bars
        for bar, score in zip(bars, dataset_max_scores.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        ax4.axhline(y=0.99, color='red', linestyle='--', alpha=0.7)
        ax4.set_ylabel('Max ROC-AUC')
        ax4.set_title('Maximum Performance by Dataset')
        ax4.set_ylim(0.8, 1.0)

        # 5. Model Performance Heatmap
        ax5 = axes[1, 1]
        pivot_df = results_df.pivot(index='Model', columns='Dataset', values='ROC-AUC')
        im = ax5.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
        ax5.set_xticks(range(len(pivot_df.columns)))
        ax5.set_xticklabels(pivot_df.columns)
        ax5.set_yticks(range(len(pivot_df.index)))
        ax5.set_yticklabels(pivot_df.index, rotation=0)
        ax5.set_title('Model Performance Heatmap')

        # Add text annotations
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                color = 'white' if value < 0.85 else 'black'
                ax5.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color=color, fontweight='bold')

        plt.colorbar(im, ax=ax5)

        # 6. Suspicious Models Detail
        ax6 = axes[1, 2]
        suspicious_data = results_df[results_df['ROC-AUC'] > 0.99]
        if len(suspicious_data) > 0:
            bars = ax6.bar(range(len(suspicious_data)), suspicious_data['ROC-AUC'],
                            color='red', alpha=0.7)
            ax6.set_xticks(range(len(suspicious_data)))
            labels = [f"{row['Dataset']}\n{row['Model']}"
                    for _, row in suspicious_data.iterrows()]
            ax6.set_xticklabels(labels, rotation=45, ha='right')
            ax6.set_ylabel('ROC-AUC')
            ax6.set_title('🚨 SUSPICIOUS MODELS (>99%)')
            ax6.set_ylim(0.99, 1.0)

            # Add exact values
            for bar, score in zip(bars, suspicious_data['ROC-AUC']):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, '✅ No Suspicious Models\n(<99% ROC-AUC)',
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax6.set_title('Model Health Status')

        plt.tight_layout()
        plt.savefig('results/plots/comprehensive_overfitting_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_comparison(self, results_df):
        """Create comprehensive visual analysis of all models."""
        print("📊 Creating comprehensive performance comparison plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔍 COMPREHENSIVE OVERFITTING ANALYSIS', fontsize=16, fontweight='bold')

        # 1. ROC-AUC by Dataset
        ax1 = axes[0, 0]
        datasets = results_df['Dataset'].unique()
        colors = ['blue', 'orange', 'green']

        for i, dataset in enumerate(datasets):
            data = results_df[results_df['Dataset'] == dataset]
            ax1.scatter(range(len(data)), data['ROC-AUC'],
                        label=dataset.capitalize(), s=100, alpha=0.7, color=colors[i])

        ax1.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='🚨 Suspicious Threshold')
        ax1.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='⚠️ High Performance')
        ax1.set_ylabel('ROC-AUC')
        ax1.set_title('ROC-AUC Performance by Dataset')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.7, 1.0)

        # 2. CV Stability Analysis  
        ax2 = axes[0, 1]
        scatter = ax2.scatter(results_df['CV Mean'], results_df['CV Std'],
                            c=results_df['ROC-AUC'], s=100, alpha=0.7, cmap='viridis')
        ax2.axhline(y=0.001, color='red', linestyle='--', alpha=0.7, label='🚨 Too Stable')
        ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='⚠️ Low Variation')
        ax2.set_xlabel('CV Mean ROC-AUC')
        ax2.set_ylabel('CV Standard Deviation')
        ax2.set_title('Cross-Validation Stability Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2)

        # 3. Train-Test Gap Analysis
        ax3 = axes[0, 2]
        train_test_gap = abs(results_df['CV Mean'] - results_df['ROC-AUC'])
        colors = ['red' if gap < 0.001 else 'orange' if gap < 0.01 else 'green'
                    for gap in train_test_gap]
        bars = ax3.bar(range(len(train_test_gap)), train_test_gap, color=colors, alpha=0.7)
        ax3.set_ylabel('|CV Mean - Test ROC-AUC|')
        ax3.set_title('Train-Test Performance Gap')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Add labels for high gaps
        for i, (gap, bar) in enumerate(zip(train_test_gap, bars)):
            if gap < 0.001:
                ax3.text(i, gap + 0.0001, '🚨', ha='center', va='bottom')

        # 4. Dataset-wise Maximum Performance
        ax4 = axes[1, 0]
        dataset_max_scores = results_df.groupby('Dataset')['ROC-AUC'].max()
        bars = ax4.bar(dataset_max_scores.index, dataset_max_scores.values)

        for bar, score in zip(bars, dataset_max_scores.values):
            if score > 0.99:
                bar.set_color('red')
                bar.set_label('🚨 Suspicious')
            elif score > 0.95:
                bar.set_color('orange')
                bar.set_label('⚠️ High')
            else:
                bar.set_color('green')
                bar.set_label('✅ Normal')

        # Add value labels on bars
        for bar, score in zip(bars, dataset_max_scores.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        ax4.axhline(y=0.99, color='red', linestyle='--', alpha=0.7)
        ax4.set_ylabel('Max ROC-AUC')
        ax4.set_title('Maximum Performance by Dataset')
        ax4.set_ylim(0.8, 1.0)

        # 5. Model Performance Heatmap
        ax5 = axes[1, 1]
        pivot_df = results_df.pivot(index='Model', columns='Dataset', values='ROC-AUC')
        im = ax5.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
        ax5.set_xticks(range(len(pivot_df.columns)))
        ax5.set_xticklabels(pivot_df.columns)
        ax5.set_yticks(range(len(pivot_df.index)))
        ax5.set_yticklabels(pivot_df.index, rotation=0)
        ax5.set_title('Model Performance Heatmap')

        # Add text annotations
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                color = 'white' if value < 0.85 else 'black'
                ax5.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color=color, fontweight='bold')

        plt.colorbar(im, ax=ax5)

        # 6. Suspicious Models Detail
        ax6 = axes[1, 2]
        suspicious_data = results_df[results_df['ROC-AUC'] > 0.99]
        if len(suspicious_data) > 0:
            bars = ax6.bar(range(len(suspicious_data)), suspicious_data['ROC-AUC'],
                            color='red', alpha=0.7)
            ax6.set_xticks(range(len(suspicious_data)))
            labels = [f"{row['Dataset']}\n{row['Model']}"
                    for _, row in suspicious_data.iterrows()]
            ax6.set_xticklabels(labels, rotation=45, ha='right')
            ax6.set_ylabel('ROC-AUC')
            ax6.set_title('🚨 SUSPICIOUS MODELS (>99%)')
            ax6.set_ylim(0.99, 1.0)

            # Add exact values
            for bar, score in zip(bars, suspicious_data['ROC-AUC']):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, '✅ No Suspicious Models\n(<99% ROC-AUC)',
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax6.set_title('Model Health Status')

        plt.tight_layout()
        plt.savefig('results/plots/comprehensive_overfitting_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Test overfitting detection
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data_processing.data_loader import DataLoader
    from data_processing.data_cleaner import DataCleaner
    from feature_engineering.feature_processor import FeatureProcessor
    from models.model_trainer import ModelTrainer

    print("🔍 Testing Overfitting Detection...")

    # Load data and results
    loader = DataLoader()
    raw_datasets = loader.load_all_datasets()

    cleaner = DataCleaner()
    clean_datasets = cleaner.clean_all_datasets(raw_datasets)

    processor = FeatureProcessor()
    processed_datasets = processor.process_all_datasets(clean_datasets, is_training=True)

    trainer = ModelTrainer()
    results = trainer.train_all_datasets(processed_datasets, optimize_hyperparams=False)
    results_df = trainer.get_results_summary()

    # Run overfitting analysis
    detector = OverfittingDetector()

    # Manual analysis (fonksiyon ismi yerine direct call)
    print("🔍 COMPREHENSIVE OVERFITTING ANALYSIS")
    print("=" * 60)

    import os
    os.makedirs('results/plots', exist_ok=True)

    stability_df = detector.analyze_cv_stability(results_df)
    leakage_flags = detector.detect_data_leakage_patterns(processed_datasets, results_df)

    print(f"\n📊 GENERATING COMPREHENSIVE VISUAL ANALYSIS...")
    detector.plot_performance_comparison(results_df)

    print("\n🎯 Overfitting Analysis completed!")