"""
EDA Visualization Module
Creates professional charts for data exploration and cleaning insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('default')
sns.set_palette("husl")


class EDAVisualizer:
    """Professional EDA visualization for churn prediction datasets."""

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize

    def plot_missing_values(self, datasets: Dict[str, pd.DataFrame]):
        """Create missing value heatmaps for all datasets."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Missing Values Analysis Across Datasets', fontsize=16, fontweight='bold')

        for idx, (name, df) in enumerate(datasets.items()):
            # Calculate missing percentages
            missing_pct = (df.isnull().sum() / len(df)) * 100
            missing_data = missing_pct[missing_pct > 0].sort_values(ascending=False)

            if not missing_data.empty:
                ax = axes[idx]
                bars = ax.bar(range(len(missing_data)), missing_data.values)
                ax.set_title(f'{name.capitalize()} Dataset', fontweight='bold')
                ax.set_ylabel('Missing Percentage (%)')
                ax.set_xticks(range(len(missing_data)))
                ax.set_xticklabels(missing_data.index, rotation=45, ha='right')

                # Color bars by severity
                for bar, pct in zip(bars, missing_data.values):
                    if pct > 20:
                        bar.set_color('red')
                    elif pct > 10:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')

                # Add percentage labels
                for bar, pct in zip(bars, missing_data.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            else:
                axes[idx].text(0.5, 0.5, 'No Missing Values!',
                            ha='center', va='center', fontsize=14,
                            transform=axes[idx].transAxes, color='green', fontweight='bold')
                axes[idx].set_title(f'{name.capitalize()} Dataset', fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/plots/missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_target_distributions(self, datasets: Dict[str, pd.DataFrame], configs: dict):
        """Plot target variable distributions."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 14))  # Slightly increased height
        fig.suptitle('Target Variable Analysis - After Cleaning', fontsize=16, fontweight='bold')

        colors = ['lightcoral', 'lightblue']

        for idx, (name, df) in enumerate(datasets.items()):
            target_col = configs['data']['datasets'][name]['target']

            # Handle website dataset special case
            if name == 'website' and 'churn' in df.columns:
                target_col = 'churn'

            if target_col in df.columns:
                # Count plot
                ax1 = axes[0, idx]
                value_counts = df[target_col].value_counts().sort_index()  # Sort by index (0,1)
                bars = ax1.bar(value_counts.index, value_counts.values, color=colors)
                ax1.set_title(f'{name.capitalize()} - Churn Distribution', fontweight='bold', pad=20)
                ax1.set_ylabel('Count')

                # Fix x-axis to show only 0 and 1
                ax1.set_xticks([0, 1])
                ax1.set_xticklabels(['No Churn (0)', 'Churn (1)'])
                ax1.set_xlim(-0.5, 1.5)  # Limit x-axis range

                # Add count labels
                for bar, count in zip(bars, value_counts.values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts)*0.01,
                            f'{count:,}', ha='center', va='bottom', fontweight='bold')

                # Percentage pie chart
                ax2 = axes[1, idx]
                labels = ['No Churn', 'Churn']
                # Ensure order is consistent (0: No Churn, 1: Churn)
                pie_values = [value_counts.get(0, 0), value_counts.get(1, 0)]
                ax2.pie(pie_values, labels=labels, autopct='%1.1f%%',
                        colors=colors, startangle=90)
                ax2.set_title(f'{name.capitalize()} - Churn Percentage', fontweight='bold', pad=15)

        # Adjust spacing to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        plt.subplots_adjust(hspace=0.4)  # More vertical space between rows
        plt.savefig('results/plots/target_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_cleaning_impact(self, raw_datasets: Dict[str, pd.DataFrame], 
                            clean_datasets: Dict[str, pd.DataFrame]):
        """Show before/after cleaning comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Cleaning Impact Analysis', fontsize=16, fontweight='bold')

        # Dataset size comparison
        ax1 = axes[0, 0]
        datasets_names = list(raw_datasets.keys())
        raw_sizes = [df.shape[0] for df in raw_datasets.values()]
        clean_sizes = [clean_datasets[name].shape[0] for name in datasets_names]

        x = np.arange(len(datasets_names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, raw_sizes, width, label='Before Cleaning', color='lightcoral')
        bars2 = ax1.bar(x + width/2, clean_sizes, width, label='After Cleaning', color='lightgreen')

        ax1.set_title('Dataset Sizes: Before vs After Cleaning', fontweight='bold')
        ax1.set_ylabel('Number of Records')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.capitalize() for name in datasets_names])
        ax1.legend()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + max(raw_sizes)*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

        # Removal percentage
        ax2 = axes[0, 1]
        removal_pcts = [(raw_sizes[i] - clean_sizes[i]) / raw_sizes[i] * 100
                        for i in range(len(raw_sizes))]

        bars = ax2.bar(datasets_names, removal_pcts, color=['orange', 'green', 'red'])
        ax2.set_title('Data Removal Percentage by Dataset', fontweight='bold')
        ax2.set_ylabel('Percentage Removed (%)')
        ax2.set_xticklabels([name.capitalize() for name in datasets_names])

        # Add percentage labels
        for bar, pct in zip(bars, removal_pcts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(removal_pcts)*0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Memory usage comparison
        ax3 = axes[1, 0]
        raw_memory = [df.memory_usage(deep=True).sum() / 1024**2 for df in raw_datasets.values()]
        clean_memory = [clean_datasets[name].memory_usage(deep=True).sum() / 1024**2
                        for name in datasets_names]

        bars1 = ax3.bar(x - width/2, raw_memory, width, label='Before', color='lightcoral')
        bars2 = ax3.bar(x + width/2, clean_memory, width, label='After', color='lightgreen')

        ax3.set_title('Memory Usage: Before vs After', fontweight='bold')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([name.capitalize() for name in datasets_names])
        ax3.legend()

        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = "📊 CLEANING SUMMARY:\n\n"
        for i, name in enumerate(datasets_names):
            removed = raw_sizes[i] - clean_sizes[i]
            summary_text += f"{name.upper()}:\n"
            summary_text += f"  • Original: {raw_sizes[i]:,} rows\n"
            summary_text += f"  • Cleaned: {clean_sizes[i]:,} rows\n"
            summary_text += f"  • Removed: {removed:,} ({removal_pcts[i]:.1f}%)\n\n"

        ax4.text(0.1, 0.9, summary_text, fontsize=12, va='top', ha='left',
                transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()
        plt.savefig('results/plots/cleaning_impact.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_full_eda_report(self, raw_datasets: Dict[str, pd.DataFrame], 
                            clean_datasets: Dict[str, pd.DataFrame], configs: dict):
        """Generate complete EDA visualization report."""
        print("📊 Generating EDA Visualization Report...")

        # Create results/plots directory
        import os
        os.makedirs('results/plots', exist_ok=True)

        # Generate all visualizations
        print("   🔍 Creating missing values analysis...")
        self.plot_missing_values(raw_datasets)

        print("   🎯 Creating target distributions...")
        self.plot_target_distributions(clean_datasets, configs)

        print("   📈 Creating cleaning impact analysis...")
        self.plot_cleaning_impact(raw_datasets, clean_datasets)

        print("✅ EDA Report generated successfully!")
        print("📁 Charts saved in: results/plots/")


if __name__ == "__main__":
    # Test visualization
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from data_processing.data_loader import DataLoader
    from data_processing.data_cleaner import DataCleaner

    print("📊 Testing EDA Visualizations...")

    # Load and clean data
    loader = DataLoader()
    raw_datasets = loader.load_all_datasets()

    cleaner = DataCleaner()
    clean_datasets = cleaner.clean_all_datasets(raw_datasets)

    # Generate visualizations
    visualizer = EDAVisualizer()
    visualizer.create_full_eda_report(raw_datasets, clean_datasets, loader.config)

    print("🎯 EDA Visualizations completed!")