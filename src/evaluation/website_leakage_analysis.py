"""
Website Dataset Leakage Investigation
Deep dive analysis for remaining suspicious features in website dataset.

CONTEXT:
- Removed points_in_wallet (0.9536 AUC) but still getting 0.9624 AUC
- This suggests additional leaky features remain in the dataset
- Need to identify and evaluate remaining suspicious features

INVESTIGATION APPROACH:
1. Single feature performance testing - identify features with >0.9 AUC
2. Business logic evaluation - assess if features could contain future information
3. Correlation analysis - check for derived/engineered features
4. Decision framework - conservative removal vs feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')


class WebsiteLeakageInvestigator:
    """
    Specialized investigator for website dataset leakage issues.
    Focuses on identifying remaining suspicious features after initial cleanup.
    """

    def __init__(self, output_dir="results/website_leakage_investigation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.investigation_results = {}
        self.business_logic_assessment = {}

    def load_website_data(self, file_path="data/ml_ready/website_ml_ready.csv"):
        """Load website ML-ready dataset"""
        print("📂 Loading website ML-ready dataset...")

        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded: {df.shape}")
            print(f"📊 Target distribution: {df['churn'].value_counts().to_dict()}")

            # Show removed features for context
            print(f"\n🚫 Previously removed features:")
            print(f"   - points_in_wallet (AUC: 0.9536) - Too high, overfitting risk")

            return df

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return None

    def quick_single_feature_test(self, df, top_n=15):
        """
        Quick single feature performance test to identify suspicious features.
        This is the main method to find remaining leaky features.
        """
        print(f"\n🎯 QUICK SINGLE FEATURE PERFORMANCE TEST")
        print("="*60)
        print("Testing each feature individually to identify suspicious patterns")

        # Prepare data
        feature_cols = [col for col in df.columns if col != 'churn']
        X = df[feature_cols].copy()
        y = df['churn'].copy()

        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

        print(f"📊 Testing {len(feature_cols)} features...")

        performance_results = []

        for col in feature_cols:
            try:
                X_single = X[[col]].values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_single, y, test_size=0.2, random_state=42, stratify=y
                )

                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)

                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)

                performance_results.append((col, auc_score))

                # Real-time suspicious feature detection
                if auc_score > 0.9:
                    print(f"🚨 HIGHLY SUSPICIOUS: {col:<30} AUC: {auc_score:.4f}")
                elif auc_score > 0.8:
                    print(f"⚠️  SUSPICIOUS: {col:<30} AUC: {auc_score:.4f}")
                elif auc_score > 0.7:
                    print(f"🔍 NOTABLE: {col:<30} AUC: {auc_score:.4f}")

            except Exception as e:
                print(f"❌ Failed to test {col}: {str(e)}")
                continue

        # Sort by performance
        performance_results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n📊 TOP {top_n} FEATURE PERFORMANCE:")
        print("-" * 60)

        suspicious_features = []
        notable_features = []

        for i, (col, auc) in enumerate(performance_results[:top_n]):
            if auc > 0.9:
                status = "🚨 CRITICAL"
                suspicious_features.append((col, auc))
            elif auc > 0.8:
                status = "⚠️  HIGH"
                suspicious_features.append((col, auc))
            elif auc > 0.7:
                status = "🔍 NOTABLE"
                notable_features.append((col, auc))
            else:
                status = "✅ NORMAL"

            print(f"{i+1:2d}. {status} {col:<30} AUC: {auc:.4f}")

        self.investigation_results['single_feature_performance'] = performance_results
        self.investigation_results['suspicious_features'] = suspicious_features
        self.investigation_results['notable_features'] = notable_features

        return performance_results, suspicious_features, notable_features

    def business_logic_evaluation(self, suspicious_features):
        """
        Evaluate suspicious features from business logic perspective.
        Determine if features could contain future information or are legitimate signals.
        """
        print(f"\n🧠 BUSINESS LOGIC EVALUATION")
        print("="*60)
        print("Evaluating suspicious features for potential data leakage")

        # Business logic patterns that indicate leakage
        leakage_patterns = {
            'temporal': ['last_', 'recent_', 'days_since', 'time_'],
            'outcome_derived': ['complaint_status', 'feedback', 'satisfaction'],
            'behavioral_post_decision': ['usage_after', 'activity_post', 'engagement_final'],
            'administrative': ['status', 'flag', 'category_final']
        }

        assessments = {}

        for feature_name, auc_score in suspicious_features:
            print(f"\n🔍 Analyzing: {feature_name} (AUC: {auc_score:.4f})")

            assessment = {
                'feature': feature_name,
                'auc_score': auc_score,
                'leakage_risk': 'LOW',
                'risk_factors': [],
                'business_rationale': '',
                'recommendation': 'KEEP'
            }

            feature_lower = feature_name.lower()

            # Check for temporal leakage patterns
            for pattern_type, patterns in leakage_patterns.items():
                for pattern in patterns:
                    if pattern in feature_lower:
                        assessment['leakage_risk'] = 'HIGH' if auc_score > 0.9 else 'MEDIUM'
                        assessment['risk_factors'].append(f"{pattern_type}: contains '{pattern}'")

            # Specific feature analysis
            if 'complaint' in feature_lower:
                if 'status' in feature_lower:
                    assessment['leakage_risk'] = 'HIGH'
                    assessment['risk_factors'].append('Complaint status likely determined after churn decision')
                    assessment['business_rationale'] = 'Complaint resolution happens after customer decides to leave'
                    assessment['recommendation'] = 'REMOVE'
                else:
                    assessment['leakage_risk'] = 'MEDIUM'
                    assessment['business_rationale'] = 'Past complaints are legitimate predictors'
                    assessment['recommendation'] = 'KEEP'

            elif 'feedback' in feature_lower:
                assessment['leakage_risk'] = 'HIGH'
                assessment['risk_factors'].append('Feedback typically given during/after churn process')
                assessment['business_rationale'] = 'Customer feedback often collected during exit surveys'
                assessment['recommendation'] = 'REMOVE'

            elif 'days_since' in feature_lower:
                if 'login' in feature_lower:
                    assessment['leakage_risk'] = 'MEDIUM'
                    assessment['business_rationale'] = 'Days since last login is predictive behavior'
                    assessment['recommendation'] = 'KEEP'
                else:
                    assessment['leakage_risk'] = 'HIGH'
                    assessment['risk_factors'].append('Days since events may be calculated post-churn')
                    assessment['recommendation'] = 'INVESTIGATE'

            # AUC-based assessment
            if auc_score > 0.95:
                if assessment['leakage_risk'] == 'LOW':
                    assessment['leakage_risk'] = 'HIGH'
                    assessment['risk_factors'].append('Extremely high AUC suggests possible leakage')
                assessment['recommendation'] = 'REMOVE'
            elif auc_score > 0.9:
                if assessment['leakage_risk'] == 'LOW':
                    assessment['leakage_risk'] = 'MEDIUM'
                    assessment['risk_factors'].append('Very high AUC - investigate further')

            # Print assessment
            risk_emoji = "🚨" if assessment['leakage_risk'] == 'HIGH' else "⚠️" if assessment['leakage_risk'] == 'MEDIUM' else "✅"
            rec_emoji = "🚫" if assessment['recommendation'] == 'REMOVE' else "🔍" if assessment['recommendation'] == 'INVESTIGATE' else "✅"

            print(f"   {risk_emoji} Risk Level: {assessment['leakage_risk']}")
            print(f"   {rec_emoji} Recommendation: {assessment['recommendation']}")

            if assessment['risk_factors']:
                print(f"   🔍 Risk Factors:")
                for factor in assessment['risk_factors']:
                    print(f"      - {factor}")

            if assessment['business_rationale']:
                print(f"   💭 Business Logic: {assessment['business_rationale']}")

            assessments[feature_name] = assessment

        self.business_logic_assessment = assessments
        return assessments

    def generate_recommendations(self):
        """Generate final recommendations based on investigation"""
        print(f"\n📋 INVESTIGATION SUMMARY & RECOMMENDATIONS")
        print("="*60)

        suspicious_features = self.investigation_results.get('suspicious_features', [])
        assessments = self.business_logic_assessment

        to_remove = []
        to_investigate = []
        to_keep = []

        for feature_name, auc_score in suspicious_features:
            if feature_name in assessments:
                recommendation = assessments[feature_name]['recommendation']

                if recommendation == 'REMOVE':
                    to_remove.append((feature_name, auc_score))
                elif recommendation == 'INVESTIGATE':
                    to_investigate.append((feature_name, auc_score))
                else:
                    to_keep.append((feature_name, auc_score))

        print(f"🚫 FEATURES TO REMOVE: {len(to_remove)}")
        for feature, auc in to_remove:
            reason = assessments[feature]['business_rationale'] or 'High leakage risk'
            print(f"   - {feature} (AUC: {auc:.4f}) - {reason}")

        if to_investigate:
            print(f"\n🔍 FEATURES TO INVESTIGATE FURTHER: {len(to_investigate)}")
            for feature, auc in to_investigate:
                print(f"   - {feature} (AUC: {auc:.4f})")

        if to_keep:
            print(f"\n✅ SUSPICIOUS BUT LEGITIMATE FEATURES: {len(to_keep)}")
            for feature, auc in to_keep:
                print(f"   - {feature} (AUC: {auc:.4f})")

        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if to_remove:
            print(f"   ⚠️  Additional data leakage detected")
            print(f"   📊 Expected AUC after removal: 0.75-0.85 (more realistic)")
            print(f"   💡 Recommendation: Remove {len(to_remove)} features and re-test")
        else:
            print(f"   🤔 No clear leakage detected - high performance may be legitimate")
            print(f"   📊 Website dataset may genuinely have strong predictive signals")
            print(f"   💡 Recommendation: Proceed with caution, validate on holdout data")

        recommendations = {
            'to_remove': to_remove,
            'to_investigate': to_investigate,
            'to_keep': to_keep,
            'next_steps': []
        }

        if to_remove:
            recommendations['next_steps'].append("Create updated ML-ready dataset without flagged features")
            recommendations['next_steps'].append("Re-run model testing to validate performance drop")

        recommendations['next_steps'].append("Proceed with hyperparameter optimization")

        return recommendations

    def run_full_investigation(self):
        """Run complete website leakage investigation"""
        print("🔍 WEBSITE DATASET LEAKAGE INVESTIGATION")
        print("="*70)
        print("Investigating remaining suspicious features after points_in_wallet removal")
        print("Current concerning performance: Random Forest AUC = 0.9624")

        # Load data
        df = self.load_website_data()
        if df is None:
            return None

        # Quick single feature test
        performance_results, suspicious_features, notable_features = self.quick_single_feature_test(df)

        if not suspicious_features:
            print(f"\n✅ No highly suspicious features found!")
            print(f"High performance may be due to legitimate strong predictive signals")
            return {'conclusion': 'no_additional_leakage', 'recommendations': []}

        # Business logic evaluation
        assessments = self.business_logic_evaluation(suspicious_features)

        # Generate recommendations
        recommendations = self.generate_recommendations()

        print(f"\n🏁 INVESTIGATION COMPLETED")
        print(f"📄 Results stored for further analysis")

        return {
            'performance_results': performance_results,
            'suspicious_features': suspicious_features,
            'assessments': assessments,
            'recommendations': recommendations
        }


# Main execution
if __name__ == "__main__":
    print("🧪 Starting Website Dataset Leakage Investigation...")

    investigator = WebsiteLeakageInvestigator()
    results = investigator.run_full_investigation()

    if results:
        print(f"\n✅ Investigation completed successfully!")

        # Show quick summary
        if results.get('recommendations'):
            rec = results['recommendations']
            print(f"\n📊 QUICK SUMMARY:")
            print(f"   🚫 Features to remove: {len(rec.get('to_remove', []))}")
            print(f"   🔍 Features to investigate: {len(rec.get('to_investigate', []))}")
            print(f"   ✅ Features to keep: {len(rec.get('to_keep', []))}")
    else:
        print(f"❌ Investigation failed - check dataset availability")