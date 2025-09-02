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
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10

class DataLeakageDetector:
    def __init__(self, output_dir="results/leakage_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def load_and_clean_data(self):
        """Load datasets and perform basic cleaning"""
        print("📂 Datasetler yükleniyor ve temizleniyor...")

        datasets = {}

        # Internet dataset
        try:
            internet_df = pd.read_csv("data/raw/internet_service_churn.csv")
            # Basic cleaning
            internet_df = internet_df.dropna()
            # Convert categorical columns to numerical
            for col in internet_df.select_dtypes(include=['object']).columns:
                if col != 'churn':  # if not target column
                    internet_df[col] = pd.Categorical(internet_df[col]).codes
            # Convert churn column to 0/1
            if 'churn' in internet_df.columns:
                internet_df['churn'] = (internet_df['churn'] == 'Yes').astype(int)

            datasets['internet'] = (internet_df, 'churn')
            print(f"✅ Internet: {internet_df.shape}")
        except Exception as e:
            print(f"❌ Internet dataset could not be loaded: {e}")

        # Banking dataset  
        try:
            banking_df = pd.read_csv("data/raw/Bank Customer Churn Prediction.csv")
            # Basic cleaning
            banking_df = banking_df.dropna()
            # Convert categorical columns to numerical
            for col in banking_df.select_dtypes(include=['object']).columns:
                if col not in ['churn', 'Exited']:  # if not target column
                    banking_df[col] = pd.Categorical(banking_df[col]).codes
            # Find target column
            target_col = 'churn' if 'churn' in banking_df.columns else 'Exited'
            if target_col in banking_df.columns:
                banking_df['churn'] = banking_df[target_col].astype(int)

            datasets['banking'] = (banking_df, 'churn')
            print(f"✅ Banking: {banking_df.shape}")
        except Exception as e:
            print(f"❌ Banking dataset could not be loaded: {e}")

        # Website dataset
        try:
            website_df = pd.read_csv("data/raw/website.csv")
            # Basic cleaning
            website_df = website_df.dropna()
            # Convert categorical columns to numerical
            for col in website_df.select_dtypes(include=['object']).columns:
                if col != 'churn_risk_score':  # if not target column
                    website_df[col] = pd.Categorical(website_df[col]).codes

            # Find target column and convert to numerical
            target_candidates = ['churn_risk_score', 'churn', 'target']
            target_col = None
            for candidate in target_candidates:
                if candidate in website_df.columns:
                    target_col = candidate
                    break

            if target_col:
                # Eğer sayısal değilse kategorik yap
                if website_df[target_col].dtype == 'object':
                    website_df[target_col] = pd.Categorical(website_df[target_col]).codes
                datasets['website'] = (website_df, target_col)
                print(f"✅ Website: {website_df.shape}, Target: {target_col}")
            else:
                print("❌ Website dataset'inde target kolonu bulunamadı")

        except Exception as e:
            print(f"❌ Website dataset could not be loaded: {e}")

        return datasets

    def plot_correlation_matrix(self, df, target_col, dataset_name):
        """Korelasyon matrisini görselleştir"""
        print(f"\n📊 {dataset_name} - Korelasyon Matrisi Oluşturuluyor...")

        # Sadece sayısal kolonları al
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        # Target ile korelasyonları ayrı göster
        target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)

        plt.figure(figsize=(15, 12))

        # Ana korelasyon matrisi
        plt.subplot(2, 2, 1)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r',
                    center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title(f'{dataset_name} - Korelasyon Matrisi')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Target korelasyonları bar plot
        plt.subplot(2, 2, 2)
        top_corrs = target_corrs[1:16]  # Target hariç top 15
        colors = ['red' if abs(x) > 0.7 else 'orange' if abs(x) > 0.5 else 'blue' for x in top_corrs]
        top_corrs.plot(kind='barh', color=colors)
        plt.title(f'{dataset_name} - Target Korelasyonları (Top 15)')
        plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Yüksek Risk (>0.7)')
        plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Orta Risk (>0.5)')
        plt.legend()
        plt.xlabel('Mutlak Korelasyon')

        # Yüksek korelasyonlu çiftler
        plt.subplot(2, 2, 3)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Yüksek korelasyon
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

        if high_corr_pairs:
            pair_df = pd.DataFrame(high_corr_pairs, columns=['Feature1', 'Feature2', 'Correlation'])
            pair_df['abs_corr'] = pair_df['Correlation'].abs()
            pair_df = pair_df.sort_values('abs_corr', ascending=True).tail(10)

            colors = ['red' if abs(x) > 0.9 else 'orange' for x in pair_df['Correlation']]
            plt.barh(range(len(pair_df)), pair_df['Correlation'], color=colors)
            plt.yticks(range(len(pair_df)), [f"{row['Feature1'][:15]}\nvs\n{row['Feature2'][:15]}"
                                            for _, row in pair_df.iterrows()])
            plt.title(f'{dataset_name} - Yüksek Korelasyonlu Çiftler')
            plt.xlabel('Korelasyon')
        else:
            plt.text(0.5, 0.5, 'Yüksek korelasyonlu\nçift bulunamadı',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{dataset_name} - Yüksek Korelasyonlu Çiftler')

        # Korelasyon dağılımı
        plt.subplot(2, 2, 4)
        all_corrs = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        plt.hist(all_corrs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.7, color='red', linestyle='--', label='Risk Sınırı (0.7)')
        plt.axvline(x=-0.7, color='red', linestyle='--')
        plt.title(f'{dataset_name} - Korelasyon Dağılımı')
        plt.xlabel('Korelasyon Değeri')
        plt.ylabel('Frekans')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name}_correlation_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        return corr_matrix, target_corrs

    def single_feature_performance(self, df, target_col, top_n=15):
        """Her feature'ı tek başına test et"""
        print(f"\n🧪 Tek Feature Performans Testi (Top {top_n})...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        performance_results = []
        X = df[numeric_cols].fillna(0)
        y = df[target_col]

        for col in numeric_cols:
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

                if auc_score > 0.9:
                    print(f"🚨 ALARM: {col} -> AUC: {auc_score:.4f}")
                elif auc_score > 0.8:
                    print(f"⚠️  YÜKSEK: {col} -> AUC: {auc_score:.4f}")

            except Exception as e:
                continue

        performance_results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n📊 En İyi {top_n} Feature:")
        for i, (col, auc) in enumerate(performance_results[:top_n]):
            status = "🚨" if auc > 0.9 else "⚠️" if auc > 0.8 else "✅"
            print(f"{i+1:2d}. {status} {col:<35} AUC: {auc:.4f}")

        return performance_results

    def comprehensive_leakage_analysis(self, df, target_col, dataset_name):
        """Kapsamlı veri sızıntısı analizi"""
        print("="*70)
        print(f"🔍 {dataset_name.upper()} VERİ SIZINTISI ANALİZİ")
        print("="*70)

        print(f"📊 Dataset boyutu: {df.shape}")
        print(f"🎯 Target kolonu: {target_col}")

        # Target dağılımı
        target_counts = df[target_col].value_counts()
        print(f"\n📈 Target Dağılımı:")
        for value, count in target_counts.items():
            ratio = count / len(df)
            print(f"  {value}: {count} ({ratio:.1%})")

        # 1. Korelasyon matrisi
        corr_matrix, target_corrs = self.plot_correlation_matrix(df, target_col, dataset_name)

        # 2. Tek feature performansı
        single_performance = self.single_feature_performance(df, target_col)

        # 3. Rapor
        perfect_corrs = [(col, corr) for col, corr in zip(target_corrs.index[1:], target_corrs.values[1:]) if abs(corr) > 0.95]
        super_features = [f for f, auc in single_performance if auc > 0.9]

        print(f"\n{'='*70}")
        print(f"📋 {dataset_name.upper()} ÖZET RAPOR")
        print(f"{'='*70}")

        risk_level = "DÜŞÜK"
        warnings = 0

        if len(perfect_corrs) > 0:
            risk_level = "ÇOK YÜKSEK"
            warnings += len(perfect_corrs)
            print(f"🚨 {len(perfect_corrs)} adet mükemmel korelasyon!")
            for feat, corr in perfect_corrs:
                print(f"   - {feat}: {corr:.4f}")

        if len(super_features) > 0:
            if risk_level != "ÇOK YÜKSEK":
                risk_level = "YÜKSEK"
            warnings += len(super_features)
            print(f"🚨 {len(super_features)} adet süper feature!")
            for feat in super_features:
                print(f"   - {feat}")

        top_3_auc = np.mean([auc for _, auc in single_performance[:3]])
        if top_3_auc > 0.95:
            risk_level = "ÇOK YÜKSEK"
            print(f"🚨 Top 3 AUC: {top_3_auc:.3f} - Çok yüksek!")

        print(f"\n🎯 RİSK SEVİYESİ: {risk_level}")
        print(f"📊 Toplam Uyarı: {warnings}")

        if risk_level in ["YÜKSEK", "ÇOK YÜKSEK"]:
            print(f"\n💡 ÖNERİLER:")
            print("1. Şüpheli feature'ları kaldır")
            print("2. Feature engineering sürecini gözden geçir")
            print("3. Cross-validation stratejisini kontrol et")

        return {
            'correlation_matrix': corr_matrix,
            'target_correlations': target_corrs,
            'single_performance': single_performance,
            'perfect_correlations': perfect_corrs,
            'super_features': super_features,
            'risk_level': risk_level
        }

# Kullanım
if __name__ == "__main__":
    detector = DataLeakageDetector()

    # Temizlenmiş dataları direkt yükle
    print("📂 Temizlenmiş datasetler yükleniyor...")

    datasets = {}

    try:
        internet_df = pd.read_csv("data/processed/internet_processed.csv")
        datasets['internet'] = (internet_df, 'churn')
        print(f"✅ Internet: {internet_df.shape}")
    except Exception as e:
        print(f"❌ Internet dataset could not be loaded: {e}")

    try:
        banking_df = pd.read_csv("data/processed/banking_processed.csv")
        datasets['banking'] = (banking_df, 'churn')
        print(f"✅ Banking: {banking_df.shape}")
    except Exception as e:
        print(f"❌ Banking dataset could not be loaded: {e}")

    try:
        website_df = pd.read_csv("data/processed/website_processed.csv")
        datasets['website'] = (website_df, 'churn')
        print(f"✅ Website: {website_df.shape}")
    except Exception as e:
        print(f"❌ Website dataset could not be loaded: {e}")

    if not datasets:
        print("❌ Hiç dataset could not be loaded!")
        exit()

    print(f"🎯 Toplam {len(datasets)} dataset yüklendi.")

    # Her dataset için analiz
    all_results = {}
    for dataset_name, (df, target_col) in datasets.items():
        print(f"\n🚀 {dataset_name.upper()} analizi başlatılıyor...")
        results = detector.comprehensive_leakage_analysis(df, target_col, dataset_name)
        all_results[dataset_name] = results

    print("\n" + "="*80)
    print("🏁 TÜM ANALİZLER TAMAMLANDI!")
    print("="*80)

    # Karşılaştırmalı özet
    print("\n📊 DATASET KARŞILAŞTIRMASI:")
    for name, results in all_results.items():
        risk = results['risk_level']
        emoji = "🚨" if risk == "ÇOK YÜKSEK" else "⚠️" if risk == "YÜKSEK" else "✅"
        perfect_count = len(results['perfect_correlations'])
        super_count = len(results['super_features'])
        top_3_auc = np.mean([auc for _, auc in results['single_performance'][:3]]) if results['single_performance'] else 0
        print(f"{emoji} {name.upper():<10} Risk: {risk:<12} Perfect: {perfect_count}  Super: {super_count}  Top3 AUC: {top_3_auc:.3f}")