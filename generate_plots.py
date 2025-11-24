"""
Generate visualization plots for the language classification dataset.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import PROCESSED_DATA_DIR, TARGET_LANGUAGES
from src.features.linguistic_features import LinguisticStatsTransformer

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# Load data
print("Loading data...")
data_path = PROCESSED_DATA_DIR / "wili_subset_train.csv"
df = pd.read_csv(data_path)

# Filter to target languages
df = df[df["language"].isin(TARGET_LANGUAGES)]
print(f"Loaded {len(df)} samples")

# Extract linguistic features
print("Extracting linguistic features...")
stats_transformer = LinguisticStatsTransformer()
features_array = stats_transformer.fit_transform(df["text"]).toarray()
feature_names = stats_transformer.feature_names_

# Create features dataframe
features_df = pd.DataFrame(features_array, columns=feature_names)
features_df["label"] = df["language"].values

# Create combined dataframe for plotting
df_plot = features_df.copy()

print(f"Features: {feature_names}")
print(f"Languages: {df_plot['label'].unique()}")

# 1. Feature heatmap (sample a subset for visualization)
print("\n1. Generating feature heatmap...")
plt.figure(figsize=(12, 8))
# Sample data for heatmap (too many rows otherwise)
sample_size = min(100, len(df_plot))
df_sample = df_plot.sample(n=sample_size, random_state=42).sort_values("label")
features_only = df_sample[feature_names].T
sns.heatmap(features_only, cmap="viridis", cbar_kws={"label": "Feature Value"})
plt.title(f"Feature Heatmap (Languages × Features) - Sample of {sample_size} samples")
plt.xlabel("Samples")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts" / "feature_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved to artifacts/feature_heatmap.png")

# 2. Feature correlation matrix
print("\n2. Generating feature correlation matrix...")
plt.figure(figsize=(10, 8))
corr_matrix = features_df[feature_names].corr()
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts" / "feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved to artifacts/feature_correlation.png")

# 3. Feature importance (using LinearSVC)
print("\n3. Generating feature importance plot...")
X = features_df[feature_names].values
y = features_df["label"].values

# Train linear SVM
model = LinearSVC(random_state=42, max_iter=10000)
model.fit(X, y)

# For multi-class, get average importance across classes
importance = np.abs(model.coef_).mean(axis=0)

indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance[indices])
plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.title("Feature Importance (Linear SVM Weights - Average Across Classes)")
plt.ylabel("Average |Weight|")
plt.xlabel("Features")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts" / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved to artifacts/feature_importance.png")

# 4. Boxplots for each feature
print("\n4. Generating boxplots...")
for i, col in enumerate(feature_names):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=features_df, x="label", y=col)
    plt.title(f"Distribution of {col} by Language")
    plt.xlabel("Language")
    plt.ylabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "artifacts" / f"boxplot_{col}.png", dpi=150, bbox_inches="tight")
    plt.close()
    if (i + 1) % 2 == 0:
        print(f"   Generated {i + 1}/{len(feature_names)} boxplots...")
print(f"   Saved {len(feature_names)} boxplots to artifacts/")

# 5. PCA plot
print("\n5. Generating PCA plot...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df[feature_names])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=features_df["label"], palette="tab10", s=50, alpha=0.6)
plt.title(f"PCA of Typological Features (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.legend(title="Language")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "artifacts" / "pca_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved to artifacts/pca_plot.png")

print("\nAll plots generated successfully!")
print(f"   Output directory: {PROJECT_ROOT / 'artifacts'}")

