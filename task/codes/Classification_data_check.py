import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load dataset ---
excel_path = r"D:\ML\ML\task\BSE. No.13-Dataset.xlsx"
sheet_name = "Balanced_Shuffled"
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Target column ---
target_column = "Cyberattack_Detected"

# --- Class distribution ---
class_counts = df[target_column].value_counts()
total_samples = class_counts.sum()
class_percentages = class_counts / total_samples * 100

print("\n📊 Sample count per class:")
print(class_counts)
print("\n📊 Class percentages:")
print(class_percentages.round(2))

# --- Red flag: class imbalance
max_class_pct = class_percentages.max()
if max_class_pct > 80:
    print(
        f"\n🚨 Red flag: Class imbalance detected. One class makes up {max_class_pct:.2f}% of samples."
    )
else:
    print("\n✅ Class distribution looks balanced.")

# --- Feature selection ---
features_df = df.drop(columns=[target_column])
numeric_features = features_df.select_dtypes(include=[np.number])

# --- Variance check ---
variances = numeric_features.var()
low_variance = variances[variances < 1e-3]

print("\n📉 Feature variances:")
print(variances.round(4))

# --- Red flag: low variance
if not low_variance.empty:
    print("\n⚠️ Red flag: Low-variance features detected (may be uninformative):")
    print(low_variance.round(4))
else:
    print("\n✅ All features have acceptable variance.")

# --- Correlation matrix ---
correlation_matrix = numeric_features.corr()

print("\n🔗 Correlation matrix:")
print(correlation_matrix.round(2))


# --- Red flag: high correlation
def find_high_correlation(corr_matrix, threshold=0.9):
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Feature1", "Feature2", "Correlation"]
    return corr_pairs[(corr_pairs["Correlation"].abs() > threshold)].sort_values(
        by="Correlation", ascending=False
    )


high_corr = find_high_correlation(correlation_matrix)

if not high_corr.empty:
    print("\n🚨 Red flag: Highly correlated feature pairs (|corr| > 0.9):")
    print(high_corr)
else:
    print("\n✅ No multicollinearity detected among features.")

# --- Optional: visualize correlation heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
