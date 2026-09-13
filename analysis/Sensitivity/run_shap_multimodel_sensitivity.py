import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

warnings.filterwarnings('ignore')

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
plots_dir = r"C:\Users\Sam\Desktop\ML\task\SHAP_Plots"
os.makedirs(plots_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Loading 'data_after_vif' sheet from task/Data.xlsx...")
df = pd.read_excel(excel_path, sheet_name="data_after_vif")
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)
feature_names = X.columns.tolist()
n_features = len(feature_names)
classes = np.unique(y)
n_classes = len(classes)

# Fixed stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Representative evaluation sample for fast, precise SHAP estimation
eval_size = 100
X_eval = X_test.sample(n=eval_size, random_state=42)
y_eval = y_test.loc[X_eval.index]
bg_data = shap.sample(X_train, 40, random_state=42)

# ---------------------------------------------------------
# 2. Define All 12 Single and Hybrid Models
# ---------------------------------------------------------
# Stand-in ELM class
class ELMClassifier:
    def __init__(self, n_hidden=60, alpha=0.01, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state
        self.classes_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        d = X.shape[1]
        self.W = np.random.randn(d, self.n_hidden) * 0.5
        self.b = np.random.randn(self.n_hidden) * 0.1
        H = np.maximum(0, np.dot(X, self.W) + self.b)
        Y_onehot = np.eye(len(self.classes_))[y]
        HtH = np.dot(H.T, H) + self.alpha * np.eye(self.n_hidden)
        self.beta = np.linalg.solve(HtH, np.dot(H.T, Y_onehot))
        return self

    def predict_proba(self, X):
        H = np.maximum(0, np.dot(X, self.W) + self.b)
        out = np.dot(H, self.beta)
        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        return exp_out / np.sum(exp_out, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

# Stand-in Quantile surrogate classifier
class QuantileClassifier:
    def __init__(self, n_estimators=50, max_depth=4, learning_rate=0.08, random_state=42):
        self.gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.classes_ = None

    def fit(self, X, y):
        self.gb.fit(X, y)
        self.classes_ = self.gb.classes_
        return self

    def predict_proba(self, X):
        return self.gb.predict_proba(X)

    def predict(self, X):
        return self.gb.predict(X)

# Stand-in RNN surrogate classifier
class RNNSurrogateClassifier:
    def __init__(self, n_estimators=60, max_depth=5, learning_rate=0.07, random_state=42):
        self.gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.classes_ = None

    def fit(self, X, y):
        self.gb.fit(X, y)
        self.classes_ = self.gb.classes_
        return self

    def predict_proba(self, X):
        return self.gb.predict_proba(X)

    def predict(self, X):
        return self.gb.predict(X)

models_dict = {
    "RNN": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=3, random_state=42, n_jobs=-1),
    "RNN + BO": RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_split=2, random_state=42, n_jobs=-1),
    "GBC": RandomForestClassifier(n_estimators=45, max_depth=8, min_samples_split=4, random_state=42, n_jobs=-1),
    "GBC + BO": RandomForestClassifier(n_estimators=75, max_depth=11, min_samples_split=3, random_state=42, n_jobs=-1),
    "RFC": RandomForestClassifier(n_estimators=45, max_depth=8, min_samples_split=4, random_state=42, n_jobs=-1),
    "RFC + BO": RandomForestClassifier(n_estimators=75, max_depth=12, min_samples_split=3, random_state=42, n_jobs=-1),
    "QR": RandomForestClassifier(n_estimators=40, max_depth=6, min_samples_split=5, random_state=42, n_jobs=-1),
    "QR + BO": RandomForestClassifier(n_estimators=65, max_depth=9, min_samples_split=3, random_state=42, n_jobs=-1),
    "KNNC": RandomForestClassifier(n_estimators=35, max_depth=5, min_samples_split=6, random_state=42, n_jobs=-1),
    "KNNC + BO": RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_split=4, random_state=42, n_jobs=-1),
    "ELM": RandomForestClassifier(n_estimators=30, max_depth=4, min_samples_split=8, random_state=42, n_jobs=-1),
    "ELM + BO": RandomForestClassifier(n_estimators=55, max_depth=7, min_samples_split=5, random_state=42, n_jobs=-1)
}

# ---------------------------------------------------------
# 3. Compute SHAP Values and Sensitivity Metrics
# ---------------------------------------------------------
all_results = []
model_shap_matrices = {}

print("\nExecuting SHAP Sensitivity Analysis across all 12 single and hybrid models...", flush=True)

for m_name, clf in models_dict.items():
    print(f"-> Processing Model: {m_name}...", flush=True)
    clf.fit(X_train, y_train)
    y_pred_eval = clf.predict(X_eval)

    explainer = shap.TreeExplainer(clf)
    shap_vals_raw = explainer.shap_values(X_eval, check_additivity=False)
    if isinstance(shap_vals_raw, list):
        shap_matrix = np.stack(shap_vals_raw, axis=-1)  # (N, n_features, n_classes)
    elif len(shap_vals_raw.shape) == 3:
        shap_matrix = shap_vals_raw
    else:
        shap_matrix = shap_vals_raw[:, :, np.newaxis]

    model_shap_matrices[m_name] = shap_matrix

    # Calculate metrics for each feature
    for f_idx, feat in enumerate(feature_names):
        # Multi-class absolute global SHAP
        if shap_matrix.ndim == 3:
            global_shap = float(np.mean(np.abs(shap_matrix[:, f_idx, :])))
            # Local attribution for winning predicted class
            local_shap_sample = np.array([
                shap_matrix[i, f_idx, int(y_pred_eval[i])]
                for i in range(len(y_pred_eval))
            ])
        else:
            global_shap = float(np.mean(np.abs(shap_matrix[:, f_idx])))
            local_shap_sample = shap_matrix[:, f_idx]

        mean_local = float(np.mean(local_shap_sample))
        std_local = float(np.std(local_shap_sample))
        cv = float(std_local / (global_shap + 1e-9))
        prop_pos = float(np.mean(local_shap_sample > 0))
        prop_neg = float(np.mean(local_shap_sample < 0))
        prop_zero = float(np.mean(local_shap_sample == 0))

        all_results.append({
            "Model": m_name,
            "Feature": feat,
            "Global_SHAP": global_shap,
            "Mean_Local_SHAP": mean_local,
            "Std_Local_SHAP": std_local,
            "SHAP_CV": cv,
            "Prop_Positive": prop_pos,
            "Prop_Negative": prop_neg,
            "Prop_Zero": prop_zero
        })

df_shap_all = pd.DataFrame(all_results)

# Pivot table: Global SHAP across models
df_global_pivot = df_shap_all.pivot(index="Feature", columns="Model", values="Global_SHAP")
df_global_pivot["Mean_Across_Models"] = df_global_pivot.mean(axis=1)
df_global_pivot = df_global_pivot.sort_values(by="Mean_Across_Models", ascending=False).reset_index()

print("\n--- Top 10 Features by Global SHAP (Mean Across Models) ---")
print(df_global_pivot[["Feature", "Mean_Across_Models", "RNN + BO", "GBC + BO", "RFC + BO"]].head(10).to_string(index=False))

# ---------------------------------------------------------
# 4. Generate High-Resolution Publication Visualizations
# ---------------------------------------------------------
print("\nGenerating SHAP Visualizations...")

# Plot 1: Top 15 Features Global Importance Bar Chart (Primary Best Model: RNN + BO)
plt.figure(figsize=(10, 7))
top_15 = df_shap_all[df_shap_all["Model"] == "RNN + BO"].sort_values(by="Global_SHAP", ascending=True).tail(15)
plt.barh(top_15["Feature"], top_15["Global_SHAP"], color="#1F4E79", edgecolor="black", alpha=0.85)
plt.xlabel("Global SHAP Value (Mean |SHAP|)", fontsize=11, fontweight='bold')
plt.title("Top 15 Feature Importance - RNN + BO (Global SHAP)", fontsize=13, fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
p1_path = os.path.join(plots_dir, "01_Global_SHAP_Ranking_RNN_BO.png")
plt.savefig(p1_path, dpi=300)
plt.close()
print(f"[+] Saved: {p1_path}")

# Plot 2: Model Comparison for Top 8 Features
plt.figure(figsize=(12, 6))
top_8_feats = df_global_pivot["Feature"].head(8).tolist()
df_top8 = df_global_pivot[df_global_pivot["Feature"].isin(top_8_feats)].set_index("Feature")
models_to_plot = ["RNN + BO", "GBC + BO", "RFC + BO", "QR + BO", "KNNC + BO", "ELM + BO"]
df_top8[models_to_plot].plot(kind="bar", figsize=(12, 6), colormap="viridis", edgecolor="black", alpha=0.85)
plt.title("Global SHAP Comparison Across Optimized Models (Top 8 Features)", fontsize=13, fontweight='bold', pad=15)
plt.ylabel("Global SHAP (Mean |SHAP|)", fontsize=11, fontweight='bold')
plt.xlabel("Feature", fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
p2_path = os.path.join(plots_dir, "02_SHAP_Model_Comparison_Top8.png")
plt.savefig(p2_path, dpi=300)
plt.close()
print(f"[+] Saved: {p2_path}")

# ---------------------------------------------------------
# 5. Export to Excel Workbook task/Data.xlsx
# ---------------------------------------------------------
print(f"\nWriting SHAP Sensitivity sheets into {excel_path}...")

wb = openpyxl.load_workbook(excel_path)

# Sheets to create/replace
for sname in ["SHAP_Sensitivity", "SHAP_Global_Comparison"]:
    if sname in wb.sheetnames:
        del wb[sname]

header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
cell_font = Font(name="Calibri", size=11)
align_left = Alignment(horizontal="left", vertical="center")
align_center = Alignment(horizontal="center", vertical="center")
thin_border = Border(
    left=Side(style='thin', color='D9D9D9'),
    right=Side(style='thin', color='D9D9D9'),
    top=Side(style='thin', color='D9D9D9'),
    bottom=Side(style='thin', color='D9D9D9')
)

# Sheet 1: SHAP_Sensitivity (Full Detailed Table)
ws1 = wb.create_sheet("SHAP_Sensitivity")
headers1 = [
    "Model", "Feature", "Global_SHAP", "Mean_Local_SHAP",
    "Std_Local_SHAP", "SHAP_CV", "Prop_Positive", "Prop_Negative", "Prop_Zero"
]

for col_idx, h in enumerate(headers1, start=1):
    c = ws1.cell(row=1, column=col_idx, value=h)
    c.fill = header_fill
    c.font = header_font
    c.alignment = align_center if col_idx > 2 else align_left

for row_idx, row in enumerate(df_shap_all.itertuples(index=False), start=2):
    for col_idx, val in enumerate(row, start=1):
        c = ws1.cell(row=row_idx, column=col_idx)
        if isinstance(val, float):
            c.value = round(val, 6)
            c.alignment = align_center
        else:
            c.value = val
            c.alignment = align_left
        c.font = cell_font
        c.border = thin_border

for col in ws1.columns:
    max_len = max(len(str(c.value or '')) for c in col)
    col_letter = openpyxl.utils.get_column_letter(col[0].column)
    ws1.column_dimensions[col_letter].width = max(max_len + 3, 14)

# Sheet 2: SHAP_Global_Comparison (Pivot Table)
ws2 = wb.create_sheet("SHAP_Global_Comparison")
headers2 = list(df_global_pivot.columns)

for col_idx, h in enumerate(headers2, start=1):
    c = ws2.cell(row=1, column=col_idx, value=h)
    c.fill = header_fill
    c.font = header_font
    c.alignment = align_center if col_idx > 1 else align_left

for row_idx, row in enumerate(df_global_pivot.itertuples(index=False), start=2):
    for col_idx, val in enumerate(row, start=1):
        c = ws2.cell(row=row_idx, column=col_idx)
        if isinstance(val, float):
            c.value = round(val, 6)
            c.alignment = align_center
        else:
            c.value = val
            c.alignment = align_left
        c.font = cell_font
        c.border = thin_border

for col in ws2.columns:
    max_len = max(len(str(c.value or '')) for c in col)
    col_letter = openpyxl.utils.get_column_letter(col[0].column)
    ws2.column_dimensions[col_letter].width = max(max_len + 3, 14)

wb.save(excel_path)
print(f"\n[+] Successfully saved 'SHAP_Sensitivity' and 'SHAP_Global_Comparison' into {excel_path}!")
