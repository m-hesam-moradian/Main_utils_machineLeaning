import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

file_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
xl = pd.ExcelFile(file_path)

if "Data_after_KFold_MLR(RFE)" in xl.sheet_names:
    sheet_name = "Data_after_KFold_MLR(RFE)"
elif "Data_after_KFold_MLR(ENN)" in xl.sheet_names:
    sheet_name = "Data_after_KFold_MLR(ENN)"
elif "Selected_Data_RFE" in xl.sheet_names:
    sheet_name = "Selected_Data_RFE"
elif "ENN_Data" in xl.sheet_names:
    sheet_name = "ENN_Data"
else:
    sheet_name = xl.sheet_names[0]


print(f"Loading data for SHAP sensitivity from sheet: '{sheet_name}'")
df = pd.read_excel(file_path, sheet_name=sheet_name)

target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLR model
model = LogisticRegression(solver='lbfgs', max_iter=500, C=1.0, random_state=42)
model.fit(X_train, y_train)

# Compute SHAP values using LinearExplainer or Explainer
print("Computing SHAP values...")
masker = shap.maskers.Independent(data=X_train)
explainer = shap.LinearExplainer(model, masker=masker)
shap_values = explainer(X_test)

# Handle multi-class shap values (n_samples, n_features, n_classes)
if len(shap_values.values.shape) == 3:
    shap_vals_matrix = shap_values.values
    classes = np.unique(y)
    
    all_class_metrics = []
    
    # Global feature importance across all classes
    mean_abs_all = np.mean(np.abs(shap_vals_matrix), axis=(0, 2))
    
    for c_idx, cls in enumerate(classes):
        shap_c = shap_vals_matrix[:, :, c_idx]
        for f_idx, feat in enumerate(X.columns):
            mean_abs = np.mean(np.abs(shap_c[:, f_idx]))
            max_s = np.max(shap_c[:, f_idx])
            min_s = np.min(shap_c[:, f_idx])
            f_vals = X_test[feat].values
            corr = np.corrcoef(f_vals, shap_c[:, f_idx])[0, 1] if np.std(f_vals) > 0 and np.std(shap_c[:, f_idx]) > 0 else 0.0
            
            all_class_metrics.append({
                "Class": f"Class {cls}",
                "Feature": feat,
                "Mean_Abs_SHAP": mean_abs,
                "Max_SHAP": max_s,
                "Min_SHAP": min_s,
                "Impact_Range": max_s - min_s,
                "Feature_Correlation": corr
            })
            
    shap_df = pd.DataFrame(all_class_metrics)
    
    # Overall summary
    overall_summary = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Abs_SHAP_Overall": mean_abs_all
    }).sort_values(by="Mean_Abs_SHAP_Overall", ascending=False).reset_index(drop=True)
else:
    mean_abs_all = np.mean(np.abs(shap_values.values), axis=0)
    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Abs_SHAP": mean_abs_all
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    overall_summary = shap_df

print("\n--- SHAP Feature Importance Summary ---")
print(overall_summary)

output_dir = r"C:\Users\Sam\Desktop\ML\task\SHAP_Plots"
os.makedirs(output_dir, exist_ok=True)

# Generate Summary Plot
try:
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Multi-Class Feature Summary", pad=20)
    plt.savefig(os.path.join(output_dir, "01_SHAP_Summary_Plot.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot to: {output_dir}")
except Exception as e:
    print(f"Note: Plot saving skipped: {e}")

# Save to Excel
with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    overall_summary.to_excel(writer, sheet_name="SHAP_Sensitivity", index=False)
    shap_df.to_excel(writer, sheet_name="SHAP_Details", index=False)

print(f"\n[+] Saved SHAP Sensitivity analysis to sheet 'SHAP_Sensitivity' and 'SHAP_Details' in {file_path}")