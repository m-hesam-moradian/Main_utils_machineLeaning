import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Paths (Output is now an Excel file instead of CSV)
# ---------------------------------------------------------
input_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
output_path = r"C:\Users\Sam\Desktop\ML\task\Test_Phase_Results.xlsx" # Changed to .xlsx

df = pd.read_excel(
    input_path,
    header=0,
    sheet_name="predicts",
)

columns = df.columns.tolist()
metrics_list = []
cm_display_data = [] # Stores formatting for the Excel tables

print("="*60)
print(" EXTRACTING TEST PHASE METRICS (80% - 90% of Data) ")
print("="*60)

# ---------------------------------------------------------
# 2. Iterate and Calculate
# ---------------------------------------------------------
for i in range(0, len(columns), 2):
    model_name = columns[i].strip()
    
    y_real = df.iloc[:, i].tolist()
    y_predict = df.iloc[:, i + 1].tolist()
    
    N = len(y_real)
    start_idx = int(N * 0.8)
    end_idx = int(N * 0.9)
    
    y_real_test = y_real[start_idx:end_idx]
    y_pred_test = y_predict[start_idx:end_idx]
    
    # Unique classes (0, 1, 2, etc.)
    unique_classes = sorted(list(set(y_real_test + y_pred_test)))
    
    # 1. Calculate Confusion Matrix
    cm = confusion_matrix(y_real_test, y_pred_test, labels=unique_classes)
    
    # 2. Calculate ROC-AUC Macro
    lb = LabelBinarizer()
    lb.fit(unique_classes)
    
    y_real_bin = lb.transform(y_real_test)
    y_pred_bin = lb.transform(y_pred_test)
    
    try:
        if len(unique_classes) > 2:
            roc_auc = roc_auc_score(y_real_bin, y_pred_bin, average='macro', multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_real_bin.ravel(), y_pred_bin.ravel(), average='macro')
    except ValueError:
        roc_auc = np.nan 
        
    print(f"Processed: {model_name: <12} | ROC-AUC: {roc_auc:.4f}")
    
    # Store ROC-AUC
    metrics_list.append({
        "Model Name": model_name,
        "ROC-AUC (Macro)": roc_auc
    })

    # -----------------------------------------------------
    # Build Beautiful Confusion Matrix for the Excel Sheet
    # -----------------------------------------------------
    # Title row
    cm_display_data.append([f"Model: {model_name}"] + [""] * len(unique_classes))
    
    # Column Headers (Predicted)
    headers = [""] + [f"Pred {cls}" for cls in unique_classes]
    cm_display_data.append(headers)
    
    # Rows (Actual)
    for r_idx, row in enumerate(cm):
        row_data = [f"Actual {unique_classes[r_idx]}"] + list(row)
        cm_display_data.append(row_data)
        
    # Blank row for spacing between models
    cm_display_data.append([]) 

# ---------------------------------------------------------
# 3. Save to Multi-Sheet Excel File
# ---------------------------------------------------------
# Note: You may need to run `pip install xlsxwriter` if it's not installed
df_metrics = pd.DataFrame(metrics_list)
df_cm = pd.DataFrame(cm_display_data)

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Save the standard metrics table
    df_metrics.to_excel(writer, sheet_name="ROC_AUC_Metrics", index=False)
    
    # Save the nicely formatted confusion matrices
    df_cm.to_excel(writer, sheet_name="Confusion_Matrices", index=False, header=False)

print("\n" + "="*60)
print(f" SUCCESS! File saved to: \n {output_path}")
print(" Open the file and look at the TWO DIFFERENT TABS at the bottom!")
print("="*60)