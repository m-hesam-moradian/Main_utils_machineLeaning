import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

# 1. Load structured data from Excel
input_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
output_path = r"C:\Users\Sam\Desktop\ML\task\ROC_Balanced_20Rows_Results.xlsx"

df = pd.read_excel(
    input_path,
    header=0,
    sheet_name="predicts(IQR)",
)

columns = df.columns.tolist()
all_roc_rows = []
N_ROWS = 20 

# 2. Loop through every pair of columns (Model Real vs Predict)
for i in range(0, len(columns), 2):
    model_name = columns[i].strip()
    
    temp_df = df.iloc[:, [i, i+1]].dropna()
    y_real = temp_df.iloc[:, 0].values
    y_predict_hard = temp_df.iloc[:, 1].values  
    
    classes = np.unique(y_real)
    
    # 3. Calculate ROC for each class
    for cls in classes:
        # Create binary target (1 if actual is this class, 0 if not)
        y_true_bin = (y_real == cls).astype(int)
        
        # --- THE FIX ---
        # Create a binary prediction (1 if model PREDICTED this class, 0 if not)
        # This prevents Python from getting confused by the "0", "1", "2" labels
        y_score_bin = (y_predict_hard == cls).astype(int)
        
        try:
            # Calculate ROC using the balanced binary scores
            fpr, tpr, thr = roc_curve(y_true_bin, y_score_bin)
            roc_auc = auc(fpr, tpr)
            
            # Interpolation to stretch the curve into 20 rows
            if len(thr) > 1 and np.isinf(thr[0]):
                thr[0] = thr[1] + 1.0 
            
            fpr_20 = np.linspace(0, 1, N_ROWS)
            tpr_20 = np.interp(fpr_20, fpr, tpr)
            thr_20 = np.interp(fpr_20, fpr, thr)
            
            # 4. Format row-by-row
            for j in range(N_ROWS):
                all_roc_rows.append({
                    "Model": model_name,
                    "Class": cls,
                    "FPR": fpr_20[j],
                    "TPR": tpr_20[j],
                    "Threshold": thr_20[j],
                    "AUC": roc_auc if j == N_ROWS - 1 else "" 
                })
        except Exception as e:
            print(f"⚠️ Could not calculate ROC for {model_name} Class {cls}: {e}")

# 5. Convert to DataFrame and Save
if all_roc_rows:
    roc_df = pd.DataFrame(all_roc_rows)
    
    roc_df.to_excel(output_path, index=False)
    print(f"✅ SUCCESS: Balanced 20-Row ROC table saved to -> {output_path}\n")
    
    try:
        roc_df.to_clipboard(index=False)
        print("📋 Table copied to clipboard! You can paste it directly into Excel.\n")
    except Exception:
        pass
        
    print("--- Preview of Output (Notice how AUC is now balanced!) ---")
    print(roc_df.head(20).to_string(index=False))
else:
    print("No data was processed.")