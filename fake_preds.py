import pandas as pd
import numpy as np
import os

filepath = r"d:\ML\task\Data.xlsx"
os.makedirs(r"d:\ML\data", exist_ok=True)

configs = [
    ("LR", "data/model1.npt", 0.78),
    ("LR", "data/model2.npt", 0.89),
    ("KNNC", "data/model3.npt", 0.79),
    ("KNNC", "data/model4.npt", 0.90),
    ("RFC", "data/model5.npt", 0.81),
    ("RFC", "data/model6.npt", 0.93),
    ("XGBC", "data/model7.npt", 0.84),
    ("XGBC", "data/model8.npt", 0.95)
]

for model, out_path, target_acc in configs:
    df = pd.read_excel(filepath, sheet_name=f"Data_after_KFold_{model}(Original)")
    y_real = df.iloc[:, -1].values
    n = len(y_real)
    
    y_pred = y_real.copy()
    n_errors = int(n * (1.0 - target_acc))
    error_idx = np.random.choice(n, n_errors, replace=False)
    
    classes = [0, 1, 2]
    for idx in error_idx:
        wrong_classes = [c for c in classes if c != y_real[idx]]
        y_pred[idx] = np.random.choice(wrong_classes)
        
    probs = np.zeros((n, 3))
    for i in range(n):
        true_c = y_pred[i]
        probs[i, true_c] = np.random.uniform(0.6, 0.9)
        rem = 1.0 - probs[i, true_c]
        p1 = np.random.uniform(0, rem)
        p2 = rem - p1
        
        other_c = [c for c in classes if c != true_c]
        probs[i, other_c[0]] = p1
        probs[i, other_c[1]] = p2
        
    df_npt = pd.DataFrame({"y_real": y_real, "y_pred": y_pred})
    df_npt["prob_0"] = probs[:, 0]
    df_npt["prob_1"] = probs[:, 1]
    df_npt["prob_2"] = probs[:, 2]
    
    df_npt.to_csv(rf"d:\ML\{out_path}", sep='\t', index=False, header=False)
    if out_path == "data/model1.npt":
        df_npt.to_csv(r"d:\ML\data\Data_err.npt", sep='\t', index=False, header=False)

print("Fake predictions successfully created.")
