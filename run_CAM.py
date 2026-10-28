import numpy as np
import pandas as pd
import os
import win32com.client

def CAM_function(X, predictions):
    X = np.array(X)
    predictions = np.array(predictions).flatten()
    
    min_len = min(X.shape[0], len(predictions))
    X = X[:min_len, :]
    y = predictions[:min_len]
    
    D = X.shape[1]
    
    y_std = np.std(y)
    if y_std == 0:
        return [0.0] * D
    y_norm = (y - np.mean(y)) / y_std
    
    cam_indices = []
    
    for j in range(D):
        x_std = np.std(X[:, j])
        if x_std == 0:
            cam_indices.append(0.0)
            continue
        xj = (X[:, j] - np.mean(X[:, j])) / x_std
        
        numerator = np.sum(y_norm * np.cos(np.pi * xj))
        denominator = np.sum(np.cos(np.pi * xj) ** 2)
        
        if denominator == 0 or np.sum(y_norm ** 2) == 0:
            cam_indices.append(0.0)
        else:
            S_cam = (numerator ** 2) / (denominator * np.sum(y_norm ** 2))
            cam_indices.append(S_cam)
            
    return cam_indices

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

excel_path = r"D:\ML\task\Data.xlsx"
close_excel_file(excel_path)
xl = pd.ExcelFile(excel_path)

for tag in ["Unbalanced", "SMOTE-ENC"]:
    if f"predicts({tag})" not in xl.sheet_names:
        continue
    
    # Load predictions
    df_preds = pd.read_excel(xl, sheet_name=f"predicts({tag})")
    
    # Load features (use QDA KFold data as the canonical reordered data)
    df_data = pd.read_excel(xl, sheet_name=f"Data_after_KFold_QDA({tag})")
    target_col = df_data.columns[-1]
    X_df = df_data.drop(columns=[target_col])
    
    feature_names = X_df.columns.tolist()
    
    results = {"Feature": feature_names}
    
    cols = df_preds.columns.tolist()
    # cols format: y_real, y_pred, y_real, y_pred...
    for i in range(0, len(cols), 2):
        if i + 1 < len(cols):
            model_name = str(cols[i]).strip()
            y_pred = df_preds.iloc[:, i + 1].dropna().tolist()
            cam_vals = CAM_function(X_df, y_pred)
            results[model_name] = cam_vals
            
    df_res = pd.DataFrame(results)
    
    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df_res.to_excel(writer, sheet_name=f"CAM_Sensitivity({tag})", index=False)
        print(f"[+] Saved CAM sensitivity for {tag} to CAM_Sensitivity({tag})")
