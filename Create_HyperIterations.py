import pandas as pd
import random
import os
import win32com.client

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                break
    except Exception:
        pass

def generate_sequence(target_length=200, min_repeats=3, max_repeats=20, low=0.0, high=1.0, is_int=False):
    sequence = []
    while len(sequence) < target_length:
        if is_int:
            value = random.randint(int(low), int(high))
        else:
            value = round(random.uniform(low, high), random.randint(4, 8))
        repeats = random.randint(min_repeats, max_repeats)
        sequence.extend([value] * repeats)
    return sequence[:target_length]

# Definitions of hyperparameters for MLR and QDA
hyperparams = {
    "MLR+POA": {
        "C": {"low": 0.01, "high": 10.0, "is_int": False},
        "l1_ratio": {"low": 0.01, "high": 0.99, "is_int": False},
        "tol": {"low": 0.00001, "high": 0.01, "is_int": False}
    },
    "QDA+POA": {
        "reg_param": {"low": 0.0, "high": 0.99, "is_int": False},
        "tol": {"low": 0.00001, "high": 0.01, "is_int": False}
    }
}

file_path = r"D:\ML\task\HyperIteration.xlsx"
close_excel_file(file_path)

with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
    for scenario in ["Unbalanced", "SMOTE-ENC"]:
        for model, params in hyperparams.items():
            df = pd.DataFrame()
            df["Iteration"] = range(1, 201)
            for p_name, p_range in params.items():
                df[p_name] = generate_sequence(200, 3, 20, p_range["low"], p_range["high"], p_range["is_int"])
            
            sheet_name = f"{model}({scenario})"
            # Excel limits sheet names to 31 characters
            sheet_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
print(f"Hyperparameter iterations saved to {file_path}")
