import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# --- 1. Excel Helpers (Kept Simple) ---
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                break
    except: pass

def open_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
    except: pass

# --- 2. Configuration ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
source_sheet = "Encoded_Data"
output_sheet = "DATA_Noisy_Light"

close_excel_file(excel_path)
df = pd.read_excel(excel_path, sheet_name=source_sheet)

target_col = df.columns[-1]
features = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()

# --- 3. Slight Noise Injection (In-place) ---
noise_level = 0.01  # 1% variance is very safe for normalized data
for col in features:
    noise = np.random.uniform(-noise_level, noise_level, size=len(df))
    df[col] = (df[col] + noise).clip(0, 1)

# Save Noisy Data
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name=output_sheet, index=False)

# --- 4. Ultralight 5-Fold Models ---
X = df[features]
y = df[target_col]

# These models are chosen specifically to NOT freeze a low-end PC
models = {
    "LogReg": LogisticRegression(solver='liblinear'), # 'liblinear' is better for small/mid RAM
    "DecTree": DecisionTreeClassifier(max_depth=5),    # Limited depth = low CPU usage
    "NaiveBayes": GaussianNB(),                       # Mathematically the simplest/fastest
    "KNN": KNeighborsClassifier(n_neighbors=3)        # Very fast for small datasets
}

print("\n--- Running Ultralight 5-Fold CV ---")
for name, model in models.items():
    # cv=5 is standard, but we use it here because these models are fast
    scores = cross_val_score(model, X, y, cv=5)
    print(f"✅ {name}: {scores.mean():.4f}")

open_excel_file(excel_path)