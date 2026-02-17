import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score

# --- 1. Excel Helper Functions ---
def close_excel_file(filepath):
    """Saves and closes the specific Excel file if it is currently open."""
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
    except Exception:
        pass

def open_excel_file(filepath):
    """Opens the specific Excel file and makes it visible."""
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("📂 Opened Excel file:", filepath)
    except Exception as e:
        print(f"❌ Could not open Excel: {e}")

# --- 2. Configuration & Data Loading ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
source_sheet = "Encoded_Data"
output_sheet = "DATA_Noisy_Updated"

close_excel_file(excel_path)

df = pd.read_excel(excel_path, sheet_name=source_sheet)

# Identify Target and Features
target_col_name = df.columns[-1]
x_features = df.drop(columns=[target_col_name]).select_dtypes(include=[np.number]).columns.tolist()

# --- 3. Slight In-Place Noise Injection ---
# Since data is normalized (0-1), 0.02 is a 2% variance—very subtle.
noise_level = 0.02 

for col in x_features:
    # Generate small random shifts (both up and down)
    noise = np.random.uniform(-noise_level, noise_level, size=len(df))
    # Update the column directly and keep within [0, 1]
    df[col] = (df[col] + noise).clip(0, 1)

# Save the noisy data back to Excel
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name=output_sheet, index=False)

# --- 4. 5-Fold Model Setup (PC Friendly) ---
X = df[x_features]
y = df[target_col_name]

models = {
    # XGBC: Fast and efficient
    "XGBC": XGBClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        tree_method='hist' # Memory saving mode
    ),

    # RFC: Reduced size for lower RAM usage
    "RFC": RandomForestClassifier(
        n_estimators=30, 
        max_depth=5,
        n_jobs=-1, # Fast parallel processing
        random_state=42
    ),

    # LOG_REG: Replaces SVC (Uses very little RAM)
    "LOG_REG": LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        solver='lbfgs',
        random_state=42
    ),

    # GPC: Note: If this crashes your PC, replace it with GaussianNB()
    "GPC": GaussianProcessClassifier(
        kernel=1.0 * RBF(1.0),
        copy_X_train=False,
        random_state=42
    )
}

# --- 5. Execution & Results ---
print("\n--- Running 5-Fold Cross-Validation ---")
for name, model in models.items():
    try:
        scores = cross_val_score(model, X, y, cv=5)
        print(f"📊 {name} Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.2f})")
    except Exception as e:
        print(f"❌ {name} failed: {e}")

# Re-open the file to see the results
open_excel_file(excel_path)
print(f"\n✅ Done! Noise injected (±{noise_level}) and models evaluated.")