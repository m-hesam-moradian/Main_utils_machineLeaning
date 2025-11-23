import pandas as pd
import numpy as np

# === Load structured data from Excel ===
df = pd.read_excel(
    r"C:\Users\Sam\Downloads\Telegram Desktop\BMM-EI. No.39-Data.xlsx",
    header=0,
    sheet_name="OFI",   # <-- change sheet name if needed
)

# === Dynamically extract model names and predictions ===
columns = df.columns.tolist()
structured_data = []

for i in range(0, len(columns), 2):
    name = columns[i].strip()
    y_real = df.iloc[:, i].to_numpy()
    y_pred = df.iloc[:, i + 1].to_numpy()
    structured_data.append({"name": name, "y_real": y_real, "y_pred": y_pred})

# === Helper functions ===
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r_value(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def rrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / np.mean(y_true)

def performance_index(y_true, y_pred):
    R = r_value(y_true, y_pred)
    RRMSE = rrmse(y_true, y_pred)
    return RRMSE / (1 + R)

def ofi(y_true_train, y_pred_train, y_true_test, y_pred_test):
    Ztr = len(y_true_train)
    Zts = len(y_true_test)
    Z = Ztr + Zts
    PI_tr = performance_index(y_true_train, y_pred_train)
    PI_ts = performance_index(y_true_test, y_pred_test)
    return ((Ztr - Zts) / Z) * PI_tr + 2 * (Zts / Z) * PI_ts

# === Calculate metrics for each model ===
results = []

# NOTE: assumes your sheet has train/test split already OR you provide indices
# If not, you can manually split (e.g. 70/30) here
split_ratio = 0.8
for entry in structured_data:
    name = entry["name"]
    y_real = entry["y_real"]
    y_pred = entry["y_pred"]

    # Split train/test
    split_idx = int(len(y_real) * split_ratio)
    y_real_train, y_pred_train = y_real[:split_idx], y_pred[:split_idx]
    y_real_test, y_pred_test = y_real[split_idx:], y_pred[split_idx:]

    # Metrics
    RMSE_train = rmse(y_real_train, y_pred_train)
    RMSE_test = rmse(y_real_test, y_pred_test)
    R_train = r_value(y_real_train, y_pred_train)
    R_test = r_value(y_real_test, y_pred_test)
    RRMSE_train = rrmse(y_real_train, y_pred_train)
    RRMSE_test = rrmse(y_real_test, y_pred_test)
    PI_train = performance_index(y_real_train, y_pred_train)
    PI_test = performance_index(y_real_test, y_pred_test)
    OFI_val = ofi(y_real_train, y_pred_train, y_real_test, y_pred_test)

    results.append({
        "Model": name,
        "RMSE_train": RMSE_train,
        "RMSE_test": RMSE_test,
        "R_train": R_train,
        "R_test": R_test,
        "RRMSE_train": RRMSE_train,
        "RRMSE_test": RRMSE_test,
        "PI_train": PI_train,
        "PI_test": PI_test,
        "OFI": OFI_val,
    })

# === Convert results to DataFrame ===
df_results = pd.DataFrame(results)

# === Display final results ===
print("\nModel Performance Metrics:")
print(df_results)

# Copy to clipboard for easy paste into Excel
df_results.to_clipboard(index=False)