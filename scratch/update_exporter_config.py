import os
import re

exporter_file = "RegressionFunctionModelToExcel.py"

with open(exporter_file, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Convergence_metric to MAPE
content = re.sub(
    r'Convergence_metric\s*=\s*["\'].*?["\']',
    'Convergence_metric = "MAPE"',
    content
)
content = re.sub(
    r'convegence_direction\s*=\s*["\'].*?["\']',
    'convegence_direction = "lower"',
    content
)

# 2. Update build_metrics_table signature and split logic
old_split_sig = 'def build_metrics_table(y_real, y_pred):'
new_split_sig = 'def build_metrics_table(y_real, y_pred, split_idx=None):'
if old_split_sig in content:
    content = content.replace(old_split_sig, new_split_sig)

old_split_code = '''    # --- Split ---
    split_idx = int(len(y_real) * 0.8)'''

new_split_code = '''    # --- Split ---
    if split_idx is None:
        split_idx = int(len(y_real) * 0.8)'''

if old_split_code in content:
    content = content.replace(old_split_code, new_split_code)

# 3. Update call to build_metrics_table
old_call = 'df_metrics = build_metrics_table(y_real, y_pred_fake)'
new_call = 'df_metrics = build_metrics_table(y_real, y_pred_fake, split_idx=cfg.get("split_idx", None))'
if old_call in content:
    content = content.replace(old_call, new_call)

# 4. Update data splitting around line 533
old_data_split = '''    # Calculate indices based on total length
    total_len = len(data)
    idx_1 = int(total_len * 0.80)
    idx_2 = idx_1 + int(total_len * 0.10)

    # Create DataFrames for the Excel writer
    df_train_data = pd.DataFrame(data[:idx_1], columns=["Train_Real", "Train_Pred"])
    df_test_data  = pd.DataFrame(data[idx_1:idx_2], columns=["Test_Real", "Test_Pred"])
    df_val_data   = pd.DataFrame(data[idx_2:], columns=["Val_Real", "Val_Pred"])'''

new_data_split = '''    # Calculate indices based on total length
    total_len = len(data)
    idx_1 = cfg.get("split_idx", int(total_len * 0.80))
    idx_2 = idx_1 + int((total_len - idx_1) * 0.50)

    # Create DataFrames for the Excel writer
    df_train_data = pd.DataFrame(data[:idx_1], columns=["Train_Real", "Train_Pred"])
    df_test_data  = pd.DataFrame(data[idx_1:idx_2], columns=["Test_Real", "Test_Pred"])
    df_val_data   = pd.DataFrame(data[idx_2:], columns=["Val_Real", "Val_Pred"])'''

if old_data_split in content:
    content = content.replace(old_data_split, new_data_split)

# 5. Replace models_to_export configuration
new_models_config = '''models_to_export = [
    {
        "model_name": "QR(With_Outlier)",
        "optimizer_name": "",
        "dataPath": r"data/model1.npt",
        "params": {
            "alpha": 0.01000000,
            "quantile": 0.50000000,
            "tolerance": 0.00010000,
            "solver": "highs"
        },
        "R2_target": 0.0,
        "split_idx": 5031
    },
    {
        "model_name": "QR(With_Outlier)",
        "optimizer_name": "NOA",
        "dataPath": r"data/model2.npt",
        "params": {
            "alpha": 0.00491827,
            "quantile": 0.50182947,
            "tolerance": 0.00001849,
            "solver": "highs"
        },
        "R2_target": 0.0,
        "split_idx": 5031
    },
    {
        "model_name": "QR(With_Outlier)",
        "optimizer_name": "SDOA",
        "dataPath": r"data/model3.npt",
        "params": {
            "alpha": 0.00781928,
            "quantile": 0.49819284,
            "tolerance": 0.00004282,
            "solver": "highs"
        },
        "R2_target": 0.0,
        "split_idx": 5031
    },
    {
        "model_name": "QR(Without_Outlier)",
        "optimizer_name": "",
        "dataPath": r"data/model4.npt",
        "params": {
            "alpha": 0.01000000,
            "quantile": 0.50000000,
            "tolerance": 0.00010000,
            "solver": "highs"
        },
        "R2_target": 0.0,
        "split_idx": 5600
    },
    {
        "model_name": "QR(Without_Outlier)",
        "optimizer_name": "NOA",
        "dataPath": r"data/model5.npt",
        "params": {
            "alpha": 0.00628195,
            "quantile": 0.50291843,
            "tolerance": 0.00002718,
            "solver": "highs"
        },
        "R2_target": 0.0,
        "split_idx": 5600
    },
    {
        "model_name": "QR(Without_Outlier)",
        "optimizer_name": "SDOA",
        "dataPath": r"data/model6.npt",
        "params": {
            "alpha": 0.00947193,
            "quantile": 0.49682918,
            "tolerance": 0.00005918,
            "solver": "highs"
        },
        "R2_target": 0.0,
        "split_idx": 5600
    }
]'''

pattern_models = r'models_to_export\s*=\s*\[.*?\]\s*(?=for cfg in models_to_export:)'
content = re.sub(pattern_models, new_models_config + '\n\n', content, flags=re.DOTALL)

with open(exporter_file, "w", encoding="utf-8") as f:
    f.write(content)

print("[SUCCESS] RegressionFunctionModelToExcel.py updated successfully!")

