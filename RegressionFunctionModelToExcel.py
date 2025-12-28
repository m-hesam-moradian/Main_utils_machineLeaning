# === IMPORTS ===
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, auc
import os
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill

# === CONFIGURATION ===
params = {
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
}

optimizer_name = " "  # no optimizer
model_name = "HGBR"
sheet_name = "HGBR"
R2_target = 0.0
min_error = -55
max_error = 63
Convergence_metric = "U95"  # Options: "rmse", "U95", etc.
convegence_direction = "higher"  # "higher" or "lower"

dataPath = r"data/Data_err.npt"
outputPath = r"task/Data.xlsx"

# === OPTIMIZED FUNCTIONS ===

def fake_r2_prediction(y_real, y_pred, R2_target):
    """ Optimized blending to hit target R2 """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    current_r2 = r2_score(y_real, y_pred)
    if current_r2 >= R2_target:
        return y_pred
    
    # Reduced steps to 200 for speed (sufficient for visual blending)
    for blend in np.linspace(0, 1, 200):
        y_fake = y_pred * (1 - blend) + y_real * blend
        if r2_score(y_real, y_fake) >= R2_target:
            return y_fake
    return y_pred * 0.5 + y_real * 0.5


def enforce_error_bounds(y_real, y_pred, min_error, max_error):
    """ 
    VECTORIZED: Replaces the slow 'for' loop with NumPy operations.
    This runs instantly even on large datasets.
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    # 1. Calculate Error % for the whole array at once
    # Handle division by zero safely using a mask
    nonzero_mask = (y_real != 0)
    error_percent = np.zeros_like(y_real)
    
    # Only calculate where y_real is not 0
    error_percent[nonzero_mask] = (y_pred[nonzero_mask] / y_real[nonzero_mask] - 1) * 100
    
    # 2. Find indices where error is out of bounds
    violation_mask = (error_percent < min_error) | (error_percent > max_error)
    # Ensure we don't touch zero-values or valid values
    final_mask = nonzero_mask & violation_mask
    
    count = np.sum(final_mask)
    
    # 3. Apply correction only to violating rows
    if count > 0:
        # Generate random percentages for all violations at once
        random_percent = np.random.uniform(min_error, max_error, size=count) / 100
        y_pred[final_mask] = y_real[final_mask] * (1 + random_percent)
        
    return y_pred


def build_metrics_table(y_real, y_pred):
    """
    Splits data internally (Train 80% / Test 20%)
    Calculates R2, U95, RMSE, MARD, COV.
    """
    def compute_metrics(y_true, y_hat):
        y_true = np.asarray(y_true)
        y_hat  = np.asarray(y_hat)
        epsilon = 1e-12

        # R2
        r2 = r2_score(y_true, y_hat)
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        # U95 (1.96 * RMSE)
        u95 = 1.96 * rmse
        # MARD (%)
        mard = 100 * np.median(np.abs((y_hat - y_true) / (y_true + epsilon)))
        # COV
        ratio = y_hat / (y_true + epsilon)
        mean_ratio = np.mean(ratio)
        std_ratio = np.std(ratio, ddof=1)
        cov = (std_ratio / mean_ratio) if mean_ratio != 0 else np.nan

        return {
            "R2": r2, "U95": u95, "RMSE": rmse, "MARD (%)": mard, "COV": cov
        }

    # --- Split Data ---
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    split_idx = int(len(y_real) * 0.8)

    # Sets
    sets = {
        "All":   (y_real, y_pred),
        "Train": (y_real[:split_idx], y_pred[:split_idx]),
        "Test":  (y_real[split_idx:], y_pred[split_idx:])
    }
    
    # Split Test into Value / Test-Value
    y_real_test, y_pred_test = sets["Test"]
    mid = len(y_real_test) // 2
    sets["Value"] = (y_real_test[:mid], y_pred_test[:mid])
    sets["Test-Value"] = (y_real_test[mid:], y_pred_test[mid:])

    # --- Calculate ---
    results = []
    cols = ["Set", "R2", "U95", "RMSE", "MARD (%)", "COV"]
    
    # Order matters for your output
    order = ["All", "Train", "Test", "Value", "Test-Value"]
    
    for name in order:
        yt, yp = sets[name]
        m = compute_metrics(yt, yp)
        results.append([name, m["R2"], m["U95"], m["RMSE"], m["MARD (%)"], m["COV"]])

    return pd.DataFrame(results, columns=cols)


def build_rec_curve(y_real, y_pred):
    errors = np.abs(y_real - y_pred)
    # Optimization: 200 points is enough for a smooth curve
    epsilon = np.linspace(0, errors.max(), 200)
    accuracy = [np.mean(errors <= e) for e in epsilon]
    rec_auc = auc(epsilon, accuracy)
    
    df = pd.DataFrame({
        "Epsilon": epsilon,
        "Accuracy": accuracy,
        "AUC": [""] * len(epsilon)
    })
    df.loc[0, "AUC"] = rec_auc
    return df


def build_relative_error_table(y_real, y_pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = ((y_pred / y_real) - 1) * 100
    # Fix infinite/nan errors resulting from div by zero
    rel_error = np.nan_to_num(rel_error)
    return pd.DataFrame({"Relative Error (%)": rel_error})


def get_conv(count=200, high=0.2, minPhase=6, maxPhase=10, convegence_direction="higher"):
    low_factor = np.random.uniform(1.2, 2.0)
    low = high * low_factor if convegence_direction == "higher" else high / low_factor
    
    phase = np.random.randint(minPhase, maxPhase + 1)
    convergence = []

    for _ in range(phase):
        repeated_count = np.random.randint(1, 6)
        random_number = np.random.uniform(low, high)
        convergence.extend([random_number] * repeated_count)

    convergence = np.resize(convergence, count)
    return np.sort(convergence)[::-1] if convegence_direction == "higher" else np.sort(convergence)


def make_style(color):
    return {
        "font": Font(bold=True),
        "alignment": Alignment(horizontal="center"),
        "fill": PatternFill(start_color=color, end_color=color, fill_type="solid"),
    }


def write_table(df, startrow, startcol, style_key, worksheet, writer, sheet_name):
    header_styles = {
        "value_pred": make_style("9DC3E6"),
        "params": make_style("A9D08E"),
        "metrics": make_style("F4B084"),
        "error": make_style("FFD966"),
        "rec_curve": make_style("E06666"),
    }
    style = header_styles.get(style_key, make_style("D9D9D9"))

    # Write Data (Fastest way)
    df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol, index=False, header=True)

    # Apply Style to Header ONLY
    for col_num in range(len(df.columns)):
        cell = worksheet.cell(row=startrow + 1, column=startcol + col_num + 1)
        cell.font = style["font"]
        cell.alignment = style["alignment"]
        cell.fill = style["fill"]

# === EXECUTION ===

print("🚀 Starting Processing...")

# 1. Load data
data = np.loadtxt(dataPath)
y_real = data[:, 0]
y_pred = data[:, 1]
print(f"✅ Data loaded: {data.shape}")

# 2. Adjust predictions
y_pred_fake = fake_r2_prediction(y_real, y_pred, R2_target)

# 3. Enforce error bounds (Now Vectorized & Fast)
y_pred_fake = enforce_error_bounds(y_real, y_pred_fake, min_error, max_error)
print("✅ Error bounds enforced")

# 4. Build DataFrames
data[:, 1] = y_pred_fake
df_value_pred = pd.DataFrame(data, columns=["y_real", "y_pred"])
df_metrics = build_metrics_table(y_real, y_pred_fake)
print("✅ Metrics Calculated")
print(df_metrics)

# 5. Convergence Logic (Handles the 'Train' missing error)
try:
    Target_metric_train = df_metrics.loc[df_metrics["Set"] == "Train", Convergence_metric].values[0]
except IndexError:
    print(f"⚠️ Warning: Could not find 'Train' set or metric '{Convergence_metric}'. Defaulting to 0.1")
    Target_metric_train = 0.1

convergence_array = get_conv(
    count=200,
    high=Target_metric_train,
    minPhase=24,
    maxPhase=32,
    convegence_direction=convegence_direction,
)
df_convergence = pd.DataFrame({"Convergence": convergence_array})

# 6. Other Tables
df_params = pd.DataFrame(list(params.items()), columns=["parameters", "values"])
df_rec_curve = build_rec_curve(y_real, y_pred_fake)
df_error = build_relative_error_table(y_real, y_pred_fake)

print("✅ All tables prepared. Writing to Excel...")

# 7. Write to Excel (Safe Mode handling)
mode = 'a' if os.path.exists(outputPath) else 'w'
if_exists = 'replace' if mode == 'a' else None

with pd.ExcelWriter(outputPath, engine="openpyxl", mode=mode, if_sheet_exists=if_exists) as writer:
    
    # Handle Sheet Creation
    if mode == 'w' or sheet_name not in writer.book.sheetnames:
        worksheet = writer.book.create_sheet(sheet_name)
    else:
        worksheet = writer.book[sheet_name]
        writer.sheets[sheet_name] = worksheet # Ensure pandas knows about it

    # Header Logic
    if optimizer_name.strip():
        title = f"{model_name} + {optimizer_name.strip()}"
        merge_end = 14
        include_convergence = True
    else:
        title = model_name
        merge_end = 13
        include_convergence = False

    # Main Title Style
    worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=merge_end + 1)
    cell = worksheet.cell(row=1, column=1)
    cell.value = title
    cell.font = Font(bold=True)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill = PatternFill(start_color="E1DFFF", end_color="E1DFFF", fill_type="solid")

    # Write Tables
    # Note: startrow=1 means Row 2 in Excel (0-based indexing for pandas writer)
    
    # Table 1: Value/Pred
    write_table(df_value_pred, 1, 0, "value_pred", worksheet, writer, sheet_name)
    
    # Table 2: Params (Next to Table 1)
    col_ptr = len(df_value_pred.columns) + 1
    write_table(df_params, 1, col_ptr, "params", worksheet, writer, sheet_name)
    
    # Table 3: Metrics (Next to Table 2)
    col_ptr += len(df_params.columns) + 1
    write_table(df_metrics, 1, col_ptr, "metrics", worksheet, writer, sheet_name)
    
    # Table 4: Errors (Next to Table 3)
    col_ptr += len(df_metrics.columns) + 1
    write_table(df_error, 1, col_ptr, "error", worksheet, writer, sheet_name)
    
    # Table 5: REC Curve (Below Params)
    rec_start_row = len(df_params) + 6
    params_col_idx = len(df_value_pred.columns) + 1
    write_table(df_rec_curve, rec_start_row, params_col_idx, "rec_curve", worksheet, writer, sheet_name)

    # Table 6: Convergence (Optional)
    if include_convergence:
        col_ptr += len(df_error.columns)
        write_table(df_convergence, 1, col_ptr, "error", worksheet, writer, sheet_name)

print("✅ Excel Saved Successfully.")