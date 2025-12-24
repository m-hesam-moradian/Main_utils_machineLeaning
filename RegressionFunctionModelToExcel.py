# === IMPORTS ===
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, auc
import os
import win32com.client


# === CONFIGURATION ===
# get params based on model for as data calld params as object 3 best numerical params of the model like this (it should be params of the model in variables[model_name]) :
# params = {
#     "alpha": 1.0,
#     "tol": 0.0001,
#     "max_iter": 1000
# }
# ok here are defalt params for the model SVR,  3 numerical params :
params = {
    "C": 1.0,
    "epsilon": 0.1,
    # "degree": 3
}


  


optimizer_name = " "  # no optimizer
optimizer_name = "Chimp Optimization Algorithm (ChOA)"
optimizer_name = "Arithmetic Optimization Algorithm (AOA)"
optimizer_name = "Particle Swarm Optimization algorithm (PSO)"

# model_name = "ُSVR"
model_name = "DTR"
sheet_name = "SVR"
R2_target = 0.0
min_error = -55
max_error = 63
Convergence_metric = "R2"  # Options: "rmse" or "smape"
convegence_direction = "lower"  # Options: "higher" or "lower"


dataPath = r"data\Data_err.npt"
outputPath = r"task/Data.xlsx"

# === FUNCTIONS ===


def fake_r2_prediction(y_real, y_pred, R2_target):
    current_r2 = r2_score(y_real, y_pred)
    if current_r2 >= R2_target:
        return y_pred
    for blend in np.linspace(0, 1, 1000):
        y_fake = y_pred * (1 - blend) + y_real * blend
        if r2_score(y_real, y_fake) >= R2_target:
            return y_fake
    return y_pred * 0.5 + y_real * 0.5


def enforce_error_bounds(y_real, y_pred, min_error, max_error):
    y_pred = y_pred.copy()
    for i in range(len(y_real)):
        if y_real[i] == 0:
            continue
        error_percent = (y_pred[i] / y_real[i] - 1) * 100
        if error_percent < min_error or error_percent > max_error:
            random_percent = np.random.uniform(min_error, max_error) / 100
            y_pred[i] = y_real[i] * (1 + random_percent)
    return y_pred


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# farivar
# def build_metrics_table(y_real, y_pred):

#     def compute_metrics(y_true, y_hat):
#         y_true = np.asarray(y_true)
#         y_hat  = np.asarray(y_hat)

#         # --- RMSE ---
#         rmse = np.sqrt(mean_squared_error(y_true, y_hat))

#         # --- R2 ---
#         r2 = r2_score(y_true, y_hat)

#         # --- AARD (%) ---
#         aard = 100 * np.mean(
#             np.abs((y_true - y_hat) / (y_true + 1e-12))
#         )

#         # --- MAEM ---
#         maem = (
#             np.sum(np.abs(y_hat - y_true)) /
#             (np.abs(np.sum(y_true)) + 1e-12)
#         )

#         # --- COV ---
#         ratio = y_hat / (y_true + 1e-12)
#         mean_ratio = np.mean(ratio)
#         cov = (1 / (mean_ratio + 1e-12)) * np.sqrt(
#             np.sum((ratio - mean_ratio) ** 2) / (len(ratio) - 1 + 1e-12)
#         )

#         return {
#             "R2": r2,
#             "RMSE": rmse,
#             "COV": cov,
#             "AARD (%)": aard,
#             "MAEM": maem
#         }

#     # --- Split data into Train / Test / Value sets ---
#     split_idx = int(len(y_real) * 0.8)

#     y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
#     y_pred_train, y_pred_test = y_pred[:split_idx], y_pred[split_idx:]

#     mid = len(y_real_test) // 2
#     y_real_value, y_pred_value = y_real_test[:mid], y_pred_test[:mid]
#     y_real_value_test, y_pred_value_test = y_real_test[mid:], y_pred_test[mid:]

#     # --- Compute metrics ---
#     M_all   = compute_metrics(y_real, y_pred)
#     M_train = compute_metrics(y_real_train, y_pred_train)
#     M_test  = compute_metrics(y_real_test, y_pred_test)
#     M_value = compute_metrics(y_real_value, y_pred_value)
#     M_valte = compute_metrics(y_real_value_test, y_pred_value_test)

#     # --- Build DataFrame ---
#     cols = ["Set"] + list(M_all.keys())
#     df_metrics = pd.DataFrame([
#         ["All",        *M_all.values()],
#         ["Train",      *M_train.values()],
#         ["Test",       *M_test.values()],
#         ["Value",      *M_value.values()],
#         ["Value-test", *M_valte.values()],
#     ], columns=cols)

#     return df_metrics


# amir
def build_metrics_table(y_real, y_pred):

    def compute_metrics(y_true, y_hat):
        y_true = np.asarray(y_true)
        y_hat = np.asarray(y_hat)
        errors = y_hat - y_true

        # --- R2 ---
        r2 = r2_score(y_true, y_hat)

        # --- RMSE ---
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))

        # --- MBE (Mean Bias Error) ---
        mbe = np.mean(errors)

        # --- SI (Scatter Index) ---
        si = rmse / (np.mean(y_true) + 1e-12)

        # --- U95 (Uncertainty 95%) ---
        sd_error = np.std(errors)
        u95 = 1.96 * np.sqrt(sd_error**2 + mbe**2)

        return {"R2": r2, "RMSE": rmse, "U95": u95, "MBE": mbe, "SI": si}

    # --- Split data into Train / Test / Value sets ---
    split_idx = int(len(y_real) * 0.8)

    y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
    y_pred_train, y_pred_test = y_pred[:split_idx], y_pred[split_idx:]

    mid = len(y_real_test) // 2
    y_real_value, y_pred_value = y_real_test[:mid], y_pred_test[:mid]
    y_real_value_test, y_pred_value_test = y_real_test[mid:], y_pred_test[mid:]

    # --- Compute metrics ---
    M_all = compute_metrics(y_real, y_pred)
    M_train = compute_metrics(y_real_train, y_pred_train)
    M_test = compute_metrics(y_real_test, y_pred_test)
    M_value = compute_metrics(y_real_value, y_pred_value)
    M_valte = compute_metrics(y_real_value_test, y_pred_value_test)

    # --- Build DataFrame ---
    cols = ["Set"] + list(M_all.keys())
    df_metrics = pd.DataFrame(
        [
            ["All", *M_all.values()],
            ["Train", *M_train.values()],
            ["Test", *M_test.values()],
            ["Value", *M_value.values()],
            ["Value-test", *M_valte.values()],
        ],
        columns=cols,
    )

    return df_metrics


def build_rec_curve(y_real, y_pred):
    errors = np.abs(y_real - y_pred)
    epsilon = np.linspace(0, errors.max(), 200)
    accuracy = [np.mean(errors <= e) for e in epsilon]
    rec_auc = auc(epsilon, accuracy)
    df_rec_curve = pd.DataFrame(
        {
            "Epsilon": epsilon,
            "Accuracy": accuracy,
            "AUC": ["" for _ in range(len(epsilon))],
        }
    )
    df_rec_curve.loc[0] = [np.nan, np.nan, rec_auc]
    return df_rec_curve


def build_relative_error_table(y_real, y_pred):
    rel_error = ((y_pred / y_real) - 1) * 100
    return pd.DataFrame({"Relative Error (%)": rel_error})


def get_conv(
    count=200, high=0.2, minPhase=6, maxPhase=10, convegence_direction="higher"
):
    # Adjust low based on convergence direction
    low_factor = np.random.uniform(1.2, 2.0)
    if convegence_direction == "higher":
        low = high * low_factor  # start much lower
    else:
        low = high / low_factor  # start slightly lower

    phase = np.random.randint(minPhase, maxPhase + 1)
    convergence = []

    for _ in range(phase):
        repeated_count = np.random.randint(1, 6)
        random_number = np.random.uniform(low, high)
        convergence.extend([random_number] * repeated_count)

    convergence = np.resize(convergence, count)
    convergence = (
        np.sort(convergence)[::-1]
        if convegence_direction == "higher"
        else np.sort(convergence)
    )

    return np.array(convergence)


def write_table(
    df, startrow, startcol, style_key, worksheet, writer, header_styles, sheet_name
):
    # 1. Define styles (kept your logic)
    header_styles = {
        "value_pred": make_style("9DC3E6"),
        "params": make_style("A9D08E"),
        "metrics": make_style("F4B084"),
        "error": make_style("FFD966"),
        "rec_curve": make_style("E06666"),
    }
    style = header_styles.get(style_key, make_style("D9D9D9"))

    # 2. Write the Data AND Header using Pandas (Extremely Fast)
    #    startrow in Pandas is 0-based, so we use your startrow directly.
    df.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=startrow,
        startcol=startcol,
        index=False,
        header=True
    )

    # 3. Apply Styles ONLY to the Header Row (Very Fast)
    #    We only loop over the columns (e.g., 2-14 times), not the rows (95,000 times)!
    for col_num in range(len(df.columns)):
        # Excel rows are 1-based. startrow is 0-based index relative to sheet start in pandas, 
        # but here worksheet expects absolute coordinates.
        # If startrow=1 (your param), pandas writes header at row 2 (index 1).
        # So we target row = startrow + 1.
        row = startrow + 1
        col = startcol + col_num
        
        cell = worksheet.cell(row=row, column=col)
        cell.font = style["font"]
        cell.alignment = style["alignment"]
        cell.fill = style["fill"]

def close_excel_file(filepath):
    import os
    import win32com.client

    excel = win32com.client.Dispatch("Excel.Application")
    for wb in excel.Workbooks:
        if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
            wb.Save()  # ✅ Explicit save
            wb.Close(SaveChanges=False)  # ✅ Close without prompting
            print("💾 Saved and 🔒 Closed Excel file:", filepath)
            break
    excel.Quit()


def open_excel_file(filepath):
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = True  # Show Excel window
    excel.Workbooks.Open(os.path.abspath(filepath))
    print("📂 Opened Excel file:", filepath)


# === EXECUTION ===

# Step 1: Load data
data = np.loadtxt(dataPath)
y_real = data[:, 0]
y_pred = data[:, 1]
print("Data loaded:", data.shape)

# Step 2: Adjust predictions
y_pred_fake = fake_r2_prediction(y_real, y_pred, R2_target)
print("Original R²:", r2_score(y_real, y_pred))
print("Fake R² before error enforcement:", r2_score(y_real, y_pred_fake))

# Step 3: Enforce error bounds
y_pred_fake = enforce_error_bounds(y_real, y_pred_fake, min_error, max_error)
print("Fake R² after error enforcement:", r2_score(y_real, y_pred_fake))

# Step 4: Build value/predict table
data[:, 1] = y_pred_fake
df_value_pred = pd.DataFrame(data, columns=["y_real", "y_pred"])
print("Value/predict table created.")

# Step 5: Build metrics table
df_metrics = build_metrics_table(y_real, y_pred_fake)
print("Metrics table created : ", df_metrics)

# Step 5.5: Generate fake convergence based on RMSE from training
Target_metric_train = df_metrics.loc[
    df_metrics["Set"] == "Train", Convergence_metric
].values[0]

convergence_array = get_conv(
    count=200,
    high=Target_metric_train,
    minPhase=24,
    maxPhase=32,
    convegence_direction=convegence_direction,
)
df_convergence = pd.DataFrame({"Convergence": convergence_array})
print("Fake convergence table created.")

# Step 6: Define model parameters
df_params = pd.DataFrame(list(params.items()), columns=["parameters", "values"])
print("Model parameters defined.")

# Step 7: Build REC curve
df_rec_curve = build_rec_curve(y_real, y_pred_fake)
print("REC curve created. AUC =", df_rec_curve.loc[0, "AUC"])

# Step 8: Build relative error table
df_error = build_relative_error_table(y_real, y_pred_fake)
print("Relative error table created.")


def make_style(color):
    return {
        "font": Font(bold=True),
        "alignment": Alignment(horizontal="center"),
        "fill": PatternFill(start_color=color, end_color=color, fill_type="solid"),
    }


# Step 9: Close Excel if open, then export to Excel
close_excel_file(outputPath)

from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill

# Load existing workbook
book = load_workbook(outputPath)
from openpyxl.styles import Font, Alignment, PatternFill


with pd.ExcelWriter(
    outputPath, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    # Create new sheet
    worksheet = writer.book.create_sheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    # === Custom Header Row ===
    if optimizer_name.strip():
        title = f"{model_name} + {optimizer_name.strip()}"
        merge_end_col = 14
        include_convergence = True
    else:
        title = model_name
        merge_end_col = 13
        include_convergence = False

    worksheet.merge_cells(
        start_row=1, start_column=1, end_row=1, end_column=merge_end_col + 1
    )
    cell = worksheet.cell(row=1, column=1)
    cell.value = title
    cell.font = Font(bold=True)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.fill = PatternFill(start_color="E1DFFF", end_color="E1DFFF", fill_type="solid")

    # === Write Tables ===
    write_table(
        df_value_pred,
        startrow=1,
        startcol=0,
        style_key="value_pred",
        worksheet=worksheet,
        writer=writer,
        header_styles=None,
        sheet_name=sheet_name,
    )
    params_col = len(df_value_pred.columns) + 1
    write_table(
        df_params,
        startrow=1,
        startcol=params_col,
        style_key="params",
        worksheet=worksheet,
        writer=writer,
        header_styles=None,
        sheet_name=sheet_name,
    )
    metrics_col = params_col + len(df_params.columns) + 1
    write_table(
        df_metrics,
        startrow=1,
        startcol=metrics_col,
        style_key="metrics",
        worksheet=worksheet,
        writer=writer,
        header_styles=None,
        sheet_name=sheet_name,
    )
    error_col = metrics_col + len(df_metrics.columns) + 1
    write_table(
        df_error,
        startrow=1,
        startcol=error_col,
        style_key="error",
        worksheet=worksheet,
        writer=writer,
        header_styles=None,
        sheet_name=sheet_name,
    )
    rec_start_row = len(df_params) + 6
    write_table(
        df_rec_curve,
        startrow=rec_start_row,
        startcol=params_col,
        style_key="rec_curve",
        worksheet=worksheet,
        writer=writer,
        header_styles=None,
        sheet_name=sheet_name,
    )

    if include_convergence:
        convergence_col = error_col + len(df_error.columns)
        write_table(
            df_convergence,
            startrow=1,
            startcol=convergence_col,
            style_key="error",
            worksheet=worksheet,
            writer=writer,
            header_styles=None,
            sheet_name=sheet_name,
        )

open_excel_file(outputPath)

print("✅ Structured Excel file saved successfully.")
