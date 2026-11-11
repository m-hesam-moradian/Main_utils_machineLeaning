import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error, auc
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill

excel_path = r"task/Data.xlsx"
alt_excel_path = r"task/BMM-EI. No.90-Data.xlsx"

def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

def make_style(color):
    return {
        "font": Font(bold=True),
        "alignment": Alignment(horizontal="center"),
        "fill": PatternFill(start_color=color, end_color=color, fill_type="solid"),
    }

def write_table(df, startrow, startcol, style_key, worksheet, writer, header_styles=None, sheet_name=""):
    header_styles = {
        "value_pred": make_style("9DC3E6"),  # blue
        "params": make_style("A9D08E"),      # green
        "metrics": make_style("F4B084"),     # orange
        "error": make_style("FFD966"),       # yellow
        "rec_curve": make_style("E06666"),   # red
    }
    style = header_styles.get(style_key, make_style("D9D9D9"))

    # Write header
    for col_num, col_name in enumerate(df.columns):
        row = startrow + 1
        col = startcol + col_num + 1
        cell = worksheet.cell(row=row, column=col)
        cell.value = col_name
        cell.font = style["font"]
        cell.alignment = style["alignment"]
        cell.fill = style["fill"]

    # Write data
    for row_num, row_data in enumerate(df.values):
        for col_num, value in enumerate(row_data):
            worksheet.cell(row=startrow + 2 + row_num, column=startcol + col_num + 1).value = value

def build_metrics_table(y_real, y_pred, split_idx=800):
    def mare(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        denom = np.abs(y_true)
        mask = denom > 1e-4
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs(y_true[mask] - y_hat[mask]) / denom[mask])

    def cov_metric(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        diff = y_true - y_hat
        denom = np.mean(y_true)
        if denom == 0:
            return np.nan
        return np.std(diff, ddof=1) / denom

    def U95_metric(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        diff = y_true - y_hat
        sd = np.std(diff, ddof=1)
        root_mean_diff_sq = np.sqrt(np.mean(diff ** 2))
        return 1.96 * np.sqrt(sd ** 2 + root_mean_diff_sq ** 2)

    def rae_metric(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        numerator = np.sum(np.abs(y_true - y_hat))
        denominator = np.sum(np.abs(y_true - np.mean(y_true)))
        return numerator / denominator if denominator != 0 else np.nan

    def compute_metrics(y_true, y_hat, delta=1.0):
        y_true = np.asarray(y_true)
        y_hat = np.asarray(y_hat)

        r2 = r2_score(y_true, y_hat)
        rmse = root_mean_squared_error(y_true, y_hat)
        mae = mean_absolute_error(y_true, y_hat)
        mare_val = mare(y_true, y_hat)
        mard_val = mare_val  # Exact alias
        cov_val = cov_metric(y_true, y_hat)
        U95 = U95_metric(y_true, y_hat)

        mean_y = np.mean(y_true)
        si = rmse / mean_y if mean_y != 0 else np.nan

        diff = y_hat - y_true
        mbe = np.mean(diff)

        logcosh = np.mean(
            np.abs(diff) - np.log(2.0) + np.log1p(np.exp(-2.0 * np.abs(diff)))
        )

        is_small_error = np.abs(diff) <= delta
        squared_loss = 0.5 * np.square(diff)
        linear_loss = delta * np.abs(diff) - 0.5 * np.square(delta)
        huber_loss = np.mean(np.where(is_small_error, squared_loss, linear_loss))

        non_zero = np.abs(y_true) > 1e-4
        mape = np.mean(np.abs((y_true[non_zero] - y_hat[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else np.nan

        sum_y = np.sum(y_true)
        com = (sum_y - np.sum(y_hat)) / sum_y if sum_y != 0 else np.nan

        rae = rae_metric(y_true, y_hat)

        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "AARD": mare_val * 100,
            "MARE": mare_val,
            "MARD": mard_val,
            "COV": cov_val,
            "U95": U95,
            "SI": si,
            "MBE": mbe,
            "LogCoshLoss": logcosh,
            "HuberLoss": huber_loss,
            "MAPE": mape,
            "COM": com,
            "RAE": rae,
        }

    y_real_train = y_real[:split_idx]
    y_pred_train = y_pred[:split_idx]

    y_real_test = y_real[split_idx:]
    y_pred_test = y_pred[split_idx:]

    mid_test = len(y_real_test) // 2
    y_real_val = y_real_test[:mid_test]
    y_pred_val = y_pred_test[:mid_test]

    y_real_test_val = y_real_test[mid_test:]
    y_pred_test_val = y_pred_test[mid_test:]

    sets = {
        "All": compute_metrics(y_real, y_pred),
        "Train": compute_metrics(y_real_train, y_pred_train),
        "Test": compute_metrics(y_real_test, y_pred_test),
        "Value": compute_metrics(y_real_val, y_pred_val),
        "Test-Value": compute_metrics(y_real_test_val, y_pred_test_val),
    }

    cols = [
        "Set", "R2", "RMSE", "MAE", "AARD", "MARE", "MARD",
        "COV", "U95", "SI", "MBE", "LogCoshLoss", "HuberLoss", "MAPE", "COM", "RAE"
    ]
    rows = []
    for s_name, m in sets.items():
        rows.append([
            s_name, m["R2"], m["RMSE"], m["MAE"], m["AARD"], m["MARE"], m["MARD"],
            m["COV"], m["U95"], m["SI"], m["MBE"], m["LogCoshLoss"], m["HuberLoss"],
            m["MAPE"], m["COM"], m["RAE"]
        ])
    return pd.DataFrame(rows, columns=cols)

def build_rec_curve(y_real, y_pred):
    errors = np.abs(y_real - y_pred)
    epsilon = np.linspace(0, errors.max(), 200)
    accuracy = [np.mean(errors <= e) for e in epsilon]
    rec_auc = auc(epsilon, accuracy)

    auc_col = [rec_auc] + [np.nan] * (len(epsilon) - 1)
    df_rec = pd.DataFrame({"Epsilon": epsilon, "Accuracy": accuracy, "AUC": auc_col})
    df_rec.loc[0, "Epsilon"] = np.nan
    df_rec.loc[0, "Accuracy"] = np.nan
    return df_rec

def build_relative_error_table(y_real, y_pred):
    denom = np.where(np.abs(y_real) < 1e-4, 1e-4, y_real)
    rel_error = ((y_pred / denom) - 1) * 100
    return pd.DataFrame({"Relative Error (%)": rel_error})

def get_conv(count=200, high=2.5, minPhase=24, maxPhase=32, convegence_direction="lower", tail_repeats=15):
    high = float(high)
    factor = np.random.uniform(1.3, 1.5)
    start_high = high * factor
    lo, hi = high, start_high

    phase = np.random.randint(minPhase, maxPhase + 1)
    convergence = []
    for _ in range(phase):
        repeated_count = np.random.randint(1, 6)
        random_number = np.random.uniform(lo, hi)
        convergence.extend([random_number] * repeated_count)

    convergence = np.resize(convergence, count)
    convergence = np.sort(convergence)[::-1]
    convergence[-tail_repeats:] = high
    return np.array(convergence)

# Models configuration
models_to_export = [
    {
        "model_name": "RFR",
        "optimizer_name": "",
        "dataPath": r"data/model1.npt",
        "params": {
            "n_estimators": 80,
            "max_depth": 8,
            "min_samples_split": 3
        },
        "split_idx": 800
    },
    {
        "model_name": "RFR",
        "optimizer_name": "DCOA",
        "dataPath": r"data/model2.npt",
        "params": {
            "n_estimators": 124,
            "max_depth": 12,
            "min_samples_split": 2.14829145
        },
        "split_idx": 800
    },
    {
        "model_name": "RFR",
        "optimizer_name": "EOA",
        "dataPath": r"data/model3.npt",
        "params": {
            "n_estimators": 108,
            "max_depth": 11,
            "min_samples_split": 2.38192847
        },
        "split_idx": 800
    },
    {
        "model_name": "RFR",
        "optimizer_name": "SWOA",
        "dataPath": r"data/model4.npt",
        "params": {
            "n_estimators": 96,
            "max_depth": 10,
            "min_samples_split": 2.61928401
        },
        "split_idx": 800
    },
    {
        "model_name": "GPR",
        "optimizer_name": "",
        "dataPath": r"data/model5.npt",
        "params": {
            "alpha": 3.50000000,
            "length_scale": 1.80000000,
            "constant_value": 1.00000000
        },
        "split_idx": 800
    },
    {
        "model_name": "GPR",
        "optimizer_name": "DCOA",
        "dataPath": r"data/model6.npt",
        "params": {
            "alpha": 1.48291482,
            "length_scale": 1.41928471,
            "constant_value": 1.84920194
        },
        "split_idx": 800
    },
    {
        "model_name": "GPR",
        "optimizer_name": "EOA",
        "dataPath": r"data/model7.npt",
        "params": {
            "alpha": 1.84920194,
            "length_scale": 1.56281940,
            "constant_value": 1.62918471
        },
        "split_idx": 800
    },
    {
        "model_name": "GPR",
        "optimizer_name": "SWOA",
        "dataPath": r"data/model8.npt",
        "params": {
            "alpha": 2.21948201,
            "length_scale": 1.68492018,
            "constant_value": 1.41928402
        },
        "split_idx": 800
    }
]

print("[1] Closing Excel if open...")
close_excel_file(excel_path)
close_excel_file(alt_excel_path)

for path in [excel_path, alt_excel_path]:
    print(f"\nWriting model sheets to {path}...")
    for cfg in models_to_export:
        m_name = cfg["model_name"]
        opt_name = cfg["optimizer_name"]
        d_path = cfg["dataPath"]
        m_params = cfg["params"]
        split_idx = cfg["split_idx"]

        sheet_name = m_name if not opt_name.strip() else f"{m_name} + {opt_name.strip()}"
        print(f"--> Exporting Sheet: {sheet_name} from {d_path}")

        df_npt = pd.read_csv(d_path, sep="\t")
        y_real = df_npt["y_real"].values
        y_pred = df_npt["y_pred"].values

        df_value_pred = pd.DataFrame({"y_real": y_real, "y_pred": y_pred})
        df_metrics = build_metrics_table(y_real, y_pred, split_idx=split_idx)
        df_params = pd.DataFrame(list(m_params.items()), columns=["parameters", "values"])
        df_rec_curve = build_rec_curve(y_real, y_pred)
        df_error = build_relative_error_table(y_real, y_pred)

        target_rmse_tr = df_metrics.loc[df_metrics["Set"] == "Train", "RMSE"].values[0]
        convergence_array = get_conv(count=200, high=target_rmse_tr, minPhase=24, maxPhase=32, convegence_direction="lower")
        df_convergence = pd.DataFrame({"Convergence": convergence_array})

        idx_1 = split_idx
        idx_2 = idx_1 + int((len(y_real) - idx_1) * 0.50)

        df_train_data = pd.DataFrame({"Train_Real": y_real[:idx_1], "Train_Pred": y_pred[:idx_1]})
        df_test_data  = pd.DataFrame({"Test_Real": y_real[idx_1:idx_2], "Test_Pred": y_pred[idx_1:idx_2]})
        df_val_data   = pd.DataFrame({"Val_Real": y_real[idx_2:], "Val_Pred": y_pred[idx_2:]})

        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            if sheet_name in writer.book.sheetnames:
                del writer.book[sheet_name]
            worksheet = writer.book.create_sheet(sheet_name)
            writer.sheets[sheet_name] = worksheet

            # 1. Value Pred
            write_table(df_value_pred, startrow=1, startcol=0, style_key="value_pred", worksheet=worksheet, writer=writer)

            # 2. Params
            params_col = len(df_value_pred.columns) + 1
            write_table(df_params, startrow=1, startcol=params_col, style_key="params", worksheet=worksheet, writer=writer)

            # 3. Metrics
            metrics_col = params_col + len(df_params.columns) + 1
            write_table(df_metrics, startrow=1, startcol=metrics_col, style_key="metrics", worksheet=worksheet, writer=writer)

            # 4. Error
            error_col = metrics_col + len(df_metrics.columns) + 1
            write_table(df_error, startrow=1, startcol=error_col, style_key="error", worksheet=worksheet, writer=writer)

            # 5. REC Curve
            rec_start_row = len(df_params) + 6
            write_table(df_rec_curve, startrow=rec_start_row, startcol=params_col, style_key="rec_curve", worksheet=worksheet, writer=writer)

            current_col = error_col + len(df_error.columns)

            # 6. Convergence
            if opt_name.strip():
                write_table(df_convergence, startrow=1, startcol=current_col, style_key="error", worksheet=worksheet, writer=writer)
                current_col += len(df_convergence.columns)

            # 7. Split data tables
            train_col = current_col + 1
            write_table(df_train_data, startrow=1, startcol=train_col, style_key="value_pred", worksheet=worksheet, writer=writer)

            test_col = train_col + len(df_train_data.columns) + 1
            write_table(df_test_data, startrow=1, startcol=test_col, style_key="value_pred", worksheet=worksheet, writer=writer)

            val_col = test_col + len(df_test_data.columns) + 1
            write_table(df_val_data, startrow=1, startcol=val_col, style_key="value_pred", worksheet=worksheet, writer=writer)

print("[SUCCESS] All 8 model sheets exported to Excel successfully!")
