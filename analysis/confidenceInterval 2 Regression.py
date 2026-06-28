import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import warnings
warnings.filterwarnings("ignore")


def build_ci_regression_reports(
        y_real,
        y_pred,
        model_name="Model",
        n_bootstrap=1000,
        ci=95):

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def mare(y_true, y_hat):
        y_true = np.asarray(y_true)
        y_hat = np.asarray(y_hat)

        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.nan

        return np.mean(
            np.abs(
                (y_true[mask] - y_hat[mask]) /
                y_true[mask]
            )
        ) * 100


    def mbe(y_true, y_hat):
        return np.mean(y_hat - y_true)


    def si(y_true, y_hat):
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        mean_obs = np.mean(y_true)

        if mean_obs == 0:
            return np.nan

        return rmse / mean_obs


    def rae(y_true, y_hat):
        den = np.sum(np.abs(y_true - np.mean(y_true)))

        if den == 0:
            return np.nan

        return np.sum(np.abs(y_true - y_hat)) / den


    def u95(y_true, y_hat):
        err = y_hat - y_true
        return 1.96 * np.std(err)


    def logcosh_loss(y_true, y_hat):
        return np.mean(np.log(np.cosh(y_hat - y_true)))


    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    def get_core_metrics(y_t, y_p):

        rmse = np.sqrt(mean_squared_error(y_t, y_p))

        return {
            "R2": r2_score(y_t, y_p),
            "RMSE": rmse,
            "MAE": mean_absolute_error(y_t, y_p),
            "MAPE": np.mean(np.abs((y_t - y_p) / y_t)) * 100,
            "MARE": mare(y_t, y_p),
            "MBE": mbe(y_t, y_p),
            "SI": si(y_t, y_p),
            "RAE": rae(y_t, y_p),
            "U95": u95(y_t, y_p),
            "LogCoshLoss": logcosh_loss(y_t, y_p)
        }


    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    def get_metrics_with_ci(y_t, y_p):

        base = get_core_metrics(y_t, y_p)

        n = len(y_t)

        boot = {k: [] for k in base}

        for _ in range(n_bootstrap):

            idx = np.random.choice(
                n,
                size=n,
                replace=True
            )

            yt = y_t[idx]
            yp = y_p[idx]

            res = get_core_metrics(yt, yp)

            for k in base:
                boot[k].append(res[k])

        lower = (100 - ci) / 2
        upper = 100 - lower

        out = {}

        for k in base:

            low = np.percentile(boot[k], lower)
            high = np.percentile(boot[k], upper)

            out[k] = f"{base[k]:.4f} ({low:.4f}-{high:.4f})"

        return out


    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    split = int(len(y_real) * 0.8)

    y_real_train = y_real[:split]
    y_real_test = y_real[split:]

    y_pred_train = y_pred[:split]
    y_pred_test = y_pred[split:]


    cols = [
        "Model",
        "Set",
        "R2",
        "RMSE",
        "MAE",
        "MAPE",
        "MARE",
        "MBE",
        "SI",
        "RAE",
        "U95",
        "LogCoshLoss"
    ]


    df = pd.DataFrame([

        [model_name,
         "All",
         *get_metrics_with_ci(
             y_real,
             y_pred
         ).values()],

        [model_name,
         "Train",
         *get_metrics_with_ci(
             y_real_train,
             y_pred_train
         ).values()],

        [model_name,
         "Test",
         *get_metrics_with_ci(
             y_real_test,
             y_pred_test
         ).values()]

    ], columns=cols)

    return df
df = pd.read_excel(
    r"C:\Users\Sam\Desktop\ML\task\Data.xlsx",
    sheet_name="predicts"
)

columns = df.columns.tolist()

all_reports = []

for i in range(0, len(columns), 2):

    raw_name = columns[i].strip()
    model_name = raw_name.split("_")[0] if "_" in raw_name else raw_name

    y_real = np.array(df.iloc[:, i].dropna())
    y_pred = np.array(df.iloc[:, i + 1].dropna())

    print(f"Bootstrapping {model_name}...")

    report = build_ci_regression_reports(
        y_real,
        y_pred,
        model_name=model_name,
        n_bootstrap=1000
    )

    all_reports.append(report)

final_metrics_df = pd.concat(all_reports, ignore_index=True)

print(final_metrics_df)

final_metrics_df.to_clipboard(index=False)

print("Copied successfully.")