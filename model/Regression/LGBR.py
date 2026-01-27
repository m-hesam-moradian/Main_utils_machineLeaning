import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
# --- Load reordered data for LGBR (after K-Fold) ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "DATA_Normalized"  # keep same sheet

df = pd.read_excel(excel_path, sheet_name=sheet_name)
target_column = df.columns[-1]


X = df.drop(columns=target_column)
y = df[target_column]


# --- Use last 20% as test set to match K-Fold logic ---
from sklearn.model_selection import train_test_split

# --- Random Train/Test Split (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% test
    random_state=42,     # for reproducibility
    shuffle=True         # random shuffle
)

# --- Define and train LGBR model ---
model = LGBMRegressor(
    # n_estimators=200,       # number of boosting rounds
    # max_depth=4,            # limit tree depth
    # learning_rate=0.05,     # shrinkage
    # subsample=0.8,          # row sampling
    # colsample_bytree=0.8,   # feature sampling
    # reg_alpha=0.1,          # L1 regularization
    # reg_lambda=1            # L2 regularization
)

model.fit(X_train, y_train)

# --- Predictions ---
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Metrics ---
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:]),
]

df_metrics = pd.DataFrame(
    [
        {
            "Set": s,
            "MAE": mean_absolute_error(y_t, y_p),
            "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
            "R2": r2_score(y_t, y_p),
        }
        for s, y_t, y_p in sets
    ]
)

print(df_metrics)

# --- Output predictions ---
df_all = pd.DataFrame({"y_real": y, "y_pred": y_pred_all})
df_train = pd.DataFrame({"y_real": y_train, "y_pred": y_pred_train})
df_test = pd.DataFrame({"y_real": y_test, "y_pred": y_pred_test})

# --- Export to clipboard ---
df_all.to_clipboard(index=False, header=False)
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)
