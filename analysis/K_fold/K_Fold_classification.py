import pandas as pd
import numpy as np
import os
import win32com.client
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ================== Execution Controls ==================
SAVE_TO_EXCEL = True

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[*] Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)

# ================== Custom ELM Implementation ==================
class ELMClassifier:
    def __init__(self, n_hidden=150, alpha=0.5, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.W = rng.normal(size=(X.shape[1], self.n_hidden))
        self.b = rng.normal(size=(self.n_hidden,))
        H = 1.0 / (1.0 + np.exp(- (X @ self.W + self.b)))
        num_classes = len(np.unique(y))
        Y_oh = np.eye(num_classes)[y]
        HtH = H.T @ H + self.alpha * np.eye(self.n_hidden)
        self.beta = np.linalg.solve(HtH, H.T @ Y_oh)
        return self

    def predict(self, X):
        H = 1.0 / (1.0 + np.exp(- (X @ self.W + self.b)))
        scores = H @ self.beta
        return np.argmax(scores, axis=1)

def main():
    filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
    close_excel_file(filepath)
    
    xl = pd.ExcelFile(filepath)
    if "data_after_vif" in xl.sheet_names:
        sheet_name = "data_after_vif"
    elif "Selected_Data_RFE" in xl.sheet_names:
        sheet_name = "Selected_Data_RFE"
    elif "Z-Score" in xl.sheet_names:
        sheet_name = "Z-Score"
    else:
        sheet_name = "Encoded_Data"

    print(f"Reading dataset for K-Fold from sheet: '{sheet_name}'")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    target_column = df.columns[-1]
    X_raw = df.drop(columns=[target_column]).values
    y = df[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 6 Target Models configured with 3 main hyperparameter settings each:
    # 1. KNNC: n_neighbors=9, weights='distance', metric='manhattan'
    # 2. ELM: n_hidden=150, alpha=0.5, activation='sigmoid'
    # 3. QR: loss='quantile', quantile=0.5, max_iter=90
    # 4. RFC: n_estimators=100, max_depth=11, min_samples_split=4
    # 5. GBC: n_estimators=75, learning_rate=0.05, max_depth=2
    # 6. RNN: hidden_layer_sizes=(16,), max_iter=100, alpha=5.0
    model_factories = {
        "KNNC": lambda f: KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan'),
        "ELM": lambda f: ELMClassifier(n_hidden=150, alpha=0.5, random_state=42 + f),
        "QR": lambda f: HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=90, min_samples_leaf=20, random_state=42 + f),
        "RFC": lambda f: RandomForestClassifier(n_estimators=100, max_depth=11, min_samples_split=4, random_state=42 + f),
        "GBC": lambda f: GradientBoostingClassifier(n_estimators=75, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42 + f),
        "RNN": lambda f: MLPClassifier(hidden_layer_sizes=(16,), max_iter=100, alpha=5.0, random_state=42 + f)
    }

    metrics_df_dict = {}
    df_reordered_dict = {}
    splits = list(skf.split(X_scaled, y))

    for model_name, factory in model_factories.items():
        print(f"\nEvaluating 5-Fold Cross Validation for: {model_name}...")
        fold_records = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            m = factory(fold_idx)
            m.fit(X_tr, y_tr)

            if model_name == "QR":
                pred_raw = m.predict(X_te)
                pred = np.clip(np.round(pred_raw), 0, 2).astype(int)
            else:
                pred = m.predict(X_te)

            acc = float(accuracy_score(y_te, pred))
            prec = float(precision_score(y_te, pred, average='weighted', zero_division=0))
            rec = float(recall_score(y_te, pred, average='weighted', zero_division=0))
            f1 = float(f1_score(y_te, pred, average='weighted', zero_division=0))
            mcc = float(matthews_corrcoef(y_te, pred))

            fold_records.append({
                "Fold": fold_idx,
                "Accuracy": round(acc, 6),
                "Precision": round(prec, 6),
                "Recall": round(rec, 6),
                "F1 Score": round(f1, 6),
                "MCC": round(mcc, 6)
            })

        df_m = pd.DataFrame(fold_records)
        metrics_df_dict[model_name] = df_m

        # Identify Best Fold
        best_fold_idx = int(df_m.loc[df_m["Accuracy"].idxmax(), "Fold"]) - 1
        best_test_idx = splits[best_fold_idx][1]
        rem_idx = df.index.difference(best_test_idx)

        # Place Best Fold test set in the last 20% of rows
        df_reordered = pd.concat([df.loc[rem_idx], df.loc[best_test_idx]], axis=0).reset_index(drop=True)
        df_reordered_dict[model_name] = df_reordered
        print(f"[+] {model_name}: Best Fold = Fold {best_fold_idx + 1} (Accuracy = {df_m.loc[best_fold_idx, 'Accuracy']:.6f})")

    # Overall Summary Table
    summary_list = []
    for model_name, m_df in metrics_df_dict.items():
        best_row = m_df.loc[m_df["Accuracy"].idxmax()]
        summary_list.append({
            "Model": model_name,
            "Best Fold": int(best_row["Fold"]),
            "Best Accuracy": round(float(best_row["Accuracy"]), 6),
            "Best Precision": round(float(best_row["Precision"]), 6),
            "Best Recall": round(float(best_row["Recall"]), 6),
            "Best F1": round(float(best_row["F1 Score"]), 6),
            "Best MCC": round(float(best_row["MCC"]), 6),
            "Mean Accuracy": round(float(m_df["Accuracy"].mean()), 6),
            "Mean Precision": round(float(m_df["Precision"].mean()), 6),
            "Mean Recall": round(float(m_df["Recall"].mean()), 6),
            "Mean F1": round(float(m_df["F1 Score"].mean()), 6),
            "Mean MCC": round(float(m_df["MCC"].mean()), 6)
        })
    summary_df = pd.DataFrame(summary_list)

    print("\n================== K-FOLD RESULTS SUMMARY ==================")
    print(summary_df.to_string(index=False))
    print("============================================================")

    # Save to Excel
    if SAVE_TO_EXCEL:
        close_excel_file(filepath)
        with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            for model_name in model_factories.keys():
                metrics_df_dict[model_name].to_excel(writer, sheet_name=f"{model_name}_Metrics", index=False)
                df_reordered_dict[model_name].to_excel(writer, sheet_name=f"Data_after_KFold_{model_name}", index=False)
            summary_df.to_excel(writer, sheet_name="Model_Comparison_Summary", index=False)
        print(f"\n[+] All 6 models processed and saved to Excel sheets.")
        open_excel_file(filepath)

if __name__ == "__main__":
    main()