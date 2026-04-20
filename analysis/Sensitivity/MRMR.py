import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.svm import LinearSVC
from lightgbm import LGBMRegressor
import os
import win32com.client

# --- Excel control helpers ---
def close_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                break
    except: pass

def open_excel_file(filepath):
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = True
    excel.Workbooks.Open(os.path.abspath(filepath))

# --- Enhanced MRMR Ranking ---
def get_mrmr_ranking(X, y):
    is_classification = y.nunique() < 20
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    if is_classification:
        relevance = mutual_info_classif(X_scaled, y, random_state=42)
    else:
        relevance = mutual_info_regression(X_scaled, y, random_state=42)
    relevance_series = pd.Series(relevance, index=X.columns)
    
    corr_matrix = X_scaled.corr().abs()
    selected_features, unselected_features = [], list(X.columns)
    mrmr_report = []

    # Step 1
    first_feature = relevance_series.idxmax()
    selected_features.append(first_feature)
    unselected_features.remove(first_feature)
    mrmr_report.append({"Rank": 1, "Feature": first_feature, "MRMR_Score": relevance_series[first_feature]})

    # Step 2 to N
    for rank in range(2, len(X.columns) + 1):
        best_score, best_feature = -np.inf, unselected_features[0]
        for feature in unselected_features:
            rel = relevance_series[feature]
            red = corr_matrix.loc[feature, selected_features].mean()
            score = rel - red
            if not np.isnan(score) and score > best_score:
                best_score, best_feature = score, feature
        
        selected_features.append(best_feature)
        unselected_features.remove(best_feature)
        mrmr_report.append({"Rank": rank, "Feature": best_feature, "MRMR_Score": best_score})

    return selected_features, pd.DataFrame(mrmr_report)

# --- Main Logic ---
if __name__ == "__main__":
    excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
    close_excel_file(excel_path)
    df = pd.read_excel(excel_path, sheet_name="Z-Score")

    target_column = df.columns[-1]
    X, y = df.drop(columns=[target_column]), df[target_column]
    is_classification = y.nunique() < 20

    # 1. Ranking
    ordered_features, ranking_df = get_mrmr_ranking(X, y)

    # 2. Evaluation Loop
    evaluation_rows = []
    print(f"🧪 Testing {len(ordered_features)} subsets to find the best one...")
    
    for i in range(1, len(ordered_features) + 1):
        current_feats = ordered_features[:i]
        X_train, X_test, y_train, y_test = train_test_split(X[current_feats], y, test_size=0.2, random_state=42)

        if is_classification:
            model = LinearSVC(random_state=42, max_iter=5000).fit(X_train, y_train)
            preds = model.predict(X_test)
            metric = f1_score(y_test, preds, average="weighted")
            evaluation_rows.append({"Features Count": i, "F1-Score": metric})
        else:
            model = LGBMRegressor(random_state=42, verbosity=-1).fit(X_train, y_train)
            preds = model.predict(X_test)
            metric = np.sqrt(mean_squared_error(y_test, preds)) # RMSE
            evaluation_rows.append({"Features Count": i, "RMSE": metric})

    eval_df = pd.DataFrame(evaluation_rows)

    # 3. Find the Optimal Number of Features
    if is_classification:
        # در طبقه‌بندی: بیشترین F1
        best_row = eval_df.loc[eval_df["F1-Score"].idxmax()]
    else:
        # در رگرسیون: کمترین RMSE
        best_row = eval_df.loc[eval_df["RMSE"].idxmin()]

    opt_count = int(best_row["Features Count"])
    best_features = ordered_features[:opt_count]
    
    # اضافه کردن وضعیت Kept/Removed به گزارش رنکینگ
    ranking_df["Status"] = ["Kept" if r <= opt_count else "Removed" for r in ranking_df["Rank"]]

    print(f"🎯 Optimal choice: {opt_count} features (Metric: {best_row.values[1]:.4f})")

    # 4. Prepare Final Dataset
    final_dataset = df[best_features + [target_column]]

    # 5. Save everything
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        ranking_df.to_excel(writer, sheet_name="MRMR_Ranking", index=False)
        eval_df.to_excel(writer, sheet_name="MRMR_Performance_Step", index=False)
        final_dataset.to_excel(writer, sheet_name="Optimal_Data_MRMR", index=False)

    print(f"✅ Success! Saved {opt_count} features to 'Optimal_Data_MRMR'.")
    open_excel_file(excel_path)