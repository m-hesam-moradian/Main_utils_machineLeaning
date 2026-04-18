import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import os
import win32com.client

# --- Excel control ---
def close_excel_file(filepath):
    excel = win32com.client.Dispatch("Excel.Application")
    for wb in excel.Workbooks:
        try:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
        except Exception:
            pass
    excel.Quit()

def open_excel_file(filepath):
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = True
    excel.Workbooks.Open(os.path.abspath(filepath))
    print("📂 Opened Excel file:", filepath)


# --- MRMRMS Feature Selection ---
def calculate_mrmrms(X, y, num_features_to_select=10):
    """
    Applies MRMRMS feature selection. 
    It standardizes the data, evaluates relevance (Mutual Information), 
    evaluates redundancy (Correlation with already selected features), 
    and iteratively picks the best features.
    """
    # Determine task type based on target unique values
    is_classification = y.nunique() < 20
    
    # 1. Standardization (Required by the paper - Section 2.4.3)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 2. Calculate Relevance (Mutual Information)
    print(f"📊 Calculating Relevance (Mutual Information) for {X.shape[1]} features...")
    if is_classification:
        relevance = mutual_info_classif(X_scaled, y, random_state=42)
    else:
        relevance = mutual_info_regression(X_scaled, y, random_state=42)
        
    relevance_series = pd.Series(relevance, index=X.columns)
    
    # 3. Redundancy matrix (Absolute Pearson Correlation)
    corr_matrix = X_scaled.corr().abs()
    
    selected_features = []
    mrmrms_records = []
    unselected_features = list(X.columns)
    
    num_features_to_select = min(num_features_to_select, len(X.columns))
    
    # Step 1: Select the very first feature with the highest relevance
    first_feature = relevance_series.idxmax()
    selected_features.append(first_feature)
    unselected_features.remove(first_feature)
    
    mrmrms_records.append({
        "Step": 1,
        "Feature": first_feature,
        "Relevance (MI)": relevance_series[first_feature],
        "Redundancy": 0.0,
        "MRMRMS_Score": relevance_series[first_feature]
    })
    
    print(f"   Step 1: Selected '{first_feature}' (Score: {relevance_series[first_feature]:.4f})")
    
    # Step 2 to N: Balance Relevance and Redundancy
    for step in range(2, num_features_to_select + 1):
        best_score = -np.inf
        best_feature = None
        best_relevance = 0
        best_redundancy = 0
        
        for feature in unselected_features:
            rel = relevance_series[feature]
            # Redundancy is the average correlation with ALREADY selected features
            red = corr_matrix.loc[feature, selected_features].mean()
            
            # MRMRMS Score calculation
            score = rel - red
            
            if score > best_score:
                best_score = score
                best_feature = feature
                best_relevance = rel
                best_redundancy = red
                
        # Register the winner of this round
        selected_features.append(best_feature)
        unselected_features.remove(best_feature)
        
        mrmrms_records.append({
            "Step": step,
            "Feature": best_feature,
            "Relevance (MI)": best_relevance,
            "Redundancy": best_redundancy,
            "MRMRMS_Score": best_score
        })
        print(f"   Step {step}: Selected '{best_feature}' (Score: {best_score:.4f})")

    mrmrms_df = pd.DataFrame(mrmrms_records)
    selected_X = X[selected_features].copy() # Return the original unscaled data for selected columns
    
    return selected_X, mrmrms_df

# --- Main Logic ---
if __name__ == "__main__":
    excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
    
    # Close excel if open to prevent permission errors
    close_excel_file(excel_path)
    
    # Load data
    df = pd.read_excel(excel_path, sheet_name="Z-Score")

    # Separate Target
    target_column = df.columns[-1]
    X_input = df.drop(columns=[target_column])
    y_input = df[target_column]

    # --- Run MRMRMS Feature Selection ---
    # Change 'num_top_features' below to select how many features you want to keep
    num_top_features = 10 
    
    print(f"\n🚀 Running MRMRMS Feature Selection (Selecting Top {num_top_features})...")
    selected_X, mrmrms_scores = calculate_mrmrms(X_input, y_input, num_features_to_select=num_top_features)

    # Reattach target column to the filtered dataset
    data_after_mrmrms = selected_X.copy()
    data_after_mrmrms[target_column] = y_input

    # Save to Excel & Open
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        mrmrms_scores.to_excel(writer, sheet_name="MRMRMS_Scores", index=False)
        data_after_mrmrms.to_excel(writer, sheet_name="Data_after_MRMRMS", index=False)

    # Copy final dataset to clipboard
    data_after_mrmrms.to_clipboard(index=False)
    
    print(f"\n✅ Done! Dataset reduced to top {num_top_features} features and saved to Excel.")
    open_excel_file(excel_path)