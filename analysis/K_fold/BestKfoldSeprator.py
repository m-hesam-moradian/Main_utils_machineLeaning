import pandas as pd
from sklearn.model_selection import StratifiedKFold
# --- USAGE ---
# Change these values
path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet = "Selected_Data"
fold_num = 4  # The fold number you want to move (e.g., 2, 4, etc.)

def process_best_fold(data_path, sheet_name, best_fold_number):
    """
    Loads data, runs Stratified K-Fold to get indices, removes the best fold rows,
    and appends them to the end. Copies result to clipboard.
    """
    # 1. Load Data
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    
    # 2. Setup K-Fold to get the exact indices used in your script
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Get indices for the requested fold number
    # Note: enumerate starts at 1, so we subtract 1 from the input
    fold_indices = list(kf.split(X, y))
    train_idx, test_idx = fold_indices[best_fold_number - 1]
    
    # 4. Separate and Concatenate
    remaining_idx = df.index.difference(test_idx)
    df_reordered = pd.concat([df.loc[remaining_idx], df.loc[test_idx]], axis=0)
    
    # 5. Copy to clipboard
    df_reordered.to_clipboard(index=False, excel=True)
    print(f"✅ Fold {best_fold_number} separated and appended to clipboard.")

process_best_fold(path, sheet, fold_num)
