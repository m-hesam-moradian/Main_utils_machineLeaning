import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

excel_path = r"D:\ML\task\Data.xlsx"
models = ["LGBC", "ETC", "BC"]

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    for m in models:
        sheet_name = f"Data_after_KFold_{m}(SMOTE)"
        print(f"Reading {sheet_name}...")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Test size is exactly 20% of original dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        train_sm_df = pd.concat([X_train_sm, y_train_sm], axis=1)
        train_sm_df = train_sm_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        test_df = pd.concat([X_test, y_test], axis=1)
        final_df = pd.concat([train_sm_df, test_df], axis=0).reset_index(drop=True)
        
        print(f"Original len: {len(df)}, New len: {len(final_df)}")
        final_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Replaced sheet {sheet_name}")
print("Done!")
