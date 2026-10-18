"""
Step 2 - Randomize Dataset + 80/20 Train/Test Split
===================================================
Randomize the Encoded_Data, split 80/20 (stratified), and save:
  - Train_Data sheet (80%)
  - Test_Data sheet (20%)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = r"/home/z/my-project/task_238/Data.xlsx"

print("Loading Encoded_Data ...")
df = pd.read_excel(DATA, sheet_name="Encoded_Data")
print(f"  Shape: {df.shape}")

target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Randomize + stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
df_test  = pd.concat([X_test,  y_test],  axis=1).reset_index(drop=True)

print(f"  Train: {df_train.shape}  |  Test: {df_test.shape}")
print(f"\nTrain target distribution:")
print(df_train[target_col].value_counts().sort_index().to_string())
print(f"\nTest target distribution:")
print(df_test[target_col].value_counts().sort_index().to_string())

# Save
with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    df_train.to_excel(writer, sheet_name="Train_Data", index=False)
    df_test.to_excel(writer, sheet_name="Test_Data", index=False)

print(f"\n[+] Saved Train_Data and Test_Data sheets.")
