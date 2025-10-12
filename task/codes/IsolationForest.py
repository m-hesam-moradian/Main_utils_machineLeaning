import pandas as pd
from sklearn.ensemble import IsolationForest

# Load your Excel file
file_path = (
    r"D:\ML\Main_utils\task\EI No. 5, Action Power-DTR-LGBR-ADAR-CPO-PRO-Data.xlsx"
)
df = pd.read_excel(file_path)

# Select numeric columns for Isolation Forest
X = df.select_dtypes(include="number")

# Fit the model
clf = IsolationForest()
clf.fit(X)

# Predict anomalies (-1 = outlier, 1 = normal)
df["outlier_flag"] = clf.predict(X)


cleaned_df = df[df["outlier_flag"] == 1].drop(columns=["outlier_flag"])

# Save cleaned data to a new sheet
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as writer:
    cleaned_df.to_excel(writer, sheet_name="Isolation_Forest", index=False)
print("Outliers detected and saved to new sheet.")
