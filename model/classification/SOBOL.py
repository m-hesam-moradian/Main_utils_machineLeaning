import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from SALib.sample import saltelli
from SALib.analyze import sobol

# --- Load dataset ---
sheet_name = "Data_after_KFold_LR"
file_path = r"C:\Users\Sam\Desktop\ML\task\BMM-EI. No.24-.xlsx"
target_column = "Fault_Status"

df = pd.read_excel(file_path, sheet_name=sheet_name).dropna()
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Train Linear Regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Define problem space for SALib ---
feature_names = X.columns.tolist()
bounds = [[float(X[col].min()), float(X[col].max())] for col in feature_names]

problem = {
    "num_vars": len(feature_names),
    "names": feature_names,
    "bounds": bounds
}

# --- Generate samples and predict ---
param_values = saltelli.sample(problem, 1024, calc_second_order=False)
predictions = model.predict(param_values)

# --- Run SOBOL analysis ---
sobol_results = sobol.analyze(problem, predictions, calc_second_order=False, print_to_console=False)

# --- Format results ---
sensitivity_df = pd.DataFrame({
    "Feature": feature_names,
    "S1": sobol_results["S1"],
    "S1_conf": sobol_results["S1_conf"]
}).sort_values(by="S1", ascending=False).reset_index(drop=True)

# --- Export to clipboard ---
sensitivity_df.to_clipboard(index=False)
print(sensitivity_df)