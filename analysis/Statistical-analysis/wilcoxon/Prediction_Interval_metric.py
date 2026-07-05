import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
filepath = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
# Update this to the exact name of the sheet where your regression predictions are!
sheet_name = "predicts" 

print(f"Loading predictions from '{sheet_name}'...")
df = pd.read_excel(filepath, sheet_name=sheet_name, header=0)
columns = df.columns.tolist()

results = []

print("="*75)
print(" CALCULATING PREDICTION INTERVAL (PI) METRICS (95% Confidence) ")
print("="*75)

# ---------------------------------------------------------
# 2. Iterate through columns (Actual, Predicted)
# ---------------------------------------------------------
for i in range(0, len(columns), 2):
    model_name = columns[i].strip()
    
    # Extract arrays and drop any NaN rows
    y_real = np.array(df.iloc[:, i].dropna())
    y_pred = np.array(df.iloc[:, i + 1].dropna())
    
    # 1. Calculate Residuals (Errors)
    residuals = y_real - y_pred
    std_residuals = np.std(residuals)
    
    # 2. Calculate 95% Prediction Interval Bounds (Z-score = 1.96)
    z_score = 1.96
    lower_bound = y_pred - (z_score * std_residuals)
    upper_bound = y_pred + (z_score * std_residuals)
    
    # 3. Calculate PICP (Prediction Interval Coverage Probability)
    # What percentage of actual values actually fall between the lower and upper bound?
    coverage = np.mean((y_real >= lower_bound) & (y_real <= upper_bound))
    
    # 4. Calculate PINAW (Prediction Interval Normalized Average Width)
    # How wide is the interval compared to the total range of the target?
    average_width = np.mean(upper_bound - lower_bound)
    target_range = np.max(y_real) - np.min(y_real)
    
    pinaw = average_width / target_range if target_range != 0 else np.nan
    
    # Store results
    results.append({
        "Model Name": model_name,
        "PICP (Coverage %)": f"{coverage * 100:.2f}%",
        "PIAW (Avg Width)": round(average_width, 4),
        "PINAW (Normalized Width)": round(pinaw, 4)
    })

# ---------------------------------------------------------
# 3. Create DataFrame and copy to clipboard
# ---------------------------------------------------------
df_results = pd.DataFrame(results)

# Print to console so you can see it
print(df_results.to_string(index=False))
print("="*75)

# Copy to clipboard
df_results.to_clipboard(index=False)
print("\n✅ Prediction Interval metrics table successfully copied to clipboard!")
print("   (You can now press Ctrl+V to paste it into Excel or Word)")