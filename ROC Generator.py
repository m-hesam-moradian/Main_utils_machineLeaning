import pandas as pd

# 1. Load the Excel file (Change 'your_data.xlsx' to your actual file name)
df = pd.read_excel('your_data.xlsx', sheet_name='ROC')

# 2. Calculate the overall Average AUC from the 5 classes
avg_auc = df['AUC'].dropna().mean()

# 3. Align the data. Since every class has exactly 12 rows, 
#    we create a "Row Number" (0 to 11) to group them together.
df['Row_Num'] = df.groupby('Class').cumcount()

# 4. Average the FPR, TPR, and Threshold across all 5 classes for each step
averaged_table = df.groupby('Row_Num')[['FPR', 'TPR', 'Threshold']].mean().reset_index(drop=True)

# 5. Format the final table so it looks exactly like your original data structure
averaged_table['AUC'] = ""  # Create an empty AUC column
averaged_table.loc[averaged_table.index[-1], 'AUC'] = round(avg_auc, 6) # Put AUC on the last row
averaged_table['Threshold'] = averaged_table['Threshold'].fillna("") # Clean up the blank cell

# 6. Print the final single table result
print("=== 1 RESULT: OVERALL MACRO-AVERAGED MODEL ===")
print(f"Average AUC: {avg_auc:.6f}\n")
print(averaged_table.to_string(index=False))
averaged_table.to_clipboard()

# ---------------------------------------------------------
# OPTIONAL: Uncomment the line below if you want to save this 
# new table to Excel to give to your figure generator!
# ---------------------------------------------------------
# averaged_table.to_excel('Averaged_ROC_1_Line.xlsx', index=False)