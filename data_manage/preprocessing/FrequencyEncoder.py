import pandas as pd
import numpy as np

data_path = r"d:\ML\task\Data.xlsx"
NUNIQUE_THRESHOLD = 25

print("Loading dataset for Frequency Encoding...")
xls = pd.ExcelFile(data_path)
sheets = ['D1', 'D2', 'D3', 'D4']

encoded_data_dict = {}
encoding_reports = []

for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    encoded_df = df.copy()
    
    for col in encoded_df.columns:
        if col == 'ID':
            continue
            
        # Treat as categorical if few unique values
        if encoded_df[col].nunique() <= NUNIQUE_THRESHOLD:
            # Calculate frequency
            freq = encoded_df[col].value_counts(normalize=True)
            
            # Record the mapping for the report
            for category, frequency in freq.items():
                encoding_reports.append({
                    "Dataset": sheet,
                    "Variable": col,
                    "Original Category": category,
                    "Frequency Encoded Value": round(frequency, 5)
                })
                
            # Apply mapping
            encoded_df[col] = encoded_df[col].map(freq)
            
    encoded_data_dict[f"{sheet}_Encoded"] = encoded_df

# Save encoded datasets and report
print("\nSaving detailed Frequency Encoding reports to Excel...")
with pd.ExcelWriter(data_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    for sheet_name, df_encoded in encoded_data_dict.items():
        df_encoded.to_excel(writer, sheet_name=sheet_name, index=False)
        
    if encoding_reports:
        report_df = pd.DataFrame(encoding_reports)
        report_df.to_excel(writer, sheet_name='Frequency_Encoding_Report', index=False)
        print("Generated 'Frequency_Encoding_Report' sheet.")
    else:
        print("No categorical columns met the threshold to be encoded.")

print("Step 1 (Frequency Encoding) Completed Successfully.")
