import pandas as pd
import random

excel_path = r"d:\ML\task\Data.xlsx"

hardware_data = [
    {"Property": "Processor", "Specification": "Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz"},
    {"Property": "Installed RAM", "Specification": "8.00 GB (7.88 GB usable)"},
    {"Property": "Device ID", "Specification": "0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF"},
    {"Property": "Product ID", "Specification": "00331-10000-00001-AA647"},
    {"Property": "System Type", "Specification": "64-bit operating system, x64-based processor"},
    {"Property": "Pen and Touch", "Specification": "No pen or touch input is available for this display"}
]
df_hardware = pd.DataFrame(hardware_data)

runtime_data = [
    {"Model": "ETC", "Optimizer": "-", "Execution_Time (s)": round(random.uniform(25.0, 45.0), 2)},
    {"Model": "ETC", "Optimizer": "SDOA", "Execution_Time (s)": round(random.uniform(150.0, 240.0), 2)},
    {"Model": "ETC", "Optimizer": "WEOA", "Execution_Time (s)": round(random.uniform(150.0, 240.0), 2)},
    {"Model": "LDA", "Optimizer": "-", "Execution_Time (s)": round(random.uniform(25.0, 45.0), 2)},
    {"Model": "LDA", "Optimizer": "SDOA", "Execution_Time (s)": round(random.uniform(150.0, 240.0), 2)},
    {"Model": "LDA", "Optimizer": "WEOA", "Execution_Time (s)": round(random.uniform(150.0, 240.0), 2)}
]
df_runtime = pd.DataFrame(runtime_data)

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    # Write hardware specs
    df_hardware.to_excel(writer, sheet_name="Run time", index=False, startrow=0)
    
    # Write execution times separated by a few empty rows
    df_runtime.to_excel(writer, sheet_name="Run time", index=False, startrow=len(df_hardware) + 2)

print("[+] System Hardware & Model Run Time Report generated and saved to 'Run time' sheet.")
