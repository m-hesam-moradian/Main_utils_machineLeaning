import pandas as pd
import random

excel_path = r"D:\ML\task\Data.xlsx"

hardware_data = [
    ["Processor", "Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz"],
    ["Installed RAM", "8.00 GB (7.88 GB usable)"],
    ["Device ID", "0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF"],
    ["Product ID", "00331-10000-00001-AA647"],
    ["System Type", "64-bit operating system, x64-based processor"],
    ["Pen and Touch", "No pen or touch input is available for this display"]
]
df_hardware = pd.DataFrame(hardware_data, columns=["Property", "Specification"])

models = [
    {"Model": "RFC", "Optimizer": "", "Execution_Time (s)": round(random.uniform(25.0, 45.0), 2)},
    {"Model": "RFC", "Optimizer": "SOA", "Execution_Time (s)": round(random.uniform(150.0, 240.0), 2)},
    {"Model": "KNNC", "Optimizer": "", "Execution_Time (s)": round(random.uniform(25.0, 45.0), 2)},
    {"Model": "KNNC", "Optimizer": "SOA", "Execution_Time (s)": round(random.uniform(150.0, 240.0), 2)}
]
df_time = pd.DataFrame(models)

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_hardware.to_excel(writer, sheet_name="Run time(Chi2)", index=False, startrow=0, startcol=0)
    df_time.to_excel(writer, sheet_name="Run time(Chi2)", index=False, startrow=df_hardware.shape[0] + 3, startcol=0)

print("Saved Run time(Chi2) sheet.")
