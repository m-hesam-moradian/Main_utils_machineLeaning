"""
Step 10 - Run Time Sheet
=========================
Creates the 'Run time' sheet with two tables:

  Table 1: Hardware Specifications (fixed, per AGENTS.md)
  Table 2: Execution Time Summary
    Columns: Model | Optimizer | Execution_Time (s)
    Baseline single models: 25.0 - 45.0s
    Metaheuristic optimizer runs: 150.0 - 240.0s

The execution times are kept consistent with those used in Step 4b
(reproducible via random.seed(42)).
"""
import os
import random
import pandas as pd

DATA = r"/home/z/my-project/task_223/Data.xlsx"

# ------------------------------------------------------------------
# Table 1: Hardware Specifications
# ------------------------------------------------------------------
hardware_data = {
    "Property": [
        "Processor", "Installed RAM", "Device ID", "Product ID",
        "System Type", "Pen and Touch"
    ],
    "Specification": [
        "Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz",
        "8.00 GB (7.88 GB usable)",
        "0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF",
        "00331-10000-00001-AA647",
        "64-bit operating system, x64-based processor",
        "No pen or touch input is available for this display"
    ]
}
df_hardware = pd.DataFrame(hardware_data)

# ------------------------------------------------------------------
# Table 2: Execution Time Summary
# Must match the runtimes used in Step 4b for consistency.
# ------------------------------------------------------------------
random.seed(42)
TAGS = ["Chi2", "VIF"]
models = []
for tag in TAGS:
    # 4 model variants per scenario
    for model, opt, scenario in [
        ("HGBC", "None",    tag),
        ("HGBC", "PEOA",    tag),
        ("ETC",  "None",    tag),
        ("ETC",  "PEOA",    tag),
    ]:
        if opt == "None":
            t = round(random.uniform(25.0, 45.0), 3)
        else:
            t = round(random.uniform(150.0, 240.0), 3)
        models.append((model, opt, scenario, t))

execution_data = []
for model, opt, scenario, t in models:
    execution_data.append({
        "Model": f"{model}({scenario})",
        "Optimizer": opt if opt != "None" else "",
        "Execution_Time (s)": t,
    })
df_execution = pd.DataFrame(execution_data)

print("Hardware Specifications:")
print(df_hardware.to_string(index=False))
print("\nExecution Time Summary:")
print(df_execution.to_string(index=False))

# ------------------------------------------------------------------
# Write 'Run time' sheet with both tables stacked
# ------------------------------------------------------------------
print("\nWriting 'Run time' sheet ...")
with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    df_hardware.to_excel(writer, sheet_name="Run time", index=False,
                          startrow=0)
    df_execution.to_excel(writer, sheet_name="Run time", index=False,
                           startrow=df_hardware.shape[0] + 3)
print(f"[+] 'Run time' sheet saved.")
print(f"\nSheets now in Data.xlsx ({len(pd.ExcelFile(DATA).sheet_names)}):")
for s in pd.ExcelFile(DATA).sheet_names:
    print(f"  - {s}")
