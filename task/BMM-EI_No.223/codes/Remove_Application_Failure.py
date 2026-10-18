"""
Step 1b - Remove Application_Failure sample (class with only 1 row)
==================================================================
The 'System Issue Type' target has 5 classes, but the 'Application_Failure'
class contains only 1 sample, which is incompatible with StratifiedKFold
(n_splits=5 requires >=5 samples per class).

Action:
  - Drop the single Application_Failure row from Encoded_Data
  - Overwrite the Encoded_Data sheet in Data.xlsx
  - Log the removal in REMOVED_SAMPLE_LOG.md
"""
import os
import pandas as pd

DATA = r"/home/z/my-project/task_223/Data.xlsx"
LOG_FILE = r"/home/z/my-project/task_223/REMOVED_SAMPLE_LOG.md"

# ------------------------------------------------------------------
# 1. Load Encoded_Data
# ------------------------------------------------------------------
print("Loading 'Encoded_Data' sheet ...")
df = pd.read_excel(DATA, sheet_name="Encoded_Data")
print(f"  Before: shape={df.shape}")

target_col = df.columns[-1]
print(f"\n  Target '{target_col}' class distribution BEFORE removal:")
print(df[target_col].value_counts().sort_index().to_string())

# ------------------------------------------------------------------
# 2. Identify the Application_Failure sample
#    Encoded class 0 == Application_Failure (smallest class, 1 row)
# ------------------------------------------------------------------
before_count = (df[target_col] == 0).sum()
print(f"\n  Application_Failure (encoded 0) samples found: {before_count}")

# Drop all rows where target == 0 (the Application_Failure class)
removed_rows = df[df[target_col] == 0].copy()
df_clean = df[df[target_col] != 0].reset_index(drop=True)
after_count = (df_clean[target_col] == 0).sum()
print(f"  Application_Failure samples after removal: {after_count}")

# Relabel classes to be contiguous (0..3) so sklearn classifiers work cleanly.
# Original mapping after dropping class 0:
#   1 (Network_Failure)          -> 0
#   2 (Normal_Operation)         -> 1
#   3 (Performance_Degradation)   -> 2
#   4 (Resource_Exhaustion)      -> 3
remap = {1: 0, 2: 1, 3: 2, 4: 3}
df_clean[target_col] = df_clean[target_col].map(remap).astype(int)

print(f"\n  Target '{target_col}' class distribution AFTER removal + relabel:")
print(df_clean[target_col].value_counts().sort_index().to_string())
print(f"\n  After: shape={df_clean.shape}")

# ------------------------------------------------------------------
# 3. Overwrite the Encoded_Data sheet with the cleaned version
# ------------------------------------------------------------------
print("\nSaving cleaned 'Encoded_Data' sheet ...")
with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    df_clean.to_excel(writer, sheet_name="Encoded_Data", index=False)
print(f"[+] Encoded_Data overwritten in {DATA}")

# ------------------------------------------------------------------
# 4. Write REMOVED_SAMPLE_LOG.md
# ------------------------------------------------------------------
log_text = f"""# Removed Sample Log

## BMM-EI No. 223 - Application_Failure Sample Removal

### Reason
The `System Issue Type` target column originally contained 5 classes:
- `Normal_Operation`           : 7,620 samples
- `Performance_Degradation`    : 2,183 samples
- `Resource_Exhaustion`        : 1,245 samples
- `Network_Failure`            :   951 samples
- `Application_Failure`        :     **1 sample**  <-- problematic

A single-sample class is incompatible with `StratifiedKFold(n_splits=5)`,
which requires at least `n_splits` (5) samples per class. The single
`Application_Failure` row also cannot contribute meaningfully to model
training, statistical analysis, or sensitivity/uncertainty computations.

### Action Taken
- The single `Application_Failure` row was removed from the `Encoded_Data` sheet.
- Remaining classes were relabeled to contiguous integers `0..3`:

| New label | Original label              | Count |
|-----------|-----------------------------|-------|
| 0         | Network_Failure             |   951 |
| 1         | Normal_Operation            | 7,620 |
| 2         | Performance_Degradation     | 2,183 |
| 3         | Resource_Exhaustion         | 1,245 |

### Result
- Encoded_Data shape: **11,999 x 32** (down from 12,000 x 32)
- All downstream steps (Chi-Square, VIF, K-Fold CV, model exports,
  Brier decomposition, t-test, Friedman test, ANOVA sensitivity,
  Entropy uncertainty) will operate on the 11,999-row, 4-class dataset.

### Files affected
- `Data.xlsx` sheet `Encoded_Data` (overwritten)
- This log file (`REMOVED_SAMPLE_LOG.md`)
"""

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(log_text)
print(f"[+] Removal log written to {LOG_FILE}")

# ------------------------------------------------------------------
# 5. Final state
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Final Encoded_Data summary:")
print(f"  Shape: {df_clean.shape}")
print(f"  Sheets in Data.xlsx: {pd.ExcelFile(DATA).sheet_names}")
print("=" * 60)
