import pandas as pd
import numpy as np

# -------------------------
# YOUR FUNCTIONS (UNCHANGED)
# -------------------------
def detect_outliers(data, new_data):
    threshold = 1.5
    data = np.array(data)

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data must contain only numbers")

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)

    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    dt = new_data.drop(new_data.index[outliers])

    return outliers, dt


def delete_iqr_loop(target, epoch, data):
    dt = data
    print("DT1", dt.shape[0])
    outlier_list = []
    outliers_shape = 0
    convergens = [[], []]

    for i in range(epoch):
        outliers, n_data = detect_outliers(dt[target], dt)
        dt = pd.DataFrame(n_data)
        outlier_list.append(outliers)

        if outliers_shape == len(outliers):
            break
        else:
            outliers_shape = len(outliers)

        print("DT", dt.shape)
        print("outliers", outliers.shape)

        convergens[0].append(i + 1)
        convergens[1].append(outliers_shape)

    return dt, outlier_list, convergens


# -------------------------
# CONFIGURATION
# -------------------------
input_file = r'C:\Users\Sam\Desktop\ML\task\Data.xlsx'
input_sheet = 'DATA_Shuffled'


output_sheet = 'IQR'
report_sheet = 'IQR_Report'

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_excel(input_file, sheet_name=input_sheet)
target_column = df.columns[-1]   

# -------------------------
# RUN IQR
# -------------------------
df_cleaned, outlier_list, convergens = delete_iqr_loop(
    target=target_column,
    epoch=1,
    data=df
)

# -------------------------
# REPORT
# -------------------------
total_removed = len(df) - len(df_cleaned)

report_data = [
    {
        "Feature / Detail": target_column,
        "Rows Triggered For Removal": total_removed
    },
    {
        "Feature / Detail": "Iterations",
        "Rows Triggered For Removal": len(convergens[0])
    },
    {
        "Feature / Detail": "-----------------------------",
        "Rows Triggered For Removal": "---"
    },
    {
        "Feature / Detail": "Total Outlier Rows Removed",
        "Rows Triggered For Removal": total_removed
    },
    {
        "Feature / Detail": "Original Row Count",
        "Rows Triggered For Removal": len(df)
    },
    {
        "Feature / Detail": "Remaining Row Count",
        "Rows Triggered For Removal": len(df_cleaned)
    }
]

report_df = pd.DataFrame(report_data)

# -------------------------
# SAVE TO EXCEL
# -------------------------
with pd.ExcelWriter(
    input_file,
    mode='a',
    engine='openpyxl',
    if_sheet_exists='replace'
) as writer:

    df_cleaned.to_excel(
        writer,
        sheet_name=output_sheet,
        index=False
    )

    report_df.to_excel(
        writer,
        sheet_name=report_sheet,
        index=False
    )

df_cleaned.to_clipboard(index=False)

print(
    f"✅ Data saved to '{output_sheet}' "
    f"and Report saved to '{report_sheet}'"
)