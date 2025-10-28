import pandas as pd

# Load your Excel data
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="BSS.No.1-Target 1")

# Apply transformations directly to original columns
# df["Packet Size"] = df["Packet Size"] * 1.20
# df["Transmission Rate"] = df["Transmission Rate"] * 1.20

# df["Active Connections"] = df["Active Connections"] * 1.25
# df["Bandwidth Utilization"] = df["Bandwidth Utilization"] * 0.85

df["Auth Failures"] = df["Auth Failures"] * 1.50
df["IDS Alerts"] = df["IDS Alerts"] * 1.50


# Copy to clipboard
df.to_clipboard(index=False)