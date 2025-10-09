import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel(
    r"D:\ML\Main_utils\task\Resource_utilization.xlsx", sheet_name="Copula"
)

# Group the data by 'feature_1'
g_data = df.groupby(df["feature_1"])

# Create a list to store the mean values
mean_values_list = []

# Iterate over the groups and calculate the mean of numerical columns
for v, groupD in g_data:
    print("v", v)
    print("groupD", groupD)
    # Select only numerical columns
    numerical_cols = groupD.select_dtypes(include=[int, float])
    # Calculate the mean of numerical columns
    mean_values = numerical_cols.mean()
    # Append the mean values to the list
    mean_values_list.append(mean_values.to_frame().T)

# Concatenate the mean values into a single DataFrame
new_data = pd.concat(mean_values_list, ignore_index=True)

# Reset the index of the new_data DataFrame
new_data.reset_index(inplace=True)

# Rename the index column to 'feature_1'
new_data.rename(columns={"index": "feature_1"}, inplace=True)

# Print the new dataset
print(new_data.head())
