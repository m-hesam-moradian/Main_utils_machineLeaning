import pandas as pd
from ctgan import CTGAN
from catboost import CatBoostClassifier

# Load original data
df_raw = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="CTAB-GAN")

# Step 1: Train model on original data
output_features = ['Memory Usage', 'Request Response Time']
target = 'Anomalous Load'

X_train = df_raw[output_features]
y_train = df_raw[target]

catboost_model = CatBoostClassifier(verbose=False)
catboost_model.fit(X_train, y_train)

# Step 2: Apply Scenario 2 changes
df_modified = df_raw.copy()
df_modified['Active Connections'] *= 1.25
df_modified['Bandwidth Utilization'] *= 0.85

# Step 3: Train CGAN to generate Memory Usage and Request Response Time
input_features = ['Active Connections', 'Bandwidth Utilization']
df_input = df_raw[input_features + output_features].dropna()

model = CTGAN(epochs=50, verbose=True)
model.fit(df_input, discrete_columns=[])

# Step 4: Generate synthetic Memory Usage and Request Response Time
synthetic = model.sample(len(df_modified))
df_modified[output_features] = synthetic[output_features]

# Step 5: Predict Anomalous Load using trained model
df_modified[target] = catboost_model.predict(df_modified[output_features])

# Step 6: Copy final result to clipboard
df_modified.to_clipboard(index=False)
print("\nSCENARIO 2 DATA COPIED TO CLIPBOARD!")
print(df_modified[target].value_counts())