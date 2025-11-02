import pandas as pd
from ctgan import CTGAN
from catboost import CatBoostClassifier

# Load original data
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="Scenario 1 Data")

# Step 1: Apply 20% increase to Packet Size and Transmission Rate
df['Packet Size'] *= 1.2
df['Transmission Rate'] *= 1.2

# Step 2: Prepare CGAN input
input_features = ['Packet Size', 'Transmission Rate']
output_features = ['Latency', 'Bandwidth Utilization']
target = 'Anomalous Load'

df_input = df[input_features + output_features]

# Train CGAN to generate Latency and Bandwidth Utilization
model = CTGAN(epochs=50, verbose=True)
model.fit(df_input, discrete_columns=[])  # Assuming all are continuous

# Generate synthetic Latency and Bandwidth Utilization
synthetic = model.sample(len(df), conditions=df[input_features])

# Step 3: Train CatBoostClassifier on original data
X_train = df[output_features]
y_train = df[target]

catboost_model = CatBoostClassifier(verbose=False)
catboost_model.fit(X_train, y_train)

# Step 4: Predict Anomalous Load on synthetic data
X_new = synthetic[output_features]
synthetic[target] = catboost_model.predict(X_new)

# Step 5: Combine inputs and predictions
final_df = pd.concat([df[input_features].reset_index(drop=True), synthetic], axis=1)

# Copy to clipboard
final_df.to_clipboard(index=False)
print("\nNEW SCENARIO DATA COPIED TO CLIPBOARD!")
print(final_df[target].value_counts())