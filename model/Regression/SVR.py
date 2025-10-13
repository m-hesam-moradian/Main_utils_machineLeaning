import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and prepare data
df = pd.read_excel(
    r"D:\ML\Main_utils\task\EL. No 6. Allocated bandwidth- SVR-ENR-SCO-POA-GGO-DATA.xlsx",
    sheet_name="String_labelEncoded",
)
y = df["allocated_bandwidth"].astype(float)
X = pd.get_dummies(df.drop(columns=["allocated_bandwidth"]), drop_first=True)

# Scale features and split
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, shuffle=False, random_state=42
)

# Train SVR model
model = SVR()
model.fit(X_train, y_train)

# Predict
y_pred_all = model.predict(X_scaled)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Split test predictions
mid = len(y_test) // 2
sets = [
    ("All", y, y_pred_all),
    ("Train", y_train, y_pred_train),
    ("Test", y_test, y_pred_test),
    ("Value", y_test[:mid], y_pred_test[:mid]),
    ("Test-Value", y_test[mid:], y_pred_test[mid:]),
]

# Compute metrics
metrics_df = pd.DataFrame(
    [
        {
            "Set": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }
        for name, y_true, y_pred in sets
    ]
)

# Print results
print("\n📊 Performance Metrics Table (SVR):")
print(metrics_df)
