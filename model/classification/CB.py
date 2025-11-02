import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and prepare data
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="Data_after_KFold")

# Encode target labels
target_column = "Anomalous Load"

y_raw = df[target_column]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)  # Converts to 0, 1, 2, ...

# Prepare features
X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
X = StandardScaler().fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

# Initialize and train CatBoostClassifier
model = CatBoostClassifier()
model.fit(X_train, y_train)

# Predict
y_pred_all = model.predict(X)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
mid = len(y_test) // 2
sets = [("All", y, y_pred_all), ("Train", y_train, y_pred_train), ("Test", y_test, y_pred_test),
        ("Value", y_test[:mid], y_pred_test[:mid]), ("Test-Value", y_test[mid:], y_pred_test[mid:])]

df_metrics = pd.DataFrame([{ "Set": s,
                             "Accuracy": accuracy_score(y_t, y_p),
                             "Precision": precision_score(y_t, y_p, average="weighted"),
                             "Recall": recall_score(y_t, y_p, average="weighted"),
                             "F1": f1_score(y_t, y_p, average="weighted")} for s, y_t, y_p in sets])

print(df_metrics)

# Output predictions
df_all = pd.DataFrame({"y_real": label_encoder.inverse_transform(y), "y_pred": label_encoder.inverse_transform(y_pred_all)})
df_train = pd.DataFrame({"y_real": label_encoder.inverse_transform(y_train), "y_pred": label_encoder.inverse_transform(y_pred_train)})
df_test = pd.DataFrame({"y_real": label_encoder.inverse_transform(y_test), "y_pred": label_encoder.inverse_transform(y_pred_test)})

# Optional: Export to clipboard or Excel
df_all.to_clipboard()
# df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)