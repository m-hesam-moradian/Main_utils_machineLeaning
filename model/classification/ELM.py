import pandas as pd
import numpy as np
from hpelm import ELM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.2-Dataset.xlsx", sheet_name="Data_after_KFold")

# Convert target to categorical labels
target_column = "Anomalous Load"
y_raw = df[target_column]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)  # Converts to 0, 1, 2, ...

# One-hot encode target for ELM
num_classes = len(np.unique(y))
y_onehot = np.eye(num_classes)[y]

# Prepare features
X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
X = StandardScaler().fit_transform(X)

# Split data
X_train, X_test, y_train_oh, y_test_oh = train_test_split(X, y_onehot, test_size=0.3, shuffle=False, random_state=42)
y_train = np.argmax(y_train_oh, axis=1)
y_test = np.argmax(y_test_oh, axis=1)

# Initialize and train ELM
elm = ELM(X.shape[1], num_classes)
elm.add_neurons(100, "sigm")
elm.train(X_train, y_train_oh)

# Predict
y_pred_all = np.argmax(elm.predict(X), axis=1)
y_pred_train = np.argmax(elm.predict(X_train), axis=1)
y_pred_test = np.argmax(elm.predict(X_test), axis=1)

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
# df_all.to_clipboard()
df_train.to_clipboard(index=False)
# df_test.to_clipboard(index=False)