import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# === Load your dataset ===
df = pd.read_excel(r"C:\Users\Sam\Desktop\ML\task\BSS.No.1-Dataset.xlsx", sheet_name="Balanced_Shuffled")

# === Separate features and target ===
X = df.drop(columns=["Anomalous Load"])
y = df["Anomalous Load"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Choose model and parameters ===
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(
#     n_estimators=100,         # Number of trees
#     max_depth=1,           # No limit on tree depth
#     min_samples_split=6,      # Minimum samples to split a node
#     min_samples_leaf=5,       # Minimum samples at a leaf
#     max_features="sqrt",      # Number of features to consider at each split
    
# )

# model = RandomForestClassifier(
#     n_estimators=250,         # Number of trees
#     max_depth=1,           # No limit on tree depth
#     min_samples_split=6,      # Minimum samples to split a node
#     min_samples_leaf=5,       # Minimum samples at a leaf
#     max_features="sqrt",      # Number of features to consider at each split
    
# )
# model = RandomForestClassifier(
#     n_estimators=100,         # Number of trees
#     max_depth=2,           # No limit on tree depth
#     min_samples_split=6,      # Minimum samples to split a node
#     min_samples_leaf=5,       # Minimum samples at a leaf
#     max_features="sqrt",      # Number of features to consider at each split
    
# )




from xgboost import XGBClassifier

model = XGBClassifier(
    # n_estimators=1,         # Number of boosting rounds (trees)
    # max_depth=16,              # Maximum tree depth
    # min_child_weight=10,       # Minimum sum of instance weight (similar to min_samples_leaf)
    # gamma=0,                  # Minimum loss reduction to make a split (optional)
    # subsample=1.0,            # Fraction of samples used per tree
    # colsample_bytree=0.707,   # ~sqrt of features (if you have ~2 features, sqrt ≈ 0.707)
    # use_label_encoder=False,  # Disable legacy encoder
    # eval_metric="logloss",    # Required to avoid warning
)
# === Train and evaluate ===
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.4f}")