import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

# Step 1: Load data
sheet_name = "LabelEncoded"
data = pd.read_excel(
    r"D:\ML\Main_utils\task\WA_Fn-UseC_-HR-Employee-Attrition.xlsx",
    sheet_name=sheet_name,
)

X = data.drop("Attrition", axis=1)
y = data["Attrition"]

feature_names = X.columns.tolist()

# مدل با 10-fold cross-validation
model = LinearDiscriminantAnalysis()
model.fit(X, y)

# ارزیابی مدل با 10-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(f"10-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# انتخاب فیچرها با threshold
importances = np.abs(model.coef_[0])  # Absolute values of coefficients
threshold = 0.1  # Define threshold for feature selection
selected_features = [name for name, score in zip(feature_names, importances) if score > threshold]

# ایجاد دیتاست جدید با فیچرهای انتخاب‌شده و ستون هدف
new_data = data[selected_features + ["Attrition"]]
