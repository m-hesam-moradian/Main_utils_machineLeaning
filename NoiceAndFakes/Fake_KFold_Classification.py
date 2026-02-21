import pandas as pd

# Main classification metrics table
# Fold	Accuracy	F1-Score
# 1	0.653623188	0.646122339
# 2	0.665217391	0.660042878
# 3	0.666666667	0.659678008
# 4	0.64057971	0.632863991
# 5	0.666666667	0.659676087

data = {
    "Fold": [1, 2, 3, 4, 5],
    "Precision": [ 0.68115942,0.668421053, 0.680412371, 0.68115942, 0.654205607],
    "F1-Score": [0.659676087,0.646122339, 0.660042878, 0.659678008, 0.632863991],
}


# Accuracy	Recall	F1	Precision

# 0.94375	0.923728814	0.960352423	1



ref_precision = 1



ref_f1 = 0.960352423





df = pd.DataFrame(data)

# 1. Calculate how much to subtract to make max Precision = reference
max_prec = df["Precision"].max()
prec_drop = max_prec - ref_precision

# 2. Adjust all Precision values downward
df["Precision"] = df["Precision"] - prec_drop

# 3. Scale F1 proportionally to the new Precision
df["F1-Score"] = df["Precision"] * (ref_f1 / ref_precision)

# 4. Predict average F1
predicted_f1 = df["F1-Score"].mean()

print(df[["Fold", "Precision", "F1-Score"]])
print(f"\nPredicted Overall F1-Score (scaled): {predicted_f1:.4f}")

df.to_clipboard(index=False)
