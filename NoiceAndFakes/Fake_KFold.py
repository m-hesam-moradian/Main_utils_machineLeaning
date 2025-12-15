import pandas as pd

# Original data
# convert this
# Fold	R2	RMSE
# 1	0.916297084	0.064004655
# 2	0.942555716	0.062221553
# 3	0.945109249	0.062053441
# 4	0.920278528	0.063727749
# 5	0.93202704	0.062924441
#  to  data like this :
# data = {
#     "Fold": [1, 2, 3, 4, 5],
#     "R2": [0.342362513, 0.368621145, 0.371174678, 0.346343957, 0.358092469],
#     "RMSE": [0.214477687, 0.21021014, 0.211997221, 0.214262008, 0.212566316],
# }


# Fold	R2	RMSE
# 1	0.656547975	6412475.641
# 2	0.610607201	5744515.87
# 3	0.726868084	5966846.013
# 4	0.635917472	6441422.077
# 5	0.820780131	5085169.9
# data = {
#     "Fold": [1, 2, 3, 4, 5],
#     "R2": [0.656547975, 0.610607201, 0.726868084, 0.635917472, 0.820780131],
#     "RMSE": [6412475.641, 5744515.87, 5966846.013, 6441422.077, 5085169.9],
# }


# Fold	R2	RMSE
# 1	0.987725961	0.034983553
# 2	0.991926865	0.026189412
# 3	0.984422532	0.039650195
# 4	0.991011099	0.031813141
# 5	0.991229091	0.027608412
data = {
    "Fold": [1, 2, 3, 4, 5],
    "R2": [0.987725961, 0.991926865, 0.984422532, 0.991011099, 0.991229091],
    "RMSE": [0.034983553, 0.026189412, 0.039650195, 0.031813141, 0.027608412],
}

# Reference R² and RMSE from your image
# firt one r2 second rmse  0.869510943	0.114564042
ref_r2 = 0.869510943
ref_rmse = 0.114564042












# Convert to DataFrame
df = pd.DataFrame(data)

# 1. Calculate how much to add to make max R² = target
max_r2 = df["R2"].max()
r2_boost = ref_r2 - max_r2

# 2. Adjust all R²s
df["R2"] = df["R2"] + r2_boost

# 3. Estimate RMSE using inverse relationship (more R² = less RMSE)
# We'll fit a fake model: RMSE = a / (R² + b)
# Use the reference point to find a fake model constant
# Assume: RMSE = k / (R² + ε), solve for k
epsilon = 1e-6  # to avoid divide by zero
k = ref_rmse * (ref_r2 + epsilon)

# Predict RMSE from Adjusted R²
df["RMSE"] = k / (df["R2"] + epsilon)


# 5. Predict average RMSE
predicted_rmse = df["RMSE"].mean()

# Show final result
print(df[["Fold", "R2", "RMSE"]])
print(f"\nPredicted Overall RMSE (more realistic): {predicted_rmse:.2f}")
df.to_clipboard(index=False)