import pandas as pd

def compute_ofi_table(rmse_data):
    """
    Takes a dictionary of model names mapped to [Train RMSE, Test RMSE],
    returns a DataFrame with Train RMSE, Test RMSE, and OFI.
    """
    rows = []
    for model, (train_rmse, test_rmse) in rmse_data.items():
        ofi = train_rmse / test_rmse if test_rmse > 1e-8 else float("inf")
        rows.append({
            "Model": model.upper(),
            "Train RMSE": round(train_rmse, 4),
            "Test RMSE": round(test_rmse, 4),
            "OFI": round(ofi, 4)
        })

    df_ofi = pd.DataFrame(rows)
    print(df_ofi)
    df_ofi.to_clipboard(index=False)
    return df_ofi

# 🧪 Example input dictionary
rmse_data = {
    "QR": [3.549366209, 3.639222593],
    "QR+DOA": [1.139942432,1.168801417 ],
    "QR+HROA": [2.547598617,2.612094075 ],
    "SF": [2.178857938,5.649078904],
    "SF+DOA": [0.46116903,1.1997198],
    "SF+HROA": [1.396683349,3.633437108],
    "LLAR": [3.886237483,3.978651769 ],
    "LLAR+DOA": [1.619983696,1.658506724],
    "LLAR+HROA": [2.817737599,2.884743079],
}

# ✅ Run and get the table
compute_ofi_table(rmse_data)