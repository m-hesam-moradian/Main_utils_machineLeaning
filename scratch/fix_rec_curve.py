with open('RegressionFunctionModelToExcel.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '    df_rec_curve = pd.DataFrame(\n        {"Epsilon": epsilon, "Accuracy": accuracy, "AUC": ["" for _ in range(len(epsilon))]}\n    )\n    df_rec_curve.loc[0] = [np.nan, np.nan, rec_auc]'

replacement = '    auc_col = [rec_auc] + [np.nan] * (len(epsilon) - 1)\n    df_rec_curve = pd.DataFrame(\n        {"Epsilon": epsilon, "Accuracy": accuracy, "AUC": auc_col}\n    )\n    df_rec_curve.loc[0, "Epsilon"] = np.nan\n    df_rec_curve.loc[0, "Accuracy"] = np.nan'

if target in code:
    code = code.replace(target, replacement)
    print("Replaced target in code!")
else:
    # Try normalizing newlines
    code_norm = code.replace('\r\n', '\n')
    assert target in code_norm, "Target not found even with newline normalization"
    code_norm = code_norm.replace(target, replacement)
    code = code_norm.replace('\n', '\r\n')
    print("Replaced target with CRLF!")

with open('RegressionFunctionModelToExcel.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("build_rec_curve updated successfully!")

