import subprocess
import sys

models = [
    ("LGBM", "0.0", "Grid Search", "data/model2.npt"),
    ("LGBM", "0.0", "Bayesian", "data/model3.npt"),
    ("LGBM", "0.0", "DSOA", "data/model4.npt"),
    ("SVC", "0.0", "NONE", "data/model5.npt"),
    ("SVC", "0.0", "Grid Search", "data/model6.npt"),
    ("SVC", "0.0", "Bayesian", "data/model7.npt"),
    ("SVC", "0.0", "DSOA", "data/model8.npt"),
]

for m_name, acc, opt, dp in models:
    full_title = f"{m_name} + {opt}" if opt != "NONE" else m_name
    print(f"\n==========================================")
    print(f"Exporting: {full_title} ({dp})...")
    print(f"==========================================")
    cmd = [
        "C:/Python314/python.exe",
        "MultiClassification(prob_upbdated).py",
        m_name,
        acc,
        opt,
        dp,
        "task/Data.xlsx"
    ]
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        print(f"Error on {full_title}!")
    else:
        print(f"[OK] {full_title} exported successfully!")

print("\n[ALL DONE] All model sheets exported to task/Data.xlsx!")

