import sys
import subprocess

tag = sys.argv[1] # e.g., "RFE" or "Chi2"

if tag == "RFE":
    better = "KNNC"
    weaker = "RFC"
else:
    better = "RFC"
    weaker = "KNNC"

cmds = [
    # Better model (Slot 1)
    [sys.executable, "MultiClassification(prob_upbdated).py", f"{better}({tag})", "0.0", "NONE", "data/model1.npt", "Recall"],
    # Better model + SOA (Slot 2)
    [sys.executable, "MultiClassification(prob_upbdated).py", f"{better}({tag})", "0.0", "SOA", "data/model2.npt", "Recall"],
    # Weaker model (Slot 4)
    [sys.executable, "MultiClassification(prob_upbdated).py", f"{weaker}({tag})", "0.0", "NONE", "data/model4.npt", "Recall"],
    # Weaker model + SOA (Slot 5)
    [sys.executable, "MultiClassification(prob_upbdated).py", f"{weaker}({tag})", "0.0", "SOA", "data/model5.npt", "Recall"]
]

for cmd in cmds:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
