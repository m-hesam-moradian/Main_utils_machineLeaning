import os
import subprocess

exporter = "MultiClassification(prob_upbdated).py"

def run_exporter(model, acc, opt, data_path):
    print(f"Exporting {model} {opt} using {data_path}...")
    cmd = ["C:/Python314/python.exe", exporter, model, str(acc), opt, data_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error on {model} {opt}:", res.stderr)
    else:
        print("Success.")

# Scenario 1 (Unbalanced)
# Model 1 = MLR
run_exporter("MLR(Unbalanced)", 0.0, "NONE", "data/model1.npt")
run_exporter("MLR(Unbalanced)", 0.0, "POA", "data/model2.npt")

# Model 2 = QDA
run_exporter("QDA(Unbalanced)", 0.0, "NONE", "data/model4.npt")
run_exporter("QDA(Unbalanced)", 0.0, "POA", "data/model5.npt")

# Scenario 2 (SMOTE-ENC)
# Model 1 = QDA
run_exporter("QDA(SMOTE-ENC)", 0.0, "NONE", "data_S2/model1.npt")
run_exporter("QDA(SMOTE-ENC)", 0.0, "POA", "data_S2/model2.npt")

# Model 2 = MLR
run_exporter("MLR(SMOTE-ENC)", 0.0, "NONE", "data_S2/model4.npt")
run_exporter("MLR(SMOTE-ENC)", 0.0, "POA", "data_S2/model5.npt")

print("All exports completed!")
