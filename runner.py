import os
import subprocess
import re

configs = [
    # No_SMOTE
    {"model": "LGBC(No_SMOTE)", "opt": "GOA", "acc": 0.892451, "path": r"data\No_SMOTE\model1.npt"},
    {"model": "LGBC(No_SMOTE)", "opt": "BOA", "acc": 0.905142, "path": r"data\No_SMOTE\model1.npt"},
    {"model": "LGBC(No_SMOTE)", "opt": "LBOA", "acc": 0.918933, "path": r"data\No_SMOTE\model1.npt"},
    # SMOTE
    {"model": "LGBC(SMOTE)", "opt": "GOA", "acc": 0.919245, "path": r"data\SMOTE\model1.npt"},
    {"model": "LGBC(SMOTE)", "opt": "BOA", "acc": 0.934812, "path": r"data\SMOTE\model1.npt"},
    {"model": "LGBC(SMOTE)", "opt": "LBOA", "acc": 0.956721, "path": r"data\SMOTE\model1.npt"},
]

script_path = r"MultiClassification(prob_upbdated).py"

with open(script_path, "r", encoding="utf-8") as f:
    content = f.read()

for c in configs:
    # replace config
    content = re.sub(r'model_name = ".*?"', lambda m: f'model_name = "{c["model"]}"', content)
    content = re.sub(r'Accuracy_target = [\d\.]+', lambda m: f'Accuracy_target = {c["acc"]}', content)
    content = re.sub(r'optimizer_name = ".*?"', lambda m: f'optimizer_name = "{c["opt"]}"', content)
    content = re.sub(r'dataPath = r".*?"', lambda m: f'dataPath = r"{c["path"]}"', content)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\n=======================================================")
    print(f"Running {c['model']} + {c['opt']} with target {c['acc']} ...")
    print(f"=======================================================\n")
    subprocess.run(["C:/Python314/python.exe", script_path])
    
print("\nAll 6 optimizer sheets successfully generated!")
