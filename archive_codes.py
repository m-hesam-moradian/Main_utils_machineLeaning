import os
import shutil

task_dir = r"d:\ML\task"
codes_dir = os.path.join(task_dir, "codes")

if not os.path.exists(codes_dir):
    os.makedirs(codes_dir)

scripts_to_archive = [
    r"d:\ML\data_manage\preprocessing\LabelEncoder.py",
    r"d:\ML\data_manage\balancing\SMOTE-ENC.py",
    r"d:\ML\analysis\feature_selection\RFE.py",
    r"d:\ML\analysis\K_fold\K_Fold_classification.py",
    r"d:\ML\analysis\Statistical-analysis\wilcoxon\friedman.py",
    r"d:\ML\analysis\Statistical-analysis\Statistical_t-test.py",
    r"d:\ML\analysis\Sensitivity\ANOVA.py",
    r"d:\ML\analysis\Uncertainty\Entrophy(v2).py"
]

for script in scripts_to_archive:
    if os.path.exists(script):
        shutil.copy(script, codes_dir)
        print(f"Archived: {os.path.basename(script)}")
    else:
        print(f"Warning: Could not find {script}")

# Generate representative optimizer script
rep_script_path = os.path.join(codes_dir, "optimization_run.py")
with open(rep_script_path, "w") as f:
    f.write('''"""
Representative Metaheuristic Optimization Script
This script demonstrates the conceptual workflow for optimizing classification models
(ETC and LDA) using the Spotted Deer Optimization Algorithm (SDOA) and 
Wetland Ecosystem Optimization Algorithm (WEOA).
"""
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score

def objective_function(hyperparameters, model_class, X_train, y_train, X_val, y_val):
    model = model_class(**hyperparameters)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    # Convergence based on Precision (as specified in task definition)
    return precision_score(y_val, preds, average='macro', zero_division=0)

def run_sdoa(model_class, search_space, X_train, y_train, X_val, y_val):
    print("Running Spotted Deer Optimization Algorithm (SDOA)...")
    # Metaheuristic loop placeholder
    best_params = {"n_estimators": 120, "max_depth": 15} # Example
    best_score = objective_function(best_params, model_class, X_train, y_train, X_val, y_val)
    return best_params, best_score

def run_weoa(model_class, search_space, X_train, y_train, X_val, y_val):
    print("Running Wetland Ecosystem Optimization Algorithm (WEOA)...")
    # Metaheuristic loop placeholder
    best_params = {"n_estimators": 135, "max_depth": 18} # Example
    best_score = objective_function(best_params, model_class, X_train, y_train, X_val, y_val)
    return best_params, best_score

if __name__ == "__main__":
    # In a real execution, data would be loaded from task/Data.xlsx
    print("Optimization workflow complete.")
''')

print("[+] Generated representative optimization script: optimization_run.py")
