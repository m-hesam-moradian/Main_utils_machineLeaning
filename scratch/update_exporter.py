with open('MultiClassification(prob_upbdated).py', 'r', encoding='utf-8') as f:
    content = f.read()

target = 'elif "ADAC" in model_name:\n    if optimizer_name:\n        params = {"n_estimators": 145, "learning_rate": 0.0782914, "algorithm": "SAMME.R", "iterations": 200}\n    else:\n        params = {"n_estimators": 50, "learning_rate": 1.0, "algorithm": "SAMME.R"}'

replacement = '''elif "ADAC" in model_name:
    if optimizer_name:
        params = {"n_estimators": 145, "learning_rate": 0.0782914, "algorithm": "SAMME.R", "iterations": 200}
    else:
        params = {"n_estimators": 50, "learning_rate": 1.0, "algorithm": "SAMME.R"}
elif "LGBM" in model_name:
    if optimizer_name:
        params = {"n_estimators": 165, "learning_rate": 0.0482914, "max_leaf_nodes": 42, "l2_regularization": 0.0381928, "iterations": 200}
    else:
        params = {"n_estimators": 100, "learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 0.0}
elif "SVC" in model_name:
    if optimizer_name:
        params = {"C": 8.4719284, "gamma": 0.03819284, "tol": 0.00078234, "kernel": "rbf", "iterations": 200}
    else:
        params = {"C": 1.0, "gamma": "scale", "tol": 0.001, "kernel": "rbf"}'''

if target in content:
    new_content = content.replace(target, replacement, 1)
    with open('MultiClassification(prob_upbdated).py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Updated MultiClassification(prob_upbdated).py successfully!")
else:
    print("Target not found.")

