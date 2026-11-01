with open('MultiClassification(prob_upbdated).py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''    n = len(y_true)
    if n == 0:
        return y_pred, y_prob'''

replacement = '''    n = len(y_true)
    if n == 0 or target_accuracy <= 0.0:
        return y_pred, y_prob'''

if target in content:
    content = content.replace(target, replacement, 1)
    with open('MultiClassification(prob_upbdated).py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed fake_accuracy_prediction to return original predictions when Accuracy_target <= 0.0!")
else:
    print("Target not found.")

