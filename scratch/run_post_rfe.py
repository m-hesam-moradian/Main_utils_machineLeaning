import subprocess
import sys

def run_script(script_path, tag):
    print(f"\n======================================")
    print(f"Running: {script_path} {tag}")
    print(f"======================================")
    try:
        result = subprocess.run([sys.executable, f"d:/ML/analysis/{script_path}", tag], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {script_path}")
        print(e.stdout)
        print(e.stderr)

scripts = [
    "Statistical-analysis/wilcoxon/DataCatcher with probability.py",
    "BS(V2).py",
    "Statistical-analysis/wilcoxon/DataCatcher.py",
    "Statistical-analysis/Statistical_t-test.py",
    "Sensitivity/MorisMethodSensivity class.py",
    "Uncertainty/Entrophy(v2).py"
]

for s in scripts:
    run_script(s, "RFE")

print("\nAll post-processing finished!")
