# Simulate and generate a detailed runtime performance report for the following classification models and optimizers on the specified device and dataset:

# 📌 Models:
# Extra Tree classification (ETC), Gaussian Boosting classification (GBC), and Random Forest Classification (RFC).

# 📌 Optimizers: Haze Optimization Algorithm (HOA) and Perfumer optimization algorithm (POA).

# 📌 Device Specifications:
# - Device Name: hesam
# - CPU: Intel Core i3-1115G4 @ 3.00GHz
# - RAM: 12.0 GB (11.7 GB usable)
# - System Type: 64-bit OS, x64-based processor

# 📌 Dataset Characteristics:
# - Classification task with 1000 samples and 8 features
# - Columns: CPU_Usage (%), RAM_Usage (MB), Disk_IO (MB/s), Network_IO (MB/s), Priority, VM_ID, Execution_Time (s), Target

# 📌 Simulation Requirements:
# 1. Report runtime metrics for each model **without optimization**.
# 2. Report runtime metrics for each model **with each optimizer applied**.
# 3. For each run, log the following:
#    - Execution_Time (s)
#    - CPU_Usage (%)
#    - RAM_Usage (MB)
#    - Disk_IO (MB/s)
#    - Network_IO (MB/s)
#    - Priority level
#    - VM_ID
#    - Accuracy or classification score (if applicable)

# 📌 Output Format:
# - Tabular report with rows for each model-optimizer pair
# - Separate section for baseline (non-optimized) model performance
# - Include summary of best-performing combination by runtime and resource efficiency

# 📌 Constraints:
# - Simulate within realistic bounds of the device specs
# - Ensure reproducibility by logging all parameters and random seeds
