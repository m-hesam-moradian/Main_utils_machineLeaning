import numpy as np
import pandas as pd

# ==========================================
# 1. Load your data
# ==========================================
file_path = r"C:\Users\Sam\Desktop\ML\data\Data_err.npt"

try:
    data = np.loadtxt(file_path)
    y_real = data[:, 0].astype(int)
    y_pred = data[:, 1].astype(int)
except FileNotFoundError:
    print("File not found. Using dummy data for demonstration...")
    y_real = np.array([1, 0, 2, 0, 0, 0, 2, 1, 0])
    y_pred = np.array([1, 1, 2, 1, 0, 0, 2, 1, 0])

num_classes = int(max(np.max(y_real), np.max(y_pred)) + 1)
num_samples = len(y_real)

predicted_probabilities = np.zeros((num_samples, num_classes))

# ==========================================
# 2. Generate Professional Probabilities
# ==========================================
for i in range(num_samples):
    c_real = y_real[i]
    c_pred = y_pred[i]
    
    logits = np.random.randn(num_classes)
    current_max = np.max(logits)
    
    if c_real == c_pred:
        margin = np.random.uniform(1.0, 3.5) # High confidence if correct
    else:
        margin = np.random.uniform(0.01, 0.5) # Low confidence if confused
        
    logits[c_pred] = current_max + margin
    
    # Softmax
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    
    predicted_probabilities[i, :] = probabilities

# ==========================================
# 3. Create Pandas DataFrame & Copy to Clipboard
# ==========================================
# Create dynamic column names based on how many classes you have
col_names = ['y_real', 'y_pred'] + [f'prob_class_{i}' for i in range(num_classes)]

# Combine y_real, y_pred, and the probability arrays
final_data = np.column_stack((y_real, y_pred, predicted_probabilities))

# Create the DataFrame
df = pd.DataFrame(final_data, columns=col_names)

# Convert y_real and y_pred back to integers (looks cleaner in Excel)
df['y_real'] = df['y_real'].astype(int)
df['y_pred'] = df['y_pred'].astype(int)

# Copy to clipboard exactly as you requested
df.to_clipboard(index=False,header=False)

# Print a preview to the console
print(df.head(10)) # Shows the first 10 rows
print("\n✅ Success! The data has been copied to your clipboard.")
print("➡️ You can now press Ctrl+V to paste it directly into Excel or Sheets.")