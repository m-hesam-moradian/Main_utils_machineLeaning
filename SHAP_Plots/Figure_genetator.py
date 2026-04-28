import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 1. Input the data directly from your image
# Rows: True labels | Columns: Predicted labels
data = np.array([
    [489, 0, 7, 0, 0, 0],
    [15, 426, 30, 0, 0, 0],
    [2, 1, 417, 0, 0, 0],
    [0, 3, 0, 410, 78, 0],
    [0, 0, 0, 110, 422, 0],
    [0, 27, 0, 0, 0, 510]
])

# 2. Define the labels for the axes
labels = ['Walking', 'Walking Up', 'Walking Down', 'Sitting', 'Standing', 'Laying']

# 3. Create a DataFrame for easier plotting
df_cm = pd.DataFrame(data, index=labels, columns=labels)

# 4. Initialize the figure
plt.figure(figsize=(10, 8))

# 5. Create the Heatmap
# annot=True: displays the numbers in the cells
# fmt='d': displays numbers as integers (not scientific notation)
# cmap='Blues': replicates the blue color scale from your reference
# linewidths/linecolor: creates the black border around each cell
ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', 
                 linewidths=1, linecolor='black', 
                 cbar_kws={'label': 'Number of Samples'})

# 6. Final Polish (Titles and Axis Labels)
plt.title('Confusion Matrix: Federated CNN on Global Test Set', fontweight='bold', fontsize=14, pad=20)
plt.ylabel('True Physical Activity (Actual)', fontweight='bold', fontsize=12)
plt.xlabel('Predicted Physical Activity (AI Guess)', fontweight='bold', fontsize=12)

# Rotate x-labels so they don't overlap
plt.xticks(rotation=45, ha='right')

# 7. Save and Show
plt.tight_layout()
plt.savefig('Confusion_Matrix_Research.png', dpi=300) # 300 DPI is standard for articles
plt.show()