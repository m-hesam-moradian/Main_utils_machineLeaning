# ===================== 1) Load Libraries =====================
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.integrate import simpson   # برای AUC در REC  # برای AUC در REC

# ===================== 2) Read Excel Data =====================

data = np.loadtxt(r"C:\Users\Sam\Desktop\ML\data\Data_err.npt")
y = data[:, 0]
predictData = data[:, 1]



# ===================== 2b) Split Data =====================
# 60% Train, 20% Test, 10% Test_Val, 10% Test_Test (مثال)
train_y, temp_y, train_pred, temp_pred = train_test_split(y, predictData, test_size=0.4, random_state=42)
test_y, test_test_y, test_pred, test_test_pred = train_test_split(temp_y, temp_pred, test_size=0.5, random_state=42)
test_val_y = test_y  # در این مثال test_val همان test_y است
test_val_pred = test_pred

# ===================== 3) Metric Function =====================
def getAllMetric(y, pred):
    y    = np.array(y).flatten()
    pred = np.array(pred).flatten()
    
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2   = r2_score(y, pred)
    mape = np.mean(np.abs((y - pred) / (y + 1e-10))) * 100
    rae  = np.sum(np.abs(y - pred)) / (np.sum(np.abs(y - np.mean(y))) + 1e-10)
    mdape= np.median(np.abs((y - pred) / (y + 1e-10))) * 100
    
    return [rmse, r2, mape, rae, mdape]

# ===================== 4) Calculate Metrics =====================
allMetric      = getAllMetric(y, predictData)
trainMetric    = getAllMetric(train_y, train_pred)
testMetric     = getAllMetric(test_y, test_pred)
testValMetric  = getAllMetric(test_val_y, test_val_pred)
testTestMetric = getAllMetric(test_test_y, test_test_pred)

all_ressss = pd.DataFrame(
    [allMetric, trainMetric, testMetric, testValMetric, testTestMetric],
    columns=['RMSE', 'R2', 'MAPE', 'RAE', 'MDAPE'],
    index=['All', 'Train', 'Test', 'Test_Val', 'Test_Test']
)

print("\n===== Final Metrics Table =====\n")
print(all_ressss)

# ===================== 5) REC Metrics =====================
def REC(y_true , y_pred):
    # initilizing the lists
    Accuracy = []
    
    # initializing the values for Epsilon
    Begin_Range = 0
    End_Range =1.95
    Interval_Size = 0.01
    
    # List of epsilons
    Epsilon = np.arange(Begin_Range , End_Range , Interval_Size)
    
    # Main Loops
    for i in range(len(Epsilon)):
        count = 0.0
        for j in range(len(y_true)):
            if np.linalg.norm(y_true[j] - y_pred[j]) / np.sqrt( np.linalg.norm(y_true[j]) **2 + np.linalg.norm(y_pred[j])**2 ) < Epsilon[i]:
                count = count + 1
        
        Accuracy.append(count/len(y_true))
    
    # Calculating Area Under Curve using Simpson's rule
    AUC = simpson(Accuracy , Epsilon ) / End_Range
        
    # returning epsilon , accuracy , area under curve    
    return Epsilon , Accuracy , AUC

# اجرای REC
# ===================== Execution & Formatting =====================
# Run the updated REC function
Epsilon, Accuracy, AUC = REC(y, predictData)

# Create an empty list/array for the AUC column filled with empty strings
auc_column = [""] * len(Epsilon)
auc_column[0] = AUC  # Place the AUC value only in the very first row

# Create the final structured DataFrame
rec_table = pd.DataFrame({
    'Epsilon': Epsilon,
    'Accuracy': Accuracy,
    'AUC': auc_column
})

# Print the result to the console
print("\n===== REC Result Table =====")
print(rec_table)

# Copy the table to clipboard (index=False hides the row numbers 0, 1, 2...)
rec_table.to_clipboard(index=False)
print("\nREC Table has been successfully copied to your clipboard!")
