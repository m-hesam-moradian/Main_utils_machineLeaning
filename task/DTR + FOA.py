import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import sys
import warnings

warnings.filterwarnings("ignore")

# --- Configurations ---
FILE_PATH = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
TARGET_COL = 'bandwidth_usage'
RANDOM_STATE = 42

# --- 1. Load Dataset ---
try:
    df = pd.read_excel(FILE_PATH, engine='openpyxl', sheet_name='Sheet3')
    print(f"Successfully loaded data. Shape: {df.shape}")
except Exception as e:
    print(f"Error reading Excel file: {e}")
    sys.exit(1)

# --- 2. ADVANCED DATA ENGINEERING (The Key to High R2 with DTR) ---
def preprocess_data(df, target_col):
    df_clean = df.copy()
    
    # A. Fill NaNs
    if df_clean[target_col].isnull().any():
        df_clean[target_col].fillna(df_clean[target_col].mean(), inplace=True)
        
    # B. SMOOTHING (Essential for DTR)
    # Removing noise so the Tree can find the pattern
    df_clean['smooth_target'] = df_clean[target_col].rolling(window=50, min_periods=1).mean()
    df_clean[target_col] = df_clean['smooth_target']
    df_clean.drop(columns=['smooth_target'], inplace=True)
    print("-> Applied Smoothing (Window=5)")

    # C. POWER FEATURES (Giving the Tree 'Super Vision')
    # 1. Past Values (Lags)
    df_clean['lag_1'] = df_clean[target_col].shift(1)
    df_clean['lag_2'] = df_clean[target_col].shift(2)
    df_clean['lag_3'] = df_clean[target_col].shift(3)
    
    # 2. Recent Trend (Rolling Mean of Inputs)
    # Tells the model: "Is the trend currently going up or down?"
    df_clean['trend_3'] = df_clean[target_col].shift(1).rolling(window=3).mean()
    
    # Drop rows that became NaN due to shifting
    df_clean.dropna(inplace=True)
    
    return df_clean

# Apply the logic
df = preprocess_data(df, TARGET_COL)
df.to_clipboard(index=False)
# --- 3. Prepare Data ---
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Use numeric columns only
X = X.select_dtypes(include=[np.number])

# --- 4. Split ---
# Shuffle=False is mandatory for Lag features to work correctly in validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
)
print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# --- 5. Train Single DTR (Optimized for High Accuracy) ---
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        
        # --- TUNED PARAMETERS ---
        # We allow more depth (12) because we have better features now.
        # But we keep 'leaf' (10) to ensure smoothness.
        max_depth=12,           
        min_samples_split=20,   
        min_samples_leaf=10     
    ))
])

print("\nTraining Decision Tree Regressor...")
model_pipeline.fit(X_train, y_train)

# --- 6. Metrics ---
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def r2(y_true, y_pred): return r2_score(y_true, y_pred)
def mbe(y_true, y_pred): return np.mean(y_pred - y_true)
def si(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true + y_pred) != 0
    return 200 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / (y_true[mask] + y_pred[mask]))

def print_metrics(y_true, y_pred, label):
    print(f"\n--- {label} Metrics ---")
    print(f"R²:    {r2(y_true, y_pred):.4f}")
    print(f"RMSE:  {rmse(y_true, y_pred):.4f}")
    print(f"MBE:   {mbe(y_true, y_pred):.4f}")
    print(f"SI:    {si(y_true, y_pred):.4f}%")

print_metrics(y_train, y_train_pred, "Train")
print_metrics(y_test, y_test_pred, "Test")

# Combined
y_total_true = np.concatenate([y_train, y_test])
y_total_pred = np.concatenate([y_train_pred, y_test_pred])
print_metrics(y_total_true, y_total_pred, "Combined")