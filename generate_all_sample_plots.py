import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as plt_sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Grid Search vs Random Search
# ==========================================
def generate_grid_vs_random():
    def f(x, y):
        # Create a contour surface with two local minima
        return np.exp(-(x**2 + y**2)/10) * np.sin(x) * np.cos(y)

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Grid Search
    grid_x = np.linspace(-2.5, 2.5, 7)
    grid_y = np.linspace(-2.5, 2.5, 7)
    GX, GY = np.meshgrid(grid_x, grid_y)
    axes[0].contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    axes[0].scatter(GX, GY, color='red', s=20, label='Grid points')
    axes[0].set_title("Grid Search")
    axes[0].set_xlabel("Parameter 1")
    axes[0].set_ylabel("Parameter 2")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].legend()

    # Random Search
    np.random.seed(42)
    RX = np.random.uniform(-2.8, 2.8, 49)
    RY = np.random.uniform(-2.8, 2.8, 49)
    axes[1].contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    axes[1].scatter(RX, RY, color='red', s=20, label='Random points')
    axes[1].set_title("Random Search")
    axes[1].set_xlabel("Parameter 1")
    axes[1].set_ylabel("Parameter 2")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(r"d:\ML\grid_vs_random.png", dpi=300)
    print("Saved grid_vs_random.png")

# ==========================================
# 2. Gaussian Process Expected Improvement
# ==========================================
def generate_gp_ei():
    # Make the target objective function smooth
    def objective(x):
        return -(np.sin(x * 6) + x * 0.5 + np.cos(x * 3) * 0.5)

    X_grid = np.linspace(0, 1000, 200).reshape(-1, 1)
    y_grid = objective(X_grid / 100)

    # Initial points
    X_sample = np.array([50, 450]).reshape(-1, 1)
    y_sample = objective(X_sample / 100)

    kernel = Matern(nu=2.5, length_scale=200)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=5)

    plt.style.use('ggplot')
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))

    for i in range(5):
        gp.fit(X_sample, y_sample)
        mu, std = gp.predict(X_grid, return_std=True)
        
        # Expected Improvement (we are minimizing cross entropy, so minimizing y)
        # But EI is usually derived for maximization. We'll flip y to maximize
        mu_sample_opt = np.min(y_sample)
        with np.errstate(divide='ignore'):
            imp = mu_sample_opt - mu - 0.01
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

        # Subplot 1: GP Mean and Variance
        ax1 = axes[i, 0]
        ax1.plot(X_grid, mu, 'C1-', label='GP mean') # C1 is blue in ggplot
        ax1.fill_between(X_grid.ravel(), mu.ravel() - 1.96*std, mu.ravel() + 1.96*std, color='C1', alpha=0.2)
        ax1.plot(X_sample, y_sample, 'ro', label='Previous steps')
        ax1.plot(X_sample[-1], y_sample[-1], 'r*', markersize=10, label='Last step')
        ax1.set_title(f"Gaussian Process\nafter {i+2} iterations", fontsize=10)
        ax1.set_ylabel("Cross entropy\n(the smaller the better)", fontsize=9)
        ax1.set_xlabel("Number of hidden units", fontsize=9)
        ax1.legend(fontsize=8)
        
        # Subplot 2: Expected Improvement
        ax2 = axes[i, 1]
        ax2.plot(X_grid, ei, 'C0-') # C0 is red in ggplot
        next_idx = np.argmax(ei)
        ax2.plot(X_grid[next_idx], ei[next_idx], 'C1*', markersize=10, label='Next step')
        ax2.set_title(f"Expected Improvement\nafter {i+2} iterations", fontsize=10)
        ax2.set_xlabel("Number of hidden units", fontsize=9)
        ax2.legend(fontsize=8)
        
        # Add next point
        X_sample = np.vstack([X_sample, X_grid[next_idx]])
        y_sample = np.vstack([y_sample, objective(X_grid[next_idx] / 100)])

    plt.tight_layout()
    plt.savefig(r"d:\ML\gaussian_process_ei.png", dpi=300)
    print("Saved gaussian_process_ei.png")

# ==========================================
# 3. Tabular Copula Sensitivity Plot (Fixed Overlap)
# ==========================================
def couples_sensitivity_analysis(model, X, y, feature_pairs, perturbation=0.1):
    original_predictions = model.predict(X)
    original_score = accuracy_score(y, original_predictions)
    sensitivity_report = []

    for feature_1, feature_2 in feature_pairs:
        X_perturbed = X.copy()
        X_perturbed[feature_1] *= 1 + perturbation
        X_perturbed[feature_2] *= 1 + perturbation

        perturbed_predictions = model.predict(X_perturbed)
        perturbed_score = accuracy_score(y, perturbed_predictions)
        # Using absolute diff or drop in accuracy
        sensitivity = abs(perturbed_score - original_score)

        sensitivity_report.append({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "sensitivity": sensitivity,
        })

    return pd.DataFrame(sensitivity_report)

def generate_copula():
    data_file = r"d:\ML\task\Data.xlsx"
    df = pd.read_excel(data_file, sheet_name="Selected_Data_RFE")
    
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    features = X.columns
    
    model = ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    feature_pairs = [(features[i], features[j]) for i in range(len(features)) for j in range(len(features))]
    
    copula = couples_sensitivity_analysis(model, X, y, feature_pairs, perturbation=0.1)
    tabular_copula = copula.pivot(index='feature_1', columns='feature_2', values='sensitivity')
    
    # Increase figsize and reduce font size to prevent overlapping numbers
    plt.figure(figsize=(16, 12))
    plt_sns.heatmap(tabular_copula, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, square=True, annot_kws={"size": 9})
    plt.title("Tabular Copula Sensitivity Plot (Classification - Accuracy Drop)", fontsize=16)
    plt.xlabel("Feature 2 (Perturbed by 10%)", fontsize=12)
    plt.ylabel("Feature 1 (Perturbed by 10%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(r"d:\ML\copula_heatmap_readable.png", dpi=300)
    print("Saved copula_heatmap_readable.png")

if __name__ == "__main__":
    generate_grid_vs_random()
    generate_gp_ei()
    generate_copula()
