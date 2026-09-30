import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

def objective(x):
    # Mock objective function representing cross-entropy with respect to hidden units
    # We want to mimic the shape in the user's image (minimum around 800-900)
    # The image shows a curve with a minimum around 850
    return (0.1 * np.sin(x / 100.0) + 0.2 * ((x - 850) / 500.0)**2 + 0.05).ravel()

def expected_improvement(X_grid, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X_grid, return_std=True)
    mu_sample_opt = np.min(Y_sample)
    
    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei, mu, sigma

def plot_bo():
    X_grid = np.linspace(0, 1000, 1000).reshape(-1, 1)
    
    # Initialize with 2 points
    X_sample = np.array([50, 450]).reshape(-1, 1)
    Y_sample = objective(X_sample)
    
    # Kernel for GP
    kernel = Matern(length_scale=200, nu=2.5)
    
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    
    for i in range(5):
        iteration = i + 2
        
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=10)
        gpr.fit(X_sample, Y_sample)
        
        ei, mu, sigma = expected_improvement(X_grid, X_sample, Y_sample, gpr)
        
        # Left plot: GP
        ax_gp = axes[i, 0]
        ax_gp.plot(X_grid, mu, 'b-', label='GP mean', color='#1f77b4')
        ax_gp.fill_between(X_grid.ravel(), mu - 1.96*sigma, mu + 1.96*sigma, alpha=0.3, color='#1f77b4')
        ax_gp.plot(X_sample[:-1], Y_sample[:-1], 'ro', markersize=6, label='Previous steps', color='#d62728')
        ax_gp.plot(X_sample[-1], Y_sample[-1], 'r*', markersize=10, label='Last step', color='#d62728')
        
        # Find next point
        next_idx = np.argmax(ei)
        next_x = X_grid[next_idx]
        ax_gp.plot(next_x, mu[next_idx], 'bo', markersize=6, color='#1f77b4')
        
        ax_gp.set_title(f"Gaussian Process\nafter {iteration} iterations", fontsize=11)
        ax_gp.set_xlabel("Number of hidden units")
        ax_gp.set_ylabel("Cross entropy\n(the smaller the better)")
        ax_gp.legend(loc='upper right')
        
        # Right plot: EI
        ax_ei = axes[i, 1]
        ax_ei.plot(X_grid, ei, 'r-', color='#e74c3c')
        ax_ei.plot(next_x, ei[next_idx], 'b*', markersize=10, label='Next step', color='#1f77b4')
        
        ax_ei.set_title(f"Expected Improvement\nafter {iteration} iterations", fontsize=11)
        ax_ei.set_xlabel("Number of hidden units")
        ax_ei.legend(loc='upper left')
        
        # Add next point for next iteration
        X_sample = np.vstack((X_sample, next_x))
        Y_sample = np.append(Y_sample, objective(np.array([[next_x[0]]])))
        
    plt.tight_layout()
    plt_path = r"D:\ML\optimization\BO_progression.png"
    plt.savefig(plt_path, dpi=300)
    print(f"Saved plot to {plt_path}")

if __name__ == "__main__":
    plot_bo()
