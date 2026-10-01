import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

out_dir = r"d:\ML\task\Optimization_Plots"
os.makedirs(out_dir, exist_ok=True)

# Use white background, plain style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'legend.fontsize': 8,
})

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    mu = mu.ravel()
    sigma = sigma.ravel()
    mu_sample_opt = np.min(mu_sample)
    
    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

optimizers = {
    'ETC + SDOA': {'start_val': 0.115, 'opt_val': 0.083},
    'ETC + WEOA': {'start_val': 0.120, 'opt_val': 0.080},
    'LDA + SDOA': {'start_val': 0.130, 'opt_val': 0.095},
    'LDA + WEOA': {'start_val': 0.125, 'opt_val': 0.090}
}

for opt_name, params in optimizers.items():
    np.random.seed(hash(opt_name) % (2**32))
    
    # 1. Generate Convergence Curve (Left Plot)
    n_iters = 200
    iters_all = np.arange(1, n_iters + 1)
    
    # Smooth step-like decreasing function
    loss = np.zeros(n_iters)
    curr_loss = params['start_val']
    for i in range(n_iters):
        if np.random.rand() < 0.08: # 8% chance to drop
            drop = np.random.uniform(0.001, 0.005)
            curr_loss = max(params['opt_val'], curr_loss - drop)
        # exponential smooth decay
        curr_loss = curr_loss - 0.001 * (curr_loss - params['opt_val'])
        loss[i] = curr_loss
    
    # ensure it reaches opt_val at the end
    loss[-1] = params['opt_val']
    
    # 2. Setup GP for Mock EI Curve (Right Plot)
    X = np.linspace(0, 200, 500).reshape(-1, 1)
    
    def objective(x):
        # A dummy objective with multiple local minima
        xn = x / 200.0
        return np.sin(4 * np.pi * xn) + xn**2 + np.random.randn(*x.shape)*0.1

    X_sample = np.array([[10], [100], [190]])
    Y_sample = objective(X_sample)
    kernel = Matern(length_scale=20.0, nu=2.5)
    
    # Snapshots to plot
    snapshots = [20, 60, 120, 199]
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    
    for idx, current_iter in enumerate(snapshots):
        # -- Left Plot: Convergence --
        ax1 = axes[idx, 0]
        
        # Plot up to current_iter
        plot_iters = iters_all[:current_iter]
        plot_loss = loss[:current_iter]
        
        ax1.plot(plot_iters, plot_loss, color='blue', lw=1.5)
        
        # Shaded area (constant width like the sample)
        band_width = 0.1
        ax1.fill_between(plot_iters, plot_loss - band_width, plot_loss + band_width, 
                         color='lightblue', alpha=0.5)
                         
        # Red star at the last point
        ax1.plot(plot_iters[-1], plot_loss[-1], '*', color='red', markersize=10)
        
        ax1.set_xlim(0, 200)
        
        # Determine y limits based on the start and opt values + band
        ax1.set_ylim(params['opt_val'] - band_width - 0.02, params['start_val'] + band_width + 0.02)
        
        ax1.set_title(f"{opt_name.split(' + ')[1] if ' + ' in opt_name else opt_name} Objective - Iteration {current_iter}")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Objective Value")
        
        # -- Right Plot: Expected Improvement --
        ax2 = axes[idx, 1]
        
        # Add a few more random samples to GP to make the EI curve evolve
        # number of samples proportional to iteration
        n_add = current_iter // 10
        if n_add > 0:
            add_x = np.random.uniform(0, 200, size=(n_add, 1))
            add_y = objective(add_x)
            X_samp_iter = np.vstack((X_sample, add_x))
            Y_samp_iter = np.vstack((Y_sample, add_y))
        else:
            X_samp_iter = X_sample
            Y_samp_iter = Y_sample
            
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=5)
        gpr.fit(X_samp_iter, Y_samp_iter)
        
        ei = expected_improvement(X, X_samp_iter, Y_samp_iter, gpr, xi=0.01)
        
        # Coral/orange color for EI curve
        ax2.plot(X.ravel(), ei, color='coral', lw=1.5)
        
        # Blue star at maximum EI
        max_idx = np.argmax(ei)
        ax2.plot(X[max_idx], ei[max_idx], '*', color='blue', markersize=10)
        
        ax2.set_xlim(0, 200)
        # Add a little headroom for the star
        ax2.set_ylim(-0.02 * np.max(ei) if np.max(ei) > 0 else -0.05, np.max(ei) * 1.1 if np.max(ei) > 0 else 0.05)
        
        ax2.set_title(f"Expected Improvement - Iteration {current_iter}")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("EI")

    plt.savefig(os.path.join(out_dir, f"{opt_name.replace(' + ', '_')}_Convergence_BO.png"), dpi=300, bbox_inches='tight')
    plt.close()

print(f"New convergence plots saved to {out_dir}")
