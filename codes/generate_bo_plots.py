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

plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
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
    'ETC + SDOA': {'x_label': 'Number of Estimators', 'bounds': (10, 1000), 'start_val': 0.17, 'opt_val': 0.05},
    'ETC + WEOA': {'x_label': 'Number of Estimators', 'bounds': (10, 1000), 'start_val': 0.17, 'opt_val': 0.04},
    'LDA + SDOA': {'x_label': 'Tolerance', 'bounds': (0.0001, 0.01), 'start_val': 0.18, 'opt_val': 0.08},
    'LDA + WEOA': {'x_label': 'Tolerance', 'bounds': (0.0001, 0.01), 'start_val': 0.18, 'opt_val': 0.07}
}

for opt_name, params in optimizers.items():
    bounds = params['bounds']
    X = np.linspace(bounds[0], bounds[1], 400).reshape(-1, 1)
    
    def objective(x):
        xn = (x - bounds[0]) / (bounds[1] - bounds[0])
        val = np.sin(4 * np.pi * xn) * (1 - xn) + xn**2
        val = val * (params['start_val'] - params['opt_val']) * 0.5 + params['start_val'] * 0.8
        return val
        
    Y_true = objective(X)
    np.random.seed(hash(opt_name) % (2**32))
    
    X_sample = np.array([bounds[0] + (bounds[1]-bounds[0])*0.1, bounds[0] + (bounds[1]-bounds[0])*0.9]).reshape(-1, 1)
    Y_sample = objective(X_sample)
    
    kernel = Matern(length_scale=(bounds[1]-bounds[0])*0.2, nu=2.5)
    
    fig, axes = plt.subplots(5, 2, figsize=(10, 13))
    fig.subplots_adjust(hspace=0.7, wspace=0.25)
    
    blue_c = '#348ABD'
    red_c = '#E24A33'
    
    for i in range(5):
        iteration = i + 2
        
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=10)
        gpr.fit(X_sample, Y_sample)
        
        mu, std = gpr.predict(X, return_std=True)
        ei = expected_improvement(X, X_sample, Y_sample, gpr, xi=0.001)
        
        # Left plot: GP
        ax1 = axes[i, 0]
        ax1.plot(X, mu, color=blue_c, lw=1.5)
        ax1.fill_between(X.ravel(), 
                         mu.ravel() - 1.96 * std, 
                         mu.ravel() + 1.96 * std, 
                         alpha=0.3, color=blue_c)
        
        ax1.plot(X_sample[:-1], Y_sample[:-1], 'o', color=blue_c, markersize=5)
        ax1.plot(X_sample[-1], Y_sample[-1], '*', color=red_c, markersize=10, label='Last step')
            
        # The user's image says "Gaussian Process \n after X iterations"
        # Since it's for their optimizers, I'll put the optimizer name.
        ax1.set_title(f"Gaussian Process ({opt_name.split(' + ')[1]})\nafter {iteration} iterations")
        ax1.set_ylabel("Error Rate\n(the smaller the better)")
        ax1.set_xlabel(params['x_label'])
        ax1.legend(loc='upper right', frameon=True)
        ax1.set_xlim(bounds[0], bounds[1])
        
        # Right plot: EI
        ax2 = axes[i, 1]
        ax2.plot(X.ravel(), ei.ravel(), color=red_c, lw=1.5)
        
        ei_flat = ei.ravel()
        next_idx = np.argmax(ei_flat)
        next_x = X.ravel()[next_idx]
        next_y = ei_flat[next_idx]
        
        ax2.plot(next_x, next_y, '*', color=blue_c, markersize=10, label='Next step')
        ax2.set_title(f"Expected Improvement\nafter {iteration} iterations")
        ax2.set_xlabel(params['x_label'])
        ax2.legend(loc='upper right' if i == 0 else 'upper left', frameon=True)
        ax2.set_xlim(bounds[0], bounds[1])
        
        X_sample = np.vstack((X_sample, next_x))
        Y_sample = np.vstack((Y_sample, objective(next_x)))
        
    plt.savefig(os.path.join(out_dir, f"{opt_name.replace(' + ', '_')}_BO_Process.png"), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Optimization plots saved to {out_dir}")
