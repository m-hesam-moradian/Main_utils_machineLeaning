import numpy as np
import matplotlib.pyplot as plt
from skopt import Optimizer
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.acquisition import gaussian_ei, gaussian_lcb
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import seaborn as sns
import pandas as pd


# 📁 Logging setup
class OptimizationLogger:
    def __init__(self):
        self.history = []

    def log(self, x, y):
        self.history.append({"x": x, "y": y})

    def to_dataframe(self):
        return pd.DataFrame(self.history)


# 🎯 Objective function with constraint
def constrained_objective(x):
    x1, x2 = x
    constraint = x1**2 + x2**2 <= 4
    if not constraint:
        return 1000  # Penalize infeasible region
    return x1 - x2 - np.sqrt(4 - x1**2 - x2**2)


# 📐 Define search space
space = [Real(-5.0, 5.0, name="x1"), Real(-5.0, 5.0, name="x2")]

# 🧪 Kernel and GP setup
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

# 🔍 Optimizer setup
opt = Optimizer(
    dimensions=space,
    base_estimator=gp,
    acq_func="EI",  # Can be "EI", "LCB", "PI"
    acq_optimizer="sampling",
    n_initial_points=10,
    random_state=42,
)

logger = OptimizationLogger()

# 🚀 Run optimization loop
n_calls = 60
for i in range(n_calls):
    next_x = opt.ask()
    next_y = constrained_objective(next_x)
    opt.tell(next_x, next_y)
    logger.log(next_x, next_y)

# 📊 Convert log to DataFrame
df = logger.to_dataframe()


# 📈 Plot convergence
def plot_convergence(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["y"], marker="o", linestyle="-", color="blue")
    plt.title("Convergence Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 🌄 Plot surrogate model landscape
def plot_surrogate_model(opt, resolution=100):
    x1 = np.linspace(-2, 2, resolution)
    x2 = np.linspace(-2, 2, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T

    mu, std = opt.base_estimator_.predict(X, return_std=True)
    Z = mu.reshape(X1.shape)

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X1, X2, Z, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.title("Surrogate Model Mean Prediction")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(*zip(*df["x"]), c="red", s=20, label="Evaluated Points")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 📉 Acquisition function visualization
def plot_acquisition(opt, acq_func=gaussian_ei, resolution=100):
    x1 = np.linspace(-2, 2, resolution)
    x2 = np.linspace(-2, 2, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T

    mu, std = opt.base_estimator_.predict(X, return_std=True)
    acq = acq_func(X, opt.base_estimator_, np.min(df["y"]))
    Z = acq.reshape(X1.shape)

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X1, X2, Z, levels=50, cmap="plasma")
    plt.colorbar(contour)
    plt.title("Acquisition Function (EI)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(*zip(*df["x"]), c="white", s=20, label="Evaluated Points")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 🖼️ Run plots
plot_convergence(df)
plot_surrogate_model(opt)
plot_acquisition(opt)

# 🏁 Final result
best_idx = np.argmin(df["y"])
best_x = df.iloc[best_idx]["x"]
best_y = df.iloc[best_idx]["y"]
print(f"Best parameters found: x = {best_x}, y = {best_y:.4f}")
