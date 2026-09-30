import numpy as np
import matplotlib.pyplot as plt

def objective_function(x, y):
    # Create a 2D landscape with multiple optima, similar to the provided image
    # We use a combination of Gaussian-like peaks and a quadratic bowl
    term1 = -1.5 * np.exp(-((x - 0.5)**2 + y**2))
    term2 = -0.8 * np.exp(-((x + 1.5)**2 + y**2))
    bowl = 0.1 * (x**2 + y**2)
    # Add a slight diagonal/cross shape to the contour
    cross = 0.2 * np.cos(x * 1.5) * np.cos(y * 1.5)
    return term1 + term2 + bowl + cross

def generate_search_space_plot():
    # Grid for contour plot
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Grid Search ---
    ax_grid = axes[0]
    # Draw contour
    levels = np.linspace(np.min(Z), np.max(Z), 15)
    contour_grid = ax_grid.contour(X, Y, Z, levels=levels, linewidths=1.0, alpha=0.8, cmap='viridis_r')
    
    # Generate Grid points
    # We want roughly the same number of points (e.g., 7x7 = 49)
    grid_x = np.linspace(-2.8, 2.8, 7)
    grid_y = np.linspace(-2.8, 2.8, 7)
    GX, GY = np.meshgrid(grid_x, grid_y)
    ax_grid.scatter(GX, GY, color='red', s=30, label='Grid points', zorder=5)
    
    # Mark the optima roughly with a star or dot if desired, but we follow the image
    ax_grid.set_title("Grid Search", fontsize=14)
    ax_grid.set_xlabel("Parameter 1", fontsize=12)
    ax_grid.set_ylabel("Parameter 2", fontsize=12)
    ax_grid.set_xlim(-3, 3)
    ax_grid.set_ylim(-3, 3)
    ax_grid.legend(loc='upper right')

    # --- Random Search ---
    ax_random = axes[1]
    # Draw contour
    contour_random = ax_random.contour(X, Y, Z, levels=levels, linewidths=1.0, alpha=0.8, cmap='viridis_r')
    
    # Generate Random points (same number as grid = 49)
    np.random.seed(42)  # For reproducibility
    rand_x = np.random.uniform(-3, 3, 49)
    rand_y = np.random.uniform(-3, 3, 49)
    ax_random.scatter(rand_x, rand_y, color='red', s=30, label='Random points', zorder=5)
    
    ax_random.set_title("Random Search", fontsize=14)
    ax_random.set_xlabel("Parameter 1", fontsize=12)
    ax_random.set_ylabel("Parameter 2", fontsize=12)
    ax_random.set_xlim(-3, 3)
    ax_random.set_ylim(-3, 3)
    ax_random.legend(loc='upper right')

    plt.tight_layout()
    plot_path = r"D:\ML\optimization\search_space_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    generate_search_space_plot()
