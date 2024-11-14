import numpy as np
import matplotlib.pyplot as plt

# 1. Lennard-Jones-Like Potential
def lennard_jones(x, A, B, n, m):
    return A / x**n - B / x**m

# 2. Generalized Rational Potential
def rational_potential(x, A, B, x0, n, m):
    return A / x**n - B / (1 + (x / x0)**m)

# 3. Asymmetric Gaussian Well with Divergence
def gaussian_well_divergence(x, A, n, B, x0, sigma):
    return A / x**n - B * np.exp(-((x - x0)**2) / (2 * sigma**2))

# Generate x values
x = np.linspace(0.1, 10, 500)  # Avoid division by zero

# Parameters for plotting
A, B, n, m, x0, sigma = 10, 5, 12, 6, 5, 1

# Plotting the functions
plt.figure(figsize=(10, 6))
plt.plot(x, lennard_jones(x, A, B, n, m), label="Lennard-Jones-Like Potential")
plt.plot(x, rational_potential(x, A, B, x0, n, m), label="Generalized Rational Potential")
plt.plot(x, gaussian_well_divergence(x, A, n, B, x0, sigma), label="Gaussian Well + Divergence")
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Functions with Divergence, Negative Well, and Flattening")
plt.grid()
plt.show()
