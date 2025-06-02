import numpy as np
import matplotlib.pyplot as plt

def lwr(x, X, y, tau):
    w = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2))
    W = np.diag(w)
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

# Data setup
np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)
X_bias = np.c_[np.ones(X.shape), X]

x_test = np.linspace(0, 2 * np.pi, 200)
x_test_bias = np.c_[np.ones(x_test.shape), x_test]
tau = 0.5

y_pred = np.array([lwr(xi, X_bias, y, tau) for xi in x_test_bias])

# Plot
plt.scatter(X, y, c='red', label='Training', alpha=0.7)
plt.plot(x_test, y_pred, c='blue', label=f'LWR (tau={tau})')
plt.legend(); plt.title("Locally Weighted Regression")
plt.grid(True); plt.show()
