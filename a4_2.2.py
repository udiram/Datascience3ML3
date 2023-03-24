import numpy as np
import matplotlib.pyplot as plt

# Load dataset (assumed to be in variables X_train and y_train)
X_train = ...
y_train = ...

# Define model parameters
N, D = X_train.shape
C = np.max(y_train) + 1

# Initialize weights
w_init = 0.1 * np.random.randn(D + 1, C)

# Add bias term to inputs
X_train = np.hstack([X_train, np.ones((N, 1))])

# Define softmax regression function
def softmax_regression(w, X):
    scores = X @ w
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    return probs

def softmax_gradient(w, X, y):
    probs = softmax_regression(w, X)
    probs[np.arange(N), y] -= 1
    grad = X.T @ probs
    return grad

# Set hyperparameters
alpha = 0.001
max_its = 5

# Train model with mini-batch gradient descent (batch size 200)
batch_size = 200
w, cost_history_batch200 = mini_batch_gradient_descent(
    softmax_gradient, w_init, X_train, y_train, alpha, max_its, batch_size)

# Train model with mini-batch gradient descent (full batch)
batch_size = N
w, cost_history_fullbatch = mini_batch_gradient_descent(
    softmax_gradient, w_init, X_train, y_train, alpha, max_its, batch_size)

# Plot cost histories
plt.plot(range(1, max_its+1), cost_history_batch200, label="Batch size 200")
plt.plot(range(1, max_its+1), cost_history_fullbatch, label="Full batch")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost history for mini-batch gradient descent")
plt.legend()
plt.show()
