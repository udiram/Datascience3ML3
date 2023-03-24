import numpy as np

# Read the data from CSV file
data = np.genfromtxt('2d_span_data_centered.csv', delimiter=',')

# Separate the features and target variable
X = data[:, :-1]
y = data[:, -1]

# Add bias term to features
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return J

# Define the gradient descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        theta = theta - (alpha/m) * np.dot(X.T, (h-y))
        J_history.append(cost_function(theta, X, y))
    return theta, J_history

# Initialize theta parameters
initial_theta = np.zeros(X.shape[1])

# Set hyperparameters
alpha = 0.01
num_iters = 1000

# Run gradient descent to get the optimal theta values
theta, J_history = gradient_descent(X, y, initial_theta, alpha, num_iters)

# Print the optimal theta values
print("Optimal theta values:", theta)

# Plot the cost function over iterations
import matplotlib.pyplot as plt
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function over iterations')
plt.show()
