import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("multiple sine waves.csv")

# Define feature transformation function
def feature_transforms(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return np.sin(a).T

# Define model function
def model(x, w):
    f = feature_transforms(x, w[0])
    a = w[1][0] + np.dot(f.T, w[1][1:])
    return a.T
def cost_function(x, y, w):
    return np.sum((y - model(x, w))**2)

def gradient_descent(x, y, w_init, alpha, max_iter):
    w = w_init.copy()
    cost_history = []
    
    for i in range(max_iter):
        grad = np.zeros_like(w)
        grad[1] = np.dot(feature_transforms(x, w[0]).T, (model(x, w) - y))
        for j in range(2):
            grad[0][j] = np.dot((model(x, w) - y), w[1][j+1] * np.cos(w[0][j] + np.dot(x, w[1][:,j+1])))

        w -= alpha * grad
        cost_history.append(cost_function(x, y, w))
        
    return w, cost_history

# Set initial parameters
w_init = [np.zeros((2,)), np.zeros((2, 3))]

# Run gradient descent
w_opt, cost_history = gradient_descent(data[['x1', 'x2']].values, data['y'].values, w_init, 10**0, 2000)

# Print optimal parameters
print("Optimal parameters:")
print(w_opt)


plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function vs Iteration")
plt.show()


#########################
# Optimal parameters:
# [array([0.02305838, 0.07671347]), array([[-0.12950636, -1.40196825, -1.30424253],
#        [-0.13812598,  3.0200682 , -1.32796415]])]
#########################
